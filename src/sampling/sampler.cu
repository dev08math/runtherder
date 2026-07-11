#include <runtherder/sampling/sampler.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/check.h>
#include <runtherder/device/check.cuh>
#include <runtherder/device/memory.cuh>

namespace runtherder::sampling {

namespace {

constexpr std::size_t kAlign = 256;

std::size_t align_up(std::size_t x) {
    return (x + kAlign - 1) & ~(kAlign - 1);
}

// Device pipeline scratch carved from the Sampler slab. The layout here must
// match scratch_total_bytes below.
struct Scratch {
    float* logits_f;     // [vocab]
    int*   idx;          // [vocab]
    float* keys_sorted;  // [vocab]
    int*   idx_sorted;   // [vocab]
    float* cumsum;       // [vocab]
    float* max_val;      // [1], the ArgMax extremum sink
    int*   token_out;    // [1]
    void*  cub_temp;     // [cub_temp_bytes]
};

// Single source of truth for the scratch layout, so the ctor sizing and the
// per call carve cannot drift. A null base tallies total bytes only. A real
// base with out set also fills the carved pointers.
std::size_t layout(std::byte* base, int vocab, std::size_t cub_temp_bytes, Scratch* out) {
    const std::size_t v   = static_cast<std::size_t>(vocab);
    std::size_t       off = 0;
    auto take = [&](std::size_t bytes) -> std::byte* {
        std::byte* const p = (base != nullptr) ? base + off : nullptr;
        off = align_up(off + bytes);
        return p;
    };
    std::byte* const p_logits      = take(v * sizeof(float));
    std::byte* const p_idx         = take(v * sizeof(int));
    std::byte* const p_keys_sorted = take(v * sizeof(float));
    std::byte* const p_idx_sorted  = take(v * sizeof(int));
    std::byte* const p_cumsum      = take(v * sizeof(float));
    std::byte* const p_max_val     = take(sizeof(float));
    std::byte* const p_token_out   = take(sizeof(int));
    std::byte* const p_cub_temp    = take(cub_temp_bytes);
    if (out != nullptr) {
        out->logits_f    = reinterpret_cast<float*>(p_logits);
        out->idx         = reinterpret_cast<int*>(p_idx);
        out->keys_sorted = reinterpret_cast<float*>(p_keys_sorted);
        out->idx_sorted  = reinterpret_cast<int*>(p_idx_sorted);
        out->cumsum      = reinterpret_cast<float*>(p_cumsum);
        out->max_val     = reinterpret_cast<float*>(p_max_val);
        out->token_out   = reinterpret_cast<int*>(p_token_out);
        out->cub_temp    = p_cub_temp;
    }
    return off;
}

__global__ void to_float_kernel(float* out, const __nv_bfloat16* in, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = static_cast<float>(in[i]);
    }
}

// Matches the host argmax exactly: bf16 widened to float is exact, so the max
// element is the same one the host loop finds.
int greedy_argmax_device(std::byte*           scratch,
                         int                  vocab,
                         std::size_t          cub_temp_bytes,
                         const __nv_bfloat16*  logits) {
    Scratch s;
    layout(scratch, vocab, cub_temp_bytes, &s);

    constexpr int block = 256;
    const int     grid  = (vocab + block - 1) / block;
    to_float_kernel<<<grid, block>>>(s.logits_f, logits, vocab);
    RUNTHERDER_CUDA_CHECK_LAST();

    std::size_t temp_bytes = cub_temp_bytes;
    RUNTHERDER_CUDA_CHECK(cub::DeviceReduce::ArgMax(
        s.cub_temp, temp_bytes, s.logits_f, s.max_val, s.token_out, vocab));

    int host_idx = 0;
    RUNTHERDER_CUDA_CHECK(cudaMemcpy(&host_idx, s.token_out, sizeof(int),
                                     cudaMemcpyDeviceToHost));
    return host_idx;
}

}  // namespace

Sampler::Sampler(int vocab_size, std::uint64_t seed)
    : vocab_size_(vocab_size), rng_(seed) {
    RUNTHERDER_CHECK(vocab_size >= 1, "Sampler needs vocab_size >= 1");

    // cub_temp_bytes_ is the max over every pipeline op, so one slab serves all.
    std::size_t argmax_b = 0;
    std::size_t max_b    = 0;
    std::size_t sum_b    = 0;
    std::size_t sort_b   = 0;
    std::size_t scan_b   = 0;
    RUNTHERDER_CUDA_CHECK(cub::DeviceReduce::ArgMax(
        nullptr, argmax_b, static_cast<float*>(nullptr),
        static_cast<float*>(nullptr), static_cast<int*>(nullptr), vocab_size));
    RUNTHERDER_CUDA_CHECK(cub::DeviceReduce::Max(
        nullptr, max_b, static_cast<float*>(nullptr),
        static_cast<float*>(nullptr), vocab_size));
    RUNTHERDER_CUDA_CHECK(cub::DeviceReduce::Sum(
        nullptr, sum_b, static_cast<float*>(nullptr),
        static_cast<float*>(nullptr), vocab_size));
    RUNTHERDER_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        nullptr, sort_b, static_cast<float*>(nullptr), static_cast<float*>(nullptr),
        static_cast<int*>(nullptr), static_cast<int*>(nullptr), vocab_size));
    RUNTHERDER_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, scan_b, static_cast<float*>(nullptr),
        static_cast<float*>(nullptr), vocab_size));

    cub_temp_bytes_ = std::max({argmax_b, max_b, sum_b, sort_b, scan_b});
    scratch_        = device::make_device_unique<std::byte>(
        layout(nullptr, vocab_size, cub_temp_bytes_, nullptr));
}

int Sampler::sample(const __nv_bfloat16* logits, const SamplingParams& params) {
    if (params.temperature == 0.0F) {
        return greedy_argmax_device(scratch_.get(), vocab_size_, cub_temp_bytes_, logits);
    }

    const std::size_t           n = static_cast<std::size_t>(vocab_size_);
    std::vector<__nv_bfloat16>  host(n);
    RUNTHERDER_CUDA_CHECK(cudaMemcpy(host.data(), logits,
                                     n * sizeof(__nv_bfloat16),
                                     cudaMemcpyDeviceToHost));

    std::vector<float> probs(n);
    const float        inv_t = 1.0F / params.temperature;
    float              max_logit = -INFINITY;
    for (std::size_t i = 0; i < n; ++i) {
        probs[i]  = static_cast<float>(host[i]) * inv_t;
        max_logit = std::max(max_logit, probs[i]);
    }

    float sum = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        probs[i] = std::exp(probs[i] - max_logit);
        sum += probs[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        probs[i] /= sum;
    }

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return probs[a] > probs[b]; });

    std::size_t keep = n;
    if (params.top_k > 0) {
        keep = std::min(keep, static_cast<std::size_t>(params.top_k));
    }
    if (params.top_p < 1.0F) {
        float       cum     = 0.0F;
        std::size_t nucleus = 0;
        for (; nucleus < keep; ++nucleus) {
            cum += probs[order[nucleus]];
            if (cum >= params.top_p) {
                ++nucleus;
                break;
            }
        }
        keep = nucleus;
    }

    float kept_sum = 0.0F;
    for (std::size_t r = 0; r < keep; ++r) {
        kept_sum += probs[order[r]];
    }

    // Draw over the unnormalized survivors. Scaling u by kept_sum is the
    // renormalization, no second pass over probs.
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    const float                           u = dist(rng_) * kept_sum;
    float                                 cum = 0.0F;
    for (std::size_t r = 0; r < keep; ++r) {
        cum += probs[order[r]];
        if (u < cum) {
            return order[r];
        }
    }
    return order[keep - 1];
}

}  // namespace runtherder::sampling
