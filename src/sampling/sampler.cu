#include <runtherder/sampling/sampler.h>

#include <algorithm>
#include <cstddef>
#include <random>

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

// Device pipeline scratch, carved from the Sampler slab by layout().
struct Scratch {
    float* logits_f;     // [vocab]
    int*   idx;          // [vocab]
    float* keys_sorted;  // [vocab]
    int*   idx_sorted;   // [vocab]
    float* prefix_mass;  // [vocab]
    float* max_val;      // [1], where Max/ArgMax writes the result
    int*   token_out;    // [1]
    void*  cub_temp;     // [cub_temp_bytes]
};

// A null base returns the total byte count. A real base with out set carves
// the slab into the pointer bundle.
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
    std::byte* const p_prefix_mass = take(v * sizeof(float));
    std::byte* const p_max_val     = take(sizeof(float));
    std::byte* const p_token_out   = take(sizeof(int));
    std::byte* const p_cub_temp    = take(cub_temp_bytes);
    if (out != nullptr) {
        out->logits_f    = reinterpret_cast<float*>(p_logits);
        out->idx         = reinterpret_cast<int*>(p_idx);
        out->keys_sorted = reinterpret_cast<float*>(p_keys_sorted);
        out->idx_sorted  = reinterpret_cast<int*>(p_idx_sorted);
        out->prefix_mass = reinterpret_cast<float*>(p_prefix_mass);
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

__global__ void scale_and_iota_kernel(float*               out,
                                      int*                 idx,
                                      const __nv_bfloat16* in,
                                      float                inv_t,
                                      int                  n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = static_cast<float>(in[i]) * inv_t;
        idx[i] = i;
    }
}

// No 1/sum normalization, it divides out of the draw downstream.
__global__ void exp_shift_kernel(float* probs, const float* max_val, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        probs[i] = expf(probs[i] - *max_val);
    }
}

// prefix_mass is nondecreasing, both cuts are binary searches over it. The
// total mass divides out of the draw, it is needed only for the top_p threshold.
__global__ void select_kernel(int*         token_out,
                              const float* prefix_mass,
                              const int*   idx_sorted,
                              int          vocab,
                              int          top_k,
                              float        top_p,
                              float        u01) {
    int keep = (top_k > 0 && top_k < vocab) ? top_k : vocab;

    if (top_p < 1.0F) {
        const float thresh = top_p * prefix_mass[vocab - 1];
        int         lo     = 0;
        int         hi     = keep;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (prefix_mass[mid] >= thresh) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        keep = (lo < keep) ? (lo + 1) : keep;
    }

    // Strict >: a zero mass token shares its neighbor's prefix and is never drawn.
    const float target = u01 * prefix_mass[keep - 1];
    int         lo     = 0;
    int         hi     = keep;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (prefix_mass[mid] > target) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    const int r = (lo < keep) ? lo : (keep - 1);
    *token_out  = idx_sorted[r];
}

// bf16 widened to float is exact, so ArgMax lands on the true maximum score.
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

// u01 is drawn host side, one rng advance per stochastic token.
int stochastic_sample_device(std::byte*            scratch,
                             int                   vocab,
                             std::size_t           cub_temp_bytes,
                             const __nv_bfloat16*  logits,
                             const SamplingParams& params,
                             float                 u01) {
    Scratch s;
    layout(scratch, vocab, cub_temp_bytes, &s);

    constexpr int block = 256;
    const int     grid  = (vocab + block - 1) / block;

    const float inv_t = 1.0F / params.temperature;
    scale_and_iota_kernel<<<grid, block>>>(s.logits_f, s.idx, logits, inv_t, vocab);
    RUNTHERDER_CUDA_CHECK_LAST();

    std::size_t temp_bytes = cub_temp_bytes;
    RUNTHERDER_CUDA_CHECK(cub::DeviceReduce::Max(
        s.cub_temp, temp_bytes, s.logits_f, s.max_val, vocab));

    exp_shift_kernel<<<grid, block>>>(s.logits_f, s.max_val, vocab);
    RUNTHERDER_CUDA_CHECK_LAST();

    // Full sort over the vocab, O(vocab), not a partial top_k selection.
    temp_bytes = cub_temp_bytes;
    RUNTHERDER_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        s.cub_temp, temp_bytes, s.logits_f, s.keys_sorted, s.idx, s.idx_sorted, vocab));

    temp_bytes = cub_temp_bytes;
    RUNTHERDER_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        s.cub_temp, temp_bytes, s.keys_sorted, s.prefix_mass, vocab));

    select_kernel<<<1, 1>>>(s.token_out, s.prefix_mass, s.idx_sorted, vocab,
                            params.top_k, params.top_p, u01);
    RUNTHERDER_CUDA_CHECK_LAST();

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
    std::size_t sort_b   = 0;
    std::size_t scan_b   = 0;
    RUNTHERDER_CUDA_CHECK(cub::DeviceReduce::ArgMax(
        nullptr, argmax_b, static_cast<float*>(nullptr),
        static_cast<float*>(nullptr), static_cast<int*>(nullptr), vocab_size));
    RUNTHERDER_CUDA_CHECK(cub::DeviceReduce::Max(
        nullptr, max_b, static_cast<float*>(nullptr),
        static_cast<float*>(nullptr), vocab_size));
    RUNTHERDER_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        nullptr, sort_b, static_cast<float*>(nullptr), static_cast<float*>(nullptr),
        static_cast<int*>(nullptr), static_cast<int*>(nullptr), vocab_size));
    RUNTHERDER_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, scan_b, static_cast<float*>(nullptr),
        static_cast<float*>(nullptr), vocab_size));

    cub_temp_bytes_ = std::max({argmax_b, max_b, sort_b, scan_b});
    scratch_        = device::make_device_unique<std::byte>(
        layout(nullptr, vocab_size, cub_temp_bytes_, nullptr));
}

int Sampler::sample(const __nv_bfloat16* logits, const SamplingParams& params) {
    if (params.temperature == 0.0F) {
        return greedy_argmax_device(scratch_.get(), vocab_size_, cub_temp_bytes_, logits);
    }

    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    const float                           u01 = dist(rng_);
    return stochastic_sample_device(scratch_.get(), vocab_size_, cub_temp_bytes_,
                                    logits, params, u01);
}

}  // namespace runtherder::sampling
