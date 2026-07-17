#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <runtherder/kernels/attention.cuh>
#include <runtherder/kernels/quantization.cuh>
#include "device_buffer.cuh"
#include <runtherder/device/check.cuh>

namespace {

using runtherder::device::DeviceBuffer;

// A bf16 KV tensor quantized to the cache's E4M3 format. Holds the device fp8
// values and their per (token, kv_head) scales for the kernel, plus the host
// dequantized values so the reference sees exactly what the kernel reads.
struct QuantizedKV {
    DeviceBuffer<__nv_fp8_e4m3> q;
    DeviceBuffer<float>         scale;
    std::vector<__nv_bfloat16>  deq;
};

QuantizedKV quantize_kv(const std::vector<__nv_bfloat16>& x,
                        int num_rows, int head_dim) {
    QuantizedKV out;
    out.q     = DeviceBuffer<__nv_fp8_e4m3>(x.size());
    out.scale = DeviceBuffer<float>(static_cast<std::size_t>(num_rows));

    DeviceBuffer<__nv_bfloat16> x_dev(x.size());
    x_dev.copy_from_host(x.data());

    // Standalone slab, the write starts at row 0.
    constexpr int     kAtZero = 0;
    DeviceBuffer<int> at_pos(1);
    at_pos.copy_from_host(&kAtZero);

    runtherder::kernels::quantize_kv_fp8(
        out.q.data(), out.scale.data(), x_dev.data(), at_pos.data(),
        /*rows_per_pos=*/1, num_rows, head_dim, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_fp8_e4m3> q_host(x.size());
    std::vector<float>         s_host(static_cast<std::size_t>(num_rows));
    out.q.copy_to_host(q_host.data());
    out.scale.copy_to_host(s_host.data());

    out.deq.resize(x.size());
    for (int row = 0; row < num_rows; ++row) {
        for (int d = 0; d < head_dim; ++d) {
            const std::size_t i = static_cast<std::size_t>(row) * head_dim + d;
            out.deq[i] = __float2bfloat16(
                runtherder::kernels::dequantize_kv_fp8(q_host[i], s_host[row]));
        }
    }
    return out;
}

std::vector<__nv_bfloat16> make_bf16_random(std::size_t   count,
                                            float         scale,
                                            std::uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<__nv_bfloat16>            out(count);
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = __float2bfloat16(dist(rng));
    }
    return out;
}

std::vector<float> to_fp32(const std::vector<__nv_bfloat16>& src) {
    std::vector<float> out(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[i] = __bfloat162float(src[i]);
    }
    return out;
}

// FP64 reference, tighter than the device FP32 path. Mirrors the kernel math:
// per (token, head) causal softmax over keys 0..token, GQA head sharing, the
// weighted v sum normalized at the end.
std::vector<__nv_bfloat16> attention_cpu_reference(
    const std::vector<__nv_bfloat16>& q,
    const std::vector<__nv_bfloat16>& k,
    const std::vector<__nv_bfloat16>& v,
    int                               num_tokens,
    int                               num_q_heads,
    int                               num_kv_heads,
    int                               head_dim,
    float                             scale) {
    std::vector<__nv_bfloat16> out(
        static_cast<std::size_t>(num_tokens) * num_q_heads * head_dim);
    const int group = num_q_heads / num_kv_heads;

    for (int qt = 0; qt < num_tokens; ++qt) {
        for (int qh = 0; qh < num_q_heads; ++qh) {
            const int         kvh    = qh / group;
            const int         n_keys = qt + 1;
            const std::size_t q_off =
                (static_cast<std::size_t>(qt) * num_q_heads + qh) * head_dim;

            std::vector<double> scores(n_keys);
            double              m = -INFINITY;
            for (int j = 0; j < n_keys; ++j) {
                const std::size_t k_off =
                    (static_cast<std::size_t>(j) * num_kv_heads + kvh) * head_dim;
                double s = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    s += static_cast<double>(__bfloat162float(q[q_off + d])) *
                         static_cast<double>(__bfloat162float(k[k_off + d]));
                }
                s *= static_cast<double>(scale);
                scores[j] = s;
                m         = std::max(m, s);
            }

            double denom = 0.0;
            for (int j = 0; j < n_keys; ++j) {
                denom += std::exp(scores[j] - m);
            }

            const std::size_t o_off =
                (static_cast<std::size_t>(qt) * num_q_heads + qh) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (int j = 0; j < n_keys; ++j) {
                    const std::size_t v_off =
                        (static_cast<std::size_t>(j) * num_kv_heads + kvh) *
                        head_dim;
                    acc += std::exp(scores[j] - m) *
                           static_cast<double>(__bfloat162float(v[v_off + d]));
                }
                out[o_off + d] =
                    __float2bfloat16(static_cast<float>(acc / denom));
            }
        }
    }
    return out;
}

struct CompareResult {
    float max_abs_err;
    float cosine_similarity;
};

CompareResult compare_bf16(const std::vector<__nv_bfloat16>& a,
                           const std::vector<__nv_bfloat16>& b) {
    const std::vector<float> af = to_fp32(a);
    const std::vector<float> bf = to_fp32(b);

    float  max_abs = 0.0f;
    double dot     = 0.0;
    double norm_a  = 0.0;
    double norm_b  = 0.0;
    for (std::size_t i = 0; i < af.size(); ++i) {
        const float diff = std::fabs(af[i] - bf[i]);
        if (diff > max_abs) {
            max_abs = diff;
        }
        dot    += static_cast<double>(af[i]) * static_cast<double>(bf[i]);
        norm_a += static_cast<double>(af[i]) * static_cast<double>(af[i]);
        norm_b += static_cast<double>(bf[i]) * static_cast<double>(bf[i]);
    }
    const double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    const float  cos   = (denom > 0.0)
                             ? static_cast<float>(dot / denom)
                             : 0.0f;
    return {max_abs, cos};
}

// Reference is built from the same fp8 values the kernel dequantizes, so the
// residual is only the bf16 rounding of those dequantized values.
constexpr float kAbsTol   = 8e-2f;
constexpr float kCosFloor = 0.999f;

void run_case(int num_tokens,
              int num_q_heads,
              int num_kv_heads,
              int head_dim) {
    const float scale =
        1.0f / std::sqrt(static_cast<float>(head_dim));
    const std::size_t q_count =
        static_cast<std::size_t>(num_tokens) * num_q_heads  * head_dim;
    const std::size_t kv_count =
        static_cast<std::size_t>(num_tokens) * num_kv_heads * head_dim;

    auto q_host = make_bf16_random(q_count,  1.0f, 0xA11CEu);
    auto k_host = make_bf16_random(kv_count, 1.0f, 0xB0Bu);
    auto v_host = make_bf16_random(kv_count, 1.0f, 0xCAFEu);

    const int         kv_rows = num_tokens * num_kv_heads;
    const QuantizedKV k = quantize_kv(k_host, kv_rows, head_dim);
    const QuantizedKV v = quantize_kv(v_host, kv_rows, head_dim);

    auto expected = attention_cpu_reference(
        q_host, k.deq, v.deq,
        num_tokens, num_q_heads, num_kv_heads, head_dim, scale);

    DeviceBuffer<__nv_bfloat16> q_dev(q_count);
    DeviceBuffer<__nv_bfloat16> out_dev(q_count);
    q_dev.copy_from_host(q_host.data());

    constexpr int     kNoCache = 0;
    DeviceBuffer<int> cache_len_dev(1);
    cache_len_dev.copy_from_host(&kNoCache);

    runtherder::kernels::attention_causal_bf16(
        out_dev.data(), q_dev.data(), k.q.data(), v.q.data(),
        k.scale.data(), v.scale.data(),
        num_tokens, cache_len_dev.data(), num_q_heads, num_kv_heads, head_dim,
        scale, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out(q_count);
    out_dev.copy_to_host(out.data());

    const auto cmp = compare_bf16(expected, out);
    EXPECT_LE(cmp.max_abs_err, kAbsTol)
        << "num_tokens=" << num_tokens << " num_q_heads=" << num_q_heads
        << " num_kv_heads=" << num_kv_heads << " head_dim=" << head_dim;
    EXPECT_GE(cmp.cosine_similarity, kCosFloor)
        << "num_tokens=" << num_tokens << " num_q_heads=" << num_q_heads
        << " num_kv_heads=" << num_kv_heads << " head_dim=" << head_dim;
}

// Cached path: query only the last n_new tokens against a cache of
// cache_len + n_new keys. Equivalent to a full prefill keeping its tail rows.
void run_decode_case(int cache_len,
                     int n_new,
                     int num_q_heads,
                     int num_kv_heads,
                     int head_dim) {
    const int         total = cache_len + n_new;
    const float       scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const std::size_t q_full =
        static_cast<std::size_t>(total) * num_q_heads * head_dim;
    const std::size_t kv_full =
        static_cast<std::size_t>(total) * num_kv_heads * head_dim;

    auto q_host = make_bf16_random(q_full,  1.0f, 0xA11CEu);
    auto k_host = make_bf16_random(kv_full, 1.0f, 0xB0Bu);
    auto v_host = make_bf16_random(kv_full, 1.0f, 0xCAFEu);

    const int         kv_rows = total * num_kv_heads;
    const QuantizedKV k = quantize_kv(k_host, kv_rows, head_dim);
    const QuantizedKV v = quantize_kv(v_host, kv_rows, head_dim);

    auto expected_full = attention_cpu_reference(
        q_host, k.deq, v.deq,
        total, num_q_heads, num_kv_heads, head_dim, scale);

    const std::size_t q_new =
        static_cast<std::size_t>(n_new) * num_q_heads * head_dim;
    const std::size_t q_off =
        static_cast<std::size_t>(cache_len) * num_q_heads * head_dim;

    DeviceBuffer<__nv_bfloat16> q_dev(q_new);
    DeviceBuffer<__nv_bfloat16> out_dev(q_new);
    q_dev.copy_from_host(q_host.data() + q_off);

    DeviceBuffer<int> cache_len_dev(1);
    cache_len_dev.copy_from_host(&cache_len);

    runtherder::kernels::attention_causal_bf16(
        out_dev.data(), q_dev.data(), k.q.data(), v.q.data(),
        k.scale.data(), v.scale.data(),
        n_new, cache_len_dev.data(), num_q_heads, num_kv_heads, head_dim, scale,
        nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out(q_new);
    out_dev.copy_to_host(out.data());

    const std::vector<__nv_bfloat16> expected(
        expected_full.begin() + static_cast<std::ptrdiff_t>(q_off),
        expected_full.end());

    const auto cmp = compare_bf16(expected, out);
    EXPECT_LE(cmp.max_abs_err, kAbsTol)
        << "cache_len=" << cache_len << " n_new=" << n_new;
    EXPECT_GE(cmp.cosine_similarity, kCosFloor)
        << "cache_len=" << cache_len << " n_new=" << n_new;
}

}  // namespace

// Single token attends only to itself, output equals v[0] (denom == 1).
TEST(AttentionPrefillBF16, SingleToken) {
    run_case(/*num_tokens=*/1, /*num_q_heads=*/4, /*num_kv_heads=*/1,
             /*head_dim=*/64);
}

// MHA, no GQA sharing (group == 1).
TEST(AttentionPrefillBF16, MultiHeadNoGQA) {
    run_case(8, 8, 8, 64);
}

TEST(AttentionPrefillBF16, Qwen3Prefill) {
    run_case(16, 32, 8, 128);
}

TEST(AttentionPrefillBF16, MaxHeadDim) {
    run_case(8, 8, 2, 256);
}

TEST(AttentionPrefillBF16, LongerPrefill) {
    run_case(64, 32, 8, 128);
}

// Single new token against a non empty cache, the decode step shape.
TEST(AttentionCausalBF16, DecodeSingleToken) {
    run_decode_case(/*cache_len=*/15, /*n_new=*/1, /*num_q_heads=*/32,
                    /*num_kv_heads=*/8, /*head_dim=*/128);
}

// Chunked prefill shape, several new tokens onto an existing cache.
TEST(AttentionCausalBF16, DecodeChunk) {
    run_decode_case(/*cache_len=*/10, /*n_new=*/6, /*num_q_heads=*/8,
                    /*num_kv_heads=*/2, /*head_dim=*/64);
}
