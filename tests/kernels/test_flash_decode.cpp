#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <runtherder/kernels/attention.cuh>
#include <runtherder/kernels/flash_decode.cuh>
#include <runtherder/kernels/quantization.cuh>
#include "device_buffer.cuh"
#include <runtherder/device/check.cuh>

namespace {

using runtherder::device::DeviceBuffer;

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

// bf16 KV quantized to the cache format, device fp8 values plus their scales.
struct QuantizedKV {
    DeviceBuffer<__nv_fp8_e4m3> q;
    DeviceBuffer<float>         scale;
};

QuantizedKV quantize_kv(const std::vector<__nv_bfloat16>& x,
                        int num_rows, int head_dim) {
    QuantizedKV out;
    out.q     = DeviceBuffer<__nv_fp8_e4m3>(x.size());
    out.scale = DeviceBuffer<float>(static_cast<std::size_t>(num_rows));

    DeviceBuffer<__nv_bfloat16> x_dev(x.size());
    x_dev.copy_from_host(x.data());
    runtherder::kernels::quantize_kv_fp8(
        out.q.data(), out.scale.data(), x_dev.data(), num_rows, head_dim, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}

struct CompareResult {
    float max_abs_err;
    float cosine_similarity;
};

CompareResult compare_bf16(const std::vector<__nv_bfloat16>& a,
                           const std::vector<__nv_bfloat16>& b) {
    float  max_abs = 0.0f;
    double dot     = 0.0;
    double norm_a  = 0.0;
    double norm_b  = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float av   = __bfloat162float(a[i]);
        const float bv   = __bfloat162float(b[i]);
        const float diff = std::fabs(av - bv);
        if (diff > max_abs) {
            max_abs = diff;
        }
        dot    += static_cast<double>(av) * static_cast<double>(bv);
        norm_a += static_cast<double>(av) * static_cast<double>(av);
        norm_b += static_cast<double>(bv) * static_cast<double>(bv);
    }
    const double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    const float  cos   = (denom > 0.0) ? static_cast<float>(dot / denom) : 1.0f;
    return {max_abs, cos};
}

// Both flash kernels and the naive oracle dequantize the same fp8 values, so
// they compute the same attention and differ only by float reassociation and
// the bf16 output rounding.
constexpr float kAbsTol   = 3e-2f;
constexpr float kCosFloor = 0.999f;

void run_case(int cache_len, int num_q_heads, int num_kv_heads,
              int head_dim, int num_splits) {
    const int         n_keys  = cache_len + 1;
    const float       scale   = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const std::size_t q_count = static_cast<std::size_t>(num_q_heads) * head_dim;
    const std::size_t kv_count =
        static_cast<std::size_t>(n_keys) * num_kv_heads * head_dim;
    const int         kv_rows = n_keys * num_kv_heads;

    const auto q_host = make_bf16_random(q_count,  1.0f, 0xA11CEu);
    const auto k_host = make_bf16_random(kv_count, 1.0f, 0xB0Bu);
    const auto v_host = make_bf16_random(kv_count, 1.0f, 0xCAFEu);

    const QuantizedKV k = quantize_kv(k_host, kv_rows, head_dim);
    const QuantizedKV v = quantize_kv(v_host, kv_rows, head_dim);

    DeviceBuffer<__nv_bfloat16> q_dev(q_count);
    q_dev.copy_from_host(q_host.data());

    // Oracle: the naive kernel at the decode shape (n_new == 1).
    DeviceBuffer<__nv_bfloat16> out_naive(q_count);
    runtherder::kernels::attention_causal_bf16(
        out_naive.data(), q_dev.data(), k.q.data(), v.q.data(),
        k.scale.data(), v.scale.data(),
        /*n_new=*/1, cache_len, num_q_heads, num_kv_heads, head_dim, scale, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> naive_host(q_count);
    out_naive.copy_to_host(naive_host.data());

    // Single block flash decode.
    DeviceBuffer<__nv_bfloat16> out_single(q_count);
    runtherder::kernels::flash_decode_bf16(
        out_single.data(), q_dev.data(), k.q.data(), v.q.data(),
        k.scale.data(), v.scale.data(),
        cache_len, num_q_heads, num_kv_heads, head_dim, scale, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> single_host(q_count);
    out_single.copy_to_host(single_host.data());

    // Split flash decode with backend owned scratch.
    DeviceBuffer<float> partial(runtherder::kernels::flash_split_scratch_floats(
        num_q_heads, head_dim, num_splits));
    DeviceBuffer<__nv_bfloat16> out_split(q_count);
    runtherder::kernels::flash_decode_split_bf16(
        out_split.data(), q_dev.data(), k.q.data(), v.q.data(),
        k.scale.data(), v.scale.data(), partial.data(), num_splits,
        cache_len, num_q_heads, num_kv_heads, head_dim, scale, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> split_host(q_count);
    out_split.copy_to_host(split_host.data());

    const auto cs = compare_bf16(naive_host, single_host);
    EXPECT_LE(cs.max_abs_err, kAbsTol)
        << "single cache_len=" << cache_len << " splits=" << num_splits;
    EXPECT_GE(cs.cosine_similarity, kCosFloor)
        << "single cache_len=" << cache_len << " splits=" << num_splits;

    const auto sp = compare_bf16(naive_host, split_host);
    EXPECT_LE(sp.max_abs_err, kAbsTol)
        << "split cache_len=" << cache_len << " splits=" << num_splits;
    EXPECT_GE(sp.cosine_similarity, kCosFloor)
        << "split cache_len=" << cache_len << " splits=" << num_splits;
}

}  // namespace

// Deep cache, GQA, a real split count.
TEST(FlashDecodeFp8, MatchesNaiveOracle) {
    run_case(/*cache_len=*/300, /*num_q_heads=*/32, /*num_kv_heads=*/8,
             /*head_dim=*/128, /*num_splits=*/4);
}

// Shallow cache: single key, one split.
TEST(FlashDecodeFp8, ShallowCache) {
    run_case(/*cache_len=*/0, /*num_q_heads=*/32, /*num_kv_heads=*/8,
             /*head_dim=*/128, /*num_splits=*/1);
}

// More splits than keys, exercising the empty split sentinel in the reduce.
TEST(FlashDecodeFp8, EmptySplitSentinel) {
    run_case(/*cache_len=*/2, /*num_q_heads=*/8, /*num_kv_heads=*/2,
             /*head_dim=*/64, /*num_splits=*/8);
}

// Deep cache at the max split count.
TEST(FlashDecodeFp8, DeepSplit) {
    run_case(/*cache_len=*/1000, /*num_q_heads=*/32, /*num_kv_heads=*/8,
             /*head_dim=*/128, /*num_splits=*/8);
}
