#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <runtherder/kernels/attention.cuh>
#include <runtherder/kernels/flash_prefill.cuh>
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

    constexpr int     kAtZero = 0;
    DeviceBuffer<int> at_pos(1);
    at_pos.copy_from_host(&kAtZero);

    runtherder::kernels::quantize_kv_fp8(
        out.q.data(), out.scale.data(), x_dev.data(), at_pos.data(),
        /*rows_per_pos=*/1, num_rows, head_dim, nullptr);
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

// The flash kernel feeds fp16 mma inputs where the naive oracle stays in fp32,
// so the two differ by fp16 rounding on top of the bf16 output rounding.
constexpr float kAbsTol   = 5e-2f;
constexpr float kCosFloor = 0.998f;

void run_case(int n_new, int cache_len, int num_q_heads, int num_kv_heads,
              int head_dim) {
    const int         n_keys  = cache_len + n_new;
    const float       scale   = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const std::size_t q_count =
        static_cast<std::size_t>(n_new) * num_q_heads * head_dim;
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

    DeviceBuffer<int> cache_len_dev(1);
    cache_len_dev.copy_from_host(&cache_len);

    DeviceBuffer<__nv_bfloat16> out_naive(q_count);
    runtherder::kernels::attention_causal_bf16(
        out_naive.data(), q_dev.data(), k.q.data(), v.q.data(),
        k.scale.data(), v.scale.data(), n_new, cache_len_dev.data(),
        num_q_heads, num_kv_heads, head_dim, scale, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> naive_host(q_count);
    out_naive.copy_to_host(naive_host.data());

    DeviceBuffer<__nv_bfloat16> out_flash(q_count);
    runtherder::kernels::flash_prefill_bf16(
        out_flash.data(), q_dev.data(), k.q.data(), v.q.data(),
        k.scale.data(), v.scale.data(), n_new, cache_len_dev.data(),
        num_q_heads, num_kv_heads, head_dim, scale, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> flash_host(q_count);
    out_flash.copy_to_host(flash_host.data());

    const auto r = compare_bf16(naive_host, flash_host);
    EXPECT_LE(r.max_abs_err, kAbsTol)
        << "n_new=" << n_new << " cache_len=" << cache_len;
    EXPECT_GE(r.cosine_similarity, kCosFloor)
        << "n_new=" << n_new << " cache_len=" << cache_len;
}

}  // namespace

// One full query tile, empty cache, GQA.
TEST(FlashPrefillFp8, OneTile) {
    run_case(/*n_new=*/16, /*cache_len=*/0, /*num_q_heads=*/32,
             /*num_kv_heads=*/8, /*head_dim=*/128);
}

// Partial last query tile and partial last key tile (20 = 16 + 4).
TEST(FlashPrefillFp8, PartialTiles) {
    run_case(/*n_new=*/20, /*cache_len=*/0, /*num_q_heads=*/32,
             /*num_kv_heads=*/8, /*head_dim=*/128);
}

// Non zero cache_len shifts the causal diagonal, keys 18 span two tiles.
TEST(FlashPrefillFp8, ChunkedPrefill) {
    run_case(/*n_new=*/8, /*cache_len=*/10, /*num_q_heads=*/32,
             /*num_kv_heads=*/8, /*head_dim=*/128);
}

// Several query and key tiles, GQA group of 3.
TEST(FlashPrefillFp8, MultiTile) {
    run_case(/*n_new=*/64, /*cache_len=*/0, /*num_q_heads=*/24,
             /*num_kv_heads=*/8, /*head_dim=*/128);
}

// Tiny sequence, heavy causal masking, GQA group of 4.
TEST(FlashPrefillFp8, ShortSequence) {
    run_case(/*n_new=*/3, /*cache_len=*/0, /*num_q_heads=*/8,
             /*num_kv_heads=*/2, /*head_dim=*/128);
}
