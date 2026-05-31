#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <runtherder/kernels/matmul.cuh>
#include "device_buffer.cuh"
#include <runtherder/device/check.cuh>

namespace {

using runtherder::device::DeviceBuffer;
using runtherder::kernels::Matmul;

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

// FP64 reference for y = x · Wᵀ, weight stored [n,k] (HF Linear). Same BF16
// inputs as the kernel. Only slack is the FP32 path and BF16 output rounding.
std::vector<__nv_bfloat16> linear_cpu_reference(
    const std::vector<__nv_bfloat16>& x,
    const std::vector<__nv_bfloat16>& weight,
    int m, int n, int k) {
    std::vector<__nv_bfloat16> out(static_cast<std::size_t>(m) * n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int l = 0; l < k; ++l) {
                const double a = __bfloat162float(x[static_cast<std::size_t>(i) * k + l]);
                const double b = __bfloat162float(weight[static_cast<std::size_t>(j) * k + l]);
                acc += a * b;
            }
            out[static_cast<std::size_t>(i) * n + j] = __float2bfloat16(static_cast<float>(acc));
        }
    }
    return out;
}

struct CompareResult {
    float max_abs_err;
    float cosine_similarity;
    float max_ref_abs;
};

CompareResult compare_bf16(const std::vector<__nv_bfloat16>& ref,
                           const std::vector<__nv_bfloat16>& got) {
    float  max_abs = 0.0f;
    float  max_ref = 0.0f;
    double dot     = 0.0;
    double norm_a  = 0.0;
    double norm_b  = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const float rf   = __bfloat162float(ref[i]);
        const float gf   = __bfloat162float(got[i]);
        const float diff = std::fabs(rf - gf);
        if (diff > max_abs) {
            max_abs = diff;
        }
        if (std::fabs(rf) > max_ref) {
            max_ref = std::fabs(rf);
        }
        dot    += static_cast<double>(rf) * static_cast<double>(gf);
        norm_a += static_cast<double>(rf) * static_cast<double>(rf);
        norm_b += static_cast<double>(gf) * static_cast<double>(gf);
    }
    const double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    const float  cos   = (denom > 0.0)
                             ? static_cast<float>(dot / denom)
                             : 0.0f;
    return {max_abs, cos, max_ref};
}

// Output magnitude scales with k. The error bound is relative to the largest
// reference value rather than absolute.
constexpr float kRelTol   = 5e-2f;
constexpr float kCosFloor = 0.9999f;

void run_case(int m, int n, int k, std::uint32_t seed = 0x1234u) {
    const std::size_t xn = static_cast<std::size_t>(m) * k;
    const std::size_t wn = static_cast<std::size_t>(n) * k;
    const std::size_t yn = static_cast<std::size_t>(m) * n;

    auto x_host = make_bf16_random(xn, 1.0f, seed);
    auto w_host = make_bf16_random(wn, 1.0f, seed ^ 0x9999u);

    const auto expected = linear_cpu_reference(x_host, w_host, m, n, k);

    DeviceBuffer<__nv_bfloat16> x_dev(xn);
    DeviceBuffer<__nv_bfloat16> w_dev(wn);
    DeviceBuffer<__nv_bfloat16> y_dev(yn);
    x_dev.copy_from_host(x_host.data());
    w_dev.copy_from_host(w_host.data());

    Matmul mm;
    mm.linear_bf16(y_dev.data(), x_dev.data(), w_dev.data(), m, n, k, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out(yn);
    y_dev.copy_to_host(out.data());

    const auto  cmp = compare_bf16(expected, out);
    const float rel = cmp.max_abs_err / (cmp.max_ref_abs + 1e-6f);
    EXPECT_LE(rel, kRelTol)                     << "m=" << m << " n=" << n << " k=" << k;
    EXPECT_GE(cmp.cosine_similarity, kCosFloor) << "m=" << m << " n=" << n << " k=" << k;
}

}  // namespace

TEST(MatmulBF16, MatchesReferenceSmall) {
    run_case(/*m=*/2, /*n=*/3, /*k=*/4);
}

TEST(MatmulBF16, MatchesReferenceQwen3QProjDecode) {
    // Qwen 3 4B q_proj, single token (decode).
    run_case(/*m=*/1, /*n=*/4096, /*k=*/2560);
}

TEST(MatmulBF16, MatchesReferenceQwen3QProjPrefill) {
    // 8 prompt tokens through the same projection.
    run_case(/*m=*/8, /*n=*/4096, /*k=*/2560);
}

TEST(MatmulBF16, MatchesReferenceQwen3DownProj) {
    // down_proj, large k.
    run_case(/*m=*/4, /*n=*/2560, /*k=*/9728);
}

TEST(MatmulBF16, MatchesReferenceBatched) {
    run_case(/*m=*/32, /*n=*/4096, /*k=*/2560);
}
