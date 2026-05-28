#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <runtherder/kernels/rmsnorm.cuh>
#include "device_buffer.cuh"
#include <runtherder/device/check.cuh>

namespace {

using runtherder::device::DeviceBuffer;

std::vector<__nv_bfloat16> make_bf16_random(std::size_t count,
                                            float        scale,
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

// CPU FP32 reference for plain RMSNorm. Reads BF16, computes in FP32,
// stores BF16 (matching the kernel's storage path).
std::vector<__nv_bfloat16> rmsnorm_cpu_reference(
    const std::vector<__nv_bfloat16>& x,
    const std::vector<__nv_bfloat16>& g,
    int                                num_tokens,
    int                                hidden_dim,
    float                              eps) {
    std::vector<__nv_bfloat16> y(static_cast<std::size_t>(num_tokens) * hidden_dim);
    for (int row = 0; row < num_tokens; ++row) {
        double ss = 0.0;
        for (int j = 0; j < hidden_dim; ++j) {
            const float xv = __bfloat162float(x[row * hidden_dim + j]);
            ss += static_cast<double>(xv) * static_cast<double>(xv);
        }
        const float rrms = 1.0f / std::sqrt(
            static_cast<float>(ss / hidden_dim) + eps);
        for (int j = 0; j < hidden_dim; ++j) {
            const float xv = __bfloat162float(x[row * hidden_dim + j]);
            const float gv = __bfloat162float(g[j]);
            y[row * hidden_dim + j] = __float2bfloat16(xv * rrms * gv);
        }
    }
    return y;
}

struct FusedReference {
    std::vector<__nv_bfloat16> h_out;
    std::vector<__nv_bfloat16> y;
};

FusedReference rmsnorm_add_cpu_reference(
    const std::vector<__nv_bfloat16>& x,
    const std::vector<__nv_bfloat16>& residual,
    const std::vector<__nv_bfloat16>& g,
    int                                num_tokens,
    int                                hidden_dim,
    float                              eps) {
    FusedReference ref;
    ref.h_out.resize(static_cast<std::size_t>(num_tokens) * hidden_dim);
    ref.y.resize(static_cast<std::size_t>(num_tokens) * hidden_dim);

    for (int row = 0; row < num_tokens; ++row) {
        std::vector<float> h(hidden_dim);
        double ss = 0.0;
        for (int j = 0; j < hidden_dim; ++j) {
            const float xv = __bfloat162float(x[row * hidden_dim + j]);
            const float rv = __bfloat162float(residual[row * hidden_dim + j]);
            h[j] = xv + rv;
            ref.h_out[row * hidden_dim + j] = __float2bfloat16(h[j]);
            ss += static_cast<double>(h[j]) * static_cast<double>(h[j]);
        }
        const float rrms = 1.0f / std::sqrt(
            static_cast<float>(ss / hidden_dim) + eps);
        for (int j = 0; j < hidden_dim; ++j) {
            const float gv = __bfloat162float(g[j]);
            ref.y[row * hidden_dim + j] = __float2bfloat16(h[j] * rrms * gv);
        }
    }
    return ref;
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

constexpr float kAbsTol   = 5e-2f;     // BF16 has ~7-bit mantissa
constexpr float kCosFloor = 0.9999f;

void run_plain_case(int num_tokens, int hidden_dim) {
    constexpr float eps = 1e-5f;

    const auto x_host = make_bf16_random(
        static_cast<std::size_t>(num_tokens) * hidden_dim, 1.0f, 0xC0FFEEu);
    const auto g_host = make_bf16_random(
        static_cast<std::size_t>(hidden_dim), 0.5f, 0xBADBEEFu);

    const auto expected = rmsnorm_cpu_reference(
        x_host, g_host, num_tokens, hidden_dim, eps);

    DeviceBuffer<__nv_bfloat16> x_dev(x_host.size());
    DeviceBuffer<__nv_bfloat16> g_dev(g_host.size());
    DeviceBuffer<__nv_bfloat16> y_dev(x_host.size());
    x_dev.copy_from_host(x_host.data());
    g_dev.copy_from_host(g_host.data());

    runtherder::kernels::rmsnorm_bf16_forward(
        y_dev.data(), x_dev.data(), g_dev.data(),
        num_tokens, hidden_dim, eps, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> y_host(x_host.size());
    y_dev.copy_to_host(y_host.data());

    const auto cmp = compare_bf16(expected, y_host);
    EXPECT_LE(cmp.max_abs_err, kAbsTol)
        << "shape=(" << num_tokens << "," << hidden_dim << ")";
    EXPECT_GE(cmp.cosine_similarity, kCosFloor)
        << "shape=(" << num_tokens << "," << hidden_dim << ")";
}

void run_fused_case(int num_tokens, int hidden_dim) {
    constexpr float eps = 1e-5f;

    const auto x_host = make_bf16_random(
        static_cast<std::size_t>(num_tokens) * hidden_dim, 1.0f, 0xFEEDFACEu);
    const auto r_host = make_bf16_random(
        static_cast<std::size_t>(num_tokens) * hidden_dim, 1.0f, 0xDEADBEEFu);
    const auto g_host = make_bf16_random(
        static_cast<std::size_t>(hidden_dim), 0.5f, 0xCAFEBABEu);

    const auto ref = rmsnorm_add_cpu_reference(
        x_host, r_host, g_host, num_tokens, hidden_dim, eps);

    DeviceBuffer<__nv_bfloat16> x_dev(x_host.size());
    DeviceBuffer<__nv_bfloat16> r_dev(r_host.size());
    DeviceBuffer<__nv_bfloat16> g_dev(g_host.size());
    DeviceBuffer<__nv_bfloat16> h_dev(x_host.size());
    DeviceBuffer<__nv_bfloat16> y_dev(x_host.size());
    x_dev.copy_from_host(x_host.data());
    r_dev.copy_from_host(r_host.data());
    g_dev.copy_from_host(g_host.data());

    runtherder::kernels::rmsnorm_add_bf16_forward(
        y_dev.data(), h_dev.data(), x_dev.data(), r_dev.data(), g_dev.data(),
        num_tokens, hidden_dim, eps, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_host(x_host.size());
    std::vector<__nv_bfloat16> y_host(x_host.size());
    h_dev.copy_to_host(h_host.data());
    y_dev.copy_to_host(y_host.data());

    const auto h_cmp = compare_bf16(ref.h_out, h_host);
    EXPECT_LE(h_cmp.max_abs_err, kAbsTol)
        << "h_out shape=(" << num_tokens << "," << hidden_dim << ")";
    EXPECT_GE(h_cmp.cosine_similarity, kCosFloor)
        << "h_out shape=(" << num_tokens << "," << hidden_dim << ")";

    const auto y_cmp = compare_bf16(ref.y, y_host);
    EXPECT_LE(y_cmp.max_abs_err, kAbsTol)
        << "y shape=(" << num_tokens << "," << hidden_dim << ")";
    EXPECT_GE(y_cmp.cosine_similarity, kCosFloor)
        << "y shape=(" << num_tokens << "," << hidden_dim << ")";
}

}  // namespace

TEST(RMSNormBF16, MatchesReferenceSmall) {
    run_plain_case(1, 128);
}

TEST(RMSNormBF16, MatchesReferenceQwen3Hidden) {
    run_plain_case(4, 2560);
}

TEST(RMSNormBF16, MatchesReferenceMaxHidden) {
    run_plain_case(2, 16384);
}

TEST(RMSNormBF16, MatchesReferenceBatched) {
    run_plain_case(32, 2560);
}

TEST(RMSNormAddBF16, MatchesReferenceSmall) {
    run_fused_case(1, 128);
}

TEST(RMSNormAddBF16, MatchesReferenceQwen3Hidden) {
    run_fused_case(4, 2560);
}

TEST(RMSNormAddBF16, MatchesReferenceMaxHidden) {
    run_fused_case(2, 16384);
}

TEST(RMSNormAddBF16, MatchesReferenceBatched) {
    run_fused_case(32, 2560);
}
