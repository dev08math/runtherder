#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <runtherder/kernels/activation.cuh>
#include <runtherder/runtime/device_buffer.cuh>
#include <runtherder/runtime/error.cuh>

namespace {

using runtherder::runtime::DeviceBuffer;

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

// CPU reference: BF16 storage, FP64 silu + multiply. Tighter than the
// device path (FP32 + __expf); kernel must still pass within BF16-storage
// tolerance.
std::vector<__nv_bfloat16> swiglu_cpu_reference(
    const std::vector<__nv_bfloat16>& gate,
    const std::vector<__nv_bfloat16>& up) {
    std::vector<__nv_bfloat16> out(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const double g = __bfloat162float(gate[i]);
        const double u = __bfloat162float(up[i]);
        const double s = 1.0 / (1.0 + std::exp(-g));
        out[i]         = __float2bfloat16(static_cast<float>(g * s * u));
    }
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
        const float af   = __bfloat162float(a[i]);
        const float bf   = __bfloat162float(b[i]);
        const float diff = std::fabs(af - bf);
        if (diff > max_abs) {
            max_abs = diff;
        }
        dot    += static_cast<double>(af) * static_cast<double>(bf);
        norm_a += static_cast<double>(af) * static_cast<double>(af);
        norm_b += static_cast<double>(bf) * static_cast<double>(bf);
    }
    const double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    const float  cos   = (denom > 0.0)
                             ? static_cast<float>(dot / denom)
                             : 0.0f;
    return {max_abs, cos};
}

constexpr float kAbsTol   = 5e-2f;
constexpr float kCosFloor = 0.9999f;

void run_case(int n) {
    const std::size_t count = static_cast<std::size_t>(n);

    auto gate_host = make_bf16_random(count, 2.0f, 0xC0FFEEu);
    auto up_host   = make_bf16_random(count, 2.0f, 0xBADBEEFu);

    const auto expected = swiglu_cpu_reference(gate_host, up_host);

    DeviceBuffer<__nv_bfloat16> gate_dev(count);
    DeviceBuffer<__nv_bfloat16> up_dev(count);
    DeviceBuffer<__nv_bfloat16> y_dev(count);
    gate_dev.copy_from_host(gate_host.data());
    up_dev.copy_from_host(up_host.data());

    runtherder::kernels::swiglu_bf16_forward(
        y_dev.data(), gate_dev.data(), up_dev.data(), n, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out(count);
    y_dev.copy_to_host(out.data());

    const auto cmp = compare_bf16(expected, out);
    EXPECT_LE(cmp.max_abs_err, kAbsTol)         << "n=" << n;
    EXPECT_GE(cmp.cosine_similarity, kCosFloor) << "n=" << n;
}

}  // namespace

TEST(SwiGLUBF16, MatchesReferenceSmall) {
    run_case(/*n=*/128);
}

TEST(SwiGLUBF16, MatchesReferenceQwen3SingleToken) {
    // Qwen 3 4B intermediate_dim = 9728, one token.
    run_case(9728);
}

TEST(SwiGLUBF16, MatchesReferenceQwen3Prefill) {
    // 8 tokens x 9728.
    run_case(8 * 9728);
}

TEST(SwiGLUBF16, MatchesReferenceLargeBatched) {
    // 32 tokens x 9728.
    run_case(32 * 9728);
}

TEST(SwiGLUBF16, AliasingYEqualsGate) {
    const int         n     = 4 * 9728;
    const std::size_t count = static_cast<std::size_t>(n);

    auto gate_host = make_bf16_random(count, 2.0f, 0x12345u);
    auto up_host   = make_bf16_random(count, 2.0f, 0x67890u);

    const auto expected = swiglu_cpu_reference(gate_host, up_host);

    DeviceBuffer<__nv_bfloat16> gate_dev(count);
    DeviceBuffer<__nv_bfloat16> up_dev(count);
    gate_dev.copy_from_host(gate_host.data());
    up_dev.copy_from_host(up_host.data());

    // y aliases gate: write back into gate buffer.
    runtherder::kernels::swiglu_bf16_forward(
        gate_dev.data(), gate_dev.data(), up_dev.data(), n, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out(count);
    gate_dev.copy_to_host(out.data());

    const auto cmp = compare_bf16(expected, out);
    EXPECT_LE(cmp.max_abs_err, kAbsTol);
    EXPECT_GE(cmp.cosine_similarity, kCosFloor);
}
