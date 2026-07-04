#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <runtherder/kernels/quantization.cuh>
#include "device_buffer.cuh"
#include <runtherder/device/check.cuh>

namespace {

using runtherder::device::DeviceBuffer;
using runtherder::kernels::dequantize_kv_fp8;
using runtherder::kernels::quantize_kv_fp8;

constexpr float kFp8E4M3Max = 448.0f;

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

struct KvRef {
    std::vector<__nv_fp8_e4m3> q;
    std::vector<float>         scale;
};

KvRef quantize_kv_reference(const std::vector<__nv_bfloat16>& x,
                            int                               num_rows,
                            int                               head_dim) {
    KvRef ref;
    ref.q.resize(static_cast<std::size_t>(num_rows) * head_dim);
    ref.scale.resize(static_cast<std::size_t>(num_rows));
    for (int row = 0; row < num_rows; ++row) {
        float amax = 0.0f;
        for (int j = 0; j < head_dim; ++j) {
            const float v = __bfloat162float(x[row * head_dim + j]);
            amax          = std::fmax(amax, std::fabs(v));
        }
        const float inv = (amax > 0.0f) ? (kFp8E4M3Max / amax) : 0.0f;
        ref.scale[row]  = amax / kFp8E4M3Max;
        for (int j = 0; j < head_dim; ++j) {
            const float v             = __bfloat162float(x[row * head_dim + j]);
            ref.q[row * head_dim + j] = __nv_fp8_e4m3(v * inv);
        }
    }
    return ref;
}

void run_case(int num_rows, int head_dim, std::uint32_t seed) {
    const auto x_host = make_bf16_random(
        static_cast<std::size_t>(num_rows) * head_dim, 1.0f, seed);

    const auto ref = quantize_kv_reference(x_host, num_rows, head_dim);

    DeviceBuffer<__nv_bfloat16> x_dev(x_host.size());
    DeviceBuffer<__nv_fp8_e4m3> q_dev(x_host.size());
    DeviceBuffer<float>         s_dev(static_cast<std::size_t>(num_rows));
    x_dev.copy_from_host(x_host.data());

    quantize_kv_fp8(q_dev.data(), s_dev.data(), x_dev.data(),
                    num_rows, head_dim, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_fp8_e4m3> q_host(x_host.size());
    std::vector<float>         s_host(static_cast<std::size_t>(num_rows));
    q_dev.copy_to_host(q_host.data());
    s_dev.copy_to_host(s_host.data());

    // Scale is a max reduction, so device and host agree to the bit.
    for (int row = 0; row < num_rows; ++row) {
        EXPECT_FLOAT_EQ(s_host[row], ref.scale[row]) << "row=" << row;
    }

    double dot = 0.0;
    double na  = 0.0;
    double nb  = 0.0;
    float  max_rel = 0.0f;
    for (int row = 0; row < num_rows; ++row) {
        const float amax_row = ref.scale[row] * kFp8E4M3Max;
        for (int j = 0; j < head_dim; ++j) {
            const std::size_t i    = static_cast<std::size_t>(row) * head_dim + j;
            const float       orig = __bfloat162float(x_host[i]);
            const float       gpu  = dequantize_kv_fp8(q_host[i], s_host[row]);
            const float       cpu  = dequantize_kv_fp8(ref.q[i], ref.scale[row]);

            EXPECT_NEAR(gpu, cpu, 1e-4f) << "row=" << row << " j=" << j;

            dot += static_cast<double>(orig) * static_cast<double>(gpu);
            na  += static_cast<double>(orig) * static_cast<double>(orig);
            nb  += static_cast<double>(gpu) * static_cast<double>(gpu);

            if (std::fabs(orig) > 0.1f * amax_row) {
                const float rel = std::fabs(gpu - orig) / std::fabs(orig);
                max_rel         = std::fmax(max_rel, rel);
            }
        }
    }

    const double denom = std::sqrt(na) * std::sqrt(nb);
    const float  cos   = (denom > 0.0) ? static_cast<float>(dot / denom) : 1.0f;
    EXPECT_GE(cos, 0.998f) << "shape=(" << num_rows << "," << head_dim << ")";
    EXPECT_LE(max_rel, 0.08f) << "shape=(" << num_rows << "," << head_dim << ")";
}

}  // namespace

TEST(QuantizeKvFp8, RoundTripHeadDim128) {
    run_case(24, 128, 0xC0FFEEu);
}

TEST(QuantizeKvFp8, RoundTripHeadDim256) {
    run_case(8, 256, 0xBADBEEFu);
}

TEST(QuantizeKvFp8, RoundTripBatched) {
    run_case(512, 128, 0xFEEDFACEu);
}

TEST(QuantizeKvFp8, ZeroRowMapsToZero) {
    constexpr int num_rows = 2;
    constexpr int head_dim = 128;

    const std::vector<__nv_bfloat16> x_host(
        static_cast<std::size_t>(num_rows) * head_dim, __float2bfloat16(0.0f));

    DeviceBuffer<__nv_bfloat16> x_dev(x_host.size());
    DeviceBuffer<__nv_fp8_e4m3> q_dev(x_host.size());
    DeviceBuffer<float>         s_dev(static_cast<std::size_t>(num_rows));
    x_dev.copy_from_host(x_host.data());

    quantize_kv_fp8(q_dev.data(), s_dev.data(), x_dev.data(),
                    num_rows, head_dim, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_fp8_e4m3> q_host(x_host.size());
    std::vector<float>         s_host(static_cast<std::size_t>(num_rows));
    q_dev.copy_to_host(q_host.data());
    s_dev.copy_to_host(s_host.data());

    for (int row = 0; row < num_rows; ++row) {
        EXPECT_FLOAT_EQ(s_host[row], 0.0f) << "row=" << row;
    }
    for (std::size_t i = 0; i < q_host.size(); ++i) {
        const float deq = dequantize_kv_fp8(q_host[i], s_host[i / head_dim]);
        EXPECT_FLOAT_EQ(deq, 0.0f) << "i=" << i;
    }
}
