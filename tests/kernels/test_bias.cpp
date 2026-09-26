#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <runtherder/kernels/bias.cuh>
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

std::vector<__nv_bfloat16> add_rows_reference(const std::vector<__nv_bfloat16>& x,
                                              const std::vector<__nv_bfloat16>& bias,
                                              int                               n,
                                              int                               dim) {
    std::vector<__nv_bfloat16> out(x.size());
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < dim; ++c) {
            const std::size_t i = static_cast<std::size_t>(r) * dim + c;
            out[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(bias[c]));
        }
    }
    return out;
}

void expect_bit_identical(const std::vector<__nv_bfloat16>& expected,
                          const std::vector<__nv_bfloat16>& actual,
                          const char*                       which) {
    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        std::uint16_t e = 0;
        std::uint16_t a = 0;
        std::memcpy(&e, &expected[i], sizeof(e));
        std::memcpy(&a, &actual[i], sizeof(a));
        ASSERT_EQ(e, a) << which << " differs at element " << i;
    }
}

void run_case(int n, int q_dim, int kv_dim) {
    const std::size_t q_count  = static_cast<std::size_t>(n) * q_dim;
    const std::size_t kv_count = static_cast<std::size_t>(n) * kv_dim;

    const auto q_host  = make_bf16_random(q_count, 4.0f, 0x1111u);
    const auto k_host  = make_bf16_random(kv_count, 4.0f, 0x2222u);
    const auto v_host  = make_bf16_random(kv_count, 4.0f, 0x3333u);
    const auto qb_host = make_bf16_random(static_cast<std::size_t>(q_dim), 2.0f, 0x4444u);
    const auto kb_host = make_bf16_random(static_cast<std::size_t>(kv_dim), 2.0f, 0x5555u);
    const auto vb_host = make_bf16_random(static_cast<std::size_t>(kv_dim), 2.0f, 0x6666u);

    DeviceBuffer<__nv_bfloat16> q_dev(q_count);
    DeviceBuffer<__nv_bfloat16> k_dev(kv_count);
    DeviceBuffer<__nv_bfloat16> v_dev(kv_count);
    DeviceBuffer<__nv_bfloat16> qb_dev(static_cast<std::size_t>(q_dim));
    DeviceBuffer<__nv_bfloat16> kb_dev(static_cast<std::size_t>(kv_dim));
    DeviceBuffer<__nv_bfloat16> vb_dev(static_cast<std::size_t>(kv_dim));
    q_dev.copy_from_host(q_host.data());
    k_dev.copy_from_host(k_host.data());
    v_dev.copy_from_host(v_host.data());
    qb_dev.copy_from_host(qb_host.data());
    kb_dev.copy_from_host(kb_host.data());
    vb_dev.copy_from_host(vb_host.data());

    runtherder::kernels::qkv_bias_add_bf16_forward(
        q_dev.data(), k_dev.data(), v_dev.data(),
        qb_dev.data(), kb_dev.data(), vb_dev.data(),
        n, q_dim, kv_dim, nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> q_out(q_count);
    std::vector<__nv_bfloat16> k_out(kv_count);
    std::vector<__nv_bfloat16> v_out(kv_count);
    q_dev.copy_to_host(q_out.data());
    k_dev.copy_to_host(k_out.data());
    v_dev.copy_to_host(v_out.data());

    expect_bit_identical(add_rows_reference(q_host, qb_host, n, q_dim), q_out, "q");
    expect_bit_identical(add_rows_reference(k_host, kb_host, n, kv_dim), k_out, "k");
    expect_bit_identical(add_rows_reference(v_host, vb_host, n, kv_dim), v_out, "v");
}

}  // namespace

TEST(QkvBiasAddBF16, SmallShape) {
    run_case(/*n=*/3, /*q_dim=*/64, /*kv_dim=*/16);
}

TEST(QkvBiasAddBF16, Qwen25OnePointFiveBDecode) {
    // Qwen2.5 1.5B: 12 heads x 128, 2 KV heads x 128, one token.
    run_case(1, 1536, 256);
}

TEST(QkvBiasAddBF16, Qwen25ThreeBPrefill) {
    // Qwen2.5 3B: 16 heads x 128, 2 KV heads x 128, 511 tokens.
    run_case(511, 2048, 256);
}
