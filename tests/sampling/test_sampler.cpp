#include <runtherder/sampling/sampler.h>

#include <cstddef>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {

using runtherder::sampling::Sampler;
using runtherder::sampling::SamplingParams;

__nv_bfloat16* upload(const std::vector<float>& values) {
    const std::size_t          n = values.size();
    std::vector<__nv_bfloat16> host(n);
    for (std::size_t i = 0; i < n; ++i) {
        host[i] = __float2bfloat16(values[i]);
    }
    __nv_bfloat16* device = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device), n * sizeof(__nv_bfloat16));
    cudaMemcpy(device, host.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    return device;
}

}  // namespace

TEST(Sampler, ZeroTemperatureIsGreedy) {
    const std::vector<float> logits = {0.1F, 0.2F, 9.0F, 0.3F};
    __nv_bfloat16*           device = upload(logits);

    Sampler        sampler(static_cast<int>(logits.size()), 1234);
    SamplingParams params;
    params.temperature = 0.0F;

    EXPECT_EQ(sampler.sample(device, params), 2);
    cudaFree(device);
}

TEST(Sampler, TopKOneCollapsesToArgmax) {
    const std::vector<float> logits = {1.0F, 5.0F, 2.0F, 0.5F};
    __nv_bfloat16*           device = upload(logits);

    Sampler        sampler(static_cast<int>(logits.size()), 1234);
    SamplingParams params;
    params.temperature = 1.0F;
    params.top_k       = 1;

    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(sampler.sample(device, params), 1);
    }
    cudaFree(device);
}

TEST(Sampler, SameSeedReproducesSequence) {
    const std::vector<float> logits = {1.0F, 1.1F, 0.9F, 1.05F, 0.95F};
    __nv_bfloat16*           device = upload(logits);

    SamplingParams params;

    Sampler a(static_cast<int>(logits.size()), 777);
    Sampler b(static_cast<int>(logits.size()), 777);
    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(a.sample(device, params), b.sample(device, params));
    }
    cudaFree(device);
}
