#include <runtherder/sampling/greedy.h>

#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {

int argmax_of(const std::vector<float>& values) {
    const int                  vocab = static_cast<int>(values.size());
    std::vector<__nv_bfloat16> host(static_cast<std::size_t>(vocab));
    for (int i = 0; i < vocab; ++i) {
        host[i] = __float2bfloat16(values[i]);
    }

    __nv_bfloat16* device = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device),
               static_cast<std::size_t>(vocab) * sizeof(__nv_bfloat16));
    cudaMemcpy(device, host.data(),
               static_cast<std::size_t>(vocab) * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);

    const int idx = runtherder::sampling::greedy_argmax(device, vocab);
    cudaFree(device);
    return idx;
}

}  // namespace

TEST(GreedyArgmax, PicksMaxInMiddle) {
    EXPECT_EQ(argmax_of({0.1F, 0.2F, 9.0F, 0.3F, 0.4F}), 2);
}

TEST(GreedyArgmax, FirstWinsOnTie) {
    EXPECT_EQ(argmax_of({5.0F, 5.0F, 1.0F}), 0);
}

TEST(GreedyArgmax, SingleElement) {
    EXPECT_EQ(argmax_of({42.0F}), 0);
}
