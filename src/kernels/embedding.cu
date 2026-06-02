#include <runtherder/kernels/embedding.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include <runtherder/device/check.cuh>

namespace runtherder::kernels {

namespace {

constexpr int kBlockSize = 256;
constexpr int kVecWidth  = 8;

__global__ void embedding_lookup_bf16_kernel(
    __nv_bfloat16* __restrict__       out,
    const __nv_bfloat16* __restrict__ embedding_table,
    const int* __restrict__           token_ids,
    int                               hidden_dim) {
    const int row    = blockIdx.x;
    const int n_vecs = hidden_dim / kVecWidth;

    const std::size_t src_off =
        static_cast<std::size_t>(token_ids[row]) * hidden_dim;
    const std::size_t dst_off =
        static_cast<std::size_t>(row) * hidden_dim;

    const float4* src = reinterpret_cast<const float4*>(embedding_table + src_off);
    float4*       dst = reinterpret_cast<float4*>(out + dst_off);

    for (int i = threadIdx.x; i < n_vecs; i += blockDim.x) {
        dst[i] = src[i];
    }
}

}  // namespace

void embedding_lookup_bf16_forward(
    __nv_bfloat16*       out,
    const __nv_bfloat16* embedding_table,
    const int*           token_ids,
    int                  num_tokens,
    int                  hidden_dim,
    cudaStream_t         stream) {
    RUNTHERDER_CHECK(num_tokens >= 1, "num_tokens must be >= 1");
    RUNTHERDER_CHECK(hidden_dim % kVecWidth == 0,
                     "hidden_dim must be divisible by 8 for BF16 vectorized copy");

    embedding_lookup_bf16_kernel<<<num_tokens, kBlockSize, 0, stream>>>(
        out, embedding_table, token_ids, hidden_dim);
    RUNTHERDER_CUDA_CHECK_LAST();
}

}  // namespace runtherder::kernels
