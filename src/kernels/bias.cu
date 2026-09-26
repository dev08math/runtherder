cin#include <runtherder/kernels/bias.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/device/check.cuh>

namespace runtherder::kernels {

namespace {

constexpr int kBlock   = 256;
constexpr int kVecBF16 = 8;  // 8 BF16 = 16 B = one float4

__device__ __forceinline__ void add_vec(__nv_bfloat16* __restrict__       x,
                                        const __nv_bfloat16* __restrict__ b) {
    float4       x_pack = *reinterpret_cast<const float4*>(x);
    const float4 b_pack = *reinterpret_cast<const float4*>(b);

    __nv_bfloat16*       xv = reinterpret_cast<__nv_bfloat16*>(&x_pack);
    const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&b_pack);

    #pragma unroll
    for (int j = 0; j < kVecBF16; ++j) {
        xv[j] = __float2bfloat16(__bfloat162float(xv[j]) + __bfloat162float(bv[j]));
    }
    *reinterpret_cast<float4*>(x) = x_pack;
}

__global__ void qkv_bias_add_bf16_kernel(
    __nv_bfloat16* __restrict__       q,
    __nv_bfloat16* __restrict__       k,
    __nv_bfloat16* __restrict__       v,
    const __nv_bfloat16* __restrict__ q_bias,
    const __nv_bfloat16* __restrict__ k_bias,
    const __nv_bfloat16* __restrict__ v_bias,
    int                               n,
    int                               q_vecs,
    int                               kv_vecs) {
    const int row_vecs = q_vecs + 2 * kv_vecs;
    const int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * row_vecs) {
        return;
    }

    const int row = tid / row_vecs;
    int       col = tid % row_vecs;

    if (col < q_vecs) {
        add_vec(q + (row * q_vecs + col) * kVecBF16, q_bias + col * kVecBF16);
        return;
    }
    col -= q_vecs;
    if (col < kv_vecs) {
        add_vec(k + (row * kv_vecs + col) * kVecBF16, k_bias + col * kVecBF16);
        return;
    }
    col -= kv_vecs;
    add_vec(v + (row * kv_vecs + col) * kVecBF16, v_bias + col * kVecBF16);
}

}  // namespace

void qkv_bias_add_bf16_forward(
    __nv_bfloat16*       q,
    __nv_bfloat16*       k,
    __nv_bfloat16*       v,
    const __nv_bfloat16* q_bias,
    const __nv_bfloat16* k_bias,
    const __nv_bfloat16* v_bias,
    int                  n,
    int                  q_dim,
    int                  kv_dim,
    cudaStream_t         stream) {
    RUNTHERDER_CHECK(n >= 1,                   "n must be >= 1");
    RUNTHERDER_CHECK(q_dim >= kVecBF16,        "q_dim must be >= 8");
    RUNTHERDER_CHECK(kv_dim >= kVecBF16,       "kv_dim must be >= 8");
    RUNTHERDER_CHECK(q_dim % kVecBF16 == 0,    "q_dim must be a multiple of 8");
    RUNTHERDER_CHECK(kv_dim % kVecBF16 == 0,   "kv_dim must be a multiple of 8");

    const int q_vecs  = q_dim / kVecBF16;
    const int kv_vecs = kv_dim / kVecBF16;
    const int total   = n * (q_vecs + 2 * kv_vecs);
    const int grid    = (total + kBlock - 1) / kBlock;
    qkv_bias_add_bf16_kernel<<<grid, kBlock, 0, stream>>>(
        q, k, v, q_bias, k_bias, v_bias, n, q_vecs, kv_vecs);
    RUNTHERDER_CUDA_CHECK_LAST();
}

}  // namespace runtherder::kernels
