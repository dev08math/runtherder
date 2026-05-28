#include <runtherder/kernels/activation.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/device/check.cuh>

namespace runtherder::kernels {

namespace {

constexpr int kBlock      = 256;
constexpr int kVecBF16    = 8;   // 8 BF16 = 16 B = one float4

__global__ void swiglu_bf16_kernel(
    __nv_bfloat16* __restrict__       y,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    int                               n_vec) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_vec) {
        return;
    }

    const float4 g_pack = reinterpret_cast<const float4*>(gate)[tid];
    const float4 u_pack = reinterpret_cast<const float4*>(up)[tid];

    const __nv_bfloat16* g = reinterpret_cast<const __nv_bfloat16*>(&g_pack);
    const __nv_bfloat16* u = reinterpret_cast<const __nv_bfloat16*>(&u_pack);

    __nv_bfloat16 o[kVecBF16];
    #pragma unroll
    for (int j = 0; j < kVecBF16; ++j) {
        const float gf = __bfloat162float(g[j]);
        const float uf = __bfloat162float(u[j]);
        const float s  = 1.0f / (1.0f + __expf(-gf));
        o[j] = __float2bfloat16(gf * s * uf);
    }

    reinterpret_cast<float4*>(y)[tid] = *reinterpret_cast<float4*>(o);
}

}  // namespace

void swiglu_bf16_forward(
    __nv_bfloat16*       y,
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    int                  n,
    cudaStream_t         stream) {
    RUNTHERDER_CHECK(n >= kVecBF16,    "n must be >= 8");
    RUNTHERDER_CHECK(n % kVecBF16 == 0, "n must be a multiple of 8");

    const int n_vec = n / kVecBF16;
    const int grid  = (n_vec + kBlock - 1) / kBlock;
    swiglu_bf16_kernel<<<grid, kBlock, 0, stream>>>(y, gate, up, n_vec);
    RUNTHERDER_CUDA_CHECK_LAST();
}

}  // namespace runtherder::kernels
