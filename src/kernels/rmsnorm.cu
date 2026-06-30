#include <runtherder/kernels/rmsnorm.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/device/check.cuh>

namespace runtherder::kernels {

namespace {

// TODO: benchmark block sizes and vector widths for various hidden_dim sizes.
constexpr int kBlockSize = 256;
constexpr int kVecWidth  = 8;
constexpr int kMaxWarps  = 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// Full sum lands on thread 0 only. Other threads return partial values.
__device__ __forceinline__ float block_reduce_sum(float v, float* s_partial) {
    const int lane      = threadIdx.x & 31;
    const int warp      = threadIdx.x >> 5;
    const int num_warps = blockDim.x  >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        s_partial[warp] = v;
    }
    __syncthreads();

    float total = (threadIdx.x < num_warps) ? s_partial[threadIdx.x] : 0.0f;
    if (warp == 0) {
        total = warp_reduce_sum(total);
    }
    return total;
}

template <int VecsPerThread, bool FuseResidual>
__global__ void rmsnorm_bf16_kernel(
    __nv_bfloat16* __restrict__       y,
    __nv_bfloat16* __restrict__       h_out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ g,
    int                               hidden_dim,
    float                             eps) {
    // One block per token. Pass 1: each thread caches h = x (+ residual when
    // fused) in registers, accumulates the local sum of squares, and writes
    // h_out on the fused path. block_reduce_sum (warp shuffle via
    // warp_reduce_sum, then shared partials) gives the row sum of squares on
    // thread 0, which forms rrms = rsqrt(ss / hidden_dim + eps). Pass 2 scales
    // from the register cache, y = h * rrms * g.
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_vecs = hidden_dim / kVecWidth;

    const __nv_bfloat16* x_row = x + row * hidden_dim;
    __nv_bfloat16*       y_row = y + row * hidden_dim;

    const __nv_bfloat16* r_row = nullptr;
    __nv_bfloat16*       h_row = nullptr;
    if constexpr (FuseResidual) {
        r_row = residual + row * hidden_dim;
        h_row = h_out + row * hidden_dim;
    }

    // Per thread register cache of h = x (+ residual), reused in pass 2.
    float h_cache[VecsPerThread * kVecWidth];
    float local_ss = 0.0f;

    int slot = 0;
    for (int i = tid; i < n_vecs; i += blockDim.x) {
        float4 xv = reinterpret_cast<const float4*>(x_row)[i];
        const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xv);

        float4 rv;
        const __nv_bfloat16* rh = nullptr;
        if constexpr (FuseResidual) {
            rv = reinterpret_cast<const float4*>(r_row)[i];
            rh = reinterpret_cast<const __nv_bfloat16*>(&rv);
        }

        #pragma unroll
        for (int j = 0; j < kVecWidth; ++j) {
            float h = __bfloat162float(xh[j]);
            if constexpr (FuseResidual) {
                h += __bfloat162float(rh[j]);
            }
            h_cache[slot * kVecWidth + j] = h;
            local_ss += h * h;
        }

        if constexpr (FuseResidual) {
            float4 hv;
            __nv_bfloat16* hh = reinterpret_cast<__nv_bfloat16*>(&hv);
            #pragma unroll
            for (int j = 0; j < kVecWidth; ++j) {
                hh[j] = __float2bfloat16(h_cache[slot * kVecWidth + j]);
            }
            reinterpret_cast<float4*>(h_row)[i] = hv;
        }

        ++slot;
    }

    __shared__ float s_partial[kMaxWarps];
    const float total_ss = block_reduce_sum(local_ss, s_partial);

    __shared__ float row_rrms;
    if (tid == 0) {
        row_rrms = rsqrtf(total_ss / static_cast<float>(hidden_dim) + eps);
    }
    __syncthreads();

    const float rrms = row_rrms;

    slot = 0;
    for (int i = tid; i < n_vecs; i += blockDim.x) {
        float4 gv = reinterpret_cast<const float4*>(g)[i];
        const __nv_bfloat16* gh = reinterpret_cast<const __nv_bfloat16*>(&gv);

        float4 yv;
        __nv_bfloat16* yh = reinterpret_cast<__nv_bfloat16*>(&yv);

        #pragma unroll
        for (int j = 0; j < kVecWidth; ++j) {
            const float h_val = h_cache[slot * kVecWidth + j];
            const float g_val = __bfloat162float(gh[j]);
            yh[j] = __float2bfloat16(h_val * rrms * g_val);
        }
        reinterpret_cast<float4*>(y_row)[i] = yv;
        ++slot;
    }
}

template <int VecsPerThread, bool FuseResidual>
inline void launch_kernel(
    __nv_bfloat16*       y,
    __nv_bfloat16*       h_out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* g,
    int                  num_tokens,
    int                  hidden_dim,
    float                eps,
    cudaStream_t         stream) {
    rmsnorm_bf16_kernel<VecsPerThread, FuseResidual>
        <<<num_tokens, kBlockSize, 0, stream>>>(
            y, h_out, x, residual, g, hidden_dim, eps);
    RUNTHERDER_CUDA_CHECK_LAST();
}

template <bool FuseResidual>
void dispatch(
    __nv_bfloat16*       y,
    __nv_bfloat16*       h_out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* g,
    int                  num_tokens,
    int                  hidden_dim,
    float                eps,
    cudaStream_t         stream) {
    RUNTHERDER_CHECK(num_tokens >= 1, "num_tokens must be >= 1");
    RUNTHERDER_CHECK(hidden_dim >= kVecWidth, "hidden_dim must be >= 8");
    RUNTHERDER_CHECK(hidden_dim % kVecWidth == 0,
                     "hidden_dim must be divisible by 8 for BF16 vectorized load");
    RUNTHERDER_CHECK(eps > 0.0f, "eps must be > 0");
    if constexpr (FuseResidual) {
        RUNTHERDER_CHECK(
            static_cast<const void*>(y) != static_cast<const void*>(h_out),
            "y and h_out must point to distinct buffers");
    }

    const int n_vecs          = hidden_dim / kVecWidth;
    const int vecs_per_thread = (n_vecs + kBlockSize - 1) / kBlockSize;

    switch (vecs_per_thread) {
        case 1:
            launch_kernel<1, FuseResidual>(
                y, h_out, x, residual, g, num_tokens, hidden_dim, eps, stream);
            break;
        case 2:
            launch_kernel<2, FuseResidual>(
                y, h_out, x, residual, g, num_tokens, hidden_dim, eps, stream);
            break;
        case 3:
        case 4:
            launch_kernel<4, FuseResidual>(
                y, h_out, x, residual, g, num_tokens, hidden_dim, eps, stream);
            break;
        case 5:
        case 6:
        case 7:
        case 8:
            launch_kernel<8, FuseResidual>(
                y, h_out, x, residual, g, num_tokens, hidden_dim, eps, stream);
            break;
        default:
            RUNTHERDER_CHECK(
                false,
                "hidden_dim too large for current dispatch ");
    }
}

}  // namespace

void rmsnorm_bf16_forward(
    __nv_bfloat16*       y,
    const __nv_bfloat16* x,
    const __nv_bfloat16* g,
    int                  num_tokens,
    int                  hidden_dim,
    float                eps,
    cudaStream_t         stream) {
    dispatch<false>(y, nullptr, x, nullptr, g,
                    num_tokens, hidden_dim, eps, stream);
}

void rmsnorm_add_bf16_forward(
    __nv_bfloat16*       y,
    __nv_bfloat16*       h_out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* g,
    int                  num_tokens,
    int                  hidden_dim,
    float                eps,
    cudaStream_t         stream) {
    dispatch<true>(y, h_out, x, residual, g,
                   num_tokens, hidden_dim, eps, stream);
}

}  // namespace runtherder::kernels
