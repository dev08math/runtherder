#include <runtherder/kernels/quantization.cuh>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <runtherder/check.h>
#include <runtherder/device/check.cuh>

namespace runtherder::kernels {

namespace {

constexpr int kBlockSize = 256;
constexpr int kVecWidth  = 8;

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

// Full max lands on thread 0 only.
__device__ __forceinline__ float block_reduce_max(float v, float* s_partial) {
    const int lane      = threadIdx.x & 31;
    const int warp      = threadIdx.x >> 5;
    const int num_warps = blockDim.x  >> 5;

    v = warp_reduce_max(v);
    if (lane == 0) {
        s_partial[warp] = v;
    }
    __syncthreads();

    float final = (threadIdx.x < num_warps) ? s_partial[threadIdx.x] : 0.0f;
    if (warp == 0) {
        final = warp_reduce_max(final);
    }
    return final;
}

template <int VecsPerThread>
__global__ void quantize_per_token_int8_kernel(
    std::int8_t* __restrict__         q,
    float* __restrict__               x_scale,
    const __nv_bfloat16* __restrict__ x,
    int                               dim) {
    // One block per token. Each thread caches its strided slice of the row in
    // registers while finding the local absmax. block_reduce_max (warp shuffle
    // via warp_reduce_max, then shared partials) gives the row absmax on thread
    // 0. The second pass requantizes from the register cache, one global read
    // per element. Zero row gives scale 0.
    const int row    = blockIdx.x;
    const int tid    = threadIdx.x;
    const int n_vecs = dim / kVecWidth;

    const __nv_bfloat16* x_row = x + row * dim;

    // Single thread will load blockDim spaced 8 elements a time.
    float cache[VecsPerThread * kVecWidth];
    float local_absmax = 0.0f;

    int slot = 0;
    for (int i = tid; i < n_vecs; i += blockDim.x) {
        float4               xv = reinterpret_cast<const float4*>(x_row)[i];
        const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xv);
        #pragma unroll
        for (int j = 0; j < kVecWidth; ++j) {
            const float v              = __bfloat162float(xh[j]);
            cache[slot * kVecWidth + j] = v;
            local_absmax               = fmaxf(local_absmax, fabsf(v));
        }
        ++slot;
    }

    __shared__ float s_partial[8];
    const float absmax = block_reduce_max(local_absmax, s_partial);

    __shared__ float row_inv_scale;
    if (tid == 0) {
        x_scale[row]  = absmax / 127.0f;
        row_inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;
    }
    __syncthreads();

    const float  inv_scale = row_inv_scale;
    std::int8_t* q_row      = q + row * dim;

    slot = 0;
    for (int i = tid; i < n_vecs; i += blockDim.x) {
        #pragma unroll
        for (int j = 0; j < kVecWidth; ++j) {
            const int qi = __float2int_rn(cache[slot * kVecWidth + j] * inv_scale);
            q_row[i * kVecWidth + j] = static_cast<std::int8_t>(max(-127, min(127, qi)));
        }
        ++slot;
    }
}

template <int VecsPerThread>
inline void launch(std::int8_t*         q,
                   float*               x_scale,
                   const __nv_bfloat16* x,
                   int                  num_tokens,
                   int                  dim,
                   cudaStream_t         stream) {
    quantize_per_token_int8_kernel<VecsPerThread>
        <<<num_tokens, kBlockSize, 0, stream>>>(q, x_scale, x, dim);
    RUNTHERDER_CUDA_CHECK_LAST();
}

}  // namespace

void quantize_per_token_int8(std::int8_t*         q,
                             float*               x_scale,
                             const __nv_bfloat16* x,
                             int                  num_tokens,
                             int                  dim,
                             cudaStream_t         stream) {
    RUNTHERDER_CHECK(num_tokens >= 1, "num_tokens must be >= 1");
    RUNTHERDER_CHECK(dim >= kVecWidth, "dim must be >= 8");
    RUNTHERDER_CHECK(dim % kVecWidth == 0, "dim must be divisible by 8");

    const int n_vecs          = dim / kVecWidth;
    const int vecs_per_thread = (n_vecs + kBlockSize - 1) / kBlockSize;

    switch (vecs_per_thread) {
        case 1:
            launch<1>(q, x_scale, x, num_tokens, dim, stream);
            break;
        case 2:
            launch<2>(q, x_scale, x, num_tokens, dim, stream);
            break;
        case 3:
        case 4:
            launch<4>(q, x_scale, x, num_tokens, dim, stream);
            break;
        case 5:
        case 6:
        case 7:
        case 8:
            launch<8>(q, x_scale, x, num_tokens, dim, stream);
            break;
        default:
            RUNTHERDER_CHECK(false, "dim too large for current dispatch");
    }
}

namespace {

__global__ void dequantize_w8a8_kernel(
    __nv_bfloat16* __restrict__       y,
    const std::int32_t* __restrict__  acc,
    const float* __restrict__         x_scale,
    const __nv_bfloat16* __restrict__ w_scale,
    int                               m,
    int                               n) {
    const long long total  = static_cast<long long>(m) * n;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride) {
        const int   row = static_cast<int>(idx / n);
        const int   col = static_cast<int>(idx % n);
        const float ws  = __bfloat162float(w_scale[col]);
        const float v   = static_cast<float>(acc[idx]) * x_scale[row] * ws;
        y[idx] = __float2bfloat16(v);
    }
}

}  // namespace

void dequantize_w8a8(__nv_bfloat16*       y,
                     const std::int32_t*  acc,
                     const float*         x_scale,
                     const __nv_bfloat16* w_scale,
                     int                  m,
                     int                  n,
                     cudaStream_t         stream) {
    RUNTHERDER_CHECK(m >= 1, "m must be >= 1");
    RUNTHERDER_CHECK(n >= 1, "n must be >= 1");

    const long long total = static_cast<long long>(m) * n;
    const int       grid  = static_cast<int>((total + kBlockSize - 1) / kBlockSize);

    dequantize_w8a8_kernel<<<grid, kBlockSize, 0, stream>>>(
        y, acc, x_scale, w_scale, m, n);
    RUNTHERDER_CUDA_CHECK_LAST();
}

namespace {

constexpr float kFp8E4M3Max = 448.0f;

template <int VecsPerThread>
__global__ void quantize_kv_fp8_kernel(
    __nv_fp8_e4m3* __restrict__       q,
    float* __restrict__               scale,
    const __nv_bfloat16* __restrict__ x,
    int                               dim) {
    const int row    = blockIdx.x;
    const int tid    = threadIdx.x;
    const int n_vecs = dim / kVecWidth;

    const __nv_bfloat16* x_row = x + row * dim;

    float cache[VecsPerThread * kVecWidth];
    float local_absmax = 0.0f;

    int slot = 0;
    for (int i = tid; i < n_vecs; i += blockDim.x) {
        float4               xv = reinterpret_cast<const float4*>(x_row)[i];
        const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xv);
        #pragma unroll
        for (int j = 0; j < kVecWidth; ++j) {
            const float v               = __bfloat162float(xh[j]);
            cache[slot * kVecWidth + j] = v;
            local_absmax                = fmaxf(local_absmax, fabsf(v));
        }
        ++slot;
    }

    __shared__ float s_partial[8];
    const float absmax = block_reduce_max(local_absmax, s_partial);

    __shared__ float row_inv_scale;
    if (tid == 0) {
        scale[row]    = absmax / kFp8E4M3Max;
        row_inv_scale = (absmax > 0.0f) ? (kFp8E4M3Max / absmax) : 0.0f;
    }
    __syncthreads();

    const float    inv_scale = row_inv_scale;
    __nv_fp8_e4m3* q_row     = q + row * dim;

    slot = 0;
    for (int i = tid; i < n_vecs; i += blockDim.x) {
        #pragma unroll
        for (int j = 0; j < kVecWidth; ++j) {
            // E4M3 ctor saturates out of range, no explicit clamp like int8.
            q_row[i * kVecWidth + j] =
                __nv_fp8_e4m3(cache[slot * kVecWidth + j] * inv_scale);
        }
        ++slot;
    }
}

template <int VecsPerThread>
inline void launch_kv_fp8(__nv_fp8_e4m3*       q,
                          float*               scale,
                          const __nv_bfloat16* x,
                          int                  num_rows,
                          int                  dim,
                          cudaStream_t         stream) {
    quantize_kv_fp8_kernel<VecsPerThread>
        <<<num_rows, kBlockSize, 0, stream>>>(q, scale, x, dim);
    RUNTHERDER_CUDA_CHECK_LAST();
}

}  // namespace

void quantize_kv_fp8(__nv_fp8_e4m3*       q,
                     float*               scale,
                     const __nv_bfloat16* x,
                     int                  num_rows,
                     int                  head_dim,
                     cudaStream_t         stream) {
    RUNTHERDER_CHECK(num_rows >= 1, "num_rows must be >= 1");
    RUNTHERDER_CHECK(head_dim >= kVecWidth, "head_dim must be >= 8");
    RUNTHERDER_CHECK(head_dim % kVecWidth == 0, "head_dim must be divisible by 8");

    const int n_vecs          = head_dim / kVecWidth;
    const int vecs_per_thread = (n_vecs + kBlockSize - 1) / kBlockSize;

    switch (vecs_per_thread) {
        case 1:
            launch_kv_fp8<1>(q, scale, x, num_rows, head_dim, stream);
            break;
        case 2:
            launch_kv_fp8<2>(q, scale, x, num_rows, head_dim, stream);
            break;
        case 3:
        case 4:
            launch_kv_fp8<4>(q, scale, x, num_rows, head_dim, stream);
            break;
        case 5:
        case 6:
        case 7:
        case 8:
            launch_kv_fp8<8>(q, scale, x, num_rows, head_dim, stream);
            break;
        default:
            RUNTHERDER_CHECK(false, "head_dim too large for current dispatch");
    }
}

}  // namespace runtherder::kernels
