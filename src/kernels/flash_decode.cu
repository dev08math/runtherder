#include <runtherder/kernels/flash_decode.cuh>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <runtherder/device/check.cuh>
#include <runtherder/kernels/quantization.cuh>

namespace runtherder::kernels {

namespace {

constexpr int kMaxHeadDim = 1024;

// One block per q head, one thread per head_dim lane. Online softmax, the q.k
// dot is a shared memory reduction across the block on every key. K and V are
// E4M3, dequantized by their per (token, kv_head) scale at the load site.
__global__ void flash_decode_bf16_kernel(
    __nv_bfloat16* __restrict__       out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k,
    const __nv_fp8_e4m3* __restrict__ v,
    const float* __restrict__         k_scale,
    const float* __restrict__         v_scale,
    int                               cache_len,
    int                               num_q_heads,
    int                               num_kv_heads,
    int                               head_dim,
    float                             scale) {
    const int qh     = blockIdx.x;
    const int d      = threadIdx.x;
    const int kvh    = qh / (num_q_heads / num_kv_heads);
    const int n_keys = cache_len + 1;

    const float q_d = __bfloat162float(q[qh * head_dim + d]);

    extern __shared__ float red[];

    float acc = 0.0f;
    float m   = -CUDART_INF_F;
    float l   = 0.0f;

    for (int j = 0; j < n_keys; ++j) {
        const long long          row   = static_cast<long long>(j) * num_kv_heads + kvh;
        const __nv_fp8_e4m3*      k_row = k + row * head_dim;
        const __nv_fp8_e4m3*      v_row = v + row * head_dim;
        const float               ks    = k_scale[row];
        const float               vs    = v_scale[row];

        red[d] = q_d * dequantize_kv_fp8(k_row[d], ks);
        __syncthreads();
        for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
            if (d < stride) {
                red[d] += red[d + stride];
            }
            __syncthreads();
        }
        const float s = red[0] * scale;
        __syncthreads();

        const float m_new = fmaxf(m, s);
        const float corr  = __expf(m - m_new);
        const float w     = __expf(s - m_new);
        l   = l * corr + w;
        acc = acc * corr + w * dequantize_kv_fp8(v_row[d], vs);
        m   = m_new;
    }

    out[qh * head_dim + d] = __float2bfloat16(acc / l);
}

// One block per (q head, split). Same body as flash_decode_bf16 over the split's
// key range, but writes the running m / l / acc to scratch instead of dividing.
// An empty split (num_splits > n_keys) writes the m = -inf, l = 0, acc = 0
// sentinel the reduce treats as a zero weight contribution.
__global__ void flash_decode_partial_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k,
    const __nv_fp8_e4m3* __restrict__ v,
    const float* __restrict__         k_scale,
    const float* __restrict__         v_scale,
    float* __restrict__               partial,
    int                               num_splits,
    int                               cache_len,
    int                               num_q_heads,
    int                               num_kv_heads,
    int                               head_dim,
    float                             scale) {
    const int qh     = blockIdx.x;
    const int si     = blockIdx.y;
    const int d      = threadIdx.x;
    const int kvh    = qh / (num_q_heads / num_kv_heads);
    const int n_keys = cache_len + 1;

    const int chunk = (n_keys + num_splits - 1) / num_splits;
    const int start = si * chunk;
    const int end   = min(start + chunk, n_keys);

    const float q_d = __bfloat162float(q[qh * head_dim + d]);

    extern __shared__ float red[];

    float acc = 0.0f;
    float m   = -CUDART_INF_F;
    float l   = 0.0f;

    for (int j = start; j < end; ++j) {
        const long long          row   = static_cast<long long>(j) * num_kv_heads + kvh;
        const __nv_fp8_e4m3*      k_row = k + row * head_dim;
        const __nv_fp8_e4m3*      v_row = v + row * head_dim;
        const float               ks    = k_scale[row];
        const float               vs    = v_scale[row];

        red[d] = q_d * dequantize_kv_fp8(k_row[d], ks);
        __syncthreads();
        for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
            if (d < stride) {
                red[d] += red[d + stride];
            }
            __syncthreads();
        }
        const float s = red[0] * scale;
        __syncthreads();

        const float m_new = fmaxf(m, s);
        const float corr  = __expf(m - m_new);
        const float w     = __expf(s - m_new);
        l   = l * corr + w;
        acc = acc * corr + w * dequantize_kv_fp8(v_row[d], vs);
        m   = m_new;
    }

    const long long hs   = static_cast<long long>(num_q_heads) * num_splits;
    const long long slot = static_cast<long long>(qh) * num_splits + si;
    if (d == 0) {
        partial[slot]      = m;   // [num_q_heads, num_splits]
        partial[hs + slot] = l;
    }
    partial[2 * hs + slot * head_dim + d] = acc;
}

// One block per q head. Merges the num_splits partials for that head onto a
// shared max, then normalizes. Thread d owns lane d, reads across splits.
__global__ void flash_decode_reduce_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__   partial,
    int                         num_splits,
    int                         num_q_heads,
    int                         head_dim) {
    const int       qh = blockIdx.x;
    const int       d  = threadIdx.x;
    const long long hs = static_cast<long long>(num_q_heads) * num_splits;

    float M = -CUDART_INF_F;
    for (int i = 0; i < num_splits; ++i) {
        M = fmaxf(M, partial[static_cast<long long>(qh) * num_splits + i]);
    }

    float acc = 0.0f;
    float L   = 0.0f;
    for (int i = 0; i < num_splits; ++i) {
        const long long slot = static_cast<long long>(qh) * num_splits + i;
        const float     a    = __expf(partial[slot] - M);
        L   += a * partial[hs + slot];
        acc += a * partial[2 * hs + slot * head_dim + d];
    }

    out[qh * head_dim + d] = __float2bfloat16(acc / L);
}

}  // namespace

void flash_decode_bf16(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const __nv_fp8_e4m3* k,
    const __nv_fp8_e4m3* v,
    const float*         k_scale,
    const float*         v_scale,
    int                  cache_len,
    int                  num_q_heads,
    int                  num_kv_heads,
    int                  head_dim,
    float                scale,
    cudaStream_t         stream) {
    RUNTHERDER_CHECK(cache_len    >= 0, "cache_len must be >= 0");
    RUNTHERDER_CHECK(num_q_heads  >= 1, "num_q_heads must be >= 1");
    RUNTHERDER_CHECK(num_kv_heads >= 1, "num_kv_heads must be >= 1");
    RUNTHERDER_CHECK(num_q_heads % num_kv_heads == 0,
                     "num_q_heads must be divisible by num_kv_heads");
    RUNTHERDER_CHECK(head_dim >= 1,           "head_dim must be >= 1");
    RUNTHERDER_CHECK(head_dim <= kMaxHeadDim, "head_dim exceeds kMaxHeadDim (1024)");
    RUNTHERDER_CHECK((head_dim & (head_dim - 1)) == 0,
                     "head_dim must be a power of two");

    const int          block = head_dim;
    const std::size_t  shmem = static_cast<std::size_t>(head_dim) * sizeof(float);
    flash_decode_bf16_kernel<<<num_q_heads, block, shmem, stream>>>(
        out, q, k, v, k_scale, v_scale, cache_len, num_q_heads, num_kv_heads,
        head_dim, scale);
    RUNTHERDER_CUDA_CHECK_LAST();
}

void flash_decode_split_bf16(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const __nv_fp8_e4m3* k,
    const __nv_fp8_e4m3* v,
    const float*         k_scale,
    const float*         v_scale,
    float*               partial,
    int                  num_splits,
    int                  cache_len,
    int                  num_q_heads,
    int                  num_kv_heads,
    int                  head_dim,
    float                scale,
    cudaStream_t         stream) {
    RUNTHERDER_CHECK(cache_len    >= 0, "cache_len must be >= 0");
    RUNTHERDER_CHECK(num_splits   >= 1, "num_splits must be >= 1");
    RUNTHERDER_CHECK(num_q_heads  >= 1, "num_q_heads must be >= 1");
    RUNTHERDER_CHECK(num_kv_heads >= 1, "num_kv_heads must be >= 1");
    RUNTHERDER_CHECK(num_q_heads % num_kv_heads == 0,
                     "num_q_heads must be divisible by num_kv_heads");
    RUNTHERDER_CHECK(head_dim >= 1,           "head_dim must be >= 1");
    RUNTHERDER_CHECK(head_dim <= kMaxHeadDim, "head_dim exceeds kMaxHeadDim (1024)");
    RUNTHERDER_CHECK((head_dim & (head_dim - 1)) == 0,
                     "head_dim must be a power of two");

    const int         block = head_dim;
    const std::size_t shmem = static_cast<std::size_t>(head_dim) * sizeof(float);

    const dim3 partial_grid(static_cast<unsigned int>(num_q_heads),
                            static_cast<unsigned int>(num_splits));
    flash_decode_partial_kernel<<<partial_grid, block, shmem, stream>>>(
        q, k, v, k_scale, v_scale, partial, num_splits, cache_len, num_q_heads,
        num_kv_heads, head_dim, scale);
    RUNTHERDER_CUDA_CHECK_LAST();

    flash_decode_reduce_kernel<<<num_q_heads, block, 0, stream>>>(
        out, partial, num_splits, num_q_heads, head_dim);
    RUNTHERDER_CUDA_CHECK_LAST();
}

}  // namespace runtherder::kernels
