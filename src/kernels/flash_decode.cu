#include <runtherder/kernels/flash_decode.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <runtherder/device/check.cuh>

namespace runtherder::kernels {

namespace {

constexpr int kMaxHeadDim = 1024;

// One block per q head, one thread per head_dim lane. Online softmax, the q.k
// dot is a shared memory reduction across the block on every key.
__global__ void flash_decode_bf16_kernel(
    __nv_bfloat16* __restrict__       out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
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
        const __nv_bfloat16* k_row =
            k + (static_cast<long long>(j) * num_kv_heads + kvh) * head_dim;
        const __nv_bfloat16* v_row =
            v + (static_cast<long long>(j) * num_kv_heads + kvh) * head_dim;

        red[d] = q_d * __bfloat162float(k_row[d]);
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
        acc = acc * corr + w * __bfloat162float(v_row[d]);
        m   = m_new;
    }

    out[qh * head_dim + d] = __float2bfloat16(acc / l);
}

}  // namespace

void flash_decode_bf16(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
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
        out, q, k, v, cache_len, num_q_heads, num_kv_heads, head_dim, scale);
    RUNTHERDER_CUDA_CHECK_LAST();
}

}  // namespace runtherder::kernels
