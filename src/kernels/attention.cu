#include <runtherder/kernels/attention.cuh>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <runtherder/device/check.cuh>
#include <runtherder/kernels/quantization.cuh>

namespace runtherder::kernels {

namespace {

// Bounds the per thread accumulator below and caps head_dim at the launch site.
constexpr int kMaxHeadDim = 256;

__device__ float qk_dot(
    const __nv_bfloat16* q_row,
    const __nv_fp8_e4m3* k_row,
    float                k_scale,
    int                  head_dim) {
    float s = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        s += __bfloat162float(q_row[d]) * dequantize_kv_fp8(k_row[d], k_scale);
    }
    return s;
}

// One thread owns one output row out[token, head, :]. Serial over keys, two
// passes (max, then exp weighted sum), recomputing the q.k dot each pass to
// avoid storing a score row. K and V are E4M3, dequantized by their per
// (token, kv_head) scale. Readable reference and the prefill path.
__global__ void attention_causal_bf16_kernel(
    __nv_bfloat16* __restrict__       out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k,
    const __nv_fp8_e4m3* __restrict__ v,
    const float* __restrict__         k_scale,
    const float* __restrict__         v_scale,
    int                               n_new,
    const int* __restrict__           cache_len,
    int                               num_q_heads,
    int                               num_kv_heads,
    int                               head_dim,
    float                             scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_new * num_q_heads) {
        return;
    }

    const int qt     = idx / num_q_heads;
    const int qh     = idx % num_q_heads;
    const int kvh    = qh / (num_q_heads / num_kv_heads);
    const int n_keys = *cache_len + qt + 1;  // causal, keys 0 .. cache_len + qt

    const __nv_bfloat16* q_row =
        q + (static_cast<long long>(qt) * num_q_heads + qh) * head_dim;

    float m = -CUDART_INF_F;
    for (int j = 0; j < n_keys; ++j) {
        const long long      row   = static_cast<long long>(j) * num_kv_heads + kvh;
        const __nv_fp8_e4m3* k_row = k + row * head_dim;
        m = fmaxf(m, qk_dot(q_row, k_row, k_scale[row], head_dim) * scale);
    }

    float acc[kMaxHeadDim];
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }

    float denom = 0.0f;
    for (int j = 0; j < n_keys; ++j) {
        const long long      row   = static_cast<long long>(j) * num_kv_heads + kvh;
        const __nv_fp8_e4m3* k_row = k + row * head_dim;
        const __nv_fp8_e4m3* v_row = v + row * head_dim;
        const float          vs    = v_scale[row];

        const float w = __expf(qk_dot(q_row, k_row, k_scale[row], head_dim) * scale - m);
        denom += w;
        for (int d = 0; d < head_dim; ++d) {
            acc[d] += w * dequantize_kv_fp8(v_row[d], vs);
        }
    }

    __nv_bfloat16* out_row =
        out + (static_cast<long long>(qt) * num_q_heads + qh) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        out_row[d] = __float2bfloat16(acc[d] / denom);
    }
}

}  // namespace

void attention_causal_bf16(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const __nv_fp8_e4m3* k,
    const __nv_fp8_e4m3* v,
    const float*         k_scale,
    const float*         v_scale,
    int                  n_new,
    const int*           cache_len,
    int                  num_q_heads,
    int                  num_kv_heads,
    int                  head_dim,
    float                scale,
    cudaStream_t         stream) {
    RUNTHERDER_CHECK(n_new        >= 1, "n_new must be >= 1");
    RUNTHERDER_CHECK(cache_len != nullptr, "cache_len must not be null");
    RUNTHERDER_CHECK(num_q_heads  >= 1, "num_q_heads must be >= 1");
    RUNTHERDER_CHECK(num_kv_heads >= 1, "num_kv_heads must be >= 1");
    RUNTHERDER_CHECK(num_q_heads % num_kv_heads == 0,
                     "num_q_heads must be divisible by num_kv_heads");
    RUNTHERDER_CHECK(head_dim >= 1,           "head_dim must be >= 1");
    RUNTHERDER_CHECK(head_dim <= kMaxHeadDim, "head_dim exceeds kMaxHeadDim (256)");

    const int total = n_new * num_q_heads;
    const int block = 128;
    const int grid  = (total + block - 1) / block;
    attention_causal_bf16_kernel<<<grid, block, 0, stream>>>(
        out, q, k, v, k_scale, v_scale, n_new, cache_len, num_q_heads,
        num_kv_heads, head_dim, scale);
    RUNTHERDER_CUDA_CHECK_LAST();
}

}  // namespace runtherder::kernels
