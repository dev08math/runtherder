#include <runtherder/kernels/attention.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <runtherder/device/check.cuh>

namespace runtherder::kernels {

namespace {

// Bounds the per thread accumulator below and caps head_dim at the launch site.
constexpr int kMaxHeadDim = 256;

__device__ float qk_dot(
    const __nv_bfloat16* q_row,
    const __nv_bfloat16* k_row,
    int                  head_dim) {
    float s = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        s += __bfloat162float(q_row[d]) * __bfloat162float(k_row[d]);
    }
    return s;
}

// One thread owns one output row out[token, head, :]. Serial over keys, two
// passes (max, then exp weighted sum), recomputing the q.k dot each pass to
// avoid storing a score row. Readable reference, replaced by a flash style
// kernel later.
__global__ void attention_prefill_bf16_kernel(
    __nv_bfloat16* __restrict__       out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int                               num_tokens,
    int                               num_q_heads,
    int                               num_kv_heads,
    int                               head_dim,
    float                             scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * num_q_heads) {
        return;
    }

    const int qt     = idx / num_q_heads;  // query token i
    const int qh     = idx % num_q_heads;  // query head h
    const int kvh    = qh / (num_q_heads / num_kv_heads);
    const int n_keys = qt + 1;  // causal, keys 0 .. qt

    const __nv_bfloat16* q_row =
        q + (static_cast<long long>(qt) * num_q_heads + qh) * head_dim;

    float m = -CUDART_INF_F;
    for (int j = 0; j < n_keys; ++j) {
        const __nv_bfloat16* k_row =
            k + (static_cast<long long>(j) * num_kv_heads + kvh) * head_dim;
        m = fmaxf(m, qk_dot(q_row, k_row, head_dim) * scale);
    }

    float acc[kMaxHeadDim];
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }

    float denom = 0.0f;
    for (int j = 0; j < n_keys; ++j) {
        const __nv_bfloat16* k_row =
            k + (static_cast<long long>(j) * num_kv_heads + kvh) * head_dim;
        const __nv_bfloat16* v_row =
            v + (static_cast<long long>(j) * num_kv_heads + kvh) * head_dim;

        const float w = __expf(qk_dot(q_row, k_row, head_dim) * scale - m);
        denom += w;
        for (int d = 0; d < head_dim; ++d) {
            acc[d] += w * __bfloat162float(v_row[d]);
        }
    }

    __nv_bfloat16* out_row =
        out + (static_cast<long long>(qt) * num_q_heads + qh) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        out_row[d] = __float2bfloat16(acc[d] / denom);
    }
}

}  // namespace

void attention_prefill_bf16(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    int                  num_tokens,
    int                  num_q_heads,
    int                  num_kv_heads,
    int                  head_dim,
    float                scale,
    cudaStream_t         stream) {
    RUNTHERDER_CHECK(num_tokens   >= 1, "num_tokens must be >= 1");
    RUNTHERDER_CHECK(num_q_heads  >= 1, "num_q_heads must be >= 1");
    RUNTHERDER_CHECK(num_kv_heads >= 1, "num_kv_heads must be >= 1");
    RUNTHERDER_CHECK(num_q_heads % num_kv_heads == 0,
                     "num_q_heads must be divisible by num_kv_heads");
    RUNTHERDER_CHECK(head_dim >= 1,           "head_dim must be >= 1");
    RUNTHERDER_CHECK(head_dim <= kMaxHeadDim, "head_dim exceeds kMaxHeadDim (256)");

    const int total = num_tokens * num_q_heads;
    const int block = 128;
    const int grid  = (total + block - 1) / block;
    attention_prefill_bf16_kernel<<<grid, block, 0, stream>>>(
        out, q, k, v, num_tokens, num_q_heads, num_kv_heads, head_dim, scale);
    RUNTHERDER_CUDA_CHECK_LAST();
}

}  // namespace runtherder::kernels
