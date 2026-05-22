#include <runtherder/kernels/rope.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/runtime/error.cuh>

namespace runtherder::kernels {

namespace {

constexpr int kMaxHeadDim = 256;

__global__ void rope_qk_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    int                         start_pos,
    int                         num_q_heads,
    int                         num_kv_heads,
    int                         head_dim,
    float                       theta_base) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int i     = threadIdx.x;
    const int half  = head_dim >> 1;

    // blockDim.x == half, so this is a no-op guard against misconfigured launches.
    if (i >= half) {
        return;
    }

    const int pos = start_pos + token;

    const float exponent =
        -2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
    const float freq  = __powf(theta_base, exponent);
    const float theta = static_cast<float>(pos) * freq;

    float s, c;
    __sincosf(theta, &s, &c);

    // grid y-dim packs Q heads first, then K heads: [0, num_q_heads) -> q,
    // [num_q_heads, num_q_heads + num_kv_heads) -> k (rebased to kv_head).
    __nv_bfloat16* row;
    if (head < num_q_heads) {
        row = q + (static_cast<long long>(token) * num_q_heads + head) * head_dim;
    } else {
        const int kv_head = head - num_q_heads;
        row = k + (static_cast<long long>(token) * num_kv_heads + kv_head) * head_dim;
    }

    const float a = __bfloat162float(row[i]);
    const float b = __bfloat162float(row[i + half]);

    row[i]        = __float2bfloat16(a * c - b * s);
    row[i + half] = __float2bfloat16(a * s + b * c);
}

}  // namespace

void rope_bf16_inplace(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    int            start_pos,
    int            seq_len,
    int            num_q_heads,
    int            num_kv_heads,
    int            head_dim,
    float          theta_base,
    cudaStream_t   stream) {
    RUNTHERDER_CHECK(seq_len      >= 1, "seq_len must be >= 1");
    RUNTHERDER_CHECK(start_pos    >= 0, "start_pos must be >= 0");
    RUNTHERDER_CHECK(num_q_heads  >= 1, "num_q_heads must be >= 1");
    RUNTHERDER_CHECK(num_kv_heads >= 1, "num_kv_heads must be >= 1");
    RUNTHERDER_CHECK(head_dim     >= 2, "head_dim must be >= 2");
    RUNTHERDER_CHECK(head_dim % 2 == 0,           "head_dim must be even");
    RUNTHERDER_CHECK(head_dim <= kMaxHeadDim,     "head_dim exceeds kMaxHeadDim (256)");
    RUNTHERDER_CHECK(theta_base > 0.0f,           "theta_base must be > 0");

    const dim3 grid(seq_len, num_q_heads + num_kv_heads);
    const dim3 block(head_dim / 2);
    rope_qk_bf16_kernel<<<grid, block, 0, stream>>>(
        q, k, start_pos, num_q_heads, num_kv_heads, head_dim, theta_base);
    RUNTHERDER_CUDA_CHECK_LAST();
}

}  // namespace runtherder::kernels
