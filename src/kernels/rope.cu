#include <runtherder/kernels/rope.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/device/check.cuh>

namespace runtherder::kernels {

namespace {

constexpr int kMaxHeadDim = 256;

__global__ void rope_qk_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const int*     __restrict__ positions,
    const float*   __restrict__ inv_freq,
    int                         num_q_heads,
    int                         num_kv_heads,
    int                         head_dim) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int i     = threadIdx.x;
    const int half  = head_dim >> 1;

    // blockDim.x == half. Guards a misconfigured launch.
    if (i >= half) {
        return;
    }

    const int pos = positions[token];

    const float freq  = inv_freq[i];
    const float theta = static_cast<float>(pos) * freq;

    float s, c;
    __sincosf(theta, &s, &c);

    // grid y dim packs Q heads first, then K heads: [0, num_q_heads) -> q,
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

void rope_bf16(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    const int*     positions,
    const float*   inv_freq,
    int            seq_len,
    int            num_q_heads,
    int            num_kv_heads,
    int            head_dim,
    cudaStream_t   stream) {
    RUNTHERDER_CHECK(seq_len      >= 1, "seq_len must be >= 1");
    RUNTHERDER_CHECK(num_q_heads  >= 1, "num_q_heads must be >= 1");
    RUNTHERDER_CHECK(num_kv_heads >= 1, "num_kv_heads must be >= 1");
    RUNTHERDER_CHECK(head_dim     >= 2, "head_dim must be >= 2");
    RUNTHERDER_CHECK(head_dim % 2 == 0,           "head_dim must be even");
    RUNTHERDER_CHECK(head_dim <= kMaxHeadDim,     "head_dim exceeds kMaxHeadDim (256)");

    const dim3 grid(seq_len, num_q_heads + num_kv_heads);
    const dim3 block(head_dim / 2);
    rope_qk_bf16_kernel<<<grid, block, 0, stream>>>(
        q, k, positions, inv_freq, num_q_heads, num_kv_heads, head_dim);
    RUNTHERDER_CUDA_CHECK_LAST();
}

}  // namespace runtherder::kernels
