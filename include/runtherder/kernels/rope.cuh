#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

/**
 * @brief In place RoPE on q and k. Token t rotates by positions[t].
 * @param q          [seq_len, num_q_heads, head_dim]
 * @param k          [seq_len, num_kv_heads, head_dim]
 * @param positions  [seq_len]
 * @param inv_freq   [head_dim / 2], per pair rotation frequency, plain or scaled by the caller
 * @note Matches a double precision reference within max abs 5e-2, cosine 0.9999.
 */
void rope_bf16(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    const int*     positions,
    const float*   inv_freq,
    int            seq_len,
    int            num_q_heads,
    int            num_kv_heads,
    int            head_dim,
    cudaStream_t   stream = nullptr);

}  // namespace runtherder::kernels
