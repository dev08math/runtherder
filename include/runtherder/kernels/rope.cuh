#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

/**
 * @brief In place RoPE on q and k. Token t rotates by positions[t].
 *        Layouts: q [seq_len, num_q_heads, head_dim],
 *        k [seq_len, num_kv_heads, head_dim].
 * @pre q and k 16 byte aligned and distinct. positions device, length seq_len,
 *      each value >= 0. seq_len >= 1, num_q_heads >= 1, num_kv_heads >= 1,
 *      2 <= head_dim <= 256, head_dim % 2 == 0, theta_base > 0.
 * @note Async on stream. Matches a double precision reference within
 *       max abs 5e-2, cosine 0.9999.
 */
void rope_bf16(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    const int*     positions,
    int            seq_len,
    int            num_q_heads,
    int            num_kv_heads,
    int            head_dim,
    float          theta_base,
    cudaStream_t   stream = nullptr);

}  // namespace runtherder::kernels
