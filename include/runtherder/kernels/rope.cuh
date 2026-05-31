#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

/**
 * @brief In place RoPE on q and k, half split convention (HF/Llama/Qwen).
 *        Layouts: q [seq_len, num_q_heads, head_dim],
 *        k [seq_len, num_kv_heads, head_dim]. Token t rotates by
 *        pos = start_pos + t.
 * @pre Device pointers 16 byte aligned, q and k distinct. seq_len >= 1,
 *      start_pos >= 0, num_q_heads >= 1, num_kv_heads >= 1,
 *      2 <= head_dim <= 256, head_dim % 2 == 0, theta_base > 0.
 * @note Async on stream. Matches a double precision reference within
 *       max abs 5e-2, cosine 0.9999.
 */
void rope_bf16_inplace(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    int            start_pos,
    int            seq_len,
    int            num_q_heads,
    int            num_kv_heads,
    int            head_dim,
    float          theta_base,
    cudaStream_t   stream = nullptr);

}  // namespace runtherder::kernels
