#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

/**
 * @brief Prefill causal self attention over all num_tokens prompt tokens at
 *        once. For query token i and head h: score against every key token
 *        j <= i (causal), softmax, weighted sum of v_j. GQA maps query head h
 *        to kv head h / (num_q_heads / num_kv_heads).
 *        Layouts: q and out [num_tokens, num_q_heads, head_dim],
 *        k and v [num_tokens, num_kv_heads, head_dim].
 * @pre out distinct from q, k, v. q, k, v post RoPE and post q/k norm.
 *      num_tokens >= 1, num_q_heads >= 1, num_kv_heads >= 1,
 *      num_q_heads % num_kv_heads == 0, 1 <= head_dim <= 256.
 * @note scale multiplies each score before softmax, normally 1 / sqrt(head_dim).
 *       Accumulates the dot and the v sum in FP32, stores BF16. Two pass softmax
 *       (max then exp sum), no flash style online rescale. One thread per output
 *       row, the readable reference. Async on stream. Matches a double precision
 *       reference within max abs 5e-2, cosine 0.9999.
 */
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
    cudaStream_t         stream = nullptr);

}  // namespace runtherder::kernels
