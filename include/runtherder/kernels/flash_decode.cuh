#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

/**
 * @brief Causal attention for a single query token against a cache.
 * @param out        [num_q_heads, head_dim]
 * @param q          [num_q_heads, head_dim]
 * @param k          [cache_len + 1, num_kv_heads, head_dim]
 * @param v          [cache_len + 1, num_kv_heads, head_dim]
 * @param cache_len  keys cached before this query. Attends cache_len + 1 keys.
 * @note head_dim maps to block threads, capped at the launch site.
 */
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
    cudaStream_t         stream = nullptr);

}  // namespace runtherder::kernels
