#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

/**
 * @brief Causal attention over an E4M3 KV cache on tensor cores. Unlike
 *        attention_causal_bf16 it feeds the scores and the value product
 *        through mma, use it for prefill (n_new > 1).
 * @param out       [n_new, num_q_heads, head_dim] bf16
 * @param q         [n_new, num_q_heads, head_dim] bf16
 * @param k         [cache_len + n_new, num_kv_heads, head_dim] E4M3
 * @param v         [cache_len + n_new, num_kv_heads, head_dim] E4M3
 * @param k_scale   [cache_len + n_new, num_kv_heads] per (token, kv_head) dequant
 * @param v_scale   [cache_len + n_new, num_kv_heads] per (token, kv_head) dequant
 * @param cache_len device resident, the keys cached before these n_new queries.
 *                  Unchecked at the launch site.
 */
void flash_prefill_bf16(
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
    cudaStream_t         stream = nullptr);

}  // namespace runtherder::kernels
