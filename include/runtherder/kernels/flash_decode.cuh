#pragma once

#include <cstddef>

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

/**
 * @brief Float count for the partial scratch of a split decode at max_splits.
 *        Layout is [num_q_heads, max_splits] m, then l, then
 *        [num_q_heads, max_splits, head_dim] acc, one contiguous slab.
 */
[[nodiscard]] constexpr std::size_t flash_split_scratch_floats(
    int num_q_heads,
    int head_dim,
    int max_splits) {
    const std::size_t heads  = static_cast<std::size_t>(num_q_heads);
    const std::size_t splits = static_cast<std::size_t>(max_splits);
    const std::size_t dim    = static_cast<std::size_t>(head_dim);
    return heads * splits * 2 + heads * splits * dim;
}

/**
 * @brief Split KV flash decode: a partial kernel over num_splits chunks of the
 *        key history, then a reduce that merges the partial softmaxes.
 * @param out         [num_q_heads, head_dim]
 * @param q           [num_q_heads, head_dim]
 * @param k           [cache_len + 1, num_kv_heads, head_dim]
 * @param v           [cache_len + 1, num_kv_heads, head_dim]
 * @param partial     caller owned scratch, at least
 *                    flash_split_scratch_floats(num_q_heads, head_dim, num_splits)
 *                    floats. Carved into m, l, acc by the launcher.
 * @param num_splits  chunk count, >= 1. partial must be sized for it.
 * @param cache_len   keys cached before this query. Attends cache_len + 1 keys.
 * @note Same shape and dtype preconditions as flash_decode_bf16. Output is the
 *       online softmax result. The split reassociates the float accumulation so
 *       it can differ from the single block kernel in the last bits, not in value.
 */
void flash_decode_split_bf16(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    float*               partial,
    int                  num_splits,
    int                  cache_len,
    int                  num_q_heads,
    int                  num_kv_heads,
    int                  head_dim,
    float                scale,
    cudaStream_t         stream = nullptr);

}  // namespace runtherder::kernels
