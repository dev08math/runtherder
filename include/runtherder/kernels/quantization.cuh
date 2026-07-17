#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace runtherder::kernels {

/**
 * @brief q = round(x / x_scale), x_scale = row amax / 127.
 * @param q        [num_tokens, dim]
 * @param x_scale  [num_tokens], symmetric so no zero point
 * @param x        [num_tokens, dim]
 */
void quantize_per_token_int8(std::int8_t*         q,
                             float*               x_scale,
                             const __nv_bfloat16* x,
                             int                  num_tokens,
                             int                  dim,
                             cudaStream_t         stream = nullptr);

/**
 * @brief y = acc * x_scale * w_scale.
 * @param y        [m, n]
 * @param acc      [m, n]
 * @param x_scale  [m], per row
 * @param w_scale  [n], per column
 */
void dequantize_w8a8(__nv_bfloat16*       y,
                     const std::int32_t*  acc,
                     const float*         x_scale,
                     const __nv_bfloat16* w_scale,
                     int                  m,
                     int                  n,
                     cudaStream_t         stream = nullptr);

/**
 * @brief q = x / scale, scale = row amax / 448. One row per quant group.
 * @param q             E4M3 destination, rows at_pos * rows_per_pos onward
 * @param scale         per row, amax / 448, symmetric. Same rows as q.
 * @param x             [num_rows, head_dim] bf16 source, indexed from row 0
 * @param at_pos        device resident, the position the write lands at.
 *                      Unchecked, the caller owns the bound against q.
 * @param rows_per_pos  rows one position spans, num_kv_heads for a KV slab
 * @note For a per (token, kv_head) KV scale pass num_rows = n_new * num_kv_heads
 *       and head_dim as the row width.
 */
void quantize_kv_fp8(__nv_fp8_e4m3*       q,
                     float*               scale,
                     const __nv_bfloat16* x,
                     const int*           at_pos,
                     int                  rows_per_pos,
                     int                  num_rows,
                     int                  head_dim,
                     cudaStream_t         stream = nullptr);

/**
 * @brief Per value inverse of quantize_kv_fp8.
 * @param scale  the stored value's row scale from quantize_kv_fp8
 */
[[nodiscard]] __host__ __device__ __forceinline__ float dequantize_kv_fp8(
    __nv_fp8_e4m3 q, float scale) {
    return static_cast<float>(q) * scale;
}

}  // namespace runtherder::kernels
