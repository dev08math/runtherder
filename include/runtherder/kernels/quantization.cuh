#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace runtherder::kernels {

/**
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
 * @param q      [num_rows, head_dim]  E4M3
 * @param scale  [num_rows]            per row, amax / 448, symmetric
 * @param x      [num_rows, head_dim]  bf16 source
 *
 * One row per quant group. For a per (token, kv_head) KV scale pass
 * num_rows = n_new * num_kv_heads and head_dim as the row width.
 */
void quantize_kv_fp8(__nv_fp8_e4m3*       q,
                     float*               scale,
                     const __nv_bfloat16* x,
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
