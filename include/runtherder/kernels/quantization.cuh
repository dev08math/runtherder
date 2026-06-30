#pragma once

#include <cuda_bf16.h>
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

}  // namespace runtherder::kernels
