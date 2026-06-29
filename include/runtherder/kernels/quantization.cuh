#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace runtherder::kernels {

// Symmetric, no zero point. dim % 8 == 0 (vectorized load).
void quantize_per_token_int8(std::int8_t*         q,
                             float*               x_scale,
                             const __nv_bfloat16* x,
                             int                  num_tokens,
                             int                  dim,
                             cudaStream_t         stream = nullptr);

// x_scale per row, w_scale per column.
void dequantize_w8a8(__nv_bfloat16*       y,
                     const std::int32_t*  acc,
                     const float*         x_scale,
                     const __nv_bfloat16* w_scale,
                     int                  m,
                     int                  n,
                     cudaStream_t         stream = nullptr);

}  // namespace runtherder::kernels
