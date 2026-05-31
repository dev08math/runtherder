#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

// Caller contract for both launchers, not checked in the kernel: device
// pointers 16 byte aligned, num_tokens >= 1, 8 <= hidden_dim <= 16384,
// hidden_dim % 8 == 0, eps > 0. Async on stream. Matches a double precision
// reference within max abs 5e-2, cosine 0.9999.

/**
 * @brief y = (x / rms(x)) * g.
 * @note y may alias x.
 */
void rmsnorm_bf16_forward(
    __nv_bfloat16*       y,
    const __nv_bfloat16* x,
    const __nv_bfloat16* g,
    int                  num_tokens,
    int                  hidden_dim,
    float                eps,
    cudaStream_t         stream = nullptr);

/**
 * @brief h_out = x + residual, then y = (h_out / rms(h_out)) * g.
 * @note h_out may alias x or residual. y must be distinct from h_out.
 */
void rmsnorm_add_bf16_forward(
    __nv_bfloat16*       y,
    __nv_bfloat16*       h_out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* g,
    int                  num_tokens,
    int                  hidden_dim,
    float                eps,
    cudaStream_t         stream = nullptr);

}  // namespace runtherder::kernels
