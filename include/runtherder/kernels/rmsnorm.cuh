#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

/**
 * @brief y = (x / rms(x)) * g.
 * @param y  [num_tokens, hidden_dim]
 * @param x  [num_tokens, hidden_dim]
 * @param g  [hidden_dim]
 * @note y may alias x. Matches a double precision reference within max abs 5e-2, cosine 0.9999.
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
 * @brief Fused residual add then rmsnorm. h_out = x + residual, then
 *        y = (h_out / rms(h_out)) * g. Unlike rmsnorm_bf16_forward it folds the
 *        residual stream in first, use it for every norm after the first.
 * @param y         [num_tokens, hidden_dim]
 * @param h_out     [num_tokens, hidden_dim]
 * @param x         [num_tokens, hidden_dim]
 * @param residual  [num_tokens, hidden_dim]
 * @param g         [hidden_dim]
 * @note h_out may alias residual. y must be distinct from h_out. Matches a
 *       double precision reference within max abs 5e-2, cosine 0.9999.
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
