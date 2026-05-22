#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

// RMSNorm, BF16 storage, FP32 reduction.
//
// Preconditions (caller, not validated in-kernel):
//   - Device pointers, 16-byte aligned.
//   - num_tokens >= 1, 8 <= hidden_dim <= 16384, hidden_dim % 8 == 0.
//   - eps > 0.
//
// Numerics: max-abs error <= 5e-2, cosine similarity >= 0.9999 vs. a
// double-precision CPU reference.
//
// Async on `stream`.

// y = (x / rms(x)) * g.
// Aliasing: y may alias x.
void rmsnorm_bf16_forward(
    __nv_bfloat16*       y,
    const __nv_bfloat16* x,
    const __nv_bfloat16* g,
    int                  num_tokens,
    int                  hidden_dim,
    float                eps,
    cudaStream_t         stream = nullptr);

// h_out = x + residual,  y = (h_out / rms(h_out)) * g.
// Aliasing: h_out may alias x or residual. y must be distinct from h_out.
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
