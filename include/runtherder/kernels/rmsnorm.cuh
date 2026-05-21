#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

// Root Mean Square layer normalization, BF16 forward.
//
// For each token row of `x` (length `hidden_dim`):
//     rms       = sqrt( mean(x_i^2) + eps )
//     y_i       = (x_i / rms) * g_i
//
// Storage is BF16; reductions and the scalar `rms` are computed in FP32
// internally for numerical stability, then the output is cast back to BF16.
//
// Shapes (row-major, contiguous):
//   x : [num_tokens, hidden_dim]   device pointer, BF16
//   y : [num_tokens, hidden_dim]   device pointer, BF16 (may alias x)
//   g : [hidden_dim]               device pointer, BF16, learned gain
//
// Preconditions (caller-enforced, kernel does not validate):
//   - All pointers refer to device memory, 16-byte aligned (true for any
//     `cudaMalloc` result).
//   - num_tokens >= 1.
//   - hidden_dim >= 8 and hidden_dim % 8 == 0  (vectorized BF16 load width).
//   - eps > 0  (from Config::rms_norm_eps).
//
// Launch: one CUDA block per token row; threads within a block cooperate on
// the reduction over `hidden_dim` via warp-shuffle + shared-memory combine.
// Asynchronous on `stream`.
void rmsnorm_bf16_forward(
    __nv_bfloat16*       y,
    const __nv_bfloat16* x,
    const __nv_bfloat16* g,
    int                  num_tokens,
    int                  hidden_dim,
    float                eps,
    cudaStream_t         stream = nullptr);

// Fused residual-add + RMSNorm, BF16 forward.
//
// For each token row:
//     h_i       = x_i + residual_i      (BF16 add via FP32)
//     rms       = sqrt( mean(h_i^2) + eps )
//     y_i       = (h_i / rms) * g_i
//
// Writes TWO outputs:
//   h_out : [num_tokens, hidden_dim]   the post-residual hidden state, fed
//                                      into the NEXT residual add in the
//                                      transformer block.
//   y     : [num_tokens, hidden_dim]   the normalized + scaled output, fed
//                                      into the immediately-following
//                                      attention or MLP sub-block.
//
// Shapes (row-major, contiguous, all BF16, all device pointers):
//   x        : [num_tokens, hidden_dim]
//   residual : [num_tokens, hidden_dim]
//   g        : [hidden_dim]
//   h_out    : [num_tokens, hidden_dim]   (may alias x or residual)
//   y        : [num_tokens, hidden_dim]   (distinct from h_out)
//
// Preconditions: same as rmsnorm_bf16_forward.
//
// Launch: one CUDA block per token row. Asynchronous on `stream`.
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
