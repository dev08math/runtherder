#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

// SwiGLU, BF16 storage, FP32 math (__expf).
// y[i] = silu(gate[i]) * up[i].
//
// Preconditions: device ptrs 16B-aligned; n >= 8, n % 8 == 0.
// Aliasing: y may alias gate or up.
// Numerics: max-abs <= 5e-2, cos-sim >= 0.9999 vs FP64 ref. Async on `stream`.
void swiglu_bf16_forward(
    __nv_bfloat16*       y,
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    int                  n,
    cudaStream_t         stream = nullptr);

}  // namespace runtherder::kernels
