#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

/**
 * @brief y[i] = silu(gate[i]) * up[i].
 * @param y     [n]
 * @param gate  [n]
 * @param up    [n]
 * @note y may alias gate or up. Matches a double precision reference within
 *       max abs 5e-2, cosine 0.9999.
 */
void swiglu_bf16_forward(
    __nv_bfloat16*       y,
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    int                  n,
    cudaStream_t         stream = nullptr);

}  // namespace runtherder::kernels
