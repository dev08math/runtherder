#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

/**
 * @brief q[r, :] += q_bias, k[r, :] += k_bias, v[r, :] += v_bias for every row r,
 *        in place.
 * @param q       [n, q_dim]
 * @param k       [n, kv_dim]
 * @param v       [n, kv_dim]
 * @param q_bias  [q_dim]
 * @param k_bias  [kv_dim]
 * @param v_bias  [kv_dim]
 * @pre q_dim and kv_dim multiples of 8. Every pointer 16 byte aligned.
 * @note Each output is bf16(float(x) + float(b)), bit identical to that reference.
 */
void qkv_bias_add_bf16_forward(
    __nv_bfloat16*       q,
    __nv_bfloat16*       k,
    __nv_bfloat16*       v,
    const __nv_bfloat16* q_bias,
    const __nv_bfloat16* k_bias,
    const __nv_bfloat16* v_bias,
    int                  n,
    int                  q_dim,
    int                  kv_dim,
    cudaStream_t         stream = nullptr);

}  // namespace runtherder::kernels
