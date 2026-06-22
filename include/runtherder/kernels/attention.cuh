#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

// Causal attention. q,out are [n_new, num_q_heads, head_dim], k,v are
// [cache_len+n_new, num_kv_heads, head_dim]. out distinct from k,v.
void attention_causal_bf16(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    int                  n_new,
    int                  cache_len,
    int                  num_q_heads,
    int                  num_kv_heads,
    int                  head_dim,
    float                scale,
    cudaStream_t         stream = nullptr);

}  // namespace runtherder::kernels
