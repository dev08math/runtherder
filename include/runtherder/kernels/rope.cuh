#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

// RoPE, BF16 storage, FP32 trig, half-split (HF/Llama/Qwen), in-place on q and k.
//
// Layouts:
//   q : [seq_len, num_q_heads,  head_dim]
//   k : [seq_len, num_kv_heads, head_dim]
// Token t rotates by pos = start_pos + t.
//
// Preconditions (caller, not validated in-kernel):
//   - Device pointers, 16-byte aligned. q and k distinct.
//   - seq_len >= 1, start_pos >= 0.
//   - num_q_heads, num_kv_heads >= 1.
//   - 2 <= head_dim <= 256, head_dim % 2 == 0.
//   - theta_base > 0.
//
// Async on `stream`.
void rope_bf16_inplace(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    int            start_pos,
    int            seq_len,
    int            num_q_heads,
    int            num_kv_heads,
    int            head_dim,
    float          theta_base,
    cudaStream_t   stream = nullptr);

}  // namespace runtherder::kernels
