#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

// Caller contract, not checked in the kernel: device pointers 16 byte
// aligned, num_tokens >= 1, hidden_dim % 8 == 0, token_ids values in
// [0, vocab_size). Async on stream. Bit exact row copy, no tolerance.

/**
 * @brief out[i] = embedding_table[token_ids[i]] for each of num_tokens rows.
 * @note out must not alias embedding_table.
 */
void embedding_lookup_bf16_forward(
    __nv_bfloat16*       out,
    const __nv_bfloat16* embedding_table,
    const int*           token_ids,
    int                  num_tokens,
    int                  hidden_dim,
    cudaStream_t         stream = nullptr);

}  // namespace runtherder::kernels
