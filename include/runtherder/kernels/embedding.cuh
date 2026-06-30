#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::kernels {

/**
 * @brief out[i] = embedding_table[token_ids[i]] for each of num_tokens rows.
 * @param out              [num_tokens, hidden_dim]
 * @param embedding_table  [vocab_size, hidden_dim]
 * @param token_ids        [num_tokens], values in [0, vocab_size)
 */
void embedding_lookup_bf16_forward(
    __nv_bfloat16*       out,
    const __nv_bfloat16* embedding_table,
    const int*           token_ids,
    int                  num_tokens,
    int                  hidden_dim,
    cudaStream_t         stream = nullptr);

}  // namespace runtherder::kernels
