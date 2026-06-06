#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::attention {

/**
 * @brief Fixed head configuration a backend is built with. Decoupled from the
 *        model layer on purpose, the attention domain does not depend on
 *        LlamaConfig. scale multiplies each score before softmax, normally
 *        1 / sqrt(head_dim).
 */
struct AttnConfig {
    int   num_q_heads;
    int   num_kv_heads;
    int   head_dim;
    float scale;
};

/**
 * @brief Attention backend swap seam. A naive prefill kernel today, a custom
 *        optimized kernel later, each satisfying run(). Head configuration is
 *        fixed at construction. Move only through the concrete type.
 *
 * Prefill only for now: every key is in the caller scratch, full causal over
 * num_tokens. KV cache and a decode path land in a later slice that revisits
 * this contract.
 */
class AttentionBackend {
public:
    virtual ~AttentionBackend() = default;

    AttentionBackend(const AttentionBackend&)            = delete;
    AttentionBackend& operator=(const AttentionBackend&) = delete;

    /**
     * @brief Full causal prefill attention over q, k, v in caller scratch.
     *        Writes attention output to out.
     * @pre q and out [num_tokens, num_q_heads, head_dim], k and v
     *      [num_tokens, num_kv_heads, head_dim], all device, post RoPE and
     *      post q/k norm. out distinct from q, k, v. num_tokens >= 1.
     * @note Async on stream. Result not valid until the stream is synchronized.
     */
    virtual void run(__nv_bfloat16*       out,
                     const __nv_bfloat16* q,
                     const __nv_bfloat16* k,
                     const __nv_bfloat16* v,
                     int                  num_tokens,
                     cudaStream_t         stream) = 0;

protected:
    AttentionBackend()                                       = default;
    AttentionBackend(AttentionBackend&&) noexcept            = default;
    AttentionBackend& operator=(AttentionBackend&&) noexcept = default;
};

}  // namespace runtherder::attention
