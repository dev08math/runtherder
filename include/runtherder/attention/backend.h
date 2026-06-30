#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/attention/kv_view.h>

namespace runtherder::attention {

/**
 * @brief Head configuration fixed at backend construction. scale multiplies
 *        each score before softmax, normally 1 / sqrt(head_dim).
 */
struct AttnConfig {
    int   num_q_heads;
    int   num_kv_heads;
    int   head_dim;
    float scale;
};

/**
 * @brief Attention backend selector.
 */
class AttentionBackend {
public:
    virtual ~AttentionBackend() = default;

    AttentionBackend(const AttentionBackend&)            = delete;
    AttentionBackend& operator=(const AttentionBackend&) = delete;

    /**
     * @brief Causal attention. out is [n_new, num_q_heads, head_dim].
     * @param q          [n_new, num_q_heads, head_dim]
     * @param kv         cached keys and values, each [cache_len + n_new, num_kv_heads, head_dim]
     * @param cache_len  keys already cached before these n_new queries
     */
    virtual void run(__nv_bfloat16*       out,
                     const __nv_bfloat16* q,
                     const KVView&        kv,
                     int                  n_new,
                     int                  cache_len,
                     cudaStream_t         stream) = 0;

protected:
    AttentionBackend()                                       = default;
    AttentionBackend(AttentionBackend&&) noexcept            = default;
    AttentionBackend& operator=(AttentionBackend&&) noexcept = default;
};

}  // namespace runtherder::attention
