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
 * @brief Interface for one causal attention implementation.
 */
class AttentionBackend {
public:
    virtual ~AttentionBackend() = default;

    AttentionBackend(const AttentionBackend&)            = delete;
    AttentionBackend& operator=(const AttentionBackend&) = delete;

    /**
     * @brief Causal attention.
     * @param out        [n_new, num_q_heads, head_dim]
     * @param q          [n_new, num_q_heads, head_dim]
     * @param kv         the layer's cache, with the n_new new keys already
     *                   appended
     * @param n_new      queries this call attends, >= 1
     * @param cache_len  device resident, the keys already cached before these
     *                   n_new queries.
     * @param stream     CUDA stream every kernel is launched on.
     */
    virtual void run(__nv_bfloat16*       out,
                     const __nv_bfloat16* q,
                     const KVView&        kv,
                     int                  n_new,
                     const int*           cache_len,
                     cudaStream_t         stream) = 0;

protected:
    AttentionBackend()                                       = default;
    AttentionBackend(AttentionBackend&&) noexcept            = default;
    AttentionBackend& operator=(AttentionBackend&&) noexcept = default;
};

}  // namespace runtherder::attention
