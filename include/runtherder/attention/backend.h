#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/attention/kv_view.h>

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
 * @brief Attention backend swap seam. A naive kernel today, a custom optimized
 *        kernel later, each satisfying run(). Head configuration is fixed at
 *        construction.
 */
class AttentionBackend {
public:
    virtual ~AttentionBackend() = default;

    AttentionBackend(const AttentionBackend&)            = delete;
    AttentionBackend& operator=(const AttentionBackend&) = delete;

    /**
     * @brief Causal attention of n_new query tokens against the cached keys and
     *        values in kv. Writes attention output to out.
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
