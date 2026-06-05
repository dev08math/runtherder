#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace runtherder::attention {

/**
 * @brief Per pass scheduling input shared by every backend. Computed once per
 *        forward pass and consumed by plan(). Positions are contiguous,
 *        start_pos .. start_pos + num_tokens - 1 (no prefix caching yet).
 * @note TODO paged KV block tables join here once the KV cache manager lands.
 *       They come from that manager, not from the backend, so their shape is
 *       fixed when it exists.
 */
struct AttnStep {
    bool prefill;
    int  start_pos;
    int  num_tokens;
};

/**
 * @brief Attention backend swap seam. FlashInfer today, a custom flash
 *        attention or any other kernel later, each satisfying plan() and run().
 *        The backend owns its KV cache and head configuration, fixed at
 *        construction. Move only through the concrete type.
 *
 * Lifecycle within one forward pass:
 *   plan(step)              once, builds the schedule reused across all layers
 *   run(out, q, k, v, l)    per layer, appends this layer KV then attends
 */
class AttentionBackend {
public:
    virtual ~AttentionBackend() = default;

    AttentionBackend(const AttentionBackend&)            = delete;
    AttentionBackend& operator=(const AttentionBackend&) = delete;

    /**
     * @brief Builds the per pass schedule from step. Prefill or decode is taken
     *        from step.prefill, so run() carries no phase argument.
     * @pre Called once before the layer loop. step describes the current pass.
     * @note Async on stream.
     */
    virtual void plan(const AttnStep& step, cudaStream_t stream) = 0;

    /**
     * @brief Appends this pass k and v for layer into the owned KV cache, then
     *        writes attention output to out. Uses the schedule from the last
     *        plan().
     * @pre plan() ran for the current pass. q [num_tokens, q_dim], k and v
     *      [num_tokens, kv_dim], all device, post RoPE and norm. out distinct
     *      from q, k, v. layer in [0, num_layers).
     * @note Async on stream. Result not valid until the stream is synchronized.
     */
    virtual void run(__nv_bfloat16*       out,
                     const __nv_bfloat16* q,
                     const __nv_bfloat16* k,
                     const __nv_bfloat16* v,
                     int                  layer,
                     cudaStream_t         stream) = 0;

protected:
    AttentionBackend()                                        = default;
    AttentionBackend(AttentionBackend&&) noexcept             = default;
    AttentionBackend& operator=(AttentionBackend&&) noexcept  = default;
};

}  // namespace runtherder::attention
