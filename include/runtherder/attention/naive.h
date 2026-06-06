#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/attention/backend.h>

namespace runtherder::attention {

/**
 * @brief Reference attention backend. run() forwards to attention_prefill_bf16,
 *        one thread per output row, no KV cache. The baseline a custom optimized
 *        backend is later validated against.
 */
class NaiveAttention final : public AttentionBackend {
public:
    explicit NaiveAttention(const AttnConfig& config) noexcept;

    void run(__nv_bfloat16*       out,
             const __nv_bfloat16* q,
             const __nv_bfloat16* k,
             const __nv_bfloat16* v,
             int                  num_tokens,
             cudaStream_t         stream) override;

private:
    AttnConfig config_;
};

}  // namespace runtherder::attention
