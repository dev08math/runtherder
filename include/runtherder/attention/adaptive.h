#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/attention/backend.h>
#include <runtherder/attention/naive.h>

namespace runtherder::attention {

/**
 * @brief Selects the attention kernel per request. n_new == 1 takes the flash
 *        decode kernel, prefill (n_new > 1) delegates to NaiveAttention.
 */
class AdaptiveAttention final : public AttentionBackend {
public:
    explicit AdaptiveAttention(const AttnConfig& config) noexcept;

    void run(__nv_bfloat16*       out,
             const __nv_bfloat16* q,
             const KVView&        kv,
             int                  n_new,
             int                  cache_len,
             cudaStream_t         stream) override;

private:
    AttnConfig     config_;
    NaiveAttention prefill_;
};

}  // namespace runtherder::attention
