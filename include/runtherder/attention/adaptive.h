#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/attention/backend.h>
#include <runtherder/attention/naive.h>
#include <runtherder/device/memory.cuh>

namespace runtherder::attention {

/**
 * @brief Selects the attention kernel per request. Prefill (n_new > 1) delegates
 *        to NaiveAttention. Decode (n_new == 1) takes the single block flash
 *        kernel for a shallow cache, or the split kernel once the cache is deep
 *        enough that one block per head underfills the GPU.
 */
class AdaptiveAttention final : public AttentionBackend {
public:
    explicit AdaptiveAttention(const AttnConfig& config);

    void run(__nv_bfloat16*       out,
             const __nv_bfloat16* q,
             const KVView&        kv,
             int                  n_new,
             const int*           cache_len,
             cudaStream_t         stream) override;

private:
    AttnConfig                     config_;
    NaiveAttention                 prefill_;
    device::DeviceUniquePtr<float> partial_;  // split decode reduction scratch
};

}  // namespace runtherder::attention
