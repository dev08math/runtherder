#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/attention/backend.h>
#include <runtherder/device/memory.cuh>

namespace runtherder::attention {

/**
 * @brief Selects the attention kernel from the token count: tensor core flash
 *        prefill for n_new > 1, split flash decode for n_new == 1.
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
    device::DeviceUniquePtr<float> partial_;  // split decode reduction scratch
};

}  // namespace runtherder::attention
