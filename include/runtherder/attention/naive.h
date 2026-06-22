#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/attention/backend.h>

namespace runtherder::attention {

// Reference attention backend, one thread per output row.
class NaiveAttention final : public AttentionBackend {
public:
    explicit NaiveAttention(const AttnConfig& config) noexcept;

    void run(__nv_bfloat16*       out,
             const __nv_bfloat16* q,
             const KVView&        kv,
             int                  n_new,
             int                  cache_len,
             cudaStream_t         stream) override;

private:
    AttnConfig config_;
};

}  // namespace runtherder::attention
