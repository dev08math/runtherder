#include <runtherder/attention/adaptive.h>

#include <runtherder/kernels/flash_decode.cuh>

namespace runtherder::attention {

AdaptiveAttention::AdaptiveAttention(const AttnConfig& config) noexcept
    : config_(config), prefill_(config) {}

void AdaptiveAttention::run(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const KVView&        kv,
    int                  n_new,
    int                  cache_len,
    cudaStream_t         stream) {
    if (n_new == 1) {
        kernels::flash_decode_bf16(
            out, q, kv.k, kv.v,
            cache_len,
            config_.num_q_heads,
            config_.num_kv_heads,
            config_.head_dim,
            config_.scale,
            stream);
        return;
    }

    prefill_.run(out, q, kv, n_new, cache_len, stream);
}

}  // namespace runtherder::attention
