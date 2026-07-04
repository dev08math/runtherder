#include <runtherder/attention/naive.h>

#include <runtherder/kernels/attention.cuh>

namespace runtherder::attention {

NaiveAttention::NaiveAttention(const AttnConfig& config) noexcept
    : config_(config) {}

void NaiveAttention::run(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const KVView&        kv,
    int                  n_new,
    int                  cache_len,
    cudaStream_t         stream) {
    kernels::attention_causal_bf16(
        out, q, kv.k, kv.v, kv.k_scale, kv.v_scale,
        n_new,
        cache_len,
        config_.num_q_heads,
        config_.num_kv_heads,
        config_.head_dim,
        config_.scale,
        stream);
}

}  // namespace runtherder::attention
