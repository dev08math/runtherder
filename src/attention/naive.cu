#include <runtherder/attention/naive.h>

#include <runtherder/kernels/attention.cuh>

namespace runtherder::attention {

NaiveAttention::NaiveAttention(const AttnConfig& config) noexcept
    : config_(config) {}

void NaiveAttention::run(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    int                  num_tokens,
    cudaStream_t         stream) {
    kernels::attention_prefill_bf16(
        out, q, k, v,
        num_tokens,
        config_.num_q_heads,
        config_.num_kv_heads,
        config_.head_dim,
        config_.scale,
        stream);
}

}  // namespace runtherder::attention
