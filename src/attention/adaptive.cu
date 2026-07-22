#include <runtherder/attention/adaptive.h>

#include <runtherder/kernels/flash_decode.cuh>
#include <runtherder/kernels/flash_prefill.cuh>

namespace runtherder::attention {

namespace {

// cache_len is device resident, the host cannot size this per step, so the
// split count is fixed and each split derives its key range on device. Higher
// gives long context decode more parallelism, empty splits stay cheap at short.
constexpr int kMaxSplits = 32;

}  // namespace

AdaptiveAttention::AdaptiveAttention(const AttnConfig& config)
    : config_(config),
      partial_(device::make_device_unique<float>(kernels::flash_split_scratch_floats(
          config.num_q_heads, config.head_dim, kMaxSplits))) {}

void AdaptiveAttention::run(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const KVView&        kv,
    int                  n_new,
    const int*           cache_len,
    cudaStream_t         stream) {
    if (n_new == 1) {
        kernels::flash_decode_split_bf16(
            out, q, kv.k, kv.v, kv.k_scale, kv.v_scale, partial_.get(),
            kMaxSplits, cache_len, config_.num_q_heads, config_.num_kv_heads,
            config_.head_dim, config_.scale, stream);
        return;
    }

    kernels::flash_prefill_bf16(
        out, q, kv.k, kv.v, kv.k_scale, kv.v_scale, n_new, cache_len,
        config_.num_q_heads, config_.num_kv_heads, config_.head_dim,
        config_.scale, stream);
}

}  // namespace runtherder::attention
