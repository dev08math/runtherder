#include <runtherder/attention/adaptive.h>

#include <runtherder/kernels/flash_decode.cuh>

namespace runtherder::attention {

namespace {

// Fixed, not sized from the history length: cache_len is device resident and
// leaves the host nothing to branch on. The partial kernel derives its chunk
// from *cache_len and sentinels the splits past the key count, correct at every
// length. 24 heads * 8 = 192 blocks on the 36 SM 4070 Laptop. A short history
// pays 8 near empty blocks where the old policy paid 1.
constexpr int kMaxSplits = 8;

}  // namespace

AdaptiveAttention::AdaptiveAttention(const AttnConfig& config)
    : config_(config),
      prefill_(config),
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

    prefill_.run(out, q, kv, n_new, cache_len, stream);
}

}  // namespace runtherder::attention
