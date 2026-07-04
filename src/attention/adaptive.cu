#include <runtherder/attention/adaptive.h>

#include <algorithm>

#include <runtherder/kernels/flash_decode.cuh>

namespace runtherder::attention {

namespace {

// Split policy. A chunk holds at least kMinChunkKeys keys to be worth its own
// block, capped at kMaxSplits so num_q_heads * kMaxSplits fills the GPU without
// oversubscribing (24 heads * 8 = 192 blocks on the 36 SM 4070 Laptop). Below
// kMinChunkKeys of history the split is not worth its second launch, so the
// single block kernel handles it.
constexpr int kMaxSplits    = 8;
constexpr int kMinChunkKeys = 256;

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
    int                  cache_len,
    cudaStream_t         stream) {
    if (n_new == 1) {
        const int n_keys = cache_len + 1;
        const int splits =
            std::min(kMaxSplits, std::max(1, (n_keys + kMinChunkKeys - 1) / kMinChunkKeys));
        if (splits > 1) {
            kernels::flash_decode_split_bf16(
                out, q, kv.k, kv.v, kv.k_scale, kv.v_scale, partial_.get(),
                splits, cache_len, config_.num_q_heads, config_.num_kv_heads,
                config_.head_dim, config_.scale, stream);
        } else {
            kernels::flash_decode_bf16(
                out, q, kv.k, kv.v, kv.k_scale, kv.v_scale, cache_len,
                config_.num_q_heads, config_.num_kv_heads, config_.head_dim,
                config_.scale, stream);
        }
        return;
    }

    prefill_.run(out, q, kv, n_new, cache_len, stream);
}

}  // namespace runtherder::attention
