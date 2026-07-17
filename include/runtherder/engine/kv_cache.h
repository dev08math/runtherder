#pragma once

#include <cstddef>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <runtherder/attention/kv_view.h>
#include <runtherder/check.h>
#include <runtherder/device/check.cuh>
#include <runtherder/device/memory.cuh>
#include <runtherder/kernels/quantization.cuh>

namespace runtherder::engine {

/**
 * @brief Single sequence contiguous KV cache stored as E4M3. K and V slabs are
 *        [num_layers, max_seq_len, num_kv_heads, head_dim]. The parallel scale
 *        slabs are [num_layers, max_seq_len, num_kv_heads], one dequant scale per
 *        (token, kv_head). append takes bf16 and quantizes on the way in, so the
 *        fp8 format never leaves the cache. Caller owns the write position.
 */
class KVCache {
public:
    KVCache(int num_layers, int max_seq_len, int num_kv_heads, int head_dim)
        : num_layers_(num_layers),
          max_seq_len_(max_seq_len),
          num_kv_heads_(num_kv_heads),
          head_dim_(head_dim),
          kv_dim_(num_kv_heads * head_dim),
          layer_stride_(max_seq_len * (num_kv_heads * head_dim)),
          scale_layer_stride_(max_seq_len * num_kv_heads) {
        RUNTHERDER_CHECK(num_layers >= 1,   "KVCache num_layers must be >= 1");
        RUNTHERDER_CHECK(max_seq_len >= 1,  "KVCache max_seq_len must be >= 1");
        RUNTHERDER_CHECK(num_kv_heads >= 1, "KVCache num_kv_heads must be >= 1");
        RUNTHERDER_CHECK(head_dim >= 1,     "KVCache head_dim must be >= 1");

        const std::size_t total = static_cast<std::size_t>(num_layers_) *
                                  static_cast<std::size_t>(layer_stride_);
        const std::size_t scale_total = static_cast<std::size_t>(num_layers_) *
                                        static_cast<std::size_t>(scale_layer_stride_);
        k_ = device::make_device_unique<__nv_fp8_e4m3>(total);
        v_ = device::make_device_unique<__nv_fp8_e4m3>(total);
        k_scale_ = device::make_device_unique<float>(scale_total);
        v_scale_ = device::make_device_unique<float>(scale_total);
    }

    KVCache(KVCache&&) noexcept            = default;
    KVCache& operator=(KVCache&&) noexcept = default;
    KVCache(const KVCache&)                = delete;
    KVCache& operator=(const KVCache&)     = delete;

    /**
     * @brief Quantizes n_new bf16 keys and values into layer at row *at_pos.
     * @param k       [n_new, num_kv_heads, head_dim] bf16
     * @param v       [n_new, num_kv_heads, head_dim] bf16
     * @param at_pos  device resident write position. Unchecked against
     *                max_seq_len, the caller owns that bound.
     */
    void append(int layer, const __nv_bfloat16* k, const __nv_bfloat16* v,
                int n_new, const int* at_pos, cudaStream_t stream) {
        RUNTHERDER_CHECK(layer >= 0 && layer < num_layers_, "KVCache append layer out of range");
        RUNTHERDER_CHECK(n_new >= 1, "KVCache append n_new must be >= 1");
        RUNTHERDER_CHECK(at_pos != nullptr, "KVCache append at_pos must not be null");

        const std::size_t dst = static_cast<std::size_t>(layer) *
                                static_cast<std::size_t>(layer_stride_);
        const std::size_t sdst = static_cast<std::size_t>(layer) *
                                 static_cast<std::size_t>(scale_layer_stride_);
        const int num_rows = n_new * num_kv_heads_;

        kernels::quantize_kv_fp8(k_.get() + dst, k_scale_.get() + sdst, k,
                                 at_pos, num_kv_heads_, num_rows, head_dim_, stream);
        kernels::quantize_kv_fp8(v_.get() + dst, v_scale_.get() + sdst, v,
                                 at_pos, num_kv_heads_, num_rows, head_dim_, stream);
    }

    [[nodiscard]] attention::KVView view(int layer) const {
        RUNTHERDER_CHECK(layer >= 0 && layer < num_layers_, "KVCache view layer out of range");

        const std::size_t base = static_cast<std::size_t>(layer) *
                                 static_cast<std::size_t>(layer_stride_);
        const std::size_t sbase = static_cast<std::size_t>(layer) *
                                  static_cast<std::size_t>(scale_layer_stride_);
        return attention::KVView{k_.get() + base, v_.get() + base,
                                 k_scale_.get() + sbase, v_scale_.get() + sbase};
    }

    [[nodiscard]] int max_seq_len() const noexcept { return max_seq_len_; }

private:
    device::DeviceUniquePtr<__nv_fp8_e4m3> k_;
    device::DeviceUniquePtr<__nv_fp8_e4m3> v_;
    device::DeviceUniquePtr<float>         k_scale_;
    device::DeviceUniquePtr<float>         v_scale_;
    int num_layers_;
    int max_seq_len_;
    int num_kv_heads_;
    int head_dim_;
    int kv_dim_;              // num_kv_heads * head_dim
    int layer_stride_;        // max_seq_len * kv_dim
    int scale_layer_stride_;  // max_seq_len * num_kv_heads
};

}  // namespace runtherder::engine
