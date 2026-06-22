#pragma once

#include <cstddef>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/attention/kv_view.h>
#include <runtherder/check.h>
#include <runtherder/device/check.cuh>
#include <runtherder/device/memory.cuh>

namespace runtherder::engine {

/**
 * @brief Single sequence contiguous KV cache. K and V slabs laid out
 *        [num_layers, max_seq_len, num_kv_heads, head_dim]. Caller owns the
 *        write position.
 */
class KVCache {
public:
    KVCache(int num_layers, int max_seq_len, int num_kv_heads, int head_dim)
        : num_layers_(num_layers),
          max_seq_len_(max_seq_len),
          kv_dim_(num_kv_heads * head_dim),
          layer_stride_(max_seq_len * (num_kv_heads * head_dim)) {
        RUNTHERDER_CHECK(num_layers >= 1,   "KVCache num_layers must be >= 1");
        RUNTHERDER_CHECK(max_seq_len >= 1,  "KVCache max_seq_len must be >= 1");
        RUNTHERDER_CHECK(num_kv_heads >= 1, "KVCache num_kv_heads must be >= 1");
        RUNTHERDER_CHECK(head_dim >= 1,     "KVCache head_dim must be >= 1");

        const std::size_t total = static_cast<std::size_t>(num_layers_) *
                                  static_cast<std::size_t>(layer_stride_);
        k_ = device::make_device_unique<__nv_bfloat16>(total);
        v_ = device::make_device_unique<__nv_bfloat16>(total);
    }

    KVCache(KVCache&&) noexcept            = default;
    KVCache& operator=(KVCache&&) noexcept = default;
    KVCache(const KVCache&)                = delete;
    KVCache& operator=(const KVCache&)     = delete;

    void append(int layer, const __nv_bfloat16* k, const __nv_bfloat16* v,
                int n_new, int at_pos, cudaStream_t stream) {
        RUNTHERDER_CHECK(layer >= 0 && layer < num_layers_, "KVCache append layer out of range");
        RUNTHERDER_CHECK(n_new >= 1, "KVCache append n_new must be >= 1");
        RUNTHERDER_CHECK(at_pos >= 0, "KVCache append at_pos must be >= 0");
        RUNTHERDER_CHECK(at_pos + n_new <= max_seq_len_, "KVCache append exceeds max_seq_len");

        const std::size_t dst = static_cast<std::size_t>(layer) * static_cast<std::size_t>(layer_stride_) +
                                static_cast<std::size_t>(at_pos) * static_cast<std::size_t>(kv_dim_);
        const std::size_t bytes =
            static_cast<std::size_t>(n_new) * static_cast<std::size_t>(kv_dim_) * sizeof(__nv_bfloat16);

        RUNTHERDER_CUDA_CHECK(cudaMemcpyAsync(k_.get() + dst, k, bytes,
                                              cudaMemcpyDeviceToDevice, stream));
        RUNTHERDER_CUDA_CHECK(cudaMemcpyAsync(v_.get() + dst, v, bytes,
                                              cudaMemcpyDeviceToDevice, stream));
    }

    [[nodiscard]] attention::KVView view(int layer, int len) const {
        RUNTHERDER_CHECK(layer >= 0 && layer < num_layers_, "KVCache view layer out of range");
        RUNTHERDER_CHECK(len >= 1 && len <= max_seq_len_, "KVCache view len out of range");

        const std::size_t base = static_cast<std::size_t>(layer) *
                                 static_cast<std::size_t>(layer_stride_);
        return attention::KVView{k_.get() + base, v_.get() + base, len};
    }

    [[nodiscard]] int max_seq_len() const noexcept { return max_seq_len_; }

private:
    device::DeviceUniquePtr<__nv_bfloat16> k_;
    device::DeviceUniquePtr<__nv_bfloat16> v_;
    int num_layers_;
    int max_seq_len_;
    int kv_dim_;        // num_kv_heads * head_dim
    int layer_stride_;  // max_seq_len * kv_dim
};

}  // namespace runtherder::engine
