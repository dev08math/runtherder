#pragma once

#include <cstddef>
#include <memory>

#include <cuda_runtime.h>

#include <runtherder/attention/backend.h>
#include <runtherder/check.h>
#include <runtherder/device/check.cuh>
#include <runtherder/device/memory.cuh>
#include <runtherder/engine/kv_cache.h>
#include <runtherder/kernels/matmul.cuh>

namespace runtherder::engine {

/**
 * @brief Runtime the engine owns and passes to forward: matmul, attention
 *        backend, KV cache, token budget.
 */
class EngineContext {
public:
    EngineContext(std::size_t                                  max_batch_tokens,
                  std::unique_ptr<attention::AttentionBackend> attention,
                  KVCache                                      kv_cache);

    EngineContext(EngineContext&&) noexcept            = default;
    EngineContext& operator=(EngineContext&&) noexcept = default;
    EngineContext(const EngineContext&)                = delete;
    EngineContext& operator=(const EngineContext&)     = delete;

    [[nodiscard]] kernels::Matmul&                         matmul()    noexcept { return matmul_; }
    [[nodiscard]] runtherder::attention::AttentionBackend& attention() noexcept { return *attention_; }
    [[nodiscard]] KVCache&                                 kv_cache()  noexcept { return kv_cache_; }
    [[nodiscard]] std::size_t max_batch_tokens() const noexcept { return max_batch_tokens_; }

    /**
     * @brief Publishes pos as the decode position every kernel reads this step.
     * @note Synchronous. Illegal inside stream capture, call it before the launch.
     */
    void set_decode_pos(int pos) {
        RUNTHERDER_CHECK(pos >= 0, "decode_pos must be >= 0");
        host_pos_ = pos;
        RUNTHERDER_CUDA_CHECK(cudaMemcpy(decode_pos_.get(), &host_pos_, sizeof(int),
                                         cudaMemcpyHostToDevice));
    }

    /**
     * @brief Decode position of the token in flight, device resident. Kernels
     *        dereference it instead of taking the position by value.
     */
    [[nodiscard]] const int* decode_pos() const noexcept { return decode_pos_.get(); }

private:
    std::size_t                                  max_batch_tokens_;
    kernels::Matmul                              matmul_;
    std::unique_ptr<attention::AttentionBackend> attention_;
    KVCache                                      kv_cache_;
    device::DeviceUniquePtr<int>                 decode_pos_ = device::make_device_unique<int>(1);
    int                                          host_pos_   = 0;
};

}  // namespace runtherder::engine
