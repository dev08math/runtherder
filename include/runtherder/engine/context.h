#pragma once

#include <cstddef>
#include <memory>

#include <runtherder/attention/backend.h>
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

private:
    std::size_t                                  max_batch_tokens_;
    kernels::Matmul                              matmul_;
    std::unique_ptr<attention::AttentionBackend> attention_;
    KVCache                                      kv_cache_;
};

}  // namespace runtherder::engine
