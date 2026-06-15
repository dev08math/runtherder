#pragma once

#include <cstddef>
#include <memory>

#include <runtherder/attention/backend.h>
#include <runtherder/kernels/matmul.cuh>

namespace runtherder::engine {

/**
 * @brief Engine owned runtime handed to the model forward: the matmul handle,
 *        the token budget, and the injected attention backend (KV cache later).
 *        The model reads what it needs and owns its own scratch.
 */
class EngineContext {
public:
    EngineContext(std::size_t                                  max_batch_tokens,
                  std::unique_ptr<attention::AttentionBackend> attention);

    EngineContext(EngineContext&&) noexcept            = default;
    EngineContext& operator=(EngineContext&&) noexcept = default;
    EngineContext(const EngineContext&)                = delete;
    EngineContext& operator=(const EngineContext&)     = delete;

    [[nodiscard]] kernels::Matmul&                         matmul()    noexcept { return matmul_; }
    [[nodiscard]] runtherder::attention::AttentionBackend& attention() noexcept { return *attention_; }
    [[nodiscard]] std::size_t max_batch_tokens() const noexcept { return max_batch_tokens_; }

private:
    std::size_t                                  max_batch_tokens_;
    kernels::Matmul                              matmul_;
    std::unique_ptr<attention::AttentionBackend> attention_;
};

}  // namespace runtherder::engine
