#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include <runtherder/attention/backend.h>
#include <runtherder/device/memory.cuh>
#include <runtherder/device/scratch_arena.cuh>
#include <runtherder/kernels/matmul.cuh>

namespace runtherder::model {

/**
 * @brief Runtime state for one served model, collaborators injected at
 *        construction. The scratch arena resets each forward, so any device
 *        pointer it returned is invalid once the next forward begins.
 */
class ModelContext {
public:
    ModelContext(std::size_t                                  scratch_bytes,
                 std::size_t                                  max_batch_tokens,
                 std::unique_ptr<attention::AttentionBackend> attention);

    ModelContext(ModelContext&&) noexcept            = default;
    ModelContext& operator=(ModelContext&&) noexcept = default;
    ModelContext(const ModelContext&)                = delete;
    ModelContext& operator=(const ModelContext&)     = delete;

    // Stages token_ids on device and returns the pointer. Synchronous copy.
    // size <= max_batch_tokens for now
    [[nodiscard]] const int* upload_token_ids(std::span<const int> token_ids);

    [[nodiscard]] device::ScratchArena& scratch()          noexcept { return scratch_; }
    [[nodiscard]] kernels::Matmul&      matmul()           noexcept { return matmul_; }
    [[nodiscard]] runtherder::attention::AttentionBackend& attention() noexcept { return *attention_; }
    [[nodiscard]] std::size_t           max_batch_tokens() const noexcept { return max_batch_tokens_; }

private:
    device::ScratchArena                         scratch_;
    device::DeviceUniquePtr<int>                 token_ids_;
    std::size_t                                  max_batch_tokens_;
    kernels::Matmul                              matmul_;
    std::unique_ptr<attention::AttentionBackend> attention_;
};

}  // namespace runtherder::model
