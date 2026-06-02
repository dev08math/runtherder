#pragma once

#include <cstddef>
#include <span>

#include <runtherder/device/memory.cuh>
#include <runtherder/device/scratch_arena.cuh>
#include <runtherder/kernels/matmul.cuh>

namespace runtherder::model {

/**
 * @brief Generic per forward runtime context, shared across architecture
 *        families. Composes the scratch arena, the reusable cuBLAS handle, and
 *        the input token id staging buffer. Move only.
 */
class ModelContext {
public:
    ModelContext(std::size_t scratch_bytes, std::size_t max_batch_tokens);

    ModelContext(ModelContext&&) noexcept            = default;
    ModelContext& operator=(ModelContext&&) noexcept = default;
    ModelContext(const ModelContext&)                = delete;
    ModelContext& operator=(const ModelContext&)     = delete;

    // Stages token_ids on device and returns the pointer. Synchronous copy.
    // size <= max_batch_tokens for now
    [[nodiscard]] const int* upload_token_ids(std::span<const int> token_ids);

    [[nodiscard]] device::ScratchArena& scratch()          noexcept { return scratch_; }
    [[nodiscard]] kernels::Matmul&      matmul()           noexcept { return matmul_; }
    [[nodiscard]] std::size_t           max_batch_tokens() const noexcept { return max_batch_tokens_; }

private:
    device::ScratchArena          scratch_;
    device::DeviceUniquePtr<int>  token_ids_;
    std::size_t                   max_batch_tokens_;
    kernels::Matmul               matmul_;
};

}  // namespace runtherder::model
