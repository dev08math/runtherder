#pragma once

#include <cstddef>

#include <runtherder/check.h>
#include <runtherder/device/memory.cuh>

namespace runtherder::device {

/**
 * @brief Bump allocator over one device buffer, a scratch pad for temporary
 *        work. alloc<T>() hands out a slice and only moves forward. reset()
 *        reclaims everything at once, nothing is freed before that.
 */
class ScratchArena {
public:
    explicit ScratchArena(std::size_t bytes);

    ScratchArena(ScratchArena&&) noexcept            = default;
    ScratchArena& operator=(ScratchArena&&) noexcept = default;
    ScratchArena(const ScratchArena&)                = delete;
    ScratchArena& operator=(const ScratchArena&)     = delete;

    void reset() noexcept { offset_ = 0; }
    
    template <typename T>
    [[nodiscard]] T* alloc(std::size_t count);

    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
    static constexpr std::size_t kAlign = 256; // cudaMalloc parity

    DeviceUniquePtr<std::byte>  data_;
    std::size_t                 bytes_;
    std::size_t                 offset_ = 0;
};

inline ScratchArena::ScratchArena(std::size_t bytes)
    : data_(make_device_unique<std::byte>(bytes)), bytes_(bytes) {
    RUNTHERDER_CHECK(bytes >= 1, "ScratchArena bytes must be >= 1");
}

template <typename T>
T* ScratchArena::alloc(std::size_t count) {
    const std::size_t aligned = (offset_ + kAlign - 1) & ~(kAlign - 1);
    const std::size_t need    = count * sizeof(T);
    RUNTHERDER_CHECK(aligned + need <= bytes_, "ScratchArena overflow");
    std::byte* const base = data_.get() + aligned;
    offset_               = aligned + need;
    return reinterpret_cast<T*>(base);
}

}  // namespace runtherder::device
