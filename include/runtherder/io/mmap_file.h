#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>

namespace runtherder::io {

/// munmap the region and close the fd. Both are silently ignored on
/// failure (destructor cleanup, nowhere to report).
struct MmapDeleter {
    std::size_t size = 0;
    int         fd   = -1;
    void operator()(void* p) const noexcept;
};

using MmapPtr = std::unique_ptr<void, MmapDeleter>;

/**
 * @brief Read only mmap of a file's full contents.
 * @note Move only via the underlying unique_ptr.
 */
class MmapFile {
public:
    /**
     * @brief Opens path read only and maps its entire contents.
     * @note Bails via RUNTHERDER_CHECK on open, fstat, or mmap failure.
     *       O_CLOEXEC is set on the fd.
     */
    [[nodiscard]] static MmapFile open(const std::filesystem::path& path);

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
    [[nodiscard]] std::size_t                size()  const noexcept { return ptr_.get_deleter().size; }

private:
    explicit MmapFile(MmapPtr ptr) noexcept : ptr_(std::move(ptr)) {}

    MmapPtr ptr_;
};

}  // namespace runtherder::io
