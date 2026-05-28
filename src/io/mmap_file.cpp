#include <runtherder/io/mmap_file.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <runtherder/check.h>

namespace runtherder::io {

void MmapDeleter::operator()(void* p) const noexcept {
    if (p != nullptr && p != MAP_FAILED) {
        ::munmap(p, size);
    }
    if (fd >= 0) {
        ::close(fd);
    }
}

MmapFile MmapFile::open(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    RUNTHERDER_CHECK(fd >= 0, "open failed");

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        RUNTHERDER_CHECK(false, "fstat failed");
    }
    const std::size_t size = static_cast<std::size_t>(st.st_size);
    RUNTHERDER_CHECK(size > 0, "file is empty");

    void* addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        ::close(fd);
        RUNTHERDER_CHECK(false, "mmap failed");
    }

    return MmapFile(MmapPtr(addr, MmapDeleter{size, fd}));
}

std::span<const std::byte> MmapFile::bytes() const noexcept {
    return { static_cast<const std::byte*>(ptr_.get()), ptr_.get_deleter().size };
}

}  // namespace runtherder::io
