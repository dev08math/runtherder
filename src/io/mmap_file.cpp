#include <runtherder/io/mmap_file.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

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
    if (fd < 0) {
        const std::string msg = "open failed: " + std::string(std::strerror(errno));
        RUNTHERDER_CHECK(false, msg.c_str());
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        // errno belongs to fstat, capture before close clobbers it.
        const std::string msg = "fstat failed: " + std::string(std::strerror(errno));
        ::close(fd);
        RUNTHERDER_CHECK(false, msg.c_str());
    }
    const std::size_t size = static_cast<std::size_t>(st.st_size);
    RUNTHERDER_CHECK(size > 0, "file is empty");

    void* addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        const std::string msg = "mmap failed: " + std::string(std::strerror(errno));
        ::close(fd);
        RUNTHERDER_CHECK(false, msg.c_str());
    }

    return MmapFile(MmapPtr(addr, MmapDeleter{size, fd}));
}

std::span<const std::byte> MmapFile::bytes() const noexcept {
    return { static_cast<const std::byte*>(ptr_.get()), ptr_.get_deleter().size };
}

}  // namespace runtherder::io
