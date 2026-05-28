#include <runtherder/model/safetensors.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

#include <runtherder/check.h>

namespace runtherder::model {

namespace {

// Reserved key in the safetensors JSON header.
constexpr std::string_view kMetadataKey = "__metadata__";

// Size of the leading little endian header length field, in bytes.
constexpr std::size_t kHeaderLenBytes = 8;

[[nodiscard]] DType parse_dtype(std::string_view s, std::string_view tensor_name) {
    if (s == "BF16") {
        return DType::BF16;
    }
    if (s == "F16") {
        return DType::FP16;
    }
    if (s == "I8") {
        return DType::INT8;
    }
    const std::string msg =
        "unsupported safetensors dtype \"" + std::string{s} +
        "\" for tensor \"" + std::string{tensor_name} + "\"";
    RUNTHERDER_CHECK(false, msg.c_str());
    return DType::BF16;  // fallback
}

}  // namespace

SafetensorsFile SafetensorsFile::open(const std::filesystem::path& path) {
    SafetensorsFile sf_file;

    sf_file.fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    RUNTHERDER_CHECK(sf_file.fd_ >= 0, "open() failed on safetensors path");

    struct stat st{};
    const int stat_rc = ::fstat(sf_file.fd_, &st);
    RUNTHERDER_CHECK(stat_rc == 0, "fstat() failed on safetensors fd");
    RUNTHERDER_CHECK(st.st_size > 0, "safetensors file is empty");
    sf_file.map_size_ = static_cast<std::size_t>(st.st_size);

    RUNTHERDER_CHECK(sf_file.map_size_ >= kHeaderLenBytes,
                     "safetensors file shorter than 8 byte header length");

    void* m = ::mmap(nullptr, sf_file.map_size_, PROT_READ, MAP_PRIVATE, sf_file.fd_, 0);
    RUNTHERDER_CHECK(m != MAP_FAILED, "mmap() failed on safetensors fd");
    sf_file.map_ = m;

    const auto* base = static_cast<const std::byte*>(sf_file.map_);

    std::uint64_t header_len = 0;
    std::memcpy(&header_len, base, kHeaderLenBytes);

    RUNTHERDER_CHECK(header_len <= sf_file.map_size_ - kHeaderLenBytes,
                     "safetensors header length exceeds file size");

    const char* header_begin = reinterpret_cast<const char*>(base + kHeaderLenBytes);
    const std::size_t data_origin = kHeaderLenBytes + static_cast<std::size_t>(header_len);
    const std::size_t data_size   = sf_file.map_size_ - data_origin;

    nlohmann::json header = nlohmann::json::parse(
        header_begin, header_begin + header_len,
        /*cb=*/nullptr, /*allow_exceptions=*/false);
    RUNTHERDER_CHECK(!header.is_discarded(), "safetensors JSON header is malformed");
    RUNTHERDER_CHECK(header.is_object(), "safetensors JSON header is not an object");

    sf_file.names_.reserve(header.size());
    sf_file.index_.reserve(header.size());

    for (auto it = header.begin(); it != header.end(); ++it) {
        const std::string& name = it.key();
        if (name == kMetadataKey) {
            sf_file.metadata_ = it.value().dump();
            continue;
        }

        const auto& v = it.value();
        RUNTHERDER_CHECK(v.is_object(),  "safetensors entry value is not an object");
        RUNTHERDER_CHECK(v.contains("dtype"),        "safetensors entry missing dtype");
        RUNTHERDER_CHECK(v.contains("shape"),        "safetensors entry missing shape");
        RUNTHERDER_CHECK(v.contains("data_offsets"), "safetensors entry missing data_offsets");

        Entry entry;
        entry.dtype = parse_dtype(v["dtype"].get<std::string>(), name);

        const auto& shape_json = v["shape"];
        RUNTHERDER_CHECK(shape_json.is_array(), "safetensors shape is not an array");
        entry.shape.reserve(shape_json.size());
        for (const auto& dim : shape_json) {
            entry.shape.push_back(dim.get<std::size_t>());
        }

        const auto& offsets = v["data_offsets"];
        RUNTHERDER_CHECK(offsets.is_array() && offsets.size() == 2,
                         "safetensors data_offsets must be a 2 element array");
        const auto begin_off = offsets[0].get<std::size_t>();
        const auto end_off   = offsets[1].get<std::size_t>();
        RUNTHERDER_CHECK(begin_off <= end_off,    "safetensors data_offsets begin > end");
        RUNTHERDER_CHECK(end_off   <= data_size,  "safetensors data_offsets exceed data section");

        entry.bytes = std::span<const std::byte>(base + data_origin + begin_off,
                                                 end_off - begin_off);

        sf_file.names_.push_back(name);
        sf_file.index_.emplace(name, std::move(entry));
    }

    return sf_file;
}

SafetensorsFile::~SafetensorsFile() {
    if (map_ != nullptr && map_size_ > 0) {
        ::munmap(map_, map_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

SafetensorsFile::SafetensorsFile(SafetensorsFile&& other) noexcept
    : fd_(other.fd_),
      map_(other.map_),
      map_size_(other.map_size_),
      metadata_(std::move(other.metadata_)),
      names_(std::move(other.names_)),
      index_(std::move(other.index_)) {
    other.fd_       = -1;
    other.map_      = nullptr;
    other.map_size_ = 0;
}

SafetensorsFile& SafetensorsFile::operator=(SafetensorsFile&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (map_ != nullptr && map_size_ > 0) {
        ::munmap(map_, map_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
    fd_       = other.fd_;
    map_      = other.map_;
    map_size_ = other.map_size_;
    metadata_ = std::move(other.metadata_);
    names_    = std::move(other.names_);
    index_    = std::move(other.index_);
    other.fd_       = -1;
    other.map_      = nullptr;
    other.map_size_ = 0;
    return *this;
}

bool SafetensorsFile::contains(std::string_view name) const {
    return index_.find(std::string{name}) != index_.end();
}

DType SafetensorsFile::dtype(std::string_view name) const {
    return lookup(name).dtype;
}

const std::vector<std::size_t>& SafetensorsFile::shape(std::string_view name) const {
    return lookup(name).shape;
}

std::span<const std::byte> SafetensorsFile::bytes(std::string_view name) const {
    return lookup(name).bytes;
}

const std::vector<std::string>& SafetensorsFile::tensor_names() const noexcept {
    return names_;
}

std::string_view SafetensorsFile::metadata() const noexcept {
    return metadata_;
}

const SafetensorsFile::Entry& SafetensorsFile::lookup(std::string_view name) const {
    const auto it = index_.find(std::string{name});
    if (it == index_.end()) [[unlikely]] {
        const std::string msg = "tensor \"" + std::string{name} + "\" not found in safetensors file";
        RUNTHERDER_CHECK(false, msg.c_str());
    }
    return it->second;
}

}  // namespace runtherder::model
