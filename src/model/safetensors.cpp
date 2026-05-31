#include <runtherder/model/safetensors.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
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
    return DType::BF16;  // unreachable, RUNTHERDER_CHECK exits
}

}  // namespace

SafetensorsFile::SafetensorsFile(io::MmapFile             mmap,
                                 std::string              metadata,
                                 std::vector<std::string> names,
                                 IndexMap                 index) noexcept
    : mmap_(std::move(mmap)),
      metadata_(std::move(metadata)),
      names_(std::move(names)),
      index_(std::move(index)) {}

SafetensorsFile SafetensorsFile::open(const std::filesystem::path& path) {
    io::MmapFile mmap = io::MmapFile::open(path);
    const auto bytes = mmap.bytes();

    RUNTHERDER_CHECK(bytes.size() >= kHeaderLenBytes,
                     "safetensors file shorter than 8 byte header length");

    const auto* base = bytes.data();

    std::uint64_t header_len = 0;
    std::memcpy(&header_len, base, kHeaderLenBytes);

    RUNTHERDER_CHECK(header_len <= bytes.size() - kHeaderLenBytes,
                     "safetensors header length exceeds file size");

    const char* header_begin = reinterpret_cast<const char*>(base + kHeaderLenBytes);
    const std::size_t data_origin = kHeaderLenBytes + static_cast<std::size_t>(header_len);
    const std::size_t data_size   = bytes.size() - data_origin;

    nlohmann::json header = nlohmann::json::parse(
        header_begin, header_begin + header_len,
        /*cb=*/nullptr, /*allow_exceptions=*/false);
    RUNTHERDER_CHECK(!header.is_discarded(), "safetensors JSON header is malformed");
    RUNTHERDER_CHECK(header.is_object(), "safetensors JSON header is not an object");

    std::string              metadata;
    std::vector<std::string> names;
    IndexMap                 index;

    names.reserve(header.size());
    index.reserve(header.size());

    for (auto it = header.begin(); it != header.end(); ++it) {
        const std::string& name = it.key();
        if (name == kMetadataKey) {
            metadata = it.value().dump();
            continue;
        }

        const auto& v = it.value();
        RUNTHERDER_CHECK(v.is_object(),              "safetensors entry value is not an object");
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
        RUNTHERDER_CHECK(begin_off <= end_off,   "safetensors data_offsets begin > end");
        RUNTHERDER_CHECK(end_off   <= data_size, "safetensors data_offsets exceed data section");

        entry.bytes = std::span<const std::byte>(base + data_origin + begin_off,
                                                 end_off - begin_off);

        names.push_back(name);
        index.emplace(name, std::move(entry));
    }

    return SafetensorsFile(std::move(mmap),
                           std::move(metadata),
                           std::move(names),
                           std::move(index));
}

bool SafetensorsFile::contains(std::string_view name) const {
    return index_.find(name) != index_.end();
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
    const auto it = index_.find(name);
    if (it == index_.end()) [[unlikely]] {
        const std::string msg = "tensor \"" + std::string{name} + "\" not found in safetensors file";
        RUNTHERDER_CHECK(false, msg.c_str());
    }
    return it->second;
}

}  // namespace runtherder::model
