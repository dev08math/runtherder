#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <runtherder/io/mmap_file.h>
#include <runtherder/model/dtype.h>
#include <runtherder/string_hash.h>

namespace runtherder::model {

/**
 * @brief Reader for one safetensors file.
 * @note Spans from bytes() and references from shape() point into the mmap
 *       region and stay valid until this object is destroyed.
 * @note dtype(), shape(), and bytes() exit via RUNTHERDER_CHECK on an unknown
 *       name. contains() is the soft query that returns false instead.
 */
class SafetensorsFile {
public:
    /**
     * @brief Open and index a safetensors file.
     * @pre The file exists, is non empty, and carries a valid header.
     * @note Any io or parse failure exits via RUNTHERDER_CHECK. No error
     *       return, no exception.
     */
    [[nodiscard]] static SafetensorsFile open(const std::filesystem::path& path);

    SafetensorsFile(SafetensorsFile&&) noexcept            = default;
    SafetensorsFile& operator=(SafetensorsFile&&) noexcept = default;
    SafetensorsFile(const SafetensorsFile&)                = delete;
    SafetensorsFile& operator=(const SafetensorsFile&)     = delete;

    [[nodiscard]] bool contains(std::string_view name) const;

    [[nodiscard]] DType                           dtype(std::string_view name) const;
    [[nodiscard]] const std::vector<std::size_t>& shape(std::string_view name) const;
    [[nodiscard]] std::span<const std::byte>      bytes(std::string_view name) const;

    /// @return Names in header order, valid until this object is destroyed.
    [[nodiscard]] const std::vector<std::string>& tensor_names() const noexcept;

    /// @return Serialized __metadata__ header object, empty when absent. The
    ///         reader does not interpret it.
    [[nodiscard]] std::string_view metadata() const noexcept;

private:
    struct Entry {
        DType                       dtype = DType::BF16;
        std::vector<std::size_t>    shape;
        std::span<const std::byte>  bytes;
    };

    using IndexMap = std::unordered_map<std::string, Entry, TransparentStringHash, std::equal_to<>>;

    SafetensorsFile(io::MmapFile             mmap,
                    std::string              metadata,
                    std::vector<std::string> names,
                    IndexMap                 index) noexcept;

    [[nodiscard]] const Entry& lookup(std::string_view name) const;

    io::MmapFile              mmap_;
    std::string               metadata_;
    std::vector<std::string>  names_;
    IndexMap                  index_;
};

}  // namespace runtherder::model
