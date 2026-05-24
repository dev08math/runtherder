#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <runtherder/model/dtype.h>

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

    ~SafetensorsFile();

    SafetensorsFile(SafetensorsFile&&) noexcept;
    SafetensorsFile& operator=(SafetensorsFile&&) noexcept;

    SafetensorsFile(const SafetensorsFile&)            = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

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
    SafetensorsFile() = default;

    struct Entry {
        DType                       dtype = DType::BF16;
        std::vector<std::size_t>    shape;
        std::span<const std::byte>  bytes;
    };

    [[nodiscard]] const Entry& lookup(std::string_view name) const;

    int                                     fd_       = -1;
    void*                                   map_      = nullptr;
    std::size_t                             map_size_ = 0;
    std::string                             metadata_;
    std::vector<std::string>                names_;
    std::unordered_map<std::string, Entry>  index_;
};

}  // namespace runtherder::model
