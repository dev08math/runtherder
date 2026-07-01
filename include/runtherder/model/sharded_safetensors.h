#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <runtherder/model/dtype.h>
#include <runtherder/model/safetensors.h>

namespace runtherder::model {

/**
 * @brief Reader over a model directory of one or many safetensors shards.
 * @note A single model.safetensors is handled as the one shard case. Many
 *       shards are discovered through model.safetensors.index.json.
 * @note Spans from bytes() and references from shape() point into the shard
 *       mmap regions and stay valid until this object is destroyed.
 * @note dtype(), shape(), and bytes() fail on an unknown name. contains() is
 *       the soft query that returns false instead.
 */
class ShardedSafetensors {
public:
    /**
     * @brief Open and index every shard in a model directory.
     * @pre dir holds either model.safetensors or model.safetensors.index.json
     *      with the shards it names.
     * @note Fails accordingly on any io or parse error.
     */
    [[nodiscard]] static ShardedSafetensors open(const std::filesystem::path& dir);

    ShardedSafetensors(ShardedSafetensors&&) noexcept            = default;
    ShardedSafetensors& operator=(ShardedSafetensors&&) noexcept = default;

    ShardedSafetensors(const ShardedSafetensors&)            = delete;
    ShardedSafetensors& operator=(const ShardedSafetensors&) = delete;

    [[nodiscard]] bool contains(std::string_view name) const;

    [[nodiscard]] DType                           dtype(std::string_view name) const;
    [[nodiscard]] const std::vector<std::size_t>& shape(std::string_view name) const;
    [[nodiscard]] std::span<const std::byte>      bytes(std::string_view name) const;

    /// @return Names across all shards, valid until this object is destroyed.
    [[nodiscard]] const std::vector<std::string>& tensor_names() const noexcept;

private:
    using NameToShard = std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>>;

    ShardedSafetensors(std::vector<SafetensorsFile> shards,
                       std::vector<std::string>     names,
                       NameToShard                  name_to_shard) noexcept;

    [[nodiscard]] const SafetensorsFile& shard_of(std::string_view name) const;

    std::vector<SafetensorsFile>  shards_;
    std::vector<std::string>      names_;
    NameToShard                   name_to_shard_;
};

}  // namespace runtherder::model
