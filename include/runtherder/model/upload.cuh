#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>

#include <runtherder/device/memory.cuh>
#include <runtherder/model/sharded_safetensors.h>
#include <runtherder/model/weights.h>
#include <runtherder/string_hash.h>

namespace runtherder::model {

using ByName = std::unordered_map<std::string, Tensor, TransparentStringHash, std::equal_to<>>;

/**
 * @brief Result of uploading every tensor in a reader into one device slab.
 * @note arena owns the slab. Every Tensor in by_name is a span into it and
 *       is invalidated when arena is freed.
 */
struct Uploaded {
    device::DeviceUniquePtr<std::byte>  arena;
    ByName                              by_name;
};

/**
 * @brief Copies every tensor in reader into a single device slab.
 * @param reader the opened safetensors shards to upload.
 * @note Architecture aware arrangement of the returned by_name into typed
 *       weights lives in each architecture's own loader.
 */
[[nodiscard]] Uploaded upload_all(const ShardedSafetensors& reader);

}  // namespace runtherder::model
