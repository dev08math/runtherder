#include <runtherder/model/upload.cuh>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <runtherder/check.h>
#include <runtherder/device/check.cuh>

namespace runtherder::model {

namespace {

// Discriminator for "could these hold the same bytes": size, dtype, shape, and the
// leading and trailing 8 bytes. Distinct weights differ at the edges, so the full
// compare below runs only for a real candidate.
[[nodiscard]] std::string payload_key(std::span<const std::byte>      src,
                                      DType                           dtype,
                                      const std::vector<std::size_t>& shape) {
    std::string key = std::to_string(src.size()) + '|' + std::to_string(static_cast<int>(dtype));
    for (const std::size_t d : shape) {
        key += '|' + std::to_string(d);
    }
    if (src.size() >= 2 * sizeof(std::uint64_t)) {
        std::uint64_t head = 0;
        std::uint64_t tail = 0;
        std::memcpy(&head, src.data(), sizeof(head));
        std::memcpy(&tail, src.data() + src.size() - sizeof(tail), sizeof(tail));
        key += '|' + std::to_string(head) + '|' + std::to_string(tail);
    }
    return key;
}

}  // namespace

Uploaded upload_all(const ShardedSafetensors& reader) {
    const auto& names = reader.tensor_names();
    RUNTHERDER_CHECK(!names.empty(), "reader contains no tensors");

    // Tensors holding identical bytes share one slab region. A tied checkpoint ships the
    // output head and the embedding table as separate copies.
    std::unordered_map<std::string, std::string>     first_with_key;
    std::unordered_map<std::string, std::size_t>     offset_of;
    std::vector<std::pair<std::string, std::size_t>> owners;

    std::size_t total_bytes = 0;
    for (const auto& name : names) {
        const auto  src = reader.bytes(name);
        std::string key = payload_key(src, reader.dtype(name), reader.shape(name));

        if (const auto it = first_with_key.find(key); it != first_with_key.end()) {
            const auto prior = reader.bytes(it->second);
            if (std::memcmp(prior.data(), src.data(), src.size()) == 0) {
                offset_of.emplace(name, offset_of.at(it->second));
                continue;
            }
        }
        first_with_key.emplace(std::move(key), name);
        offset_of.emplace(name, total_bytes);
        owners.emplace_back(name, total_bytes);
        total_bytes += src.size();
    }
    RUNTHERDER_CHECK(total_bytes > 0, "reader tensors are all zero size");

    auto arena = device::make_device_unique<std::byte>(total_bytes);

    for (const auto& [name, offset] : owners) {
        const auto src = reader.bytes(name);
        RUNTHERDER_CUDA_CHECK(cudaMemcpy(arena.get() + offset,
                                         src.data(),
                                         src.size(),
                                         cudaMemcpyHostToDevice));
    }

    ByName by_name;
    by_name.reserve(names.size());
    for (const auto& name : names) {
        Tensor t;
        t.data  = std::span<std::byte>(arena.get() + offset_of.at(name), reader.bytes(name).size());
        t.shape = reader.shape(name);
        t.dtype = reader.dtype(name);
        by_name.emplace(name, std::move(t));
    }

    return Uploaded{std::move(arena), std::move(by_name)};
}

}  // namespace runtherder::model
