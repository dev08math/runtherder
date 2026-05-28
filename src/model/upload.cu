#include <runtherder/model/upload.cuh>

#include <cstddef>
#include <span>
#include <utility>

#include <runtherder/check.h>
#include <runtherder/device/check.cuh>

namespace runtherder::model {

// TODO loading sits on tier 1 of a three tier staircase. Climb it only if
// the load shows up in a profile.
//   1 (now) direct cudaMemcpy (sync) per tensor from the mmap span into
//           the slab. Driver stages each pageable copy through its own
//           pinned pool. Simplest, slowest.
//   2 cudaHostRegister the whole mmap data region, one cudaMemcpyAsync of
//     all bytes into the slab, cudaHostUnregister. Best case when the OS
//     allows pinning the full mapping.
//   3 Fallback when register fails. Two small pinned buffers A and B, ping
//     pong: fill A, kick DMA on A, fill B, wait A, kick DMA on B, swap.
//     Bounded host RAM, overlaps refill with transfer.
Uploaded upload_all(const ShardedSafetensors& reader) {
    const auto& names = reader.tensor_names();
    RUNTHERDER_CHECK(!names.empty(), "reader contains no tensors");

    std::size_t total_bytes = 0;
    for (const auto& name : names) {
        total_bytes += reader.bytes(name).size();
    }
    RUNTHERDER_CHECK(total_bytes > 0, "reader tensors are all zero size");

    auto arena = device::make_device_unique<std::byte>(total_bytes);

    ByName by_name;
    by_name.reserve(names.size());

    std::size_t offset = 0;
    for (const auto& name : names) {
        const auto src = reader.bytes(name);

        RUNTHERDER_CUDA_CHECK(cudaMemcpy(arena.get() + offset,
                                         src.data(),
                                         src.size(),
                                         cudaMemcpyHostToDevice));

        Tensor t;
        t.data  = std::span<std::byte>(arena.get() + offset, src.size());
        t.shape = reader.shape(name);
        t.dtype = reader.dtype(name);
        by_name.emplace(name, std::move(t));

        offset += src.size();
    }

    return Uploaded{std::move(arena), std::move(by_name)};
}

}  // namespace runtherder::model
