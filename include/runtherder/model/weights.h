#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include <runtherder/device/memory.cuh>
#include <runtherder/model/dtype.h>

namespace runtherder::model {

// TODO:  quantized weights not loaded yet. Move scales and zeros into the
// slab as Tensors when they are, to match the rest of the graph.
struct QuantMeta {
    device::DeviceUniquePtr<half> scales;
    device::DeviceUniquePtr<half> zeros;
    std::size_t group_size = 0;
};

/**
 * @brief One weight tensor inside a model's arena.
 * @note data is invalidated when the parent weights graph is destroyed.
 */
struct Tensor {
    std::span<std::byte>      data;
    std::vector<std::size_t>  shape;
    DType                     dtype = DType::BF16;
    std::optional<QuantMeta>  quant;
};

}  // namespace runtherder::model
