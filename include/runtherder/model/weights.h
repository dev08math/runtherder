#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include <runtherder/model/dtype.h>

namespace runtherder::model {

/**
 * @brief Dequant scales and their layout for a quantized weight Tensor.
 * @note scales views the same device slab as the weights, not separately owned.
 *       group_size 0 means one scale per output channel.
 */
struct QuantMeta {
    std::span<std::byte>  scales;
    DType                 scale_dtype = DType::FP16;
    std::size_t           group_size  = 0;
};

/**
 * @brief One weight tensor inside a model's arena.
 * @note data is invalidated when the arena backing it is freed.
 */
struct Tensor {
    std::span<std::byte>      data;
    std::vector<std::size_t>  shape;
    DType                     dtype = DType::BF16;
    std::optional<QuantMeta>  quant;
};

}  // namespace runtherder::model
