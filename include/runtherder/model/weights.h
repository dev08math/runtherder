#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <runtherder/model/config.h>
#include <runtherder/runtime/device_buffer.cuh>

namespace runtherder::model {

enum class DType : std::uint8_t {
    BF16,
    FP16,
    INT8,
    INT4_AWQ,
    INT4_GPTQ,
};

[[nodiscard]] constexpr bool is_float_dtype(DType dtype) noexcept {
    switch (dtype) {
        case DType::BF16:
        case DType::FP16:
            return true;
        case DType::INT8:
        case DType::INT4_AWQ:
        case DType::INT4_GPTQ:
            return false;
    }
    return false;
}

struct QuantMeta {
    runtime::DeviceUniquePtr<half> scales;
    runtime::DeviceUniquePtr<half> zeros;
    std::size_t group_size = 0;
};

struct Tensor {
    runtime::DeviceUniquePtr<std::byte> data;
    std::vector<std::size_t>            shape;
    DType                               dtype = DType::BF16;
    std::optional<QuantMeta>            quant;
};

struct LayerWeights {
    Tensor attn_norm;

    Tensor wq;
    Tensor wk;
    Tensor wv;
    Tensor wo;

    Tensor ffn_norm;

    Tensor w_gate;
    Tensor w_up;
    Tensor w_down;
};

struct Weights {
    Tensor                    token_embedding;
    std::vector<LayerWeights> layers;
    Tensor                    final_norm;
    Tensor                    lm_head;
};

}  // namespace runtherder::model
