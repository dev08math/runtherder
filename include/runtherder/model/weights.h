#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <optional>
#include <vector>

#include <runtherder/model/config.h>
#include <runtherder/model/dtype.h>
#include <runtherder/device/memory.cuh>

namespace runtherder::model {

struct QuantMeta {
    device::DeviceUniquePtr<half> scales;
    device::DeviceUniquePtr<half> zeros;
    std::size_t group_size = 0;
};

struct Tensor {
    device::DeviceUniquePtr<std::byte> data;
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
