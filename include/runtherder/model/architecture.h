#pragma once

#include <cstddef>
#include <span>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/engine/context.h>

namespace runtherder::model {

struct Logits {
    const __nv_bfloat16* data;
    int                  vocab_size;
};

/**
 * @brief Architecture neutral handle the engine drives. forward() runs the full
 *        stack and returns logits for the last token.
 * @note Logits.data points into the model's own scratch and stays valid only
 *       until the next forward() on the same model.
 */
class ModelArchitecture {
public:
    virtual ~ModelArchitecture() = default;

    [[nodiscard]] virtual Logits forward(engine::EngineContext& ctx,
                                         std::span<const int>   token_ids,
                                         cudaStream_t           stream = nullptr) = 0;
};

}  // namespace runtherder::model
