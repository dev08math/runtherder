#pragma once

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
 * @brief Interface the engine drives to run any model architecture.
 */
class ModelArchitecture {
public:
    virtual ~ModelArchitecture() = default;

    /**
     * @brief Uploads token_ids and publishes start_pos as the decode position,
     *        then holds both for the next forward().
     * @param ctx       engine context (matmul, attention, KV cache).
     * @param token_ids the tokens to run, non empty.
     * @param start_pos position of the first token in the sequence.
     * @note Host to device copies. Illegal inside stream capture, call it before
     *       the capture or replay.
     */
    virtual void stage(engine::EngineContext& ctx,
                       std::span<const int>   token_ids,
                       int                    start_pos) = 0;

    /**
     * @brief Runs one forward pass over the staged tokens and returns the last
     *        token's logits, for the caller to sample the next token from.
     * @param stream  CUDA stream every kernel is launched on.
     * @note Fails unless a stage() precedes it. Device work only, capturable.
     * @note Logits.data points into the model's own scratch and stays valid only
     *       until the next forward() on the same model.
     */
    [[nodiscard]] virtual Logits forward(engine::EngineContext& ctx,
                                         cudaStream_t           stream = nullptr) = 0;
};

}  // namespace runtherder::model
