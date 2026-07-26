#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include <runtherder/device/scratch_arena.cuh>
#include <runtherder/engine/context.h>
#include <runtherder/model/supported/llama/llama.h>
#include <runtherder/model/upload.cuh>

namespace runtherder::model::qwen3 {

/**
 * @brief Builds the Qwen3 weights from an uploaded tensor map.
 * @param uploaded device tensors from upload_all, consumed here.
 * @param config   the loaded model hyperparameters.
 * @note Fails if a required tensor is missing or one is left unused.
 */
[[nodiscard]] LlamaWeights arrange(Uploaded uploaded, const LlamaConfig& config);

/**
 * @brief Runs one Qwen3 forward pass over n tokens. Returns only the last
 *        token's logits.
 * @param weights       Qwen3 weights from arrange().
 * @param config        the loaded model hyperparameters.
 * @param scratch       forward scratch arena, sized by scratch_bytes().
 * @param dev_token_ids device buffer of n token ids.
 * @param positions     [n] device resident, the sequence position of each token.
 * @param ctx           engine runtime.
 * @param n             token count, non empty.
 * @param stream        CUDA stream every kernel is launched on.
 * @pre weights arranged by arrange() against config.
 * @note Device work only. Host to device staging belongs to the caller.
 */
[[nodiscard]] Logits forward(const LlamaWeights&    weights,
                             const LlamaConfig&     config,
                             device::ScratchArena&  scratch,
                             const int*             dev_token_ids,
                             const int*             positions,
                             engine::EngineContext& ctx,
                             std::size_t            n,
                             cudaStream_t           stream);

/**
 * @brief Bytes the Qwen3 forward scratch arena needs for up to
 *        max_batch_tokens tokens.
 * @param config           the loaded model hyperparameters.
 * @param max_batch_tokens most tokens a single forward pass will hold.
 * @param quantized        reserves the W8A8 staging. Must match what forward()
 *                         derives from the same weights, or the arena overruns.
 */
[[nodiscard]] std::size_t scratch_bytes(const LlamaConfig& config,
                                        std::size_t        max_batch_tokens,
                                        bool               quantized);

}  // namespace runtherder::model::qwen3
