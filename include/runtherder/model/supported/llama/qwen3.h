#pragma once

#include <cstddef>
#include <span>

#include <cuda_runtime.h>

#include <runtherder/device/scratch_arena.cuh>
#include <runtherder/engine/context.h>
#include <runtherder/model/supported/llama/llama.h>
#include <runtherder/model/upload.cuh>

namespace runtherder::model::qwen3 {

/**
 * @brief Drains an uploaded tensor map into the Qwen 3 weights graph. upload_all
 *        does the device copy, this only arranges what it produced.
 * @pre uploaded.by_name holds exactly the Qwen 3 tensor set for
 *      config.base().num_layers() layers.
 * @note Bails via RUNTHERDER_CHECK on a missing expected tensor, and on any
 *       tensor left in the map after arrangement.
 */
[[nodiscard]] LlamaWeights arrange(Uploaded uploaded, const LlamaConfig& config);

/**
 * @brief Qwen 3 forward skeleton: shared kernel steps plus q_norm/k_norm.
 * @pre weights arranged by arrange() against config. token_ids non empty.
 * @note Async. Returns last token logits.
 */
[[nodiscard]] LlamaLogits forward(const LlamaWeights&    weights,
                                  const LlamaConfig&     config,
                                  device::ScratchArena&  scratch,
                                  const int*             dev_token_ids,
                                  engine::EngineContext& ctx,
                                  std::size_t            n,
                                  int                    start_pos,
                                  cudaStream_t           stream = nullptr);

[[nodiscard]] std::size_t scratch_bytes(const LlamaConfig& config,
                                        std::size_t        max_batch_tokens);

}  // namespace runtherder::model::qwen3
