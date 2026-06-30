#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include <runtherder/device/scratch_arena.cuh>
#include <runtherder/engine/context.h>
#include <runtherder/model/supported/llama/llama.h>
#include <runtherder/model/upload.cuh>

namespace runtherder::model::llama3 {

/**
 * @brief Drains an uploaded tensor map into the Llama 3 weights graph. upload_all
 *        does the device copy, this only arranges what it produced.
 * @pre uploaded.by_name holds exactly the Llama 3 tensor set for
 *      config.base().num_layers() layers. Linear weights are I8 with a sibling
 *      _scale. lm_head is BF16 and present even when tie_word_embeddings is set.
 * @note Bails via RUNTHERDER_CHECK on a missing expected tensor, and on any
 *       tensor left in the map after arrangement.
 */
[[nodiscard]] LlamaWeights arrange(Uploaded uploaded, const LlamaConfig& config);

/**
 * @brief Llama 3 forward skeleton: shared kernel steps, no q_norm/k_norm, INT8
 *        projections via W8A8, BF16 lm_head.
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

}  // namespace runtherder::model::llama3
