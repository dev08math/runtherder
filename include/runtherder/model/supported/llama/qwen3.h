#pragma once

#include <span>

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
 * @note Async. Partial (layer 0).
 */
[[nodiscard]] LlamaActivations forward(const LlamaWeights&  weights,
                                       const LlamaConfig&   config,
                                       ModelContext&        ctx,
                                       std::span<const int> token_ids);

}  // namespace runtherder::model::qwen3
