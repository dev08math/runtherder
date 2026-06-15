#pragma once

#include <span>

#include <runtherder/engine/context.h>
#include <runtherder/engine/output_sink.h>
#include <runtherder/engine/sequence_state.h>
#include <runtherder/model/architecture.h>
#include <runtherder/sampling/sampler.h>

namespace runtherder::engine {

/**
 * @brief Autoregressive decode loop for one sequence.
 * @note Re feeds the whole sequence to the forward each step (no KV cache), so
 *       the model's scratch must be sized for prompt.size() + max_new_tokens.
 *       One sequence per call.
 */
class Generator {
public:
    Generator(model::ModelArchitecture& model,
              EngineContext&            ctx,
              sampling::Sampler&        sampler) noexcept;

    void generate(SequenceState& seq, std::span<const int> prompt, OutputSink& sink);

private:
    model::ModelArchitecture& model_;
    EngineContext&            ctx_;
    sampling::Sampler&        sampler_;
};

}  // namespace runtherder::engine
