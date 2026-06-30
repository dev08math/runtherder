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
 */
class Generator {
public:
    Generator(model::ModelArchitecture& model,
              EngineContext&            ctx,
              sampling::Sampler&        sampler) noexcept;

    /**
     * @brief Prefills prompt, then samples and emits to sink until EOS or max new
     *        tokens. The EOS token is not emitted.
     * @param seq   sampling params and stop condition, advanced per token
     * @param sink  receives each token, flushed at the end
     */
    void generate(SequenceState& seq, std::span<const int> prompt, OutputSink& sink);

private:
    model::ModelArchitecture& model_;
    EngineContext&            ctx_;
    sampling::Sampler&        sampler_;
};

}  // namespace runtherder::engine
