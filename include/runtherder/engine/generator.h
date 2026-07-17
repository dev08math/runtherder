#pragma once

#include <span>

#include <runtherder/device/graph.cuh>
#include <runtherder/engine/context.h>
#include <runtherder/engine/output_sink.h>
#include <runtherder/engine/sequence_state.h>
#include <runtherder/model/architecture.h>
#include <runtherder/sampling/sampler.h>

namespace runtherder::engine {

/**
 * @brief Autoregressive decode loop for one sequence. Captures the decode step
 *        into a CUDA graph and replays it per token.
 */
class Generator {
public:
    /**
     * @param enforce_eager  launches every step kernel by kernel, no capture.
     *                       The reference path the graph is diffed against.
     */
    Generator(model::ModelArchitecture& model,
              EngineContext&            ctx,
              sampling::Sampler&        sampler,
              bool                      enforce_eager = false);

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
    bool                      enforce_eager_;
    device::CudaStream        stream_;
    device::CudaGraph         decode_graph_;
    // Where the captured graph writes its logits. Valid across replays only
    // because every decode carves the arena identically.
    model::Logits             decode_logits_{nullptr, 0};
};

}  // namespace runtherder::engine
