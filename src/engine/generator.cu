#include <runtherder/engine/generator.h>

#include <span>
#include <vector>

#include <cuda_runtime.h>

#include <runtherder/device/check.cuh>
#include <runtherder/model/architecture.h>

namespace runtherder::engine {

Generator::Generator(model::ModelArchitecture& model,
                     EngineContext&            ctx,
                     sampling::Sampler&        sampler) noexcept
    : model_(model), ctx_(ctx), sampler_(sampler) {}

void Generator::generate(SequenceState& seq, std::span<const int> prompt, OutputSink& sink) {
    std::vector<int> tokens(prompt.begin(), prompt.end());

    for (;;) {
        const model::Logits out = model_.forward(ctx_, tokens);
        RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

        const int next = sampler_.sample(out.data, seq.sampling());

        if (seq.is_eos(next)) {
            break;
        }

        sink.on_token(seq.seq_id(), next);
        seq.advance();

        if (seq.reached_max()) {
            break;
        }

        tokens.push_back(next);
    }

    sink.flush();
}

}  // namespace runtherder::engine
