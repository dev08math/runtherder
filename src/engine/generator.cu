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
    std::vector<int> input(prompt.begin(), prompt.end());
    int              start_pos = 0;

    for (;;) {
        const model::Logits out = model_.forward(ctx_, input, start_pos);
        RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

        start_pos += static_cast<int>(input.size());

        const int next = sampler_.sample(out.data, seq.sampling());

        if (seq.is_eos(next)) {
            break;
        }

        sink.on_token(seq.seq_id(), next);
        seq.advance();

        if (seq.reached_max()) {
            break;
        }

        input.assign(1, next);
    }

    sink.flush();
}

}  // namespace runtherder::engine
