#include <runtherder/engine/generator.h>

#include <span>
#include <vector>

#include <cuda_runtime.h>

#include <runtherder/device/check.cuh>
#include <runtherder/model/architecture.h>

namespace runtherder::engine {

Generator::Generator(model::ModelArchitecture& model,
                     EngineContext&            ctx,
                     sampling::Sampler&        sampler,
                     bool                      enforce_eager)
    : model_(model), ctx_(ctx), sampler_(sampler), enforce_eager_(enforce_eager) {}

void Generator::generate(SequenceState& seq, std::span<const int> prompt, OutputSink& sink) {
    std::vector<int> input(prompt.begin(), prompt.end());
    int              start_pos    = 0;
    int              decode_steps = 0;

    for (;;) {
        model_.stage(ctx_, input, start_pos);

        const bool decode = input.size() == 1;
        if (decode) {
            ++decode_steps;
        }

        // Capture waits for the second decode step. The first one still misses
        // the cuBLASLt plan cache at the decode shape, and the heuristic search
        // that fills it is not capturable. Prefill never captures, its arena
        // carve depends on the prompt length.
        model::Logits out{nullptr, 0};
        if (!enforce_eager_ && decode && decode_steps >= 2) {
            if (!decode_graph_.captured()) {
                decode_graph_.capture(stream_.get(), [&] {
                    decode_logits_ = model_.forward(ctx_, stream_.get());
                });
            }
            decode_graph_.launch(stream_.get());
            out = decode_logits_;
        } else {
            out = model_.forward(ctx_, stream_.get());
        }

        start_pos += static_cast<int>(input.size());

        // sample() reads logits on the default stream, which does not order
        // against stream_.
        stream_.synchronize();
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
