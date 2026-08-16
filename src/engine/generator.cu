#include <runtherder/engine/generator.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include <cuda_runtime.h>

#include <runtherder/check.h>
#include <runtherder/device/check.cuh>
#include <runtherder/model/architecture.h>

namespace runtherder::engine {

namespace {

[[nodiscard]] std::size_t reusable_prefix(const std::vector<int>& resident,
                                          std::span<const int>    prompt) {
    const std::size_t limit = std::min(resident.size(), prompt.size() - 1);
    std::size_t       i     = 0;
    while (i < limit && resident[i] == prompt[i]) {
        ++i;
    }
    return i;
}

}  // namespace

Generator::Generator(model::ModelArchitecture& model,
                     EngineContext&            ctx,
                     sampling::Sampler&        sampler,
                     bool                      enforce_eager)
    : model_(model), ctx_(ctx), sampler_(sampler), enforce_eager_(enforce_eager) {}

void Generator::generate(SequenceState& seq, std::span<const int> prompt, OutputSink& sink) {
    RUNTHERDER_CHECK(!prompt.empty(), "generate needs a non empty prompt");

    const std::size_t reuse = reusable_prefix(resident_, prompt);

    std::vector<int> resident(prompt.begin(), prompt.end());
    std::vector<int> input(prompt.begin() + static_cast<std::ptrdiff_t>(reuse), prompt.end());
    int              start_pos    = static_cast<int>(reuse);
    int              decode_steps = 0;

    for (;;) {
        model_.stage(ctx_, input, start_pos);

        const bool decode = input.size() == 1;
        if (decode) {
            ++decode_steps;
        }

        // The first decode step misses the cuBLASLt plan cache at the decode
        // shape, and the heuristic search that fills it is not capturable.
        // Prefill never captures, its arena carve depends on the prompt length.
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
        resident.push_back(next);
    }

    resident_ = std::move(resident);

    sink.flush();
}

}  // namespace runtherder::engine
