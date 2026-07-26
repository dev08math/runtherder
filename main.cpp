#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include <runtherder/check.h>
#include <runtherder/engine/engine.h>
#include <runtherder/engine/output_sink.h>
#include <runtherder/engine/sequence_state.h>
#include <runtherder/sampling/sampler.h>
#include <runtherder/tokenizer/tokenizer.h>

namespace {

namespace eng = runtherder::engine;

class TimingSink final : public eng::OutputSink {
public:
    void on_token([[maybe_unused]] int seq_id, int token_id) override {
        if (tokens_.empty()) {
            first_token_ = std::chrono::steady_clock::now();
        }
        tokens_.push_back(token_id);
    }
    void flush() override {}

    [[nodiscard]] const std::vector<int>& tokens() const noexcept { return tokens_; }
    [[nodiscard]] std::chrono::steady_clock::time_point first_token_time() const noexcept {
        return first_token_;
    }

private:
    std::vector<int>                      tokens_;
    std::chrono::steady_clock::time_point first_token_{};
};

class DiscardSink final : public eng::OutputSink {
public:
    void on_token([[maybe_unused]] int seq_id, [[maybe_unused]] int token_id) override {}
    void flush() override {}
};

// Emits each token id to stdout, one per line, flushed per token for a parent
// process to read live.
class StdoutStreamSink final : public eng::OutputSink {
public:
    void on_token([[maybe_unused]] int seq_id, int token_id) override {
        std::printf("%d\n", token_id);
        std::fflush(stdout);
    }
    void flush() override { std::fflush(stdout); }
};

struct CliArgs {
    std::filesystem::path                model_dir;
    std::string                          prompt;
    std::optional<std::filesystem::path> prompt_file;
    std::optional<unsigned int>          seed;
    std::optional<float>                 temperature;
    std::optional<float>                 top_p;
    std::optional<int>                   top_k;
    int                                  max_tokens = 256;
    std::optional<int>                   max_model_len;
    int                                  warmup        = 0;
    bool                                 enforce_eager = false;
    bool                                 stream        = false;
    bool                                 serve         = false;
};

CliArgs parse_args(int argc, char** argv) {
    CliArgs                  args;
    std::vector<std::string> prompt_words;

    CLI::App app{"runtherder, a CUDA LLM inference engine"};
    app.add_option("model", args.model_dir, "model directory")
        ->required()
        ->check(CLI::ExistingDirectory);
    app.add_option("prompt", prompt_words, "prompt text");
    app.add_option("--prompt-file", args.prompt_file,
                   "read the prompt from a file, overrides the positional")
        ->check(CLI::ExistingFile);
    app.add_option("--seed", args.seed, "rng seed");
    app.add_option("--temperature", args.temperature, "sampling temperature, 0 is greedy");
    app.add_option("--top-p", args.top_p, "nucleus cutoff, 1 disables");
    app.add_option("--top-k", args.top_k, "top k cap, 0 disables");
    app.add_option("--max-tokens", args.max_tokens, "max new tokens to generate");
    app.add_option("--max-model-len", args.max_model_len, "KV cache depth in tokens");
    app.add_option("--warmup", args.warmup,
                   "discarded passes before the timed run, for benchmarking");
    app.add_flag("--enforce-eager", args.enforce_eager, "disable CUDA graph capture");
    app.add_flag("--stream", args.stream, "emit token ids to stdout, one per line, no stats");
    app.add_flag("--serve", args.serve, "persistent stdin/stdout ids server, model loads once");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    if (args.prompt_file) {
        std::ifstream in(*args.prompt_file, std::ios::binary);
        RUNTHERDER_CHECK(in.good(), "cannot open --prompt-file");
        args.prompt.assign(std::istreambuf_iterator<char>(in),
                           std::istreambuf_iterator<char>());
    } else {
        for (const auto& word : prompt_words) {
            if (!args.prompt.empty()) {
                args.prompt += ' ';
            }
            args.prompt += word;
        }
    }
    if (args.prompt.empty()) {
        args.prompt = "Runtherder is operational.";
    }
    return args;
}

[[nodiscard]] eng::SamplingOverrides sampling_overrides(const CliArgs& args) {
    return eng::SamplingOverrides{args.temperature, args.top_p, args.top_k, args.seed};
}

void run_serve(const CliArgs& args) {
    const std::size_t max_len = static_cast<std::size_t>(args.max_model_len.value_or(4096));

    eng::Engine engine(args.model_dir, max_len, max_len, sampling_overrides(args),
                       args.enforce_eager);
    StdoutStreamSink sink;

    std::printf("READY\n");
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::vector<int>   prompt;
        std::istringstream iss(line);
        int                tok = 0;
        while (iss >> tok) {
            prompt.push_back(tok);
        }
        if (prompt.empty()) {
            continue;
        }

        eng::SequenceState seq =
            engine.make_sequence(static_cast<int>(prompt.size()), args.max_tokens);
        engine.generate(seq, prompt, sink);

        std::printf("END\n");
        std::fflush(stdout);
    }
}

void run_generate(const CliArgs& args) {
    auto                   tokenizer = runtherder::tokenizer::Tokenizer::load(args.model_dir);
    const std::vector<int> ids       = tokenizer.encode(args.prompt);
    RUNTHERDER_CHECK(!ids.empty(), "prompt encoded to zero tokens");

    const std::size_t max_batch_tokens = ids.size();
    const std::size_t max_seq_len =
        args.max_model_len ? static_cast<std::size_t>(*args.max_model_len)
                           : ids.size() + static_cast<std::size_t>(args.max_tokens);

    eng::Engine engine(args.model_dir, max_batch_tokens, max_seq_len, sampling_overrides(args),
                       args.enforce_eager);

    eng::SequenceState seq = engine.make_sequence(static_cast<int>(ids.size()), args.max_tokens);

    if (args.stream) {
        StdoutStreamSink stream_sink;
        engine.generate(seq, ids, stream_sink);
        return;
    }

    // Same prompt length, so the cuBLASLt plan cache is keyed on the m the timed
    // run will use. Two tokens forces one decode step, which captures the graph.
    // Advances the host RNG, so a seeded run with warmup diverges from one without.
    for (int i = 0; i < args.warmup; ++i) {
        eng::SequenceState warm_seq = engine.make_sequence(static_cast<int>(ids.size()), 2);
        DiscardSink        warm_sink;
        engine.generate(warm_seq, ids, warm_sink);
    }

    TimingSink sink;
    const auto t0 = std::chrono::steady_clock::now();
    engine.generate(seq, ids, sink);
    const auto t1 = std::chrono::steady_clock::now();

    const std::size_t n_gen     = sink.tokens().size();
    const double      elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    const double      prefill_s =
        (n_gen > 0)
            ? std::chrono::duration<double>(sink.first_token_time() - t0).count()
            : elapsed_s;
    const double decode_s = elapsed_s - prefill_s;

    const double prefill_tps = static_cast<double>(ids.size()) / prefill_s;
    const double decode_tps =
        (n_gen > 1) ? static_cast<double>(n_gen - 1) / decode_s : 0.0;

    const runtherder::sampling::SamplingParams& params     = engine.params();
    const std::string                           completion = tokenizer.decode(sink.tokens());

    std::printf("prompt:        %.60s%s\n", args.prompt.c_str(),
                args.prompt.size() > 60 ? " ..." : "");
    std::printf("exec mode:     %s\n", args.enforce_eager ? "eager" : "cuda graph");
    std::printf("sampling:      temperature %.2f, top_p %.2f, top_k %d\n",
                static_cast<double>(params.temperature),
                static_cast<double>(params.top_p), params.top_k);
    if (params.temperature != 0.0F) {
        std::printf("seed:          %u\n", engine.seed());
    }
    std::printf("warmup:        %d passes\n", args.warmup);
    std::printf("prompt tokens: %zu\n", ids.size());
    std::printf("prefill:       %zu tokens in %.3fs  (%.1f tok/s, TTFT)\n",
                ids.size(), prefill_s, prefill_tps);
    std::printf("decode:        %zu tokens in %.3fs  (%.2f tok/s)\n",
                (n_gen > 0 ? n_gen - 1 : 0), decode_s, decode_tps);
    std::printf("completion:    \"%s\"\n", completion.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    const CliArgs args = parse_args(argc, argv);

    if (args.serve) {
        run_serve(args);
    } else {
        run_generate(args);
    }
    return 0;
}
