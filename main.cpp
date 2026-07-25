#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <CLI/CLI.hpp>
#include <cuda_runtime.h>

#include <runtherder/attention/backend.h>
#include <runtherder/attention/factory.h>
#include <runtherder/check.h>
#include <runtherder/device/check.cuh>
#include <runtherder/engine/context.h>
#include <runtherder/engine/generator.h>
#include <runtherder/engine/kv_cache.h>
#include <runtherder/model/generation_config.h>
#include <runtherder/model/supported/llama/llama.h>
#include <runtherder/sampling/sampler.h>
#include <runtherder/tokenizer/tokenizer.h>

namespace {

class TimingSink final : public runtherder::engine::OutputSink {
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

// Emits each token id to stdout, one per line, flushed per token for a parent
// process to read live.
class StdoutStreamSink final : public runtherder::engine::OutputSink {
public:
    void on_token([[maybe_unused]] int seq_id, int token_id) override {
        std::printf("%d\n", token_id);
        std::fflush(stdout);
    }
    void flush() override { std::fflush(stdout); }
};

struct CliArgs {
    std::filesystem::path       model_dir;
    std::string                 prompt;
    std::optional<unsigned int> seed;
    std::optional<float>        temperature;
    std::optional<float>        top_p;
    std::optional<int>          top_k;
    int                         max_tokens    = 256;
    std::optional<int>          max_model_len;
    bool                        enforce_eager = false;
    bool                        stream        = false;
    bool                        serve         = false;
};

CliArgs parse_args(int argc, char** argv) {
    CliArgs                  args;
    std::vector<std::string> prompt_words;

    CLI::App app{"runtherder, a CUDA LLM inference engine"};
    app.add_option("model", args.model_dir, "model directory")
        ->required()
        ->check(CLI::ExistingDirectory);
    app.add_option("prompt", prompt_words, "prompt text");
    app.add_option("--seed", args.seed, "rng seed");
    app.add_option("--temperature", args.temperature, "sampling temperature, 0 is greedy");
    app.add_option("--top-p", args.top_p, "nucleus cutoff, 1 disables");
    app.add_option("--top-k", args.top_k, "top k cap, 0 disables");
    app.add_option("--max-tokens", args.max_tokens, "max new tokens to generate");
    app.add_option("--max-model-len", args.max_model_len, "KV cache depth in tokens");
    app.add_flag("--enforce-eager", args.enforce_eager, "disable CUDA graph capture");
    app.add_flag("--stream", args.stream, "emit token ids to stdout, one per line, no stats");
    app.add_flag("--serve", args.serve, "persistent stdin/stdout ids server, model loads once");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    for (const auto& word : prompt_words) {
        if (!args.prompt.empty()) {
            args.prompt += ' ';
        }
        args.prompt += word;
    }
    if (args.prompt.empty()) {
        args.prompt = "Runtherder is operational.";
    }
    return args;
}

int run_serve(const CliArgs& args) {
    namespace m   = runtherder::model;
    namespace eng = runtherder::engine;

    const std::size_t max_len = static_cast<std::size_t>(args.max_model_len.value_or(4096));

    auto model = m::LlamaModel::load(args.model_dir, max_len);

    const int                               head_dim = static_cast<int>(model.config().head_dim());
    const runtherder::attention::AttnConfig attn{
        static_cast<int>(model.config().num_heads()),
        static_cast<int>(model.config().num_kv_heads()),
        head_dim,
        1.0F / std::sqrt(static_cast<float>(head_dim)),
    };
    auto backend = runtherder::attention::make_attention_backend(
        runtherder::attention::BackendKind::Adaptive, attn);

    eng::KVCache kv_cache(
        static_cast<int>(model.config().base().num_layers()),
        static_cast<int>(max_len),
        static_cast<int>(model.config().num_kv_heads()),
        head_dim);
    eng::EngineContext ctx(max_len, std::move(backend), std::move(kv_cache));

    std::random_device rd;
    const unsigned int seed_value = args.seed.value_or(rd());
    runtherder::sampling::Sampler sampler(
        static_cast<int>(model.config().base().vocab_size()), seed_value);

    const m::GenerationConfig            gen = m::GenerationConfig::load(args.model_dir);
    runtherder::sampling::SamplingParams params;
    params.temperature = args.temperature.value_or(gen.temperature());
    params.top_p       = args.top_p.value_or(gen.top_p());
    params.top_k       = args.top_k.value_or(gen.top_k());

    eng::Generator   generator(model, ctx, sampler, args.enforce_eager);
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

        const eng::StopCondition stop{model.config().base().eos_token_ids(), args.max_tokens};
        eng::SequenceState       seq =
            eng::SequenceState::create(0, static_cast<int>(prompt.size()), params, stop);
        generator.generate(seq, prompt, sink);

        std::printf("END\n");
        std::fflush(stdout);
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    namespace m   = runtherder::model;
    namespace eng = runtherder::engine;

    const CliArgs args = parse_args(argc, argv);
    if (args.serve) {
        return run_serve(args);
    }
    const auto& dir = args.model_dir;

    auto                   tokenizer = runtherder::tokenizer::Tokenizer::load(dir);
    const std::vector<int> ids       = tokenizer.encode(args.prompt);
    RUNTHERDER_CHECK(!ids.empty(), "prompt encoded to zero tokens");

    const std::size_t max_batch_tokens = ids.size();
    const std::size_t max_seq_len =
        args.max_model_len ? static_cast<std::size_t>(*args.max_model_len)
                           : ids.size() + static_cast<std::size_t>(args.max_tokens);

    auto model = m::LlamaModel::load(dir, max_batch_tokens);

    const int                               head_dim = static_cast<int>(model.config().head_dim());
    const runtherder::attention::AttnConfig attn{
        static_cast<int>(model.config().num_heads()),
        static_cast<int>(model.config().num_kv_heads()),
        head_dim,
        1.0F / std::sqrt(static_cast<float>(head_dim)),
    };
    auto backend = runtherder::attention::make_attention_backend(
        runtherder::attention::BackendKind::Adaptive, attn);

    eng::KVCache kv_cache(
        static_cast<int>(model.config().base().num_layers()),
        static_cast<int>(max_seq_len),
        static_cast<int>(model.config().num_kv_heads()),
        head_dim);
    eng::EngineContext ctx(max_batch_tokens, std::move(backend), std::move(kv_cache));

    std::random_device rd;
    const unsigned int seed_value = args.seed.value_or(rd());
    runtherder::sampling::Sampler sampler(
        static_cast<int>(model.config().base().vocab_size()), seed_value);

    const m::GenerationConfig     gen = m::GenerationConfig::load(dir);
    runtherder::sampling::SamplingParams params;
    params.temperature = args.temperature.value_or(gen.temperature());
    params.top_p       = args.top_p.value_or(gen.top_p());
    params.top_k       = args.top_k.value_or(gen.top_k());

    const eng::StopCondition stop{model.config().base().eos_token_ids(), args.max_tokens};
    eng::SequenceState seq = eng::SequenceState::create(0, static_cast<int>(ids.size()), params, stop);

    eng::Generator generator(model, ctx, sampler, args.enforce_eager);

    if (args.stream) {
        StdoutStreamSink stream_sink;
        generator.generate(seq, ids, stream_sink);
        return 0;
    }

    TimingSink sink;
    const auto t0 = std::chrono::steady_clock::now();
    generator.generate(seq, ids, sink);
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

    const std::string completion = tokenizer.decode(sink.tokens());

    std::printf("prompt:        %.60s%s\n", args.prompt.c_str(),
                args.prompt.size() > 60 ? " ..." : "");
    std::printf("exec mode:     %s\n", args.enforce_eager ? "eager" : "cuda graph");
    std::printf("sampling:      temperature %.2f, top_p %.2f, top_k %d\n",
                static_cast<double>(params.temperature),
                static_cast<double>(params.top_p), params.top_k);
    if (params.temperature != 0.0F) {
        std::printf("seed:          %u\n", seed_value);
    }
    std::printf("prompt tokens: %zu\n", ids.size());
    std::printf("prefill:       %zu tokens in %.3fs  (%.1f tok/s, TTFT)\n",
                ids.size(), prefill_s, prefill_tps);
    std::printf("decode:        %zu tokens in %.3fs  (%.2f tok/s)\n",
                (n_gen > 0 ? n_gen - 1 : 0), decode_s, decode_tps);
    std::printf("completion:    \"%s\"\n", completion.c_str());

    return 0;
}
