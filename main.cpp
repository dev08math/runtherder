#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include <runtherder/check.h>
#include <runtherder/engine/engine.h>
#include <runtherder/engine/output_sink.h>
#include <runtherder/engine/sequence_state.h>
#include <runtherder/io/text_file.h>
#include <runtherder/sampling/sampler.h>
#include <runtherder/tokenizer/chat_template.h>
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

class ChatStreamSink final : public eng::OutputSink {
public:
    ChatStreamSink(runtherder::tokenizer::Tokenizer& tok, bool show_reasoning)
        : tok_{tok},
          show_reasoning_{show_reasoning},
          think_open_{probe_single(tok, "<think>")},
          think_close_{probe_single(tok, "</think>")} {}

    void on_token([[maybe_unused]] int seq_id, int token_id) override {
        ids_.push_back(token_id);
        if (token_id == think_open_) {
            in_reasoning_ = true;
        }
        const bool closing = token_id == think_close_ && in_reasoning_;
        if (closing) {
            in_reasoning_ = false;
            answer_start_ = ids_.size();
        }

        const std::string text = tok_.decode(ids_);
        std::string_view  ready{text};
        while (ready.ends_with(kReplacement)) {
            ready.remove_suffix(kReplacement.size());
        }
        if (ready.size() <= shown_) {
            return;
        }

        std::string_view fresh = ready.substr(shown_);
        shown_                 = ready.size();

        if (!show_reasoning_ && (in_reasoning_ || closing)) {
            return;
        }
        if (!show_reasoning_ && !answer_open_) {
            while (!fresh.empty() && (fresh.front() == '\n' || fresh.front() == ' ')) {
                fresh.remove_prefix(1);
            }
            if (fresh.empty()) {
                return;
            }
            answer_open_ = true;
        }

        std::fwrite(fresh.data(), 1, fresh.size(), stdout);
        std::fflush(stdout);
    }

    void flush() override { std::fflush(stdout); }

    /// @brief The reply with any reasoning block removed.
    [[nodiscard]] std::string text() const {
        return tok_.decode(std::span<const int>{ids_}.subspan(answer_start_));
    }

    [[nodiscard]] std::size_t reasoning_tokens() const noexcept { return answer_start_; }
    [[nodiscard]] std::size_t answer_tokens() const noexcept {
        return ids_.size() - answer_start_;
    }

    void reset() {
        ids_.clear();
        shown_        = 0;
        answer_start_ = 0;
        in_reasoning_ = false;
        answer_open_  = false;
    }

private:
    static constexpr std::string_view kReplacement = "\xEF\xBF\xBD";

    // A checkpoint that marks its reasoning block with added tokens encodes
    // each tag to a single id.
    [[nodiscard]] static int probe_single(runtherder::tokenizer::Tokenizer& tok,
                                          std::string_view                  tag) {
        const std::vector<int> ids = tok.encode(tag);
        return ids.size() == 1 ? ids.front() : -1;
    }

    runtherder::tokenizer::Tokenizer& tok_;
    bool                              show_reasoning_;
    int                               think_open_;
    int                               think_close_;
    std::vector<int>                  ids_;
    std::size_t                       shown_        = 0;
    std::size_t                       answer_start_ = 0;
    bool                              in_reasoning_ = false;
    bool                              answer_open_  = false;
};

struct CliArgs {
    std::filesystem::path                model_dir;
    std::string                          prompt;
    std::optional<std::filesystem::path> prompt_file;
    std::optional<unsigned int>          seed;
    std::optional<float>                 temperature;
    std::optional<float>                 top_p;
    std::optional<int>                   top_k;
    std::optional<int>                   max_tokens;
    std::optional<int>                   max_model_len;
    int                                  warmup        = 0;
    bool                                 enforce_eager = false;
    bool                                 stream        = false;
    bool                                 serve         = false;
    bool                                 conversation  = false;
    bool                                 show_reasoning = false;
    std::optional<std::string>           system_prompt;
};

CliArgs parse_args(int argc, char** argv) {
    CliArgs                  args;
    std::vector<std::string> prompt_words;

    CLI::App app{"runtherder, a CUDA LLM inference engine"};
    app.allow_non_standard_option_names();
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
    app.add_flag("-cnv,--conversation", args.conversation,
                 "interactive chat, applies the checkpoint's chat template");
    app.add_option("--system", args.system_prompt, "system prompt, conversation mode only");
    app.add_flag("--show-reasoning", args.show_reasoning,
                 "print the model's reasoning block, conversation mode only");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    if (args.prompt_file) {
        args.prompt = runtherder::io::read_text_file(*args.prompt_file);
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

// Interactive modes size the KV cache up front, since neither knows how long the
// session will run. One shot generation sizes from the prompt it already has.
constexpr int kDefaultInteractiveLen = 4096;

constexpr int kDefaultMaxTokens = 256;

constexpr std::size_t kMinReplyBudget = 1024;

[[nodiscard]] eng::Engine make_interactive_engine(const CliArgs& args, std::size_t max_len) {
    return eng::Engine{args.model_dir, max_len, max_len, sampling_overrides(args),
                       args.enforce_eager};
}

[[nodiscard]] std::size_t interactive_len(const CliArgs& args) {
    return static_cast<std::size_t>(args.max_model_len.value_or(kDefaultInteractiveLen));
}

void run_serve(const CliArgs& args) {
    eng::Engine      engine = make_interactive_engine(args, interactive_len(args));
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

        eng::SequenceState seq = engine.make_sequence(
            static_cast<int>(prompt.size()), args.max_tokens.value_or(kDefaultMaxTokens));
        engine.generate(seq, prompt, sink);

        std::printf("END\n");
        std::fflush(stdout);
    }
}

void run_chat(const CliArgs& args) {
    namespace tk = runtherder::tokenizer;

    const std::size_t max_len = interactive_len(args);

    tk::Tokenizer          tokenizer = tk::Tokenizer::load(args.model_dir);
    const tk::ChatTemplate tmpl      = tk::ChatTemplate::load(args.model_dir);

    eng::Engine    engine = make_interactive_engine(args, max_len);
    ChatStreamSink sink(tokenizer, args.show_reasoning);

    std::vector<tk::Message> messages;
    if (args.system_prompt) {
        messages.push_back({tk::Role::System, *args.system_prompt});
    }

    std::string line;
    while (true) {
        std::fputs("> ", stdout);
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) {
            std::fputs("\n", stdout);
            break;
        }
        if (line.empty()) {
            continue;
        }

        messages.push_back({tk::Role::User, line});

        // Drop whole oldest exchanges until the prompt plus room to answer fits.
        // A system prompt is pinned, and the live user turn is never evicted.
        const std::size_t pinned  = args.system_prompt ? 1U : 0U;
        const std::size_t reserve = std::min<std::size_t>(kMinReplyBudget, max_len / 4);

        std::vector<int> ids = tmpl.encode(tokenizer, messages, true);
        while (ids.size() + reserve > max_len && messages.size() > pinned + 1) {
            const std::size_t drop =
                std::min<std::size_t>(2, messages.size() - pinned - 1);
            messages.erase(messages.begin() + static_cast<std::ptrdiff_t>(pinned),
                           messages.begin() + static_cast<std::ptrdiff_t>(pinned + drop));
            ids = tmpl.encode(tokenizer, messages, true);
        }
        RUNTHERDER_CHECK(ids.size() < max_len,
                         "one message is longer than --max-model-len on its own");

        sink.reset();
        const int budget = args.max_tokens.value_or(static_cast<int>(max_len - ids.size()));
        eng::SequenceState seq = engine.make_sequence(static_cast<int>(ids.size()), budget);
        engine.generate(seq, ids, sink);
        std::fputs("\n", stdout);
        if (sink.reasoning_tokens() > 0) {
            std::printf("[reasoning %zu tokens, answer %zu tokens]\n",
                        sink.reasoning_tokens(), sink.answer_tokens());
        }
        std::fputs("\n", stdout);

        messages.push_back({tk::Role::Assistant, sink.text()});
    }
}

struct RunStats {
    std::size_t prompt_tokens;
    std::size_t generated;
    double      prefill_s;
    double      decode_s;
    double      prefill_tps;
    double      decode_tps;
};

// Prefill is charged up to the first token, so TTFT and prefill tok/s are the
// same measurement. Decode excludes that token for the same reason.
[[nodiscard]] RunStats measure(const TimingSink&                     sink,
                               std::size_t                           prompt_tokens,
                               std::chrono::steady_clock::time_point t0,
                               std::chrono::steady_clock::time_point t1) {
    const std::size_t n_gen     = sink.tokens().size();
    const double      elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    const double      prefill_s =
        (n_gen > 0)
            ? std::chrono::duration<double>(sink.first_token_time() - t0).count()
            : elapsed_s;
    const double decode_s = elapsed_s - prefill_s;

    return RunStats{
        .prompt_tokens = prompt_tokens,
        .generated     = (n_gen > 0) ? n_gen - 1 : 0,
        .prefill_s     = prefill_s,
        .decode_s      = decode_s,
        .prefill_tps   = static_cast<double>(prompt_tokens) / prefill_s,
        .decode_tps    = (n_gen > 1) ? static_cast<double>(n_gen - 1) / decode_s : 0.0,
    };
}

void print_run_stats(const CliArgs&     args,
                     const eng::Engine& engine,
                     const RunStats&    stats,
                     const std::string& completion) {
    const runtherder::sampling::SamplingParams& params = engine.params();

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
    std::printf("prompt tokens: %zu\n", stats.prompt_tokens);
    std::printf("prefill:       %zu tokens in %.3fs  (%.1f tok/s, TTFT)\n",
                stats.prompt_tokens, stats.prefill_s, stats.prefill_tps);
    std::printf("decode:        %zu tokens in %.3fs  (%.2f tok/s)\n",
                stats.generated, stats.decode_s, stats.decode_tps);
    std::printf("completion:    \"%s\"\n", completion.c_str());
}

void run_generate(const CliArgs& args) {
    auto                   tokenizer = runtherder::tokenizer::Tokenizer::load(args.model_dir);
    const std::vector<int> ids       = tokenizer.encode(args.prompt);
    RUNTHERDER_CHECK(!ids.empty(), "prompt encoded to zero tokens");

    const int         max_new           = args.max_tokens.value_or(kDefaultMaxTokens);
    const std::size_t max_batch_tokens  = ids.size();
    const std::size_t max_seq_len =
        args.max_model_len ? static_cast<std::size_t>(*args.max_model_len)
                           : ids.size() + static_cast<std::size_t>(max_new);

    eng::Engine engine(args.model_dir, max_batch_tokens, max_seq_len, sampling_overrides(args),
                       args.enforce_eager);

    eng::SequenceState seq = engine.make_sequence(static_cast<int>(ids.size()), max_new);

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

    print_run_stats(args, engine, measure(sink, ids.size(), t0, t1),
                    tokenizer.decode(sink.tokens()));
}

}  // namespace

int main(int argc, char** argv) {
    const CliArgs args = parse_args(argc, argv);

    if (args.serve) {
        run_serve(args);
    } else if (args.conversation) {
        run_chat(args);
    } else {
        run_generate(args);
    }
    return 0;
}
