#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <runtherder/attention/backend.h>
#include <runtherder/attention/factory.h>
#include <runtherder/check.h>
#include <runtherder/device/check.cuh>
#include <runtherder/engine/context.h>
#include <runtherder/engine/generator.h>
#include <runtherder/engine/kv_cache.h>
#include <runtherder/model/supported/llama/llama.h>
#include <runtherder/model/weights.h>
#include <runtherder/sampling/sampler.h>
#include <runtherder/tokenizer/tokenizer.h>

namespace {

std::size_t param_count(const runtherder::model::Tensor& t) {
    std::size_t n = 1;
    for (const auto d : t.shape) {
        n *= d;
    }
    return n;
}

// Stamps the first emitted token, the prefill / decode boundary (TTFT).
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: runtherder <model_dir> [--seed <n>] [--temp <f>] "
                     "[--enforce-eager] [prompt...]\n");
        return EXIT_FAILURE;
    }

    namespace m   = runtherder::model;
    namespace eng = runtherder::engine;

    const std::filesystem::path dir = argv[1];
    RUNTHERDER_CHECK(std::filesystem::is_directory(dir),
                     "first argument must be the model dir");

    std::string                 prompt;
    std::optional<unsigned int> seed;
    float                       temperature   = 0.7F;
    bool                        enforce_eager = false;

    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--enforce-eager") {
            enforce_eager = true;
            continue;
        }
        if (arg == "--temp") {
            RUNTHERDER_CHECK(i + 1 < argc, "--temp needs a value");
            temperature = std::stof(argv[++i]);
            RUNTHERDER_CHECK(temperature >= 0.0F, "--temp must be >= 0");
            continue;
        }
        if (arg == "--seed") {
            RUNTHERDER_CHECK(i + 1 < argc, "--seed needs a value");
            seed = static_cast<unsigned int>(std::stoul(argv[++i]));
            continue;
        }
        if (!prompt.empty()) {
            prompt += ' ';
        }
        prompt += argv[i];
    }
    if (prompt.empty()) {
        prompt = "Runtherder is operational.";
    }

    constexpr int max_new_tokens = 256;

    auto                   tokenizer = runtherder::tokenizer::Tokenizer::load(dir);
    const std::vector<int> ids       = tokenizer.encode(prompt);
    RUNTHERDER_CHECK(!ids.empty(), "prompt encoded to zero tokens");

    const std::size_t max_batch_tokens = ids.size();
    const std::size_t max_seq_len      = ids.size() + static_cast<std::size_t>(max_new_tokens);

    auto model = m::LlamaModel::load(dir, max_batch_tokens);

    std::size_t total = 0;
    total += param_count(model.weights().token_embedding());
    total += param_count(model.weights().final_norm());
    for (const auto& layer : model.weights().layers()) {
        total += param_count(layer.attn_norm);
        total += param_count(layer.wq);
        total += param_count(layer.wk);
        total += param_count(layer.wv);
        total += param_count(layer.wo);
        if (layer.q_norm) {
            total += param_count(*layer.q_norm);
        }
        if (layer.k_norm) {
            total += param_count(*layer.k_norm);
        }
        total += param_count(layer.ffn_norm);
        total += param_count(layer.w_gate);
        total += param_count(layer.w_up);
        total += param_count(layer.w_down);
    }
    if (!model.config().base().tie_word_embeddings()) {
        total += param_count(model.weights().lm_head());
    }

    std::printf("model dir:     %s\n", dir.string().c_str());
    std::printf("layers:        %zu\n", model.config().base().num_layers());
    std::printf("vocab:         %zu\n", model.config().base().vocab_size());
    std::printf("total params:  %.3f B  (%zu)\n", static_cast<double>(total) / 1e9, total);

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
    const unsigned int seed_value = seed.value_or(rd());
    runtherder::sampling::Sampler sampler(
        static_cast<int>(model.config().base().vocab_size()), seed_value);

    runtherder::sampling::SamplingParams params;
    // temperature 0 selects greedy argmax, which draws no rng.
    params.temperature = temperature;
    params.top_p       = 0.8F;
    params.top_k       = 20;

    const eng::StopCondition stop{model.config().base().eos_token_ids(), max_new_tokens};
    eng::SequenceState seq = eng::SequenceState::create(0, static_cast<int>(ids.size()), params, stop);

    TimingSink      sink;
    eng::Generator  generator(model, ctx, sampler, enforce_eager);

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
    const double blended_tps = static_cast<double>(n_gen) / elapsed_s;

    const std::string completion = tokenizer.decode(sink.tokens());

    std::printf("prompt:        %.60s%s\n", prompt.c_str(),
                prompt.size() > 60 ? " ..." : "");
    std::printf("exec mode:     %s\n", enforce_eager ? "eager" : "cuda graph");
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
    std::printf("generated:     %zu tokens in %.3fs  (%.2f tok/s blended)\n",
                n_gen, elapsed_s, blended_tps);
    std::printf("completion:    \"%s\"\n", completion.c_str());

    return 0;
}
