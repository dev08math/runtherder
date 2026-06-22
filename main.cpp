#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: runtherder <model_dir> [prompt...]\n");
        return EXIT_FAILURE;
    }

    namespace m   = runtherder::model;
    namespace eng = runtherder::engine;

    const std::filesystem::path dir = argv[1];

    std::string prompt;
    for (int i = 2; i < argc; ++i) {
        if (i > 2) {
            prompt += ' ';
        }
        prompt += argv[i];
    }
    if (prompt.empty()) {
        prompt = "Runtherder is operational.";
    }

    constexpr int max_new_tokens = 32;

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
        runtherder::attention::BackendKind::Naive, attn);

    eng::KVCache kv_cache(
        static_cast<int>(model.config().base().num_layers()),
        static_cast<int>(max_seq_len),
        static_cast<int>(model.config().num_kv_heads()),
        head_dim);
    eng::EngineContext ctx(max_batch_tokens, std::move(backend), std::move(kv_cache));

    std::random_device            rd;
    runtherder::sampling::Sampler sampler(static_cast<int>(model.config().base().vocab_size()), rd());

    runtherder::sampling::SamplingParams params;
    params.temperature = 0.7F;
    params.top_p       = 0.8F;
    params.top_k       = 20;

    const eng::StopCondition stop{model.config().base().eos_token_id(), max_new_tokens};
    eng::SequenceState seq = eng::SequenceState::create(0, static_cast<int>(ids.size()), params, stop);

    eng::VectorSink sink;
    eng::Generator  generator(model, ctx, sampler);
    generator.generate(seq, ids, sink);

    const std::string completion = tokenizer.decode(sink.tokens());

    std::printf("prompt:        %s\n", prompt.c_str());
    std::printf("prompt tokens: %zu\n", ids.size());
    std::printf("generated:     %zu tokens\n", sink.tokens().size());
    std::printf("completion:    \"%s\"\n", completion.c_str());

    return 0;
}
