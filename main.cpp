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
#include <runtherder/model/context.h>
#include <runtherder/model/sharded_safetensors.h>
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
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: runtherder <model_dir> [prompt]\n");
        return EXIT_FAILURE;
    }

    namespace m = runtherder::model;

    const std::filesystem::path dir    = argv[1];
    const std::string           prompt = (argc == 3) ? argv[2] : "Runtherder is opertational";

    auto config  = m::LlamaConfig::load(dir);
    auto reader  = m::ShardedSafetensors::open(dir);
    auto weights = m::LlamaWeights::load(reader, config);

    std::size_t total = 0;
    total += param_count(weights.token_embedding());
    total += param_count(weights.final_norm());
    for (const auto& layer : weights.layers()) {
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
    if (!config.base().tie_word_embeddings()) {
        total += param_count(weights.lm_head());
    }

    std::printf("model dir:     %s\n", dir.string().c_str());
    std::printf("layers:        %zu\n", config.base().num_layers());
    std::printf("vocab:         %zu\n", config.base().vocab_size());
    std::printf("total params:  %.3f B  (%zu)\n", static_cast<double>(total) / 1e9, total);

    auto                   tokenizer = runtherder::tokenizer::Tokenizer::load(dir);
    const std::vector<int> ids       = tokenizer.encode(prompt);
    RUNTHERDER_CHECK(!ids.empty(), "prompt encoded to zero tokens");

    const std::size_t max_batch_tokens = ids.size();

    const int                         head_dim = static_cast<int>(config.head_dim());
    const runtherder::attention::AttnConfig attn{
        static_cast<int>(config.num_heads()),
        static_cast<int>(config.num_kv_heads()),
        head_dim,
        1.0F / std::sqrt(static_cast<float>(head_dim)),
    };
    auto backend = runtherder::attention::make_attention_backend(
        runtherder::attention::BackendKind::Naive, attn);

    const std::size_t scratch = m::llama_scratch_bytes(config, max_batch_tokens);
    m::ModelContext   ctx(scratch, max_batch_tokens, std::move(backend));

    const m::LlamaLogits out = m::llama_forward(weights, config, ctx, ids);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

    std::random_device                   rd;
    runtherder::sampling::Sampler        sampler(out.vocab_size, rd());
    runtherder::sampling::SamplingParams params;
    params.temperature = 0.7F;
    params.top_p       = 0.8F;
    params.top_k       = 20;

    const int         next  = sampler.sample(out.logits, params);
    const std::string piece = tokenizer.decode(std::vector<int>{next});

    std::printf("prompt:        %s\n", prompt.c_str());
    std::printf("prompt tokens: %zu\n", ids.size());
    std::printf("next token id: %d\n", next);
    std::printf("next token:    \"%s\"\n", piece.c_str());

    return 0;
}
