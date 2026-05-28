#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include <runtherder/model/sharded_safetensors.h>
#include <runtherder/model/supported/llama.h>
#include <runtherder/model/weights.h>

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
    if (argc != 2) {
        std::fprintf(stderr, "usage: runtherder <model_dir>\n");
        return EXIT_FAILURE;
    }

    namespace m = runtherder::model;

    const std::filesystem::path dir = argv[1];

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
    std::printf("hidden_dim:    %zu\n", config.base().hidden_dim());
    std::printf("intermediate:  %zu\n", config.intermediate_dim());
    std::printf("heads (Q/KV):  %zu / %zu\n", config.num_heads(), config.num_kv_heads());
    std::printf("head_dim:      %zu\n", config.head_dim());
    std::printf("vocab:         %zu\n", config.base().vocab_size());
    std::printf("max_seq_len:   %zu\n", config.base().max_seq_len());
    std::printf("tied embeds:   %s\n", config.base().tie_word_embeddings() ? "yes" : "no");
    std::printf("total params:  %.3f B  (%zu)\n", static_cast<double>(total) / 1e9, total);

    return 0;
}
