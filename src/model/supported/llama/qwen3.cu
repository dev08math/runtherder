#include <runtherder/model/supported/llama/qwen3.h>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <runtherder/check.h>

namespace runtherder::model::qwen3 {

namespace {

[[nodiscard]] Tensor take(ByName& m, const std::string& name) {
    auto it = m.find(name);
    if (it == m.end()) [[unlikely]] {
        const std::string msg = "expected tensor \"" + name + "\" missing from model";
        RUNTHERDER_CHECK(false, msg.c_str());
    }
    Tensor out = std::move(it->second);
    m.erase(it);
    return out;
}

[[nodiscard]] std::string layer_name(std::size_t i, std::string_view suffix) {
    return "model.layers." + std::to_string(i) + "." + std::string{suffix};
}

}  // namespace

LlamaWeights arrange(Uploaded uploaded, const LlamaConfig& config) {
    ByName& m = uploaded.by_name;

    Tensor token_embedding = take(m, "model.embed_tokens.weight");
    Tensor final_norm      = take(m, "model.norm.weight");

    Tensor lm_head;
    if (config.base().tie_word_embeddings()) {
        // Tensor is move only because of QuantMeta. Build a fresh view onto the
        // embedding span rather than copy assigning.
        lm_head = Tensor{
            token_embedding.data,
            token_embedding.shape,
            token_embedding.dtype,
            std::nullopt,
        };
    } else {
        lm_head = take(m, "lm_head.weight");
    }

    const std::size_t num_layers = config.base().num_layers();
    std::vector<LlamaLayerWeights> layers;
    layers.reserve(num_layers);

    for (std::size_t i = 0; i < num_layers; ++i) {
        LlamaLayerWeights layer;
        layer.attn_norm = take(m, layer_name(i, "input_layernorm.weight"));
        layer.wq        = take(m, layer_name(i, "self_attn.q_proj.weight"));
        layer.wk        = take(m, layer_name(i, "self_attn.k_proj.weight"));
        layer.wv        = take(m, layer_name(i, "self_attn.v_proj.weight"));
        layer.wo        = take(m, layer_name(i, "self_attn.o_proj.weight"));
        layer.q_norm    = take(m, layer_name(i, "self_attn.q_norm.weight"));
        layer.k_norm    = take(m, layer_name(i, "self_attn.k_norm.weight"));
        layer.ffn_norm  = take(m, layer_name(i, "post_attention_layernorm.weight"));
        layer.w_gate    = take(m, layer_name(i, "mlp.gate_proj.weight"));
        layer.w_up      = take(m, layer_name(i, "mlp.up_proj.weight"));
        layer.w_down    = take(m, layer_name(i, "mlp.down_proj.weight"));
        layers.push_back(std::move(layer));
    }

    RUNTHERDER_CHECK(m.empty(), "model directory contains tensors not consumed by the loader");

    return LlamaWeights{
        std::move(uploaded.arena),
        std::move(token_embedding),
        std::move(layers),
        std::move(final_norm),
        std::move(lm_head),
    };
}

}  // namespace runtherder::model::qwen3
