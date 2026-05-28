#include <runtherder/model/supported/llama.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

#include <runtherder/check.h>
#include <runtherder/io/json_helpers.h>
#include <runtherder/model/upload.cuh>

namespace runtherder::model {

namespace {

constexpr std::string_view kConfigFileName = "config.json";

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

LlamaConfig::LlamaConfig(Config       base,
                         std::size_t  num_heads,
                         std::size_t  num_kv_heads,
                         std::size_t  head_dim,
                         std::size_t  intermediate_dim,
                         float        rms_norm_eps,
                         float        rope_theta) noexcept
    : base_(std::move(base)),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      intermediate_dim_(intermediate_dim),
      rms_norm_eps_(rms_norm_eps),
      rope_theta_(rope_theta) {}

LlamaConfig LlamaConfig::load(const std::filesystem::path& model_dir) {
    Config base = Config::load(model_dir);

    const nlohmann::json cfg = io::read_json_file(model_dir / kConfigFileName);

    const auto num_heads        = io::require_field<std::size_t>(cfg, "num_attention_heads");
    const auto num_kv_heads     = io::require_field<std::size_t>(cfg, "num_key_value_heads");
    const auto head_dim         = io::require_field<std::size_t>(cfg, "head_dim");
    const auto intermediate_dim = io::require_field<std::size_t>(cfg, "intermediate_size");
    const auto rms_norm_eps     = io::require_field<float>(cfg, "rms_norm_eps");
    const auto rope_theta       = io::require_field<float>(cfg, "rope_theta");

    RUNTHERDER_CHECK(num_heads > 0,        "num_attention_heads must be positive");
    RUNTHERDER_CHECK(num_kv_heads > 0,     "num_key_value_heads must be positive");
    RUNTHERDER_CHECK(num_heads % num_kv_heads == 0,
                     "num_key_value_heads must divide num_attention_heads");
    RUNTHERDER_CHECK(head_dim > 0,         "head_dim must be positive");
    RUNTHERDER_CHECK(intermediate_dim > 0, "intermediate_size must be positive");
    RUNTHERDER_CHECK(rms_norm_eps > 0.0f,  "rms_norm_eps must be positive");
    RUNTHERDER_CHECK(rope_theta > 0.0f,    "rope_theta must be positive");

    return LlamaConfig(std::move(base), num_heads, num_kv_heads, head_dim,
                       intermediate_dim, rms_norm_eps, rope_theta);
}

LlamaWeights LlamaWeights::load(const ShardedSafetensors& reader, const LlamaConfig& config) {
    switch (config.base().architecture()) {
        case ArchitectureKind::Qwen3:
            return load_qwen3(reader, config);
    }
    RUNTHERDER_CHECK(false, "architecture not in Llama family");
    return LlamaWeights{};
}

LlamaWeights LlamaWeights::load_qwen3(const ShardedSafetensors& reader, const LlamaConfig& config) {
    Uploaded uploaded = upload_all(reader);
    ByName& m = uploaded.by_name;

    LlamaWeights out;
    out.arena_           = std::move(uploaded.arena);
    out.token_embedding_ = take(m, "model.embed_tokens.weight");
    out.final_norm_      = take(m, "model.norm.weight");

    if (config.base().tie_word_embeddings()) {
        // Share the span and metadata. Tensor is move only because of QuantMeta,
        // so build a fresh one with the same view rather than copy assigning.
        out.lm_head_ = Tensor{
            out.token_embedding_.data,
            out.token_embedding_.shape,
            out.token_embedding_.dtype,
            std::nullopt,
        };
    } else {
        out.lm_head_ = take(m, "lm_head.weight");
    }

    const std::size_t num_layers = config.base().num_layers();
    out.layers_.reserve(num_layers);

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
        out.layers_.push_back(std::move(layer));
    }

    RUNTHERDER_CHECK(m.empty(), "model directory contains tensors not consumed by the loader");

    return out;
}

}  // namespace runtherder::model
