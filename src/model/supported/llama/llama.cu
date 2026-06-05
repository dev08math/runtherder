#include <runtherder/model/supported/llama/llama.h>

#include <cstddef>
#include <span>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

#include <runtherder/check.h>
#include <runtherder/io/json_helpers.h>
#include <runtherder/model/supported/llama/qwen3.h>
#include <runtherder/model/upload.cuh>

namespace runtherder::model {

namespace {

constexpr std::string_view kConfigFileName = "config.json";

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
      q_dim_(num_heads * head_dim),
      kv_dim_(num_kv_heads * head_dim),
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

LlamaWeights::LlamaWeights(device::DeviceUniquePtr<std::byte> arena,
                           Tensor                             token_embedding,
                           std::vector<LlamaLayerWeights>     layers,
                           Tensor                             final_norm,
                           Tensor                             lm_head) noexcept
    : arena_(std::move(arena)),
      token_embedding_(std::move(token_embedding)),
      layers_(std::move(layers)),
      final_norm_(std::move(final_norm)),
      lm_head_(std::move(lm_head)) {}

LlamaWeights LlamaWeights::load(const ShardedSafetensors& reader, const LlamaConfig& config) {
    switch (config.base().architecture()) {
        case ArchitectureKind::Qwen3:
            return qwen3::arrange(upload_all(reader), config);
    }
    RUNTHERDER_CHECK(false, "architecture not in Llama family");
    return LlamaWeights{};
}

LlamaActivations llama_forward(const LlamaWeights&  weights,
                               const LlamaConfig&   config,
                               ModelContext&        ctx,
                               std::span<const int> token_ids) {
    switch (config.base().architecture()) {
        case ArchitectureKind::Qwen3:
            return qwen3::forward(weights, config, ctx, token_ids);
    }
    RUNTHERDER_CHECK(false, "architecture not in Llama family");
    return LlamaActivations{};
}

}  // namespace runtherder::model
