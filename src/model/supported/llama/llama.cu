#include <runtherder/model/supported/llama/llama.h>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <runtherder/check.h>
#include <runtherder/device/check.cuh>
#include <runtherder/io/json_helpers.h>
#include <runtherder/model/supported/llama/llama3.h>
#include <runtherder/model/supported/llama/qwen3.h>
#include <runtherder/model/upload.cuh>

namespace runtherder::model {

namespace {

constexpr std::string_view kConfigFileName = "config.json";

[[nodiscard]] RopeType parse_rope_type(std::string_view raw) {
    if (raw == "llama3") {
        return RopeType::Llama3;
    }
    const std::string msg = "unsupported rope_type \"" + std::string{raw} + "\"";
    RUNTHERDER_CHECK(false, msg.c_str());
    return RopeType::Default;
}

}  // namespace

LlamaConfig::LlamaConfig(Config                     base,
                         std::size_t                num_heads,
                         std::size_t                num_kv_heads,
                         std::size_t                head_dim,
                         std::size_t                intermediate_dim,
                         float                      rms_norm_eps,
                         float                      rope_theta,
                         RopeType                   rope_type,
                         std::optional<RopeScaling> rope_scaling) noexcept
    : base_(std::move(base)),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      q_dim_(num_heads * head_dim),
      kv_dim_(num_kv_heads * head_dim),
      intermediate_dim_(intermediate_dim),
      rms_norm_eps_(rms_norm_eps),
      rope_theta_(rope_theta),
      rope_type_(rope_type),
      rope_scaling_(std::move(rope_scaling)) {}

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

    RopeType                   rope_type = RopeType::Default;
    std::optional<RopeScaling> rope_scaling;
    if (cfg.contains("rope_scaling") && !cfg.at("rope_scaling").is_null()) {
        const nlohmann::json& rs = cfg.at("rope_scaling");
        rope_type = parse_rope_type(io::require_field<std::string>(rs, "rope_type"));
        const RopeScaling scaling{
            io::require_field<float>(rs, "factor"),
            io::require_field<float>(rs, "low_freq_factor"),
            io::require_field<float>(rs, "high_freq_factor"),
            io::require_field<std::size_t>(rs, "original_max_position_embeddings"),
        };
        RUNTHERDER_CHECK(scaling.factor > 0.0f, "rope_scaling.factor must be positive");
        RUNTHERDER_CHECK(scaling.low_freq_factor > 0.0f,
                         "rope_scaling.low_freq_factor must be positive");
        RUNTHERDER_CHECK(scaling.high_freq_factor > scaling.low_freq_factor,
                         "rope_scaling.high_freq_factor must exceed low_freq_factor");
        RUNTHERDER_CHECK(scaling.original_max_position_embeddings > 0,
                         "rope_scaling.original_max_position_embeddings must be positive");
        rope_scaling = scaling;
    }

    return LlamaConfig(std::move(base), num_heads, num_kv_heads, head_dim,
                       intermediate_dim, rms_norm_eps, rope_theta, rope_type,
                       std::move(rope_scaling));
}

LlamaWeights::LlamaWeights(device::DeviceUniquePtr<std::byte> arena,
                           Tensor                             token_embedding,
                           std::vector<LlamaLayerWeights>     layers,
                           Tensor                             final_norm,
                           Tensor                             lm_head,
                           device::DeviceUniquePtr<float>     inv_freq) noexcept
    : arena_(std::move(arena)),
      token_embedding_(std::move(token_embedding)),
      layers_(std::move(layers)),
      final_norm_(std::move(final_norm)),
      lm_head_(std::move(lm_head)),
      inv_freq_(std::move(inv_freq)) {}

    // Builds the head_dim / 2 entry inverse frequency table for plain RoPE.
    // Entry i is theta_base^(-2i / head_dim), evaluated in double and narrowed
    // to float, not float powf.
std::vector<float> rope_inv_freq_plain(std::size_t head_dim, float theta_base) {

    RUNTHERDER_CHECK(head_dim >= 2,     "head_dim must be >= 2");
    RUNTHERDER_CHECK(head_dim % 2 == 0, "head_dim must be even");
    RUNTHERDER_CHECK(theta_base > 0.0f, "theta_base must be > 0");

    const std::size_t  half = head_dim / 2;
    std::vector<float> inv_freq(half);
    for (std::size_t i = 0; i < half; ++i) {
        const double exponent =
            -2.0 * static_cast<double>(i) / static_cast<double>(head_dim);
        inv_freq[i] =
            static_cast<float>(std::pow(static_cast<double>(theta_base), exponent));
    }
    return inv_freq;
}

device::DeviceUniquePtr<float> upload_rope_inv_freq(const std::vector<float>& host) {
    device::DeviceUniquePtr<float> dev = device::make_device_unique<float>(host.size());
    RUNTHERDER_CUDA_CHECK(cudaMemcpy(dev.get(), host.data(),
                                     host.size() * sizeof(float), cudaMemcpyHostToDevice));
    return dev;
}

Tensor take(ByName& m, const std::string& name) {
    auto it = m.find(name);
    if (it == m.end()) [[unlikely]] {
        const std::string msg = "expected tensor \"" + name + "\" missing from model";
        RUNTHERDER_CHECK(false, msg.c_str());
    }
    Tensor out = std::move(it->second);
    m.erase(it);
    if (!is_float_dtype(out.dtype)) {
        Tensor scale = take(m, name + "_scale");
        out.quant    = QuantMeta{scale.data, scale.dtype, 0};
    }
    return out;
}

std::string layer_name(std::size_t i, std::string_view suffix) {
    return "model.layers." + std::to_string(i) + "." + std::string{suffix};
}

const __nv_bfloat16* bf16(const Tensor& t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.data.data());
}

LlamaWeights LlamaWeights::load(const ShardedSafetensors& reader, const LlamaConfig& config) {
    switch (config.base().model_type()) {
        case ModelType::Qwen3:
            return qwen3::arrange(upload_all(reader), config);
        case ModelType::Llama3:
            return llama3::arrange(upload_all(reader), config);
    }
    RUNTHERDER_CHECK(false, "model not in Llama family");
    return LlamaWeights{};
}

std::size_t llama_scratch_bytes(const LlamaConfig& config, std::size_t max_batch_tokens) {
    switch (config.base().model_type()) {
        case ModelType::Qwen3:
            return qwen3::scratch_bytes(config, max_batch_tokens);
        case ModelType::Llama3:
            return llama3::scratch_bytes(config, max_batch_tokens);
    }
    RUNTHERDER_CHECK(false, "model not in Llama family");
    return 0;
}

LlamaModel::LlamaModel(LlamaConfig config, LlamaWeights weights, std::size_t max_batch_tokens)
    : config_(std::move(config)),
      weights_(std::move(weights)),
      max_batch_tokens_(max_batch_tokens),
      scratch_(llama_scratch_bytes(config_, max_batch_tokens)),
      staging_(device::make_device_unique<int>(max_batch_tokens)),
      positions_(device::make_device_unique<int>(max_batch_tokens)) {
    RUNTHERDER_CHECK(max_batch_tokens_ >= 1, "max_batch_tokens must be >= 1");
}

LlamaModel LlamaModel::load(const std::filesystem::path& model_dir,
                            std::size_t                  max_batch_tokens) {
    LlamaConfig  config  = LlamaConfig::load(model_dir);
    auto         reader  = ShardedSafetensors::open(model_dir);
    LlamaWeights weights = LlamaWeights::load(reader, config);
    return LlamaModel(std::move(config), std::move(weights), max_batch_tokens);
}

void LlamaModel::stage(engine::EngineContext& ctx,
                       std::span<const int>   token_ids,
                       int                    start_pos) {
    const std::size_t n = token_ids.size();
    RUNTHERDER_CHECK(n >= 1, "stage needs at least one token");
    RUNTHERDER_CHECK(n <= max_batch_tokens_, "token count exceeds model capacity");
    RUNTHERDER_CHECK(start_pos >= 0, "start_pos must be >= 0");
    // The only host side view of the write position. append() takes it device
    // resident and cannot bound it.
    RUNTHERDER_CHECK(start_pos + static_cast<int>(n) <= ctx.kv_cache().max_seq_len(),
                     "sequence exceeds KV cache max_seq_len");

    RUNTHERDER_CUDA_CHECK(cudaMemcpy(staging_.get(), token_ids.data(),
                                     n * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<int> positions_host(n);
    std::iota(positions_host.begin(), positions_host.end(), start_pos);
    RUNTHERDER_CUDA_CHECK(cudaMemcpy(positions_.get(), positions_host.data(),
                                     n * sizeof(int), cudaMemcpyHostToDevice));

    ctx.set_decode_pos(start_pos);

    staged_n_         = n;
}

Logits LlamaModel::forward(engine::EngineContext& ctx, cudaStream_t stream) {
    RUNTHERDER_CHECK(staged_n_ >= 1, "forward called before stage");

    switch (config_.base().model_type()) {
        case ModelType::Qwen3:
            return qwen3::forward(weights_, config_, scratch_, staging_.get(),
                                  positions_.get(), ctx, staged_n_, stream);
        case ModelType::Llama3:
            return llama3::forward(weights_, config_, scratch_, staging_.get(),
                                   positions_.get(), ctx, staged_n_, stream);
    }
    RUNTHERDER_CHECK(false, "model not in Llama family");
    return Logits{nullptr, 0};
}

}  // namespace runtherder::model
