#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/device/memory.cuh>
#include <runtherder/device/scratch_arena.cuh>
#include <runtherder/engine/context.h>
#include <runtherder/model/architecture.h>
#include <runtherder/model/config.h>
#include <runtherder/model/sharded_safetensors.h>
#include <runtherder/model/upload.cuh>
#include <runtherder/model/weights.h>

namespace runtherder::model {

enum class RopeType {
    Default,
    Llama3,
};

struct RopeScaling {
    float       factor;
    float       low_freq_factor;
    float       high_freq_factor;
    std::size_t original_max_position_embeddings;
};

/**
 * @brief Hyperparameters of one Llama family model. Composes the universal
 *        Config and adds family specific fields (heads, MLP dim, RoPE,
 *        RMSNorm epsilon). rope_scaling is set only when config.json carries a
 *        rope_scaling block (Llama 3). It stays nullopt for plain RoPE (Qwen).
 * @note Only constructable via load(). Cross field invariants are checked
 *       there before the instance is returned.
 */
class LlamaConfig {
public:
    /**
     * @brief Parses config.json under model_dir into a validated LlamaConfig.
     * @note Bails via RUNTHERDER_CHECK on missing file, malformed JSON,
     *       missing required fields, a non positive field, or num_kv_heads
     *       not dividing num_heads.
     */
    [[nodiscard]] static LlamaConfig load(const std::filesystem::path& model_dir);

    [[nodiscard]] const Config& base()             const noexcept { return base_; }
    [[nodiscard]] std::size_t   num_heads()        const noexcept { return num_heads_; }
    [[nodiscard]] std::size_t   num_kv_heads()     const noexcept { return num_kv_heads_; }
    [[nodiscard]] std::size_t   head_dim()         const noexcept { return head_dim_; }
    [[nodiscard]] std::size_t   q_dim()            const noexcept { return q_dim_; }
    [[nodiscard]] std::size_t   kv_dim()           const noexcept { return kv_dim_; }
    [[nodiscard]] std::size_t   intermediate_dim() const noexcept { return intermediate_dim_; }
    [[nodiscard]] float         rms_norm_eps()     const noexcept { return rms_norm_eps_; }
    [[nodiscard]] float         rope_theta()       const noexcept { return rope_theta_; }
    [[nodiscard]] RopeType      rope_type()        const noexcept { return rope_type_; }
    [[nodiscard]] const std::optional<RopeScaling>& rope_scaling() const noexcept { return rope_scaling_; }

private:
    LlamaConfig(Config                     base,
                std::size_t                num_heads,
                std::size_t                num_kv_heads,
                std::size_t                head_dim,
                std::size_t                intermediate_dim,
                float                      rms_norm_eps,
                float                      rope_theta,
                RopeType                   rope_type,
                std::optional<RopeScaling> rope_scaling) noexcept;

    Config                     base_;
    std::size_t                num_heads_;
    std::size_t                num_kv_heads_;
    std::size_t                head_dim_;
    std::size_t                q_dim_;
    std::size_t                kv_dim_;
    std::size_t                intermediate_dim_;
    float                      rms_norm_eps_;
    float                      rope_theta_;
    RopeType                   rope_type_;
    std::optional<RopeScaling> rope_scaling_;
};

/**
 * @brief One transformer block in the Llama family layout (Qwen, Llama,
 *        Mistral). q_norm and k_norm are populated only by architectures
 *        that carry them (Qwen 3).
 */
struct LlamaLayerWeights {
    Tensor attn_norm;

    Tensor wq;
    Tensor wk;
    Tensor wv;
    Tensor wo;

    std::optional<Tensor> q_norm;
    std::optional<Tensor> k_norm;

    Tensor ffn_norm;

    Tensor w_gate;
    Tensor w_up;
    Tensor w_down;
};

/**
 * @brief Loaded weights for one Llama family model. arena owns every byte
 *        the tensor spans point at. Immutable after load(). Move only.
 */
class LlamaWeights {
public:
    /**
     * @brief Dispatches on config.base().model_type() to the right per
     *        model loader within the family.
     * @note Bails via RUNTHERDER_CHECK on missing tensors, shape mismatch
     *       against config, or a model not in the family.
     */
    [[nodiscard]] static LlamaWeights load(const ShardedSafetensors& reader,
                                           const LlamaConfig& config);

    /**
     * @brief Assembles the graph from parts a per architecture arranger drained
     *        out of an uploaded tensor map. arena owns the bytes every part
     *        spans into.
     */
    LlamaWeights(device::DeviceUniquePtr<std::byte> arena,
                 Tensor                             token_embedding,
                 std::vector<LlamaLayerWeights>     layers,
                 Tensor                             final_norm,
                 Tensor                             lm_head,
                 device::DeviceUniquePtr<float>     inv_freq) noexcept;

    LlamaWeights(LlamaWeights&&) noexcept            = default;
    LlamaWeights& operator=(LlamaWeights&&) noexcept = default;
    LlamaWeights(const LlamaWeights&)                = delete;
    LlamaWeights& operator=(const LlamaWeights&)     = delete;

    [[nodiscard]] const Tensor&                          token_embedding() const noexcept { return token_embedding_; }
    [[nodiscard]] const std::vector<LlamaLayerWeights>&  layers()          const noexcept { return layers_; }
    [[nodiscard]] const Tensor&                          final_norm()      const noexcept { return final_norm_; }
    [[nodiscard]] const Tensor&                          lm_head()         const noexcept { return lm_head_; }
    [[nodiscard]] const float*                           inv_freq()        const noexcept { return inv_freq_.get(); }

private:
    LlamaWeights() = default;

    device::DeviceUniquePtr<std::byte>     arena_;
    Tensor                                 token_embedding_;
    std::vector<LlamaLayerWeights>         layers_;
    Tensor                                 final_norm_;
    Tensor                                 lm_head_;
    device::DeviceUniquePtr<float>         inv_freq_;
};

/**
 * @brief Removes and returns a tensor from the uploaded map. An integer tensor
 *        also drains its sibling <name>_scale and attaches it as QuantMeta. A
 *        float tensor returns with no quant.
 * @note Bails via RUNTHERDER_CHECK if name (or a required _scale) is absent.
 */
[[nodiscard]] Tensor take(ByName& m, const std::string& name);

[[nodiscard]] std::string layer_name(std::size_t i, std::string_view suffix);

[[nodiscard]] const __nv_bfloat16* bf16(const Tensor& t);

/**
 * @brief RoPE inverse frequency table for the plain (unscaled) rotation,
 *        entry i = theta_base^(-2i / head_dim) for i in [0, head_dim / 2).
 * @pre head_dim even and >= 2, theta_base > 0.
 */
[[nodiscard]] std::vector<float> rope_inv_freq_plain(std::size_t head_dim, float theta_base);

[[nodiscard]] device::DeviceUniquePtr<float> upload_rope_inv_freq(const std::vector<float>& host);

struct LlamaLogits {
    const __nv_bfloat16* logits;
    int                  vocab_size;
};

[[nodiscard]] std::size_t llama_scratch_bytes(const LlamaConfig& config,
                                              std::size_t        max_batch_tokens);

/**
 * @brief Llama family model behind the ModelArchitecture seam. Covers Qwen 3
 *        through the family loader. Owns config, weights, and its own forward
 *        scratch plus token staging, both sized for max_batch_tokens.
 * @note forward() bails via RUNTHERDER_CHECK if token_ids exceeds
 *       max_batch_tokens.
 */
class LlamaModel final : public ModelArchitecture {
public:
    [[nodiscard]] static LlamaModel load(const std::filesystem::path& model_dir,
                                         std::size_t                  max_batch_tokens);

    [[nodiscard]] Logits forward(engine::EngineContext& ctx,
                                 std::span<const int>   token_ids,
                                 int                    start_pos,
                                 cudaStream_t           stream = nullptr) override;

    [[nodiscard]] const LlamaConfig&  config()  const noexcept { return config_; }
    [[nodiscard]] const LlamaWeights& weights() const noexcept { return weights_; }

private:
    LlamaModel(LlamaConfig config, LlamaWeights weights, std::size_t max_batch_tokens);

    LlamaConfig                  config_;
    LlamaWeights                 weights_;
    std::size_t                  max_batch_tokens_;
    device::ScratchArena         scratch_;
    device::DeviceUniquePtr<int> staging_;
};

}  // namespace runtherder::model
