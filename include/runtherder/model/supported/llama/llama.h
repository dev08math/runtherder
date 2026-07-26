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
#include <runtherder/kernels/matmul.cuh>
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
 * @brief Hyperparameters of one Llama style model. Composes the universal
 *        Config with architecture specific fields.
 * @note The Llama prefix names the architecture shape, not the lineage. Qwen3
 *       loads through here and is not a Llama descendant.
 * @note rope_scaling is set only when config.json carries a rope_scaling block,
 *       and stays nullopt otherwise.
 */
class LlamaConfig {
public:
    /**
     * @brief Parses config.json under model_dir into a validated LlamaConfig.
     * @note Validates required fields and cross field invariants, and fails
     *       accordingly on a malformed config.
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
 * @brief Weights of one transformer block in the Llama style layout: the
 *        attention and FFN norms, the Q/K/V/O projections, and the gate/up/down
 *        MLP projections.
 * @note q_norm and k_norm are populated only by architectures that carry them,
 *       and stay nullopt otherwise.
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
 * @brief Loaded weights for one Llama style model. Immutable after load().
 */
class LlamaWeights {
public:
    /**
     * @brief Dispatches on config.base().model_type() to the right per model
     *        loader for that architecture.
     * @note Fails accordingly on an architecture outside the Llama style set,
     *       or weights that do not match the config.
     */
    [[nodiscard]] static LlamaWeights load(const ShardedSafetensors& reader,
                                           const LlamaConfig& config);

    /**
     * @brief Takes the arranged weight parts drained from an uploaded tensor map
     *        by a per architecture loader. arena must own the bytes every tensor
     *        part points into.
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
 * @brief Extracts the tensor named `name` from the uploaded tensor map. An
 *        integer tensor also extracts its `<name>_scale` sibling into QuantMeta.
 * @param m    uploaded tensor map, erased from in place.
 * @param name full safetensors key.
 * @note Fails if `name` or a required scale is absent.
 */
[[nodiscard]] Tensor take(ByName& m, const std::string& name);

[[nodiscard]] std::string layer_name(std::size_t i, std::string_view suffix);

[[nodiscard]] const __nv_bfloat16* bf16(const Tensor& t);

/**
 * @brief y = x.Wt, taking the int8 path when w has an integer dtype and the
 *        bf16 path otherwise.
 * @param w   weight [n, k]. An integer dtype requires w.quant populated.
 * @param w8  int8 staging carved from the forward scratch. Untouched on the
 *            bf16 path, so a caller with only bf16 weights may pass anything.
 * @param m   rows of x
 * @param n   rows of w
 * @param k   shared inner dimension
 */
/**
 * @brief True when any projection weight carries an integer dtype, meaning the
 *        forward takes the int8 path and needs the W8A8 staging reserved.
 */
[[nodiscard]] bool has_quantized_weights(const LlamaWeights& weights);

void linear(kernels::Matmul&           matmul,
            const kernels::W8A8Buffer& w8,
            __nv_bfloat16*             y,
            const __nv_bfloat16*       x,
            const Tensor&              w,
            int                        m,
            int                        n,
            int                        k,
            cudaStream_t               stream);

/**
 * @brief RoPE inverse frequency table for the plain (unscaled) rotation,
 *        entry i = theta_base^(-2i / head_dim) for i in [0, head_dim / 2).
 * @pre head_dim even and >= 2, theta_base > 0.
 */
[[nodiscard]] std::vector<float> rope_inv_freq_plain(std::size_t head_dim, float theta_base);

[[nodiscard]] device::DeviceUniquePtr<float> upload_rope_inv_freq(const std::vector<float>& host);

/**
 * @brief Bytes the forward scratch arena needs for up to max_batch_tokens
 *        tokens. Dispatches on model_type to the per architecture sizing.
 * @param config           the loaded model hyperparameters.
 * @param weights          the loaded weights, read for their dtypes only.
 * @param max_batch_tokens most tokens a single forward pass will hold.
 * @note The W8A8 staging is reserved only when weights are quantized. A forward
 *       that carves it against bf16 weights overruns the arena.
 */
[[nodiscard]] std::size_t llama_scratch_bytes(const LlamaConfig&  config,
                                              const LlamaWeights& weights,
                                              std::size_t         max_batch_tokens);

/**
 * @brief Llama style implementation of the ModelArchitecture interface. Owns
 *        its forward scratch, token staging, and position staging, all sized for
 *        max_batch_tokens.
 * @note stage() fails if the token count exceeds max_batch_tokens.
 */
class LlamaModel final : public ModelArchitecture {
public:
    [[nodiscard]] static LlamaModel load(const std::filesystem::path& model_dir,
                                         std::size_t                  max_batch_tokens);

    void stage(engine::EngineContext& ctx,
               std::span<const int>   token_ids,
               int                    start_pos) override;

    [[nodiscard]] Logits forward(engine::EngineContext& ctx,
                                 cudaStream_t           stream) override;

    [[nodiscard]] const LlamaConfig&  config()  const noexcept { return config_; }
    [[nodiscard]] const LlamaWeights& weights() const noexcept { return weights_; }

private:
    LlamaModel(LlamaConfig config, LlamaWeights weights, std::size_t max_batch_tokens);

    LlamaConfig                  config_;
    LlamaWeights                 weights_;
    std::size_t                  max_batch_tokens_;
    device::ScratchArena         scratch_;
    // A captured graph froze these addresses. stage() rewrites the contents,
    // never the pointers.
    device::DeviceUniquePtr<int> staging_;
    device::DeviceUniquePtr<int> positions_;
    std::size_t                  staged_n_         = 0;
};

}  // namespace runtherder::model
