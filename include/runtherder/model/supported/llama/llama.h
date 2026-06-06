#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/device/memory.cuh>
#include <runtherder/model/config.h>
#include <runtherder/model/context.h>
#include <runtherder/model/sharded_safetensors.h>
#include <runtherder/model/weights.h>

namespace runtherder::model {

/**
 * @brief Hyperparameters of one Llama family model. Composes the universal
 *        Config and adds family specific fields (heads, MLP dim, RoPE,
 *        RMSNorm epsilon).
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

private:
    LlamaConfig(Config       base,
                std::size_t  num_heads,
                std::size_t  num_kv_heads,
                std::size_t  head_dim,
                std::size_t  intermediate_dim,
                float        rms_norm_eps,
                float        rope_theta) noexcept;

    Config       base_;
    std::size_t  num_heads_;
    std::size_t  num_kv_heads_;
    std::size_t  head_dim_;
    std::size_t  q_dim_;
    std::size_t  kv_dim_;
    std::size_t  intermediate_dim_;
    float        rms_norm_eps_;
    float        rope_theta_;
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
     * @brief Dispatches on config.base().architecture() to the right per
     *        architecture loader within the family.
     * @note Bails via RUNTHERDER_CHECK on missing tensors, shape mismatch
     *       against config, or an architecture not in the family.
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
                 Tensor                             lm_head) noexcept;

    LlamaWeights(LlamaWeights&&) noexcept            = default;
    LlamaWeights& operator=(LlamaWeights&&) noexcept = default;
    LlamaWeights(const LlamaWeights&)                = delete;
    LlamaWeights& operator=(const LlamaWeights&)     = delete;

    [[nodiscard]] const Tensor&                          token_embedding() const noexcept { return token_embedding_; }
    [[nodiscard]] const std::vector<LlamaLayerWeights>&  layers()          const noexcept { return layers_; }
    [[nodiscard]] const Tensor&                          final_norm()      const noexcept { return final_norm_; }
    [[nodiscard]] const Tensor&                          lm_head()         const noexcept { return lm_head_; }

private:
    LlamaWeights() = default;

    device::DeviceUniquePtr<std::byte>     arena_;
    Tensor                                 token_embedding_;
    std::vector<LlamaLayerWeights>         layers_;
    Tensor                                 final_norm_;
    Tensor                                 lm_head_;
};

// TODO need to remove once forward returns logits. Interim and test only: forward
// writes its activations into transient ctx.scratch(), so these pointers are
// the only way to test q/k/v and attn.
struct LlamaActivations {
    const __nv_bfloat16* hidden;
    const __nv_bfloat16* normed;
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const __nv_bfloat16* attn;
    int                  num_tokens;
};

// Llama family forward. Partial: runs layer 0 only and returns interim
// activations carved from ctx.scratch(), not logits. Async on stream.
[[nodiscard]] LlamaActivations llama_forward(const LlamaWeights&  weights,
                                             const LlamaConfig&   config,
                                             ModelContext&        ctx,
                                             std::span<const int> token_ids,
                                             cudaStream_t         stream = nullptr);

}  // namespace runtherder::model
