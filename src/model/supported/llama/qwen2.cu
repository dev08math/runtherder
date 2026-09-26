#include <runtherder/model/supported/llama/qwen2.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/attention/backend.h>
#include <runtherder/check.h>
#include <runtherder/device/check.cuh>
#include <runtherder/kernels/activation.cuh>
#include <runtherder/kernels/bias.cuh>
#include <runtherder/kernels/embedding.cuh>
#include <runtherder/kernels/matmul.cuh>
#include <runtherder/kernels/rmsnorm.cuh>
#include <runtherder/kernels/rope.cuh>

namespace runtherder::model::qwen2 {

LlamaWeights arrange(Uploaded uploaded, const LlamaConfig& config) {
    ByName& m = uploaded.by_name;

    Tensor token_embedding = take(m, "model.embed_tokens.weight");
    Tensor final_norm      = take(m, "model.norm.weight");

    Tensor lm_head;
    if (config.base().tie_word_embeddings()) {
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
        layer.q_bias    = take(m, layer_name(i, "self_attn.q_proj.bias"));
        layer.k_bias    = take(m, layer_name(i, "self_attn.k_proj.bias"));
        layer.v_bias    = take(m, layer_name(i, "self_attn.v_proj.bias"));
        layer.ffn_norm  = take(m, layer_name(i, "post_attention_layernorm.weight"));
        layer.w_gate    = take(m, layer_name(i, "mlp.gate_proj.weight"));
        layer.w_up      = take(m, layer_name(i, "mlp.up_proj.weight"));
        layer.w_down    = take(m, layer_name(i, "mlp.down_proj.weight"));
        layers.push_back(std::move(layer));
    }

    RUNTHERDER_CHECK(m.empty(), "model directory contains tensors not consumed by the loader");

    device::DeviceUniquePtr<float> inv_freq =
        upload_rope_inv_freq(rope_inv_freq_plain(config.head_dim(), config.rope_theta()));

    return LlamaWeights{
        std::move(uploaded.arena),
        std::move(token_embedding),
        std::move(layers),
        std::move(final_norm),
        std::move(lm_head),
        std::move(inv_freq),
    };
}

std::size_t scratch_bytes(const LlamaConfig& config, std::size_t max_batch_tokens,
                          bool quantized) {
    const std::size_t n                = max_batch_tokens;
    const std::size_t hidden_dim       = config.base().hidden_dim();
    const std::size_t q_dim            = config.q_dim();
    const std::size_t kv_dim           = config.kv_dim();
    const std::size_t intermediate_dim = config.intermediate_dim();
    const std::size_t vocab            = config.base().vocab_size();

    constexpr std::size_t kAlign      = 256;
    constexpr std::size_t kAllocs     = 12;
    constexpr std::size_t kW8A8Allocs = 3;

    const std::size_t bf16_elems =
        n * (5 * hidden_dim + 2 * q_dim + 2 * kv_dim + 2 * intermediate_dim) + vocab;

    std::size_t w8a8_bytes  = 0;
    std::size_t w8a8_allocs = 0;
    if (quantized) {
        const std::size_t n_max = std::max({q_dim, kv_dim, hidden_dim, intermediate_dim});
        const std::size_t k_max = std::max({hidden_dim, q_dim, intermediate_dim});

        w8a8_bytes = n * k_max * sizeof(std::int8_t) +
                     n * sizeof(float) +
                     n * n_max * sizeof(std::int32_t);
        w8a8_allocs = kW8A8Allocs;
    }

    return bf16_elems * sizeof(__nv_bfloat16) + w8a8_bytes +
           (kAllocs + w8a8_allocs) * kAlign;
}

Logits forward(const LlamaWeights&    weights,
               const LlamaConfig&     config,
               device::ScratchArena&  scratch,
               const int*             dev_token_ids,
               const int*             positions,
               engine::EngineContext& ctx,
               std::size_t            n,
               cudaStream_t           stream) {
    RUNTHERDER_CHECK(n >= 1, "forward needs at least one token");

    const std::size_t hidden_dim = config.base().hidden_dim();
    const std::size_t q_dim      = config.q_dim();
    const std::size_t kv_dim     = config.kv_dim();

    const int n_int      = static_cast<int>(n);
    const int hidden_int = static_cast<int>(hidden_dim);
    const int q_int      = static_cast<int>(q_dim);
    const int kv_int     = static_cast<int>(kv_dim);

    scratch.reset();

    const int num_q_heads  = static_cast<int>(config.num_heads());
    const int num_kv_heads = static_cast<int>(config.num_kv_heads());
    const int head_dim     = static_cast<int>(config.head_dim());

    const std::size_t intermediate_dim = config.intermediate_dim();
    const int         inter_int        = static_cast<int>(intermediate_dim);

    __nv_bfloat16* hidden     = scratch.alloc<__nv_bfloat16>(n * hidden_dim);
    __nv_bfloat16* normed     = scratch.alloc<__nv_bfloat16>(n * hidden_dim);
    __nv_bfloat16* q          = scratch.alloc<__nv_bfloat16>(n * q_dim);
    __nv_bfloat16* k          = scratch.alloc<__nv_bfloat16>(n * kv_dim);
    __nv_bfloat16* v          = scratch.alloc<__nv_bfloat16>(n * kv_dim);
    __nv_bfloat16* attn       = scratch.alloc<__nv_bfloat16>(n * q_dim);
    __nv_bfloat16* attn_proj  = scratch.alloc<__nv_bfloat16>(n * hidden_dim);
    __nv_bfloat16* ffn_normed = scratch.alloc<__nv_bfloat16>(n * hidden_dim);
    __nv_bfloat16* gate       = scratch.alloc<__nv_bfloat16>(n * intermediate_dim);
    __nv_bfloat16* up         = scratch.alloc<__nv_bfloat16>(n * intermediate_dim);
    __nv_bfloat16* mlp_out    = scratch.alloc<__nv_bfloat16>(n * hidden_dim);

    // Carve only when scratch_bytes reserved it. Both sides read the same weights.
    kernels::W8A8Buffer w8{};
    if (has_quantized_weights(weights)) {
        const int n_max =
            static_cast<int>(std::max({q_dim, kv_dim, hidden_dim, intermediate_dim}));
        const int k_max = static_cast<int>(std::max({hidden_dim, q_dim, intermediate_dim}));
        w8 = kernels::W8A8Buffer::from_arena(scratch, n_int, n_max, k_max);
    }

    kernels::embedding_lookup_bf16_forward(
        hidden, bf16(weights.token_embedding()), dev_token_ids, n_int, hidden_int, stream);

    const std::vector<LlamaLayerWeights>& layers = weights.layers();
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const LlamaLayerWeights& layer = layers[i];

        if (i == 0) {
            kernels::rmsnorm_bf16_forward(
                normed, hidden, bf16(layer.attn_norm), n_int, hidden_int,
                config.rms_norm_eps(), stream);
        } else {
            kernels::rmsnorm_add_bf16_forward(
                normed, hidden, mlp_out, hidden, bf16(layer.attn_norm),
                n_int, hidden_int, config.rms_norm_eps(), stream);
        }

        linear(ctx.matmul(), w8, q, normed, layer.wq, n_int, q_int, hidden_int, stream);
        linear(ctx.matmul(), w8, k, normed, layer.wk, n_int, kv_int, hidden_int, stream);
        linear(ctx.matmul(), w8, v, normed, layer.wv, n_int, kv_int, hidden_int, stream);

        kernels::qkv_bias_add_bf16_forward(
            q, k, v, bf16(*layer.q_bias), bf16(*layer.k_bias), bf16(*layer.v_bias),
            n_int, q_int, kv_int, stream);

        kernels::rope_bf16(
            q, k, positions, weights.inv_freq(), n_int, num_q_heads, num_kv_heads,
            head_dim, stream);

        ctx.kv_cache().append(static_cast<int>(i), k, v, n_int, ctx.decode_pos(), stream);
        const attention::KVView kv = ctx.kv_cache().view(static_cast<int>(i));
        ctx.attention().run(attn, q, kv, n_int, ctx.decode_pos(), stream);

        linear(ctx.matmul(), w8, attn_proj, attn, layer.wo, n_int, hidden_int, q_int, stream);

        kernels::rmsnorm_add_bf16_forward(
            ffn_normed, hidden, attn_proj, hidden, bf16(layer.ffn_norm),
            n_int, hidden_int, config.rms_norm_eps(), stream);

        linear(ctx.matmul(), w8, gate, ffn_normed, layer.w_gate,
               n_int, inter_int, hidden_int, stream);
        linear(ctx.matmul(), w8, up, ffn_normed, layer.w_up,
               n_int, inter_int, hidden_int, stream);

        kernels::swiglu_bf16_forward(gate, gate, up, n_int * inter_int, stream);

        linear(ctx.matmul(), w8, mlp_out, gate, layer.w_down,
               n_int, hidden_int, inter_int, stream);
    }

    kernels::rmsnorm_add_bf16_forward(
        normed, hidden, mlp_out, hidden, bf16(weights.final_norm()),
        n_int, hidden_int, config.rms_norm_eps(), stream);

    const int      vocab_int = static_cast<int>(config.base().vocab_size());
    __nv_bfloat16* logits    = scratch.alloc<__nv_bfloat16>(vocab_int);

    const __nv_bfloat16* last = normed + static_cast<std::size_t>(n_int - 1) * hidden_dim;
    ctx.matmul().linear_bf16(logits, last, bf16(weights.lm_head()),
                             1, vocab_int, hidden_int, stream);

    return Logits{logits, vocab_int};
}

}  // namespace runtherder::model::qwen2
