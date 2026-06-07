#include <runtherder/model/supported/llama/qwen3.h>

#include <cstddef>
#include <numeric>
#include <optional>
#include <span>
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
#include <runtherder/kernels/embedding.cuh>
#include <runtherder/kernels/rmsnorm.cuh>
#include <runtherder/kernels/rope.cuh>

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

[[nodiscard]] const __nv_bfloat16* bf16(const Tensor& t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.data.data());
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

LlamaActivations forward(const LlamaWeights&  weights,
                         const LlamaConfig&   config,
                         ModelContext&        ctx,
                         std::span<const int> token_ids,
                         cudaStream_t         stream) {
    const std::size_t n = token_ids.size();
    RUNTHERDER_CHECK(n >= 1, "forward needs at least one token");

    const std::size_t hidden_dim = config.base().hidden_dim();
    const std::size_t q_dim      = config.q_dim();
    const std::size_t kv_dim     = config.kv_dim();

    const int n_int      = static_cast<int>(n);
    const int hidden_int = static_cast<int>(hidden_dim);

    ctx.scratch().reset();
    const int* ids = ctx.upload_token_ids(token_ids);

    __nv_bfloat16* hidden = ctx.scratch().alloc<__nv_bfloat16>(n * hidden_dim);
    __nv_bfloat16* normed = ctx.scratch().alloc<__nv_bfloat16>(n * hidden_dim);
    __nv_bfloat16* q      = ctx.scratch().alloc<__nv_bfloat16>(n * q_dim);
    __nv_bfloat16* k      = ctx.scratch().alloc<__nv_bfloat16>(n * kv_dim);
    __nv_bfloat16* v      = ctx.scratch().alloc<__nv_bfloat16>(n * kv_dim);

    kernels::embedding_lookup_bf16_forward(
        hidden, bf16(weights.token_embedding()), ids, n_int, hidden_int);

    const LlamaLayerWeights& layer = weights.layers().front();

    kernels::rmsnorm_bf16_forward(
        normed, hidden, bf16(layer.attn_norm), n_int, hidden_int, config.rms_norm_eps());

    ctx.matmul().linear_bf16(q, normed, bf16(layer.wq),
                             n_int, static_cast<int>(q_dim), hidden_int);
    ctx.matmul().linear_bf16(k, normed, bf16(layer.wk),
                             n_int, static_cast<int>(kv_dim), hidden_int);
    ctx.matmul().linear_bf16(v, normed, bf16(layer.wv),
                             n_int, static_cast<int>(kv_dim), hidden_int);

    const int num_q_heads  = static_cast<int>(config.num_heads());
    const int num_kv_heads = static_cast<int>(config.num_kv_heads());
    const int head_dim     = static_cast<int>(config.head_dim());

    kernels::rmsnorm_bf16_forward(
        q, q, bf16(*layer.q_norm), n_int * num_q_heads, head_dim, config.rms_norm_eps());
    kernels::rmsnorm_bf16_forward(
        k, k, bf16(*layer.k_norm), n_int * num_kv_heads, head_dim, config.rms_norm_eps());

    // Single sequence prefill, so positions are contiguous 0..n-1.
    int*             positions = ctx.scratch().alloc<int>(n);
    std::vector<int> positions_host(n);
    std::iota(positions_host.begin(), positions_host.end(), 0);
    RUNTHERDER_CUDA_CHECK(cudaMemcpy(positions, positions_host.data(),
                                     n * sizeof(int), cudaMemcpyHostToDevice));

    kernels::rope_bf16(
        q, k, positions, n_int, num_q_heads, num_kv_heads, head_dim,
        config.rope_theta());

    __nv_bfloat16* attn = ctx.scratch().alloc<__nv_bfloat16>(n * q_dim);
    ctx.attention().run(attn, q, k, v, n_int, stream);

    __nv_bfloat16* attn_proj = ctx.scratch().alloc<__nv_bfloat16>(n * hidden_dim);
    ctx.matmul().linear_bf16(attn_proj, attn, bf16(layer.wo),
                             n_int, hidden_int, static_cast<int>(q_dim));

    __nv_bfloat16* ffn_normed = ctx.scratch().alloc<__nv_bfloat16>(n * hidden_dim);
    kernels::rmsnorm_add_bf16_forward(
        ffn_normed, hidden, attn_proj, hidden, bf16(layer.ffn_norm),
        n_int, hidden_int, config.rms_norm_eps());

    const std::size_t intermediate_dim = config.intermediate_dim();
    const int         inter_int        = static_cast<int>(intermediate_dim);

    __nv_bfloat16* gate = ctx.scratch().alloc<__nv_bfloat16>(n * intermediate_dim);
    __nv_bfloat16* up   = ctx.scratch().alloc<__nv_bfloat16>(n * intermediate_dim);

    ctx.matmul().linear_bf16(gate, ffn_normed, bf16(layer.w_gate),
                             n_int, inter_int, hidden_int);
    ctx.matmul().linear_bf16(up, ffn_normed, bf16(layer.w_up),
                             n_int, inter_int, hidden_int);

    // swiglu writes silu(gate) * up back into gate.
    kernels::swiglu_bf16_forward(gate, gate, up, n_int * inter_int);

    __nv_bfloat16* mlp_out = ctx.scratch().alloc<__nv_bfloat16>(n * hidden_dim);
    ctx.matmul().linear_bf16(mlp_out, gate, bf16(layer.w_down),
                             n_int, hidden_int, inter_int);

    return LlamaActivations{hidden, normed, q, k, v, attn, mlp_out, n_int};
}

}  // namespace runtherder::model::qwen3
