#include <runtherder/model/supported/llama/llama3.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/attention/backend.h>
#include <runtherder/check.h>
#include <runtherder/device/check.cuh>
#include <runtherder/kernels/activation.cuh>
#include <runtherder/kernels/embedding.cuh>
#include <runtherder/kernels/matmul.cuh>
#include <runtherder/kernels/rmsnorm.cuh>
#include <runtherder/kernels/rope.cuh>

namespace runtherder::model::llama3 {

namespace {

std::vector<float> rope_inv_freq_llama3(std::size_t        head_dim,
                                        float              theta_base,
                                        const RopeScaling& scaling) {
    std::vector<float> inv_freq = rope_inv_freq_plain(head_dim, theta_base);

    const double factor           = static_cast<double>(scaling.factor);
    const double low_freq_factor  = static_cast<double>(scaling.low_freq_factor);
    const double high_freq_factor = static_cast<double>(scaling.high_freq_factor);
    const double old_context_len =
        static_cast<double>(scaling.original_max_position_embeddings);

    const double low_freq_wavelen  = old_context_len / low_freq_factor;
    const double high_freq_wavelen = old_context_len / high_freq_factor;
    constexpr double two_pi        = 2.0 * 3.14159265358979323846;

    for (float& entry : inv_freq) {
        const double f       = static_cast<double>(entry);
        const double wavelen = two_pi / f;

        double scaled = f;
        if (wavelen > low_freq_wavelen) {
            scaled = f / factor;
        } else if (wavelen >= high_freq_wavelen) {
            const double smooth = (old_context_len / wavelen - low_freq_factor) /
                                  (high_freq_factor - low_freq_factor);
            scaled = (1.0 - smooth) * (f / factor) + smooth * f;
        }
        entry = static_cast<float>(scaled);
    }
    return inv_freq;
}

void linear(kernels::Matmul&           matmul,
            const kernels::W8A8Buffer& w8,
            __nv_bfloat16*             y,
            const __nv_bfloat16*       x,
            const Tensor&              w,
            int                        m,
            int                        n,
            int                        k,
            cudaStream_t               stream) {
    if (is_float_dtype(w.dtype)) {
        matmul.linear_bf16(y, x, bf16(w), m, n, k, stream);
        return;
    }
    const auto* weight  = reinterpret_cast<const std::int8_t*>(w.data.data());
    const auto* w_scale = reinterpret_cast<const __nv_bfloat16*>(w.quant->scales.data());
    matmul.linear_w8a8(y, x, weight, w_scale, w8, m, n, k, stream);
}

}  // namespace

LlamaWeights arrange(Uploaded uploaded, const LlamaConfig& config) {
    ByName& m = uploaded.by_name;

    Tensor token_embedding = take(m, "model.embed_tokens.weight");
    Tensor final_norm      = take(m, "model.norm.weight");

    Tensor lm_head;
    if (m.find("lm_head.weight") != m.end()) {
        lm_head = take(m, "lm_head.weight");
    } else {
        lm_head = Tensor{
            token_embedding.data,
            token_embedding.shape,
            token_embedding.dtype,
            std::nullopt,
        };
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
        layer.ffn_norm  = take(m, layer_name(i, "post_attention_layernorm.weight"));
        layer.w_gate    = take(m, layer_name(i, "mlp.gate_proj.weight"));
        layer.w_up      = take(m, layer_name(i, "mlp.up_proj.weight"));
        layer.w_down    = take(m, layer_name(i, "mlp.down_proj.weight"));
        layers.push_back(std::move(layer));
    }

    RUNTHERDER_CHECK(m.empty(), "model directory contains tensors not consumed by the loader");

    std::vector<float> inv_freq_host =
        config.rope_scaling().has_value()
            ? rope_inv_freq_llama3(config.head_dim(), config.rope_theta(), *config.rope_scaling())
            : rope_inv_freq_plain(config.head_dim(), config.rope_theta());
    device::DeviceUniquePtr<float> inv_freq = upload_rope_inv_freq(inv_freq_host);

    return LlamaWeights{
        std::move(uploaded.arena),
        std::move(token_embedding),
        std::move(layers),
        std::move(final_norm),
        std::move(lm_head),
        std::move(inv_freq),
    };
}

std::size_t scratch_bytes(const LlamaConfig& config, std::size_t max_batch_tokens) {
    const std::size_t n                = max_batch_tokens;
    const std::size_t hidden_dim       = config.base().hidden_dim();
    const std::size_t q_dim            = config.q_dim();
    const std::size_t kv_dim           = config.kv_dim();
    const std::size_t intermediate_dim = config.intermediate_dim();
    const std::size_t vocab            = config.base().vocab_size();

    constexpr std::size_t kAlign  = 256;
    constexpr std::size_t kAllocs = 16;

    const std::size_t bf16_elems =
        n * (5 * hidden_dim + 2 * q_dim + 2 * kv_dim + 2 * intermediate_dim) + vocab;

    const std::size_t n_max = std::max({q_dim, kv_dim, hidden_dim, intermediate_dim});
    const std::size_t k_max = std::max({hidden_dim, q_dim, intermediate_dim});

    const std::size_t w8a8_bytes = n * k_max * sizeof(std::int8_t) +
                                   n * sizeof(float) +
                                   n * n_max * sizeof(std::int32_t);

    return bf16_elems * sizeof(__nv_bfloat16) + n * sizeof(int) + w8a8_bytes + kAllocs * kAlign;
}

LlamaLogits forward(const LlamaWeights&    weights,
                    const LlamaConfig&     config,
                    device::ScratchArena&  scratch,
                    const int*             dev_token_ids,
                    engine::EngineContext& ctx,
                    std::size_t            n,
                    int                    start_pos,
                    cudaStream_t           stream) {
    RUNTHERDER_CHECK(n >= 1, "forward needs at least one token");

    const std::size_t hidden_dim = config.base().hidden_dim();
    const std::size_t q_dim      = config.q_dim();
    const std::size_t kv_dim     = config.kv_dim();

    const int n_int      = static_cast<int>(n);
    const int hidden_int = static_cast<int>(hidden_dim);

    scratch.reset();
    const int* ids = dev_token_ids;

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

    kernels::embedding_lookup_bf16_forward(
        hidden, bf16(weights.token_embedding()), ids, n_int, hidden_int);

    int*             positions = scratch.alloc<int>(n);
    std::vector<int> positions_host(n);
    std::iota(positions_host.begin(), positions_host.end(), start_pos);
    RUNTHERDER_CUDA_CHECK(cudaMemcpy(positions, positions_host.data(),
                                     n * sizeof(int), cudaMemcpyHostToDevice));

    const int n_max = static_cast<int>(std::max({q_dim, kv_dim, hidden_dim, intermediate_dim}));
    const int k_max = static_cast<int>(std::max({hidden_dim, q_dim, intermediate_dim}));
    const kernels::W8A8Buffer w8 = kernels::W8A8Buffer::from_arena(scratch, n_int, n_max, k_max);

    const std::vector<LlamaLayerWeights>& layers = weights.layers();
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const LlamaLayerWeights& layer = layers[i];

        if (i == 0) {
            kernels::rmsnorm_bf16_forward(
                normed, hidden, bf16(layer.attn_norm), n_int, hidden_int,
                config.rms_norm_eps());
        } else {
            kernels::rmsnorm_add_bf16_forward(
                normed, hidden, mlp_out, hidden, bf16(layer.attn_norm),
                n_int, hidden_int, config.rms_norm_eps());
        }

        linear(ctx.matmul(), w8, q, normed, layer.wq,
               n_int, static_cast<int>(q_dim), hidden_int, stream);
        linear(ctx.matmul(), w8, k, normed, layer.wk,
               n_int, static_cast<int>(kv_dim), hidden_int, stream);
        linear(ctx.matmul(), w8, v, normed, layer.wv,
               n_int, static_cast<int>(kv_dim), hidden_int, stream);

        kernels::rope_bf16(
            q, k, positions, weights.inv_freq(), n_int, num_q_heads, num_kv_heads,
            head_dim);

        ctx.kv_cache().append(static_cast<int>(i), k, v, n_int, start_pos, stream);
        const attention::KVView kv = ctx.kv_cache().view(static_cast<int>(i), start_pos + n_int);
        ctx.attention().run(attn, q, kv, n_int, start_pos, stream);

        linear(ctx.matmul(), w8, attn_proj, attn, layer.wo,
               n_int, hidden_int, static_cast<int>(q_dim), stream);

        kernels::rmsnorm_add_bf16_forward(
            ffn_normed, hidden, attn_proj, hidden, bf16(layer.ffn_norm),
            n_int, hidden_int, config.rms_norm_eps());

        linear(ctx.matmul(), w8, gate, ffn_normed, layer.w_gate,
               n_int, inter_int, hidden_int, stream);
        linear(ctx.matmul(), w8, up, ffn_normed, layer.w_up,
               n_int, inter_int, hidden_int, stream);

        kernels::swiglu_bf16_forward(gate, gate, up, n_int * inter_int);

        linear(ctx.matmul(), w8, mlp_out, gate, layer.w_down,
               n_int, hidden_int, inter_int, stream);
    }

    kernels::rmsnorm_add_bf16_forward(
        normed, hidden, mlp_out, hidden, bf16(weights.final_norm()),
        n_int, hidden_int, config.rms_norm_eps());

    const int      vocab_int = static_cast<int>(config.base().vocab_size());
    __nv_bfloat16* logits    = scratch.alloc<__nv_bfloat16>(vocab_int);

    const __nv_bfloat16* last = normed + static_cast<std::size_t>(n_int - 1) * hidden_dim;
    ctx.matmul().linear_bf16(logits, last, bf16(weights.lm_head()),
                             1, vocab_int, hidden_int);

    return LlamaLogits{logits, vocab_int};
}

}  // namespace runtherder::model::llama3
