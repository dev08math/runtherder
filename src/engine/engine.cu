#include <runtherder/engine/engine.h>

#include <cmath>
#include <random>
#include <utility>

#include <runtherder/attention/factory.h>
#include <runtherder/model/generation_config.h>

namespace runtherder::engine {

namespace {

[[nodiscard]] sampling::SamplingParams resolve_params(const std::filesystem::path& model_dir,
                                                      const SamplingOverrides&     overrides) {
    const model::GenerationConfig gen = model::GenerationConfig::load(model_dir);

    sampling::SamplingParams params;
    params.temperature = overrides.temperature.value_or(gen.temperature());
    params.top_p       = overrides.top_p.value_or(gen.top_p());
    params.top_k       = overrides.top_k.value_or(gen.top_k());
    return params;
}

[[nodiscard]] unsigned int resolve_seed(const SamplingOverrides& overrides) {
    if (overrides.seed) {
        return *overrides.seed;
    }
    std::random_device rd;
    return rd();
}

}  // namespace

std::unique_ptr<attention::AttentionBackend> Engine::make_backend(
    const model::LlamaConfig& config) {
    const int head_dim = static_cast<int>(config.head_dim());

    const attention::AttnConfig attn{
        static_cast<int>(config.num_heads()),
        static_cast<int>(config.num_kv_heads()),
        head_dim,
        1.0F / std::sqrt(static_cast<float>(head_dim)),
    };
    return attention::make_attention_backend(attention::BackendKind::Adaptive, attn);
}

KVCache Engine::make_kv_cache(const model::LlamaConfig& config, std::size_t max_seq_len) {
    return KVCache(static_cast<int>(config.base().num_layers()),
                   static_cast<int>(max_seq_len),
                   static_cast<int>(config.num_kv_heads()),
                   static_cast<int>(config.head_dim()));
}

Engine::Engine(const std::filesystem::path& model_dir,
               std::size_t                  max_batch_tokens,
               std::size_t                  max_seq_len,
               const SamplingOverrides&     overrides,
               bool                         enforce_eager)
    : seed_(resolve_seed(overrides)),
      params_(resolve_params(model_dir, overrides)),
      model_(model::LlamaModel::load(model_dir, max_batch_tokens)),
      ctx_(max_batch_tokens,
           make_backend(model_.config()),
           make_kv_cache(model_.config(), max_seq_len)),
      sampler_(static_cast<int>(model_.config().base().vocab_size()), seed_),
      generator_(model_, ctx_, sampler_, enforce_eager) {}

SequenceState Engine::make_sequence(int prompt_len, int max_new_tokens) const {
    StopCondition stop{model_.config().base().eos_token_ids(), max_new_tokens};
    return SequenceState::create(0, prompt_len, params_, std::move(stop));
}

}  // namespace runtherder::engine
