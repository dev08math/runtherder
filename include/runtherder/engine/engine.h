#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>

#include <runtherder/attention/backend.h>
#include <runtherder/engine/context.h>
#include <runtherder/engine/generator.h>
#include <runtherder/engine/kv_cache.h>
#include <runtherder/engine/output_sink.h>
#include <runtherder/engine/sequence_state.h>
#include <runtherder/model/supported/llama/llama.h>
#include <runtherder/sampling/sampler.h>

namespace runtherder::engine {

/**
 * @brief Sampling values a caller pins explicitly. An unset field falls back to
 *        the model's generation_config.json. An unset seed draws from
 *        std::random_device.
 */
struct SamplingOverrides {
    std::optional<float>        temperature;
    std::optional<float>        top_p;
    std::optional<int>          top_k;
    std::optional<unsigned int> seed;
};

/**
 * @brief Composition root for a single sequence generation. Outlives every
 *        SequenceState it hands out and every prompt run through it.
 */
class Engine {
public:
    /**
     * @param model_dir         directory holding config.json and the safetensors shards
     * @param max_batch_tokens  most tokens one forward pass will hold, sizes the
     *                          forward scratch and caps prompt length
     * @param max_seq_len       KV cache depth in tokens, >= max_batch_tokens
     * @param overrides         sampling values that win over generation_config.json
     * @param enforce_eager     launches every decode step kernel by kernel, no capture
     * @note Loads weights to VRAM. Fails accordingly on a malformed config, a
     *       missing shard, or an allocation the device cannot satisfy.
     */
    Engine(const std::filesystem::path& model_dir,
           std::size_t                  max_batch_tokens,
           std::size_t                  max_seq_len,
           const SamplingOverrides&     overrides,
           bool                         enforce_eager);

    // generator_ binds references to model_, ctx_, and sampler_. Moving would
    // leave those bound to the moved from object.
    Engine(Engine&&)                 = delete;
    Engine& operator=(Engine&&)      = delete;
    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    /**
     * @brief Prefills prompt, then samples and emits to sink until EOS or the
     *        sequence's max new tokens. The EOS token is not emitted.
     * @param seq     advanced per token, built by make_sequence()
     * @param prompt  tokens to prefill, non empty, at most max_batch_tokens
     * @param sink    receives each token, flushed at the end
     */
    void generate(SequenceState& seq, std::span<const int> prompt, OutputSink& sink) {
        generator_.generate(seq, prompt, sink);
    }

    /**
     * @brief Builds a sequence carrying the resolved sampling params and the
     *        model's EOS token set.
     * @param prompt_len      number of prompt tokens, >= 1
     * @param max_new_tokens  cap on generated tokens, >= 1
     */
    [[nodiscard]] SequenceState make_sequence(int prompt_len, int max_new_tokens) const;

    [[nodiscard]] const sampling::SamplingParams& params() const noexcept { return params_; }
    [[nodiscard]] unsigned int                    seed()   const noexcept { return seed_; }

private:
    [[nodiscard]] static std::unique_ptr<attention::AttentionBackend> make_backend(
        const model::LlamaConfig& config);
    [[nodiscard]] static KVCache make_kv_cache(const model::LlamaConfig& config,
                                               std::size_t               max_seq_len);

    // Each member's initializer reads the ones above it. Reordering compiles and
    // then reads uninitialized state.
    unsigned int             seed_;
    sampling::SamplingParams params_;
    model::LlamaModel        model_;
    EngineContext            ctx_;
    sampling::Sampler        sampler_;
    Generator                generator_;
};

}  // namespace runtherder::engine
