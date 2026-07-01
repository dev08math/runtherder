#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace runtherder::model {

enum class ModelType {
    Qwen3,
    Llama3,
};

/**
 * @brief Universal hyperparameters every LLM carries. Family specific
 *        extensions live in their family's own config under supported/.
 */
class Config {
public:
    /**
     * @brief Parses the universal portion of config.json under model_dir.
     * @note Validates the parsed fields, and fails accordingly on a malformed config.
     */
    [[nodiscard]] static Config load(const std::filesystem::path& model_dir);

    [[nodiscard]] ModelType        model_type()          const noexcept { return model_type_; }
    [[nodiscard]] std::size_t      hidden_dim()          const noexcept { return hidden_dim_; }
    [[nodiscard]] std::size_t      num_layers()          const noexcept { return num_layers_; }
    [[nodiscard]] std::size_t      vocab_size()          const noexcept { return vocab_size_; }
    [[nodiscard]] const std::vector<int>& eos_token_ids() const noexcept { return eos_token_ids_; }
    [[nodiscard]] std::size_t      max_seq_len()         const noexcept { return max_seq_len_; }
    [[nodiscard]] bool             tie_word_embeddings() const noexcept { return tie_word_embeddings_; }

private:
    Config() = default;

    ModelType        model_type_          = ModelType::Qwen3;
    std::size_t      hidden_dim_          = 0;
    std::size_t      num_layers_          = 0;
    std::size_t      vocab_size_          = 0;
    std::vector<int> eos_token_ids_;
    std::size_t      max_seq_len_         = 0;
    bool             tie_word_embeddings_ = false;
};

}  // namespace runtherder::model
