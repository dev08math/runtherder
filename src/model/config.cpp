#include <runtherder/model/config.h>

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include <runtherder/check.h>
#include <runtherder/io/json_helpers.h>
#include <runtherder/string_hash.h>

namespace runtherder::model {

namespace {

constexpr std::string_view kConfigFileName = "config.json";

// Registry of supported model_type strings.
[[nodiscard]] const auto& model_type_registry() {
    static const std::unordered_map<std::string, ModelType,
                                    TransparentStringHash, std::equal_to<>> table{
        {"qwen2", ModelType::Qwen2},
        {"qwen3", ModelType::Qwen3},
        {"llama", ModelType::Llama3},
    };
    return table;
}

[[nodiscard]] ModelType parse_model_type(std::string_view raw) {
    const auto& reg = model_type_registry();
    const auto  it  = reg.find(raw);
    if (it == reg.end()) [[unlikely]] {
        const std::string msg = "unsupported model_type \"" + std::string{raw} + "\"";
        RUNTHERDER_CHECK(false, msg.c_str());
    }
    return it->second;
}

[[nodiscard]] std::vector<int> parse_eos_token_ids(const nlohmann::json& cfg) {
    constexpr const char* kKey = "eos_token_id";
    if (!cfg.contains(kKey)) [[unlikely]] {
        RUNTHERDER_CHECK(false, "json missing required field eos_token_id");
    }
    const nlohmann::json& node = cfg.at(kKey);
    std::vector<int> ids;
    if (node.is_array()) {
        ids = node.get<std::vector<int>>();
    } else {
        ids.push_back(node.get<int>());
    }
    RUNTHERDER_CHECK(!ids.empty(), "eos_token_id must not be empty");
    return ids;
}

}  // namespace

Config Config::load(const std::filesystem::path& model_dir) {
    const nlohmann::json cfg = io::read_json_file(model_dir / kConfigFileName);

    Config out;
    out.model_type_          = parse_model_type(io::require_field<std::string>(cfg, "model_type"));
    out.hidden_dim_          = io::require_field<std::size_t>(cfg, "hidden_size");
    out.num_layers_          = io::require_field<std::size_t>(cfg, "num_hidden_layers");
    out.vocab_size_          = io::require_field<std::size_t>(cfg, "vocab_size");
    out.eos_token_ids_       = parse_eos_token_ids(cfg);
    out.max_seq_len_         = io::require_field<std::size_t>(cfg, "max_position_embeddings");
    out.tie_word_embeddings_ = io::require_field<bool>(cfg, "tie_word_embeddings");

    RUNTHERDER_CHECK(out.hidden_dim_  > 0, "hidden_size must be positive");
    RUNTHERDER_CHECK(out.num_layers_  > 0, "num_hidden_layers must be positive");
    RUNTHERDER_CHECK(out.vocab_size_  > 0, "vocab_size must be positive");
    RUNTHERDER_CHECK(out.max_seq_len_ > 0, "max_position_embeddings must be positive");

    return out;
}

}  // namespace runtherder::model
