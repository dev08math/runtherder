#include <runtherder/model/config.h>

#include <string>
#include <string_view>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include <runtherder/check.h>
#include <runtherder/io/json_helpers.h>
#include <runtherder/string_hash.h>

namespace runtherder::model {

namespace {

constexpr std::string_view kConfigFileName = "config.json";

// Registry of supported model_type strings.
[[nodiscard]] const auto& architecture_registry() {
    static const std::unordered_map<std::string, ArchitectureKind,
                                    TransparentStringHash, std::equal_to<>> table{
        {"qwen3", ArchitectureKind::Qwen3},
    };
    return table;
}

[[nodiscard]] ArchitectureKind parse_architecture(std::string_view raw) {
    const auto& reg = architecture_registry();
    const auto  it  = reg.find(raw);
    if (it == reg.end()) [[unlikely]] {
        const std::string msg = "unsupported model_type \"" + std::string{raw} + "\"";
        RUNTHERDER_CHECK(false, msg.c_str());
    }
    return it->second;
}

}  // namespace

Config Config::load(const std::filesystem::path& model_dir) {
    const nlohmann::json cfg = io::read_json_file(model_dir / kConfigFileName);

    Config out;
    out.architecture_        = parse_architecture(io::require_field<std::string>(cfg, "model_type"));
    out.hidden_dim_          = io::require_field<std::size_t>(cfg, "hidden_size");
    out.num_layers_          = io::require_field<std::size_t>(cfg, "num_hidden_layers");
    out.vocab_size_          = io::require_field<std::size_t>(cfg, "vocab_size");
    out.eos_token_id_        = io::require_field<int>(cfg, "eos_token_id");
    out.max_seq_len_         = io::require_field<std::size_t>(cfg, "max_position_embeddings");
    out.tie_word_embeddings_ = io::require_field<bool>(cfg, "tie_word_embeddings");

    RUNTHERDER_CHECK(out.hidden_dim_  > 0, "hidden_size must be positive");
    RUNTHERDER_CHECK(out.num_layers_  > 0, "num_hidden_layers must be positive");
    RUNTHERDER_CHECK(out.vocab_size_  > 0, "vocab_size must be positive");
    RUNTHERDER_CHECK(out.max_seq_len_ > 0, "max_position_embeddings must be positive");

    return out;
}

}  // namespace runtherder::model
