#include <runtherder/tokenizer/chat_template.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>

#include <runtherder/check.h>
#include <runtherder/io/json_helpers.h>
#include <runtherder/io/text_file.h>

namespace runtherder::tokenizer {

namespace {

[[nodiscard]] const char* role_name(Role role) {
    switch (role) {
        case Role::System:    return "system";
        case Role::User:      return "user";
        case Role::Assistant: return "assistant";
    }
    RUNTHERDER_CHECK(false, "unhandled Role");
    return "";
}

[[nodiscard]] std::string token_string(const nlohmann::json& field) {
    if (field.is_string()) {
        return field.get<std::string>();
    }
    if (field.is_object() && field.contains("content")) {
        return field.at("content").get<std::string>();
    }
    return {};
}

[[nodiscard]] std::size_t auto_prefix_len(Tokenizer& tok) {
    const std::vector<int> a = tok.encode("a");
    const std::vector<int> b = tok.encode("b");

    const auto mismatch = std::ranges::mismatch(a, b);
    return static_cast<std::size_t>(std::distance(a.begin(), mismatch.in1));
}

}  // namespace

ChatTemplate::ChatTemplate(std::string source, std::string bos, std::string eos)
    : source_{std::move(source)}, bos_{std::move(bos)}, eos_{std::move(eos)} {}

ChatTemplate ChatTemplate::load(const std::filesystem::path& model_dir) {
    const std::filesystem::path jinja = model_dir / "chat_template.jinja";
    const std::filesystem::path config = model_dir / "tokenizer_config.json";

    std::string source;
    std::string bos;
    std::string eos;

    if (std::filesystem::exists(config)) {
        const nlohmann::json cfg = io::read_json_file(config);
        if (cfg.contains("chat_template") && cfg.at("chat_template").is_string()) {
            source = cfg.at("chat_template").get<std::string>();
        }
        if (cfg.contains("bos_token")) {
            bos = token_string(cfg.at("bos_token"));
        }
        if (cfg.contains("eos_token")) {
            eos = token_string(cfg.at("eos_token"));
        }
    }

    if (source.empty() && std::filesystem::exists(jinja)) {
        source = io::read_text_file(jinja);
    }

    RUNTHERDER_CHECK(!source.empty(),
                     "checkpoint carries no chat template, run without conversation mode");

    return ChatTemplate{std::move(source), std::move(bos), std::move(eos)};
}

std::vector<int> ChatTemplate::encode(Tokenizer&               tok,
                                      std::span<const Message> messages,
                                      bool                     add_generation_prompt) const {
    nlohmann::ordered_json turns = nlohmann::ordered_json::array();
    for (const Message& m : messages) {
        turns.push_back({{"role", role_name(m.role)}, {"content", m.content}});
    }

    minja::chat_template tmpl(source_, bos_, eos_);

    minja::chat_template_inputs inputs;
    inputs.messages               = std::move(turns);
    inputs.add_generation_prompt  = add_generation_prompt;

    const std::string rendered = tmpl.apply(inputs, minja::chat_template_options{});

    std::vector<int> ids = tok.encode(rendered);

    const std::size_t added = auto_prefix_len(tok);
    RUNTHERDER_CHECK(added <= ids.size(), "tokenizer prepended more tokens than it produced");
    ids.erase(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(added));

    return ids;
}

}  // namespace runtherder::tokenizer
