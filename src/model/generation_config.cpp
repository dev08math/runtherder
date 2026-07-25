#include <runtherder/model/generation_config.h>

#include <string_view>

#include <nlohmann/json.hpp>

#include <runtherder/io/json_helpers.h>

namespace runtherder::model {

namespace {

constexpr std::string_view kGenerationConfigFileName = "generation_config.json";

template <typename T>
[[nodiscard]] T optional_field(const nlohmann::json& obj, const char* key, T fallback) {
    if (!obj.contains(key) || obj.at(key).is_null()) {
        return fallback;
    }
    return obj.at(key).get<T>();
}

}  // namespace

GenerationConfig GenerationConfig::load(const std::filesystem::path& model_dir) {
    GenerationConfig out;

    const std::filesystem::path path = model_dir / kGenerationConfigFileName;
    if (!std::filesystem::exists(path)) {
        return out;
    }

    const nlohmann::json cfg = io::read_json_file(path);
    out.temperature_ = optional_field<float>(cfg, "temperature", out.temperature_);
    out.top_p_       = optional_field<float>(cfg, "top_p", out.top_p_);
    out.top_k_       = optional_field<int>(cfg, "top_k", out.top_k_);

    return out;
}

}  // namespace runtherder::model
