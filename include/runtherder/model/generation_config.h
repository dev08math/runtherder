#pragma once

#include <filesystem>

namespace runtherder::model {

/**
 * @brief Model's recommended sampling defaults from generation_config.json.
 *        CLI flags override these at the launch site.
 */
class GenerationConfig {
public:
    /**
     * @brief Loads generation_config.json under model_dir when present.
     * @note An absent file or an absent field yields the neutral default
     *       (temperature 1.0, top_p 1.0, top_k 0), leaving the distribution
     *       unaltered.
     */
    [[nodiscard]] static GenerationConfig load(const std::filesystem::path& model_dir);

    [[nodiscard]] float temperature() const noexcept { return temperature_; }
    [[nodiscard]] float top_p()       const noexcept { return top_p_; }
    [[nodiscard]] int   top_k()       const noexcept { return top_k_; }

private:
    GenerationConfig() = default;

    float temperature_ = 1.0F;
    float top_p_       = 1.0F;
    int   top_k_       = 0;
};

}  // namespace runtherder::model
