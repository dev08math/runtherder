#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <tokenizers_cpp.h>

namespace runtherder::tokenizer {

/**
 * @brief HF tokenizer over the model dir's tokenizer.json, via tokenizers-cpp.
 * @note Only constructable via load().
 */
class Tokenizer {
public:
    /**
     * @brief Reads tokenizer.json under model_dir and builds the HF tokenizer.
     * @note Bails via RUNTHERDER_CHECK on missing file.
     */
    [[nodiscard]] static Tokenizer load(const std::filesystem::path& model_dir);

    [[nodiscard]] std::vector<int> encode(std::string_view text);
    [[nodiscard]] std::string      decode(std::span<const int> ids);

private:
    explicit Tokenizer(std::unique_ptr<tokenizers::Tokenizer> impl);

    std::unique_ptr<tokenizers::Tokenizer> impl_;
};

}  // namespace runtherder::tokenizer
