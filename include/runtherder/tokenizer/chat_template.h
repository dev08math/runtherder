#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <runtherder/tokenizer/tokenizer.h>

namespace runtherder::tokenizer {

/**
 * @brief Who authored a message in a conversation.
 */
enum class Role {
    System,
    User,
    Assistant,
};

/**
 * @brief One turn of a conversation.
 */
struct Message {
    Role        role;
    std::string content;
};

/**
 * @brief The checkpoint's own Jinja chat template, rendered over a conversation.
 *
 * @note The template is the authority on the prompt format. Special tokens come
 *       from it as literal text, so nothing is prepended on the caller's behalf.
 */
class ChatTemplate {
public:
    /**
     * @brief Reads the template from the checkpoint.
     *
     * @param model_dir directory holding tokenizer_config.json, or a
     *        chat_template.jinja sibling
     */
    [[nodiscard]] static ChatTemplate load(const std::filesystem::path& model_dir);

    /**
     * @brief Renders the conversation and tokenizes it.
     *
     * @param tok tokenizer built from the same checkpoint
     * @param messages conversation so far, oldest first
     * @param add_generation_prompt append the assistant header the model
     *        continues from
     *
     * @note Strips whatever the tokenizer's post processor prepends, so a
     *       template that emits its own BOS does not produce two.
     */
    [[nodiscard]] std::vector<int> encode(Tokenizer&                 tok,
                                          std::span<const Message>   messages,
                                          bool                       add_generation_prompt) const;

private:
    ChatTemplate(std::string source, std::string bos, std::string eos);

    std::string source_;
    std::string bos_;
    std::string eos_;
};

}  // namespace runtherder::tokenizer
