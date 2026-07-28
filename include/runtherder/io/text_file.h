#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <runtherder/check.h>

namespace runtherder::io {

/**
 * @brief Reads the whole file at path into a string.
 *
 * @param path file to read, opened in binary so line endings survive
 *
 * @note Header only so the tokenizer and model libraries can both use it
 *       without one linking the other. Fails accordingly when path cannot be
 *       opened.
 */
[[nodiscard]] inline std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    RUNTHERDER_CHECK(in.is_open(), "failed to open file");

    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

}  // namespace runtherder::io
