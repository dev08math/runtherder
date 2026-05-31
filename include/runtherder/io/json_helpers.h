#pragma once

#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

#include <runtherder/check.h>

namespace runtherder::io {

/**
 * @brief Reads path into a parsed nlohmann::json object.
 * @note Bails via RUNTHERDER_CHECK on open failure, malformed JSON, or a
 *       non object root.
 */
[[nodiscard]] nlohmann::json read_json_file(const std::filesystem::path& path);

/**
 * @brief Extracts a required field from a JSON object, typed as T.
 * @pre The value at key is convertible to T. A type mismatch throws an
 *      nlohmann exception instead of bailing via RUNTHERDER_CHECK.
 * @note Bails via RUNTHERDER_CHECK if the key is missing.
 */
template <typename T>
[[nodiscard]] T require_field(const nlohmann::json& obj, const char* key) {
    if (!obj.contains(key)) [[unlikely]] {
        const std::string msg = std::string{"json missing required field "} + key;
        RUNTHERDER_CHECK(false, msg.c_str());
    }
    return obj.at(key).get<T>();
}

}  // namespace runtherder::io
