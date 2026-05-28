#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

namespace runtherder {

/**
 * @brief Heterogeneous hash for std::unordered_map<std::string, V>. Lets
 *        find/contains take std::string_view directly without allocating
 *        a temporary std::string per call.
 * @note Pair with std::equal_to<> as the KeyEqual to enable heterogeneous
 *       lookup end to end.
 */
struct TransparentStringHash {
    using is_transparent = void;
    [[nodiscard]] std::size_t operator()(std::string_view s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }
    [[nodiscard]] std::size_t operator()(const std::string& s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }
    [[nodiscard]] std::size_t operator()(const char* s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }
};

}  // namespace runtherder
