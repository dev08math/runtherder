#include <runtherder/io/json_helpers.h>

#include <runtherder/io/text_file.h>

namespace runtherder::io {

nlohmann::json read_json_file(const std::filesystem::path& path) {
    auto parsed =
        nlohmann::json::parse(read_text_file(path), /*cb=*/nullptr, /*allow_exceptions=*/false);
    RUNTHERDER_CHECK(!parsed.is_discarded(), "json file is malformed");
    RUNTHERDER_CHECK(parsed.is_object(),     "json root is not an object");
    return parsed;
}

}  // namespace runtherder::io
