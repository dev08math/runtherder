#include <runtherder/io/json_helpers.h>

#include <fstream>
#include <sstream>

namespace runtherder::io {

nlohmann::json read_json_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    RUNTHERDER_CHECK(in.is_open(), "failed to open json file");

    std::stringstream buffer;
    buffer << in.rdbuf();

    auto parsed = nlohmann::json::parse(buffer.str(), /*cb=*/nullptr, /*allow_exceptions=*/false);
    RUNTHERDER_CHECK(!parsed.is_discarded(), "json file is malformed");
    RUNTHERDER_CHECK(parsed.is_object(),     "json root is not an object");
    return parsed;
}

}  // namespace runtherder::io
