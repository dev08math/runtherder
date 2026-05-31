#include <runtherder/model/sharded_safetensors.h>

#include <cstddef>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

#include <runtherder/check.h>

namespace runtherder::model {

namespace {

constexpr std::string_view kIndexFileName  = "model.safetensors.index.json";
constexpr std::string_view kSingleFileName = "model.safetensors";

// Reads the shard filenames named by the index weight_map, deduplicated.
// The std::set gives a stable order across runs.
[[nodiscard]] std::set<std::string> read_shard_names(const std::filesystem::path& index_path) {
    std::ifstream in(index_path, std::ios::binary);
    RUNTHERDER_CHECK(in.is_open(), "failed to open safetensors index file");

    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();

    nlohmann::json index = nlohmann::json::parse(
        text, /*cb=*/nullptr, /*allow_exceptions=*/false);
    RUNTHERDER_CHECK(!index.is_discarded(), "safetensors index json is malformed");
    RUNTHERDER_CHECK(index.is_object(),     "safetensors index json is not an object");
    RUNTHERDER_CHECK(index.contains("weight_map"), "safetensors index missing weight_map");

    const auto& weight_map = index["weight_map"];
    RUNTHERDER_CHECK(weight_map.is_object(), "safetensors index weight_map is not an object");

    std::set<std::string> shard_names;
    for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
        shard_names.insert(it.value().get<std::string>());
    }
    RUNTHERDER_CHECK(!shard_names.empty(), "safetensors index weight_map is empty");
    return shard_names;
}

}  // namespace

ShardedSafetensors::ShardedSafetensors(std::vector<SafetensorsFile> shards,
                                       std::vector<std::string>     names,
                                       NameToShard                  name_to_shard) noexcept
    : shards_(std::move(shards)),
      names_(std::move(names)),
      name_to_shard_(std::move(name_to_shard)) {}

ShardedSafetensors ShardedSafetensors::open(const std::filesystem::path& dir) {
    const std::filesystem::path index_path  = dir / kIndexFileName;
    const std::filesystem::path single_path = dir / kSingleFileName;

    std::vector<std::filesystem::path> shard_paths;
    if (std::filesystem::exists(index_path)) {
        const std::set<std::string> shard_names = read_shard_names(index_path);
        shard_paths.reserve(shard_names.size());
        for (const std::string& name : shard_names) {
            shard_paths.push_back(dir / name);
        }
    } else {
        RUNTHERDER_CHECK(std::filesystem::exists(single_path),
                         "model dir has neither the index json nor model.safetensors");
        shard_paths.push_back(single_path);
    }

    std::vector<SafetensorsFile> shards;
    shards.reserve(shard_paths.size());
    for (const std::filesystem::path& path : shard_paths) {
        shards.push_back(SafetensorsFile::open(path));
    }

    std::vector<std::string> names;
    NameToShard              name_to_shard;

    // Names come from each shard's own header, not the index weight_map.
    for (std::size_t shard_index = 0; shard_index < shards.size(); ++shard_index) {
        for (const std::string& name : shards[shard_index].tensor_names()) {
            const auto [inserted_it, inserted] = name_to_shard.emplace(name, shard_index);
            RUNTHERDER_CHECK(inserted, "tensor name appears in more than one shard");
            names.push_back(name);
        }
    }

    return ShardedSafetensors(std::move(shards), std::move(names), std::move(name_to_shard));
}

bool ShardedSafetensors::contains(std::string_view name) const {
    return name_to_shard_.find(name) != name_to_shard_.end();
}

DType ShardedSafetensors::dtype(std::string_view name) const {
    return shard_of(name).dtype(name);
}

const std::vector<std::size_t>& ShardedSafetensors::shape(std::string_view name) const {
    return shard_of(name).shape(name);
}

std::span<const std::byte> ShardedSafetensors::bytes(std::string_view name) const {
    return shard_of(name).bytes(name);
}

const std::vector<std::string>& ShardedSafetensors::tensor_names() const noexcept {
    return names_;
}

const SafetensorsFile& ShardedSafetensors::shard_of(std::string_view name) const {
    const auto it = name_to_shard_.find(name);
    if (it == name_to_shard_.end()) [[unlikely]] {
        const std::string msg = "tensor \"" + std::string{name} + "\" not found in any shard";
        RUNTHERDER_CHECK(false, msg.c_str());
    }
    return shards_[it->second];
}

}  // namespace runtherder::model
