#include <runtherder/tokenizer/tokenizer.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <runtherder/check.h>
#include <runtherder/io/mmap_file.h>

namespace runtherder::tokenizer {

Tokenizer Tokenizer::load(const std::filesystem::path& model_dir) {
    const io::MmapFile               file = io::MmapFile::open(model_dir / "tokenizer.json");
    const std::span<const std::byte> raw  = file.bytes();

    const std::string blob(reinterpret_cast<const char*>(raw.data()), raw.size());

    std::unique_ptr<tokenizers::Tokenizer> impl = tokenizers::Tokenizer::FromBlobJSON(blob);
    RUNTHERDER_CHECK(impl != nullptr, "failed to build tokenizer from tokenizer.json");

    return Tokenizer{std::move(impl)};
}

Tokenizer::Tokenizer(std::unique_ptr<tokenizers::Tokenizer> impl) : impl_{std::move(impl)} {}

std::vector<int> Tokenizer::encode(std::string_view text) {
    return impl_->Encode(std::string{text});
}

std::string Tokenizer::decode(std::span<const int> ids) {
    return impl_->Decode(std::vector<std::int32_t>{ids.begin(), ids.end()});
}

}  // namespace runtherder::tokenizer
