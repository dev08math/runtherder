#include <runtherder/engine/context.h>

#include <cstddef>
#include <memory>
#include <utility>

#include <runtherder/check.h>

namespace runtherder::engine {

EngineContext::EngineContext(std::size_t                                  max_batch_tokens,
                             std::unique_ptr<attention::AttentionBackend> attention,
                             KVCache                                      kv_cache)
    : max_batch_tokens_(max_batch_tokens),
      attention_(std::move(attention)),
      kv_cache_(std::move(kv_cache)) {
    RUNTHERDER_CHECK(max_batch_tokens_ >= 1, "max_batch_tokens must be >= 1");
    RUNTHERDER_CHECK(attention_ != nullptr, "attention backend must not be null");
}

}  // namespace runtherder::engine
