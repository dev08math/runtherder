#pragma once

#include <memory>

#include <runtherder/attention/backend.h>

namespace runtherder::attention {

enum class BackendKind {
    Naive,
};

[[nodiscard]] std::unique_ptr<AttentionBackend> make_attention_backend(
    BackendKind       kind,
    const AttnConfig& config);

}  // namespace runtherder::attention
