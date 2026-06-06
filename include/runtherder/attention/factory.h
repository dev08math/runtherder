#pragma once

#include <memory>

#include <runtherder/attention/backend.h>

namespace runtherder::attention {

/**
 * @brief Selects which attention backend to construct. One entry today, gains a
 *        case per backend as they land (a custom optimized kernel next).
 */
enum class BackendKind {
    Naive,
};

/**
 * @brief Builds the chosen attention backend behind the ABC, fixed to config.
 * @note Definition lives in factory.cu so call sites depend on the seam, not on
 *       any concrete type.
 */
[[nodiscard]] std::unique_ptr<AttentionBackend> make_attention_backend(
    BackendKind       kind,
    const AttnConfig& config);

}  // namespace runtherder::attention
