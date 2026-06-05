#pragma once

#include <memory>

#include <runtherder/attention/backend.h>

namespace runtherder::attention {

/**
 * @brief Selects which attention backend to construct. One entry today, gains a
 *        case per backend as they land.
 */
enum class BackendKind {
    FlashInfer,
};

/**
 * @brief Builds the chosen attention backend behind the ABC.
 * @note Definition lands with the first concrete backend. Declared here so call
 *       sites depend on the seam, not on any concrete type.
 */
[[nodiscard]] std::unique_ptr<AttentionBackend> make_attention_backend(BackendKind kind);

}  // namespace runtherder::attention
