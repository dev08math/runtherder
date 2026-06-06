#include <runtherder/attention/factory.h>

#include <memory>

#include <runtherder/attention/naive.h>
#include <runtherder/check.h>

namespace runtherder::attention {

std::unique_ptr<AttentionBackend> make_attention_backend(
    BackendKind       kind,
    const AttnConfig& config) {
    switch (kind) {
        case BackendKind::Naive:
            return std::make_unique<NaiveAttention>(config);
    }
    RUNTHERDER_CHECK(false, "unknown attention BackendKind");
}

}  // namespace runtherder::attention
