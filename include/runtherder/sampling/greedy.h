#pragma once

#include <cuda_bf16.h>

namespace runtherder::sampling {

// The temperature == 0 path of the sampler: a deterministic argmax, no draw.
[[nodiscard]] int greedy_argmax(const __nv_bfloat16* logits, int vocab);

}  // namespace runtherder::sampling
