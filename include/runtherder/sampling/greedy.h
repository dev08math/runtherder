#pragma once

#include <cuda_bf16.h>

namespace runtherder::sampling {

// logits is a device pointer to vocab bf16 values. vocab >= 1.
[[nodiscard]] int greedy_argmax(const __nv_bfloat16* logits, int vocab);

}  // namespace runtherder::sampling
