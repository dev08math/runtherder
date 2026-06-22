#pragma once

#include <cuda_bf16.h>

namespace runtherder::attention {

// Non owning window into one layer of the KV cache. k, v span [len, num_kv_heads, head_dim].
struct KVView {
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    int                  len;
};

}  // namespace runtherder::attention
