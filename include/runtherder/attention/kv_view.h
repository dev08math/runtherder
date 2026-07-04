#pragma once

#include <cuda_fp8.h>

namespace runtherder::attention {

/**
 * @brief View of one KV cache layer. k, v are E4M3 [len, num_kv_heads, head_dim].
 *        k_scale, v_scale are the per (token, kv_head) dequant scales
 *        [len, num_kv_heads]. Recover one value: float(k[i]) * k_scale[token, kvh].
 */
struct KVView {
    const __nv_fp8_e4m3* k;
    const __nv_fp8_e4m3* v;
    const float*         k_scale;
    const float*         v_scale;
    int                  len;
};

}  // namespace runtherder::attention
