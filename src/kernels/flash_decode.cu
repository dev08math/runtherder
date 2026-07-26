#include <runtherder/kernels/flash_decode.cuh>

#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <runtherder/device/check.cuh>
#include <runtherder/kernels/quantization.cuh>

namespace runtherder::kernels {

namespace {

constexpr int kBlockThreads = 128;
constexpr int kWarpSize     = 32;
constexpr int kElemsPerLane = 16;  // one 16 byte vector load of E4M3 per lane

// Lanes of a warp that share this thread's key. A full warp mask would hang:
// key groups leave the loop at different iteration counts, so a collective
// naming absent lanes never completes.
template <int LanesPerKey>
[[nodiscard]] __device__ __forceinline__ unsigned int group_lane_mask(int lane) {
    if constexpr (LanesPerKey == kWarpSize) {
        return 0xffffffffu;
    } else {
        return ((1u << LanesPerKey) - 1u) << (lane & ~(LanesPerKey - 1));
    }
}

// q for this lane's slice, held in registers for the whole key loop.
template <int HeadDim>
__device__ __forceinline__ void load_q_slice(
    float (&q_reg)[kElemsPerLane],
    const __nv_bfloat16* __restrict__ q,
    int qh,
    int base) {
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
        q_reg[i] = __bfloat162float(q[qh * HeadDim + base + i]);
    }
}

// Unscaled q.k for one key. LanesPerKey lanes each cover kElemsPerLane elements
// and combine through the group mask, so the reduction never leaves the warp.
template <int LanesPerKey>
[[nodiscard]] __device__ __forceinline__ float dot_one_key(
    const float (&q_reg)[kElemsPerLane],
    const __nv_fp8_e4m3* __restrict__ k_row,
    float                             k_s,
    unsigned int                      mask) {
    const uint4          bits = *reinterpret_cast<const uint4*>(k_row);
    const __nv_fp8_e4m3* vals = reinterpret_cast<const __nv_fp8_e4m3*>(&bits);

    float p = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
        p += q_reg[i] * dequantize_kv_fp8(vals[i], k_s);
    }
#pragma unroll
    for (int off = 1; off < LanesPerKey; off <<= 1) {
        p += __shfl_xor_sync(mask, p, off);
    }
    return p;
}

// Fold one key into the running online softmax state.
__device__ __forceinline__ void accumulate_key(
    float (&acc)[kElemsPerLane],
    float&                            m,
    float&                            l,
    float                             score,
    const __nv_fp8_e4m3* __restrict__ v_row,
    float                             v_s) {
    const float m_new = fmaxf(m, score);
    const float corr  = __expf(m - m_new);
    const float w     = __expf(score - m_new);

    const uint4          bits = *reinterpret_cast<const uint4*>(v_row);
    const __nv_fp8_e4m3* vals = reinterpret_cast<const __nv_fp8_e4m3*>(&bits);
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
        acc[i] = acc[i] * corr + w * dequantize_kv_fp8(vals[i], v_s);
    }
    l = l * corr + w;
    m = m_new;
}

// Combine the key groups sharing a warp. Partners under xor by off hold the
// same head_dim slice, so acc combines elementwise.
template <int LanesPerKey>
__device__ __forceinline__ void merge_groups_in_warp(
    float (&acc)[kElemsPerLane],
    float& m,
    float& l) {
#pragma unroll
    for (int off = LanesPerKey; off < kWarpSize; off <<= 1) {
        const float m_other = __shfl_xor_sync(0xffffffffu, m, off);
        const float l_other = __shfl_xor_sync(0xffffffffu, l, off);
        const float m_new   = fmaxf(m, m_other);
        const float c_self  = __expf(m - m_new);
        const float c_other = __expf(m_other - m_new);
#pragma unroll
        for (int i = 0; i < kElemsPerLane; ++i) {
            const float a_other = __shfl_xor_sync(0xffffffffu, acc[i], off);
            acc[i] = acc[i] * c_self + a_other * c_other;
        }
        l = l * c_self + l_other * c_other;
        m = m_new;
    }
}

// One block per (q head, split). HeadDim / 16 lanes cooperate on a key, so
// 128 / that many keys are in flight per pass. K and V are E4M3, dequantized by
// their per (token, kv_head) scale at the load site. Writes the running
// m / l / acc to scratch instead of dividing. m starts at -FLT_MAX/2 rather
// than -inf: merging two never entered groups would evaluate -inf minus -inf.
template <int HeadDim>
__global__ void flash_decode_partial_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k,
    const __nv_fp8_e4m3* __restrict__ v,
    const float* __restrict__         k_scale,
    const float* __restrict__         v_scale,
    float* __restrict__               partial,
    int                               num_splits,
    const int* __restrict__           cache_len,
    int                               num_q_heads,
    int                               num_kv_heads,
    float                             scale) {
    constexpr int kLanesPerKey = HeadDim / kElemsPerLane;
    constexpr int kGroups      = kBlockThreads / kLanesPerKey;
    constexpr int kWarps       = kBlockThreads / kWarpSize;

    const int qh  = blockIdx.x;
    const int si  = blockIdx.y;
    const int tid = threadIdx.x;

    const int group   = tid / kLanesPerKey;
    const int base    = (tid % kLanesPerKey) * kElemsPerLane;
    const int warp_id = tid / kWarpSize;
    const int lane    = tid % kWarpSize;

    const int kvh    = qh / (num_q_heads / num_kv_heads);
    const int n_keys = *cache_len + 1;
    const int chunk  = (n_keys + num_splits - 1) / num_splits;
    const int start  = si * chunk;
    const int end    = min(start + chunk, n_keys);

    const unsigned int mask = group_lane_mask<kLanesPerKey>(lane);

    float q_reg[kElemsPerLane];
    load_q_slice<HeadDim>(q_reg, q, qh, base);

    float acc[kElemsPerLane];
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
        acc[i] = 0.0f;
    }
    float m = -FLT_MAX / 2.0f;
    float l = 0.0f;

    for (int j = start + group; j < end; j += kGroups) {
        const long long row = static_cast<long long>(j) * num_kv_heads + kvh;
        const float     ks  = k_scale[row];
        const float     vs  = v_scale[row];

        const float score =
            dot_one_key<kLanesPerKey>(q_reg, k + row * HeadDim + base, ks, mask) * scale;
        accumulate_key(acc, m, l, score, v + row * HeadDim + base, vs);
    }

    merge_groups_in_warp<kLanesPerKey>(acc, m, l);

    __shared__ float s_acc[kWarps * HeadDim];
    __shared__ float s_m[kWarps];
    __shared__ float s_l[kWarps];

    if (lane < kLanesPerKey) {
#pragma unroll
        for (int i = 0; i < kElemsPerLane; ++i) {
            s_acc[warp_id * HeadDim + base + i] = acc[i];
        }
        if (lane == 0) {
            s_m[warp_id] = m;
            s_l[warp_id] = l;
        }
    }
    __syncthreads();

    float M = s_m[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) {
        M = fmaxf(M, s_m[w]);
    }

    const long long hs   = static_cast<long long>(num_q_heads) * num_splits;
    const long long slot = static_cast<long long>(qh) * num_splits + si;

    for (int d = tid; d < HeadDim; d += kBlockThreads) {
        float a = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            a += s_acc[w * HeadDim + d] * __expf(s_m[w] - M);
        }
        partial[2 * hs + slot * HeadDim + d] = a;
    }

    if (tid == 0) {
        float L = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            L += s_l[w] * __expf(s_m[w] - M);
        }
        partial[slot]      = M;   // [num_q_heads, num_splits]
        partial[hs + slot] = L;
    }
}

// One block per q head. Merges the num_splits partials for that head onto a
// per thread max, then normalizes. Thread d owns lane d, reads across splits.
__global__ void flash_decode_reduce_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__   partial,
    int                         num_splits,
    int                         num_q_heads,
    int                         head_dim) {
    const int       qh = blockIdx.x;
    const int       d  = threadIdx.x;
    const long long hs = static_cast<long long>(num_q_heads) * num_splits;

    float M = -FLT_MAX;
    for (int i = 0; i < num_splits; ++i) {
        M = fmaxf(M, partial[static_cast<long long>(qh) * num_splits + i]);
    }

    float acc = 0.0f;
    float L   = 0.0f;
    for (int i = 0; i < num_splits; ++i) {
        const long long slot = static_cast<long long>(qh) * num_splits + i;
        const float     a    = __expf(partial[slot] - M);
        L   += a * partial[hs + slot];
        acc += a * partial[2 * hs + slot * head_dim + d];
    }

    out[qh * head_dim + d] = __float2bfloat16(acc / L);
}

template <int HeadDim>
void launch_partial(
    const dim3&          grid,
    const __nv_bfloat16* q,
    const __nv_fp8_e4m3* k,
    const __nv_fp8_e4m3* v,
    const float*         k_scale,
    const float*         v_scale,
    float*               partial,
    int                  num_splits,
    const int*           cache_len,
    int                  num_q_heads,
    int                  num_kv_heads,
    float                scale,
    cudaStream_t         stream) {
    flash_decode_partial_kernel<HeadDim><<<grid, kBlockThreads, 0, stream>>>(
        q, k, v, k_scale, v_scale, partial, num_splits, cache_len, num_q_heads,
        num_kv_heads, scale);
}

}  // namespace

void flash_decode_split_bf16(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const __nv_fp8_e4m3* k,
    const __nv_fp8_e4m3* v,
    const float*         k_scale,
    const float*         v_scale,
    float*               partial,
    int                  num_splits,
    const int*           cache_len,
    int                  num_q_heads,
    int                  num_kv_heads,
    int                  head_dim,
    float                scale,
    cudaStream_t         stream) {
    RUNTHERDER_CHECK(cache_len != nullptr, "cache_len must not be null");
    RUNTHERDER_CHECK(num_splits   >= 1, "num_splits must be >= 1");
    RUNTHERDER_CHECK(num_q_heads  >= 1, "num_q_heads must be >= 1");
    RUNTHERDER_CHECK(num_kv_heads >= 1, "num_kv_heads must be >= 1");
    RUNTHERDER_CHECK(num_q_heads % num_kv_heads == 0,
                     "num_q_heads must be divisible by num_kv_heads");

    const dim3 grid(static_cast<unsigned int>(num_q_heads),
                    static_cast<unsigned int>(num_splits));

    switch (head_dim) {
        case 32:
            launch_partial<32>(grid, q, k, v, k_scale, v_scale, partial, num_splits,
                               cache_len, num_q_heads, num_kv_heads, scale, stream);
            break;
        case 64:
            launch_partial<64>(grid, q, k, v, k_scale, v_scale, partial, num_splits,
                               cache_len, num_q_heads, num_kv_heads, scale, stream);
            break;
        case 128:
            launch_partial<128>(grid, q, k, v, k_scale, v_scale, partial, num_splits,
                                cache_len, num_q_heads, num_kv_heads, scale, stream);
            break;
        case 256:
            launch_partial<256>(grid, q, k, v, k_scale, v_scale, partial, num_splits,
                                cache_len, num_q_heads, num_kv_heads, scale, stream);
            break;
        case 512:
            launch_partial<512>(grid, q, k, v, k_scale, v_scale, partial, num_splits,
                                cache_len, num_q_heads, num_kv_heads, scale, stream);
            break;
        default:
            RUNTHERDER_CHECK(false, "head_dim must be one of 32, 64, 128, 256, 512");
    }
    RUNTHERDER_CUDA_CHECK_LAST();

    flash_decode_reduce_kernel<<<num_q_heads, head_dim, 0, stream>>>(
        out, partial, num_splits, num_q_heads, head_dim);
    RUNTHERDER_CUDA_CHECK_LAST();
}

}  // namespace runtherder::kernels
