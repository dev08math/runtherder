#include <runtherder/kernels/flash_prefill.cuh>

#include <cmath>
#include <cstdint>

#include <runtherder/check.h>
#include <runtherder/device/ldmatrix.cuh>
#include <runtherder/device/mma.cuh>
#include <runtherder/kernels/quantization.cuh>

// Flash attention prefill on tensor cores. One block owns a BR row query tile for
// one q head, split across BR / 16 warps (grid.x = query tiles, grid.y = q heads).
// Q is staged to SMEM once, then a loop over the causal key tiles: dequant the fp8
// KV into SMEM, QKᵀ through mma, causal plus boundary mask, online softmax, PV
// through mma, normalize and write. fp16 mma inputs, fp32 accumulate.

namespace runtherder::kernels {

namespace {

using device::FragmentA;
using device::FragmentB;
using device::FragmentC;
using device::kMmaK;
using device::kMmaM;
using device::kMmaN;

[[nodiscard]] __device__ __forceinline__ std::uint32_t cvt_f16x2(float lo, float hi) {
    std::uint32_t r;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

template <int D, int BR, int BC>
__global__ void flash_prefill_kernel(__nv_bfloat16*       out,
                                     const __nv_bfloat16* q,
                                     const __nv_fp8_e4m3* k,
                                     const __nv_fp8_e4m3* v,
                                     const float*         k_scale,
                                     const float*         v_scale,
                                     int                  n_new,
                                     const int*           cache_len,
                                     int                  num_q_heads,
                                     int                  num_kv_heads,
                                     float                scale) {
    constexpr int QK_K = D / kMmaK;
    constexpr int QK_N = BC / kMmaN;
    constexpr int PV_N = D / kMmaN;
    constexpr int PV_K = BC / kMmaK;
    static_assert(BC % kMmaK == 0, "BC must be a whole number of PV mma steps");
    const int lane     = threadIdx.x & 31;
    const int warp_row = (threadIdx.x >> 5) * kMmaM;
    const int q_tile   = blockIdx.x;
    const int qh       = blockIdx.y;
    const int kvh      = qh / (num_q_heads / num_kv_heads);

    const int clen       = *cache_len;
    const int total_keys = clen + n_new;
    const int q_base     = q_tile * BR;

    __shared__ __half smem_q[BR * D];
    __shared__ __half smem_k[BC * D];
    __shared__ __half smem_v[BC * D];

    for (int idx = threadIdx.x; idx < BR * D; idx += blockDim.x) {
        const int r   = idx / D;
        const int c   = idx % D;
        const int tok = q_base + r;
        __half    h   = __float2half(0.0f);
        if (tok < n_new) {
            const __nv_bfloat16 qb =
                q[(static_cast<std::int64_t>(tok) * num_q_heads + qh) * D + c];
            h = __float2half(__bfloat162float(qb));
        }
        smem_q[device::swizzle_offset<D>(r, c)] = h;
    }
    __syncthreads();

    FragmentC O[PV_N];
    for (int n = 0; n < PV_N; ++n) {
        O[n].fill(0.0f);
    }
    float m[2] = {-INFINITY, -INFINITY};
    float l[2] = {0.0f, 0.0f};

    // Causal tile skip: this block only touches key tiles up to its last real
    // query position, fully future tiles are all masked anyway.
    const int last_tok   = min(q_base + BR, n_new) - 1;
    const int max_qpos   = clen + last_tok;
    const int all_tiles  = (total_keys + BC - 1) / BC;
    const int n_kv_tiles = min(all_tiles, max_qpos / BC + 1);

    for (int j = 0; j < n_kv_tiles; ++j) {
        for (int idx = threadIdx.x; idx < BC * D; idx += blockDim.x) {
            const int r   = idx / D;
            const int c   = idx % D;
            const int key = j * BC + r;
            __half    kh  = __float2half(0.0f);
            __half    vh  = __float2half(0.0f);
            if (key < total_keys) {
                const std::int64_t base =
                    (static_cast<std::int64_t>(key) * num_kv_heads + kvh);
                kh = __float2half(dequantize_kv_fp8(k[base * D + c], k_scale[base]));
                vh = __float2half(dequantize_kv_fp8(v[base * D + c], v_scale[base]));
            }
            smem_k[device::swizzle_offset<D>(r, c)] = kh;
            smem_v[device::swizzle_offset<D>(r, c)] = vh;
        }
        __syncthreads();

        FragmentC S[QK_N];
        for (int n = 0; n < QK_N; ++n) {
            S[n].fill(0.0f);
        }
        for (int kk = 0; kk < QK_K; ++kk) {
            FragmentA qf;
            device::load_q_fragment<D>(qf, smem_q, warp_row, kk * kMmaK);
#pragma unroll
            for (int kn = 0; kn < QK_N / 2; ++kn) {
                FragmentB kf0, kf1;
                device::load_kt_fragments<D>(kf0, kf1, smem_k, kn * kMmaK, kk * kMmaK);
                device::mma(S[2 * kn], qf, kf0);
                device::mma(S[2 * kn + 1], qf, kf1);
            }
        }

        for (int n = 0; n < QK_N; ++n) {
            for (int i = 0; i < FragmentC::kElements; ++i) {
                const int qr   = warp_row + FragmentC::get_row(lane, i);
                const int kc   = n * kMmaN + FragmentC::get_col(lane, i);
                const int qpos = clen + q_base + qr;
                const int kpos = j * BC + kc;
                float     s    = S[n].reg[i] * scale;
                if (kpos > qpos || kpos >= total_keys) {
                    s = -INFINITY;
                }
                S[n].reg[i] = s;
            }
        }

        // The rmax and rowsum reductions are warp collectives, every lane must
        // reach them. A fully masked row guards its state writes, never a shuffle.
        for (int row = 0; row < 2; ++row) {
            float rmax = -INFINITY;
            for (int n = 0; n < QK_N; ++n) {
                rmax = fmaxf(rmax, S[n].reg[row * 2]);
                rmax = fmaxf(rmax, S[n].reg[row * 2 + 1]);
            }
            rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, 1));
            rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, 2));

            const bool  masked = (rmax == -INFINITY);
            const float alpha  = masked ? 1.0f : __expf(m[row] - rmax);
            if (!masked) {
                for (int n = 0; n < PV_N; ++n) {
                    O[n].reg[row * 2]     *= alpha;
                    O[n].reg[row * 2 + 1] *= alpha;
                }
                l[row] *= alpha;
                m[row]  = rmax;
            }

            float rowsum = 0.0f;
            for (int n = 0; n < QK_N; ++n) {
                const float e0 = masked ? 0.0f : __expf(S[n].reg[row * 2] - rmax);
                const float e1 = masked ? 0.0f : __expf(S[n].reg[row * 2 + 1] - rmax);
                S[n].reg[row * 2]     = e0;
                S[n].reg[row * 2 + 1] = e1;
                rowsum += e0 + e1;
            }
            rowsum += __shfl_xor_sync(0xffffffffu, rowsum, 1);
            rowsum += __shfl_xor_sync(0xffffffffu, rowsum, 2);
            if (!masked) {
                l[row] += rowsum;
            }
        }

#pragma unroll
        for (int kc = 0; kc < PV_K; ++kc) {
            FragmentA P;
            P.reg[0] = cvt_f16x2(S[2 * kc].reg[0], S[2 * kc].reg[1]);
            P.reg[1] = cvt_f16x2(S[2 * kc].reg[2], S[2 * kc].reg[3]);
            P.reg[2] = cvt_f16x2(S[2 * kc + 1].reg[0], S[2 * kc + 1].reg[1]);
            P.reg[3] = cvt_f16x2(S[2 * kc + 1].reg[2], S[2 * kc + 1].reg[3]);

#pragma unroll
            for (int n = 0; n < PV_N; n += 2) {
                FragmentB vf0, vf1;
                device::load_vt_fragments<D>(vf0, vf1, smem_v, kc * kMmaK, n * kMmaN);
                device::mma(O[n], P, vf0);
                device::mma(O[n + 1], P, vf1);
            }
        }

        __syncthreads();
    }

    for (int n = 0; n < PV_N; ++n) {
        for (int i = 0; i < FragmentC::kElements; ++i) {
            const int tok = q_base + warp_row + FragmentC::get_row(lane, i);
            if (tok < n_new) {
                const int   c = n * kMmaN + FragmentC::get_col(lane, i);
                const float o = O[n].reg[i] / l[i >> 1];
                out[(static_cast<std::int64_t>(tok) * num_q_heads + qh) * D + c] =
                    __float2bfloat16(o);
            }
        }
    }
}

}  // namespace

void flash_prefill_bf16(__nv_bfloat16*       out,
                        const __nv_bfloat16* q,
                        const __nv_fp8_e4m3* k,
                        const __nv_fp8_e4m3* v,
                        const float*         k_scale,
                        const float*         v_scale,
                        int                  n_new,
                        const int*           cache_len,
                        int                  num_q_heads,
                        int                  num_kv_heads,
                        int                  head_dim,
                        float                scale,
                        cudaStream_t         stream) {
    RUNTHERDER_CHECK(head_dim == 128, "flash_prefill_bf16 supports head_dim 128 only");
    RUNTHERDER_CHECK(num_q_heads % num_kv_heads == 0,
                     "num_q_heads must be a multiple of num_kv_heads");

    constexpr int D  = 128;
    constexpr int BR = 64;  // 4 warps, one 16 row M tile each
    constexpr int BC = 64;

    const int  n_q_tiles = (n_new + BR - 1) / BR;
    const dim3 grid(static_cast<unsigned>(n_q_tiles), static_cast<unsigned>(num_q_heads));
    const dim3 block(BR / kMmaM * 32);

    flash_prefill_kernel<D, BR, BC><<<grid, block, 0, stream>>>(
        out, q, k, v, k_scale, v_scale, n_new, cache_len, num_q_heads, num_kv_heads,
        scale);
}

}  // namespace runtherder::kernels
