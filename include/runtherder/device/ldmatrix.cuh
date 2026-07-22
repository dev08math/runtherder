#pragma once

#include <cstdint>

#include <cuda_fp16.h>

#include <runtherder/device/mma.cuh>

namespace runtherder::device {

/**
 * @brief SMEM element offset of (row, col) under the XOR swizzle, applied to
 *        8 element (16 byte) groups. Store and load must both route through
 *        this. Keeps every group 16 byte aligned.
 */
template <int kStride>
[[nodiscard]] __device__ __forceinline__ int swizzle_offset(int row, int col) {
    return row * kStride + (((col >> 3) ^ (row & 7)) << 3) + (col & 7);
}

/**
 * @brief Loads a 16x16 row major tile at (row_offset, col_offset) into one
 *        FragmentA via ldmatrix.x4.
 */
template <int kStride>
__device__ __forceinline__ void load_q_fragment(FragmentA&    frag,
                                                 const __half* smem_tile,
                                                 int           row_offset,
                                                 int           col_offset) {
    const int lane        = threadIdx.x & 31;
    const int tile_id     = lane >> 3;
    const int row_in_tile = lane & 7;
    const int row         = row_offset + (tile_id & 1) * 8 + row_in_tile;
    const int col         = col_offset + (tile_id >> 1) * 8;

    const std::uint32_t addr =
        __cvta_generic_to_shared(&smem_tile[swizzle_offset<kStride>(row, col)]);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1]), "=r"(frag.reg[2]), "=r"(frag.reg[3])
        : "r"(addr));
}

/**
 * @brief Loads a 16x16 tile of K (row major, [n, k]) into the two FragmentB
 *        halves of Kᵀ for the QKᵀ mma. b0 holds n=0..7, b1 holds n=8..15.
 */
template <int kStride>
__device__ __forceinline__ void load_kt_fragments(FragmentB&    b0,
                                                   FragmentB&    b1,
                                                   const __half* smem_tile,
                                                   int           row_offset,
                                                   int           col_offset) {
    const int lane        = threadIdx.x & 31;
    const int tile_id     = lane >> 3;
    const int row_in_tile = lane & 7;
    const int row         = row_offset + (tile_id & 1) * 8 + row_in_tile;
    const int col         = col_offset + (tile_id >> 1) * 8;

    const std::uint32_t addr =
        __cvta_generic_to_shared(&smem_tile[swizzle_offset<kStride>(row, col)]);

    std::uint32_t r0, r1, r2, r3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));

    b0.reg[0] = r0;
    b0.reg[1] = r2;
    b1.reg[0] = r1;
    b1.reg[1] = r3;
}

/**
 * @brief Loads a 16x16 tile of V (row major, [k, d]) transposed into the two
 *        FragmentB halves for the PV mma. b0 holds d=0..7, b1 holds d=8..15.
 */
template <int kStride>
__device__ __forceinline__ void load_vt_fragments(FragmentB&    b0,
                                                   FragmentB&    b1,
                                                   const __half* smem_tile,
                                                   int           row_offset,
                                                   int           col_offset) {
    const int lane        = threadIdx.x & 31;
    const int tile_id     = lane >> 3;
    const int row_in_tile = lane & 7;
    const int row         = row_offset + (tile_id & 1) * 8 + row_in_tile;
    const int col         = col_offset + (tile_id >> 1) * 8;

    const std::uint32_t addr =
        __cvta_generic_to_shared(&smem_tile[swizzle_offset<kStride>(row, col)]);

    std::uint32_t r0, r1, r2, r3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));

    b0.reg[0] = r0;
    b0.reg[1] = r1;
    b1.reg[0] = r2;
    b1.reg[1] = r3;
}

}  // namespace runtherder::device
