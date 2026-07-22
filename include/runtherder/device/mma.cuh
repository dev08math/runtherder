#pragma once

#include <cstdint>

#include <cuda_fp16.h>

namespace runtherder::device {

inline constexpr int kMmaM = 16;
inline constexpr int kMmaN = 8;
inline constexpr int kMmaK = 16;

/**
 * @brief 'A' operand of the m16n8k16 mma, a 16x16 row major fp16 tile scattered
 *        across the warp. get_row/get_col map (lane, element) to the (row, col)
 *        that element holds, the distribution the instruction mandates.
 */
struct FragmentA {
    static constexpr int kElements = 8;
    static constexpr int kRegs     = 4;

    std::uint32_t reg[kRegs];  // two packed fp16 per register

    [[nodiscard]] static __device__ __forceinline__ int get_row(int lane, int i) {
        return (lane >> 2) + 8 * ((i >> 1) & 1);
    }
    [[nodiscard]] static __device__ __forceinline__ int get_col(int lane, int i) {
        return (lane & 3) * 2 + (i & 1) + 8 * (i >> 2);
    }
};

/**
 * @brief 'B' operand of the m16n8k16 mma, a 16x8 col major fp16 tile scattered
 *        across the warp.
 */
struct FragmentB {
    static constexpr int kElements = 4;
    static constexpr int kRegs     = 2;

    std::uint32_t reg[kRegs];  // two packed fp16 per register

    [[nodiscard]] static __device__ __forceinline__ int get_row(int lane, int i) {
        return (lane & 3) * 2 + (i & 1) + 8 * (i >> 1);
    }
    [[nodiscard]] static __device__ __forceinline__ int get_col(int lane, int i) {
        return lane >> 2;
    }
};

/**
 * @brief Accumulator and result of the m16n8k16 mma, a 16x8 fp32 tile scattered
 *        across the warp. Element i of a lane lives in reg[i].
 */
struct FragmentC {
    static constexpr int kElements = 4;
    static constexpr int kRegs     = 4;

    float reg[kRegs];

    __device__ __forceinline__ void fill(float val) {
        reg[0] = val;
        reg[1] = val;
        reg[2] = val;
        reg[3] = val;
    }

    [[nodiscard]] static __device__ __forceinline__ int get_row(int lane, int i) {
        return (lane >> 2) + 8 * (i >> 1);
    }
    [[nodiscard]] static __device__ __forceinline__ int get_col(int lane, int i) {
        return (lane & 3) * 2 + (i & 1);
    }
};

/**
 * @brief D = A * B + C, one m16n8k16 tensor core multiply add. fp16 operands,
 *        fp32 accumulate.
 */
__device__ __forceinline__ void mma(FragmentC&       D,
                                    const FragmentA& A,
                                    const FragmentB& B,
                                    const FragmentC& C) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D.reg[0]), "=f"(D.reg[1]), "=f"(D.reg[2]), "=f"(D.reg[3])
        : "r"(A.reg[0]), "r"(A.reg[1]), "r"(A.reg[2]), "r"(A.reg[3]),
          "r"(B.reg[0]), "r"(B.reg[1]),
          "f"(C.reg[0]), "f"(C.reg[1]), "f"(C.reg[2]), "f"(C.reg[3]));
}

/**
 * @brief acc += A * B, one m16n8k16 tensor core multiply add in place.
 */
__device__ __forceinline__ void mma(FragmentC&       acc,
                                    const FragmentA& A,
                                    const FragmentB& B) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc.reg[0]), "+f"(acc.reg[1]), "+f"(acc.reg[2]), "+f"(acc.reg[3])
        : "r"(A.reg[0]), "r"(A.reg[1]), "r"(A.reg[2]), "r"(A.reg[3]),
          "r"(B.reg[0]), "r"(B.reg[1]));
}

}  // namespace runtherder::device
