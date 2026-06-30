#pragma once

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include <runtherder/device/memory.cuh>
#include <runtherder/device/scratch_arena.cuh>

namespace runtherder::kernels {

/**
 * @brief Scratch memory for one linear_w8a8 call, carved from the arena.
 */
struct W8A8Buffer {
    std::int8_t*  q;        // [m, k]
    float*        x_scale;  // [m]
    std::int32_t* acc;      // [m, n]

    [[nodiscard]] static W8A8Buffer from_arena(device::ScratchArena& arena,
                                               int m, int n, int k);
};

/**
 * @brief Reusable matmul context for bf16 and int8, built once.
 */
class Matmul {
public:
    Matmul();

    Matmul(Matmul&&) noexcept            = default;
    Matmul& operator=(Matmul&&) noexcept = default;
    Matmul(const Matmul&)                = delete;
    Matmul& operator=(const Matmul&)     = delete;

    /**
     * @brief y = x · Wᵀ. bf16 storage, fp32 accumulation.
     * @param y       [m, n]
     * @param x       [m, k]
     * @param weight  [n, k]
     * @note Matches a double precision reference within max abs 5e-2, cosine 0.9999.
     */
    void linear_bf16(__nv_bfloat16*       y,
                     const __nv_bfloat16* x,
                     const __nv_bfloat16* weight,
                     int                  m,
                     int                  n,
                     int                  k,
                     cudaStream_t         stream = nullptr);

    /**
     * @brief y = x · Wᵀ, int8 compute, bf16 out.
     * @param y        [m, n]
     * @param x        [m, k]
     * @param weight   [n, k]
     * @param w_scale  [n], per output channel
     * @note Requires a GPU with int8 tensor core support.
     */
    void linear_w8a8(__nv_bfloat16*       y,
                     const __nv_bfloat16* x,
                     const std::int8_t*   weight,
                     const __nv_bfloat16* w_scale,
                     const W8A8Buffer&    scratch,
                     int                  m,
                     int                  n,
                     int                  k,
                     cudaStream_t         stream = nullptr);

private:
    struct LtDeleter {
        void operator()(cublasLtHandle_t handle) const noexcept {
            if (handle != nullptr) {
                cublasLtDestroy(handle);
            }
        }
    };
    // cublasLtHandle_t is itself a pointer. remove_pointer_t strips one level so
    // unique_ptr owns a single handle, not a pointer to one.
    using LtHandle = std::unique_ptr<std::remove_pointer_t<cublasLtHandle_t>, LtDeleter>;

    // Recommended minimum workspace size for cublasLtMatmulAlgo_t with BF16 inputs and FP32 accumulation is 32 MiB by NVIDIA
    static constexpr std::size_t kWorkspaceBytes = std::size_t{32} << 20;

    LtHandle                            handle_;
    device::DeviceUniquePtr<std::byte>  workspace_;
};

}  // namespace runtherder::kernels
