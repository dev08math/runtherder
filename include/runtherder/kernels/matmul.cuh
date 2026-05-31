#pragma once

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <type_traits>

#include <runtherder/device/memory.cuh>

namespace runtherder::kernels {

/**
 * @brief Reusable cuBLASLt context for BF16 matmuls. Owns the library handle
 *        and a device workspace, created once and reused across calls.
 * @note Move only.
 */
class Matmul {
public:
    Matmul();

    Matmul(Matmul&&) noexcept            = default;
    Matmul& operator=(Matmul&&) noexcept = default;
    Matmul(const Matmul&)                = delete;
    Matmul& operator=(const Matmul&)     = delete;

    /**
     * @brief y = x · Wᵀ. BF16 storage, FP32 accumulation, no bias. Row major:
     *        x [m, k], weight [n, k] (HF Linear), y [m, n].
     * @pre Device pointers. m, n, k >= 1. y distinct from x and weight.
     * @note Async on stream. Result not valid until the stream is synchronized.
     * @note Matches a double precision reference within max abs 5e-2,
     *       cosine 0.9999.
     */
    void linear_bf16(__nv_bfloat16*       y,
                     const __nv_bfloat16* x,
                     const __nv_bfloat16* weight,
                     int                  m,
                     int                  n,
                     int                  k,
                     cudaStream_t         stream = nullptr);

private:
    // The handle comes from cublasLtCreate and is released by cublasLtDestroy,
    // not delete. unique_ptr defaults to delete, so the cleanup is wired here.
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
