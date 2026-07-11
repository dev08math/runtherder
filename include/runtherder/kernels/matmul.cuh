#pragma once

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
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
    using LtHandle = std::unique_ptr<std::remove_pointer_t<cublasLtHandle_t>, LtDeleter>;

    struct MatmulDescDeleter {
        void operator()(cublasLtMatmulDesc_t desc) const noexcept {
            if (desc != nullptr) {
                cublasLtMatmulDescDestroy(desc);
            }
        }
    };
    struct MatrixLayoutDeleter {
        void operator()(cublasLtMatrixLayout_t layout) const noexcept {
            if (layout != nullptr) {
                cublasLtMatrixLayoutDestroy(layout);
            }
        }
    };

    using DescPtr   = std::unique_ptr<std::remove_pointer_t<cublasLtMatmulDesc_t>, MatmulDescDeleter>;
    using LayoutPtr = std::unique_ptr<std::remove_pointer_t<cublasLtMatrixLayout_t>, MatrixLayoutDeleter>;

    // Per shape and dtype plan: the desc, layouts, and chosen algo.
    struct GemmPlan {
        DescPtr              desc;
        LayoutPtr            layout_w;
        LayoutPtr            layout_x;
        LayoutPtr            layout_y;
        cublasLtMatmulAlgo_t algo;
    };

    // (m, n, k) plus the full dtype signature (compute, scale, ab, out) as ints.
    // The dtype fields keep distinct quant paths from sharing a plan.
    using GemmKey = std::tuple<int, int, int, int, int, int, int>;

    // Miss builds the desc and layouts and runs the heuristic once, then inserts.
    // Hit returns the stored plan.
    [[nodiscard]] const GemmPlan& get_or_build_plan(int                 m,
                                                    int                 n,
                                                    int                 k,
                                                    cublasComputeType_t compute,
                                                    cudaDataType_t      scale_type,
                                                    cudaDataType_t      ab_type,
                                                    cudaDataType_t      out_type);

    // cuBLASLt workspace cap, shared by every GEMM dtype. 32 MiB follows NVIDIA's
    // cuBLASLt workspace recommendation.
    static constexpr std::size_t kWorkspaceBytes = std::size_t{32} << 20;

    LtHandle                            handle_;
    device::DeviceUniquePtr<std::byte>  workspace_;
    std::map<GemmKey, GemmPlan>         plan_cache_;
};

}  // namespace runtherder::kernels
