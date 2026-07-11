#include <runtherder/kernels/matmul.cuh>

#include <cstddef>
#include <cstdint>
#include <utility>

#include <runtherder/check.h>
#include <runtherder/device/cublas_check.cuh>
#include <runtherder/kernels/quantization.cuh>

namespace runtherder::kernels {

// cuBLASLt backs both linear_bf16 and linear_w8a8.
Matmul::Matmul() {
    cublasLtHandle_t raw = nullptr;
    RUNTHERDER_CUBLAS_CHECK(cublasLtCreate(&raw));
    handle_.reset(raw);
    workspace_ = device::make_device_unique<std::byte>(kWorkspaceBytes);
}

const Matmul::GemmPlan& Matmul::get_or_build_plan(int                 m,
                                                  int                 n,
                                                  int                 k,
                                                  cublasComputeType_t compute,
                                                  cudaDataType_t      scale_type,
                                                  cudaDataType_t      ab_type,
                                                  cudaDataType_t      out_type) {
    const GemmKey key{m, n, k,
                      static_cast<int>(compute), static_cast<int>(scale_type),
                      static_cast<int>(ab_type), static_cast<int>(out_type)};
    if (const auto it = plan_cache_.find(key); it != plan_cache_.end()) {
        return it->second;
    }

    // cuBLASLt is column major. Each row major buffer reads as its transpose,
    // and the call targets yᵀ = W · xᵀ. Mapped dims M = n, N = m, K = k.
    const std::int64_t lt_m = n;
    const std::int64_t lt_n = m;
    const std::int64_t lt_k = k;

    cublasLtMatmulDesc_t raw_desc = nullptr;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulDescCreate(&raw_desc, compute, scale_type));
    DescPtr desc(raw_desc);

    const cublasOperation_t transa = CUBLAS_OP_T;  // weight, stored [n,k]
    const cublasOperation_t transb = CUBLAS_OP_N;  // x, stored [m,k]
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc.get(), CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc.get(), CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Layouts describe the physical column major storage of each row major
    // buffer. weight [n,k] reads as column major k×n, x [m,k] as k×m, y [m,n]
    // as n×m. Leading dim is the row major width of each buffer. Plain COL for
    // the output. The COL32 reorder trap applies only to int8 OUTPUT, not our
    // int32 or bf16 output.
    cublasLtMatrixLayout_t raw_w = nullptr;
    cublasLtMatrixLayout_t raw_x = nullptr;
    cublasLtMatrixLayout_t raw_y = nullptr;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&raw_w, ab_type, lt_k, lt_m, lt_k));
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&raw_x, ab_type, lt_k, lt_n, lt_k));
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&raw_y, out_type, lt_m, lt_n, lt_m));
    LayoutPtr layout_w(raw_w);
    LayoutPtr layout_x(raw_x);
    LayoutPtr layout_y(raw_y);

    cublasLtMatmulPreference_t pref = nullptr;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    const std::size_t ws_bytes = kWorkspaceBytes;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle_.get(), desc.get(), layout_w.get(), layout_x.get(),
        layout_y.get(), layout_y.get(), pref, 1, &heuristic, &returned));
    cublasLtMatmulPreferenceDestroy(pref);
    RUNTHERDER_CHECK(returned > 0, "no cuBLASLt algorithm for this matmul shape");

    GemmPlan plan{
        std::move(desc),
        std::move(layout_w),
        std::move(layout_x),
        std::move(layout_y),
        heuristic.algo,
    };
    return plan_cache_.emplace(key, std::move(plan)).first->second;
}

void Matmul::linear_bf16(__nv_bfloat16*       y,
                         const __nv_bfloat16* x,
                         const __nv_bfloat16* weight,
                         int                  m,
                         int                  n,
                         int                  k,
                         cudaStream_t         stream) {
    RUNTHERDER_CHECK(m >= 1 && n >= 1 && k >= 1, "matmul dims must be positive");

    const GemmPlan& plan = get_or_build_plan(
        m, n, k, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF);

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmul(
        handle_.get(), plan.desc.get(),
        &alpha,
        weight, plan.layout_w.get(),
        x,      plan.layout_x.get(),
        &beta,
        y,      plan.layout_y.get(),
        y,      plan.layout_y.get(),
        &plan.algo,
        workspace_.get(), kWorkspaceBytes,
        stream));
}

W8A8Buffer W8A8Buffer::from_arena(device::ScratchArena& arena, int m, int n, int k) {
    W8A8Buffer b;
    b.q       = arena.alloc<std::int8_t>(static_cast<std::size_t>(m) * k);
    b.x_scale = arena.alloc<float>(static_cast<std::size_t>(m));
    b.acc     = arena.alloc<std::int32_t>(static_cast<std::size_t>(m) * n);
    return b;
}

void Matmul::linear_w8a8(__nv_bfloat16*       y,
                         const __nv_bfloat16* x,
                         const std::int8_t*   weight,
                         const __nv_bfloat16* w_scale,
                         const W8A8Buffer&    scratch,
                         int                  m,
                         int                  n,
                         int                  k,
                         cudaStream_t         stream) {
    RUNTHERDER_CHECK(m >= 1 && n >= 1 && k >= 1, "matmul dims must be positive");
    RUNTHERDER_CHECK(k % 4 == 0, "int8 gemm requires k divisible by 4");

    quantize_per_token_int8(scratch.q, scratch.x_scale, x, m, k, stream);

    const GemmPlan& plan = get_or_build_plan(
        m, n, k, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_8I, CUDA_R_32I);

    const std::int32_t alpha = 1;
    const std::int32_t beta  = 0;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmul(
        handle_.get(), plan.desc.get(),
        &alpha,
        weight,      plan.layout_w.get(),
        scratch.q,   plan.layout_x.get(),
        &beta,
        scratch.acc, plan.layout_y.get(),
        scratch.acc, plan.layout_y.get(),
        &plan.algo,
        workspace_.get(), kWorkspaceBytes,
        stream));

    dequantize_w8a8(y, scratch.acc, scratch.x_scale, w_scale, m, n, stream);
}

}  // namespace runtherder::kernels
