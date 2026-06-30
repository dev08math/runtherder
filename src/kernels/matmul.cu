#include <runtherder/kernels/matmul.cuh>

#include <cstddef>
#include <cstdint>

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

void Matmul::linear_bf16(__nv_bfloat16*       y,
                         const __nv_bfloat16* x,
                         const __nv_bfloat16* weight,
                         int                  m,
                         int                  n,
                         int                  k,
                         cudaStream_t         stream) {
    RUNTHERDER_CHECK(m >= 1 && n >= 1 && k >= 1, "matmul dims must be positive");

    // cuBLASLt is column major. Each row major buffer reads as its transpose,
    // and the call targets yᵀ = W · xᵀ. Mapped dims M = n, N = m, K = k.
    const std::int64_t lt_m = n;
    const std::int64_t lt_n = m;
    const std::int64_t lt_k = k;

    cublasLtMatmulDesc_t desc = nullptr;
    RUNTHERDER_CUBLAS_CHECK(
        cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    const cublasOperation_t transa = CUBLAS_OP_T;  // weight, stored [n,k]
    const cublasOperation_t transb = CUBLAS_OP_N;  // x, stored [m,k]
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Layouts describe the physical column major storage of each row major
    // buffer. weight [n,k] reads as column major k×n, x [m,k] as k×m, y [m,n]
    // as n×m. Leading dim is the row major width of each buffer.
    cublasLtMatrixLayout_t layout_w = nullptr;
    cublasLtMatrixLayout_t layout_x = nullptr;
    cublasLtMatrixLayout_t layout_y = nullptr;
    RUNTHERDER_CUBLAS_CHECK(
        cublasLtMatrixLayoutCreate(&layout_w, CUDA_R_16BF, lt_k, lt_m, lt_k));
    RUNTHERDER_CUBLAS_CHECK(
        cublasLtMatrixLayoutCreate(&layout_x, CUDA_R_16BF, lt_k, lt_n, lt_k));
    RUNTHERDER_CUBLAS_CHECK(
        cublasLtMatrixLayoutCreate(&layout_y, CUDA_R_16BF, lt_m, lt_n, lt_m));

    cublasLtMatmulPreference_t pref = nullptr;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    const std::size_t ws_bytes = kWorkspaceBytes;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle_.get(), desc, layout_w, layout_x, layout_y, layout_y,
        pref, 1, &heuristic, &returned));
    RUNTHERDER_CHECK(returned > 0, "no cuBLASLt algorithm for this matmul shape");

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmul(
        handle_.get(), desc,
        &alpha,
        weight, layout_w,
        x,      layout_x,
        &beta,
        y,      layout_y,
        y,      layout_y,
        &heuristic.algo,
        workspace_.get(), ws_bytes,
        stream));

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layout_y);
    cublasLtMatrixLayoutDestroy(layout_x);
    cublasLtMatrixLayoutDestroy(layout_w);
    cublasLtMatmulDescDestroy(desc);
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

    const std::int64_t lt_m = n;
    const std::int64_t lt_n = m;
    const std::int64_t lt_k = k;

    cublasLtMatmulDesc_t desc = nullptr;
    RUNTHERDER_CUBLAS_CHECK(
        cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));

    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Plain COL layout. The COL32 reorder is required only for int8 output, the
    // int32 output path takes plain column major.
    cublasLtMatrixLayout_t layout_w = nullptr;
    cublasLtMatrixLayout_t layout_x = nullptr;
    cublasLtMatrixLayout_t layout_y = nullptr;
    RUNTHERDER_CUBLAS_CHECK(
        cublasLtMatrixLayoutCreate(&layout_w, CUDA_R_8I, lt_k, lt_m, lt_k));
    RUNTHERDER_CUBLAS_CHECK(
        cublasLtMatrixLayoutCreate(&layout_x, CUDA_R_8I, lt_k, lt_n, lt_k));
    RUNTHERDER_CUBLAS_CHECK(
        cublasLtMatrixLayoutCreate(&layout_y, CUDA_R_32I, lt_m, lt_n, lt_m));

    cublasLtMatmulPreference_t pref = nullptr;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    const std::size_t ws_bytes = kWorkspaceBytes;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle_.get(), desc, layout_w, layout_x, layout_y, layout_y,
        pref, 1, &heuristic, &returned));
    RUNTHERDER_CHECK(returned > 0, "no cuBLASLt algorithm for this int8 matmul shape");

    const std::int32_t alpha = 1;
    const std::int32_t beta  = 0;
    RUNTHERDER_CUBLAS_CHECK(cublasLtMatmul(
        handle_.get(), desc,
        &alpha,
        weight,     layout_w,
        scratch.q,  layout_x,
        &beta,
        scratch.acc, layout_y,
        scratch.acc, layout_y,
        &heuristic.algo,
        workspace_.get(), ws_bytes,
        stream));

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layout_y);
    cublasLtMatrixLayoutDestroy(layout_x);
    cublasLtMatrixLayoutDestroy(layout_w);
    cublasLtMatmulDescDestroy(desc);

    dequantize_w8a8(y, scratch.acc, scratch.x_scale, w_scale, m, n, stream);
}

}  // namespace runtherder::kernels
