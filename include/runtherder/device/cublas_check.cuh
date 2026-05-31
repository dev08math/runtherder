#pragma once

#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>

#include <runtherder/check.h>

namespace runtherder::device {

inline void check_cublas(cublasStatus_t status, const char* expr,
                         const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) [[unlikely]] {
        std::fprintf(stderr, "cuBLAS error at %s:%d -- %s\n  %s\n",
                     file, line, cublasGetStatusString(status), expr);
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace runtherder::device

#define RUNTHERDER_CUBLAS_CHECK(expr) \
    ::runtherder::device::check_cublas((expr), #expr, __FILE__, __LINE__)
