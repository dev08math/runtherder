#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include <runtherder/runtime/check.h>

namespace runtherder::runtime {

inline void check_cuda(cudaError_t err, const char* expr,
                       const char* file, int line) {
    if (err != cudaSuccess) [[unlikely]] {
        std::fprintf(stderr, "CUDA error at %s:%d -- %s\n  %s\n",
                     file, line, cudaGetErrorString(err), expr);
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace runtherder::runtime

#define RUNTHERDER_CUDA_CHECK(expr) \
    ::runtherder::runtime::check_cuda((expr), #expr, __FILE__, __LINE__)

#define RUNTHERDER_CUDA_CHECK_LAST() \
    RUNTHERDER_CUDA_CHECK(cudaGetLastError())
