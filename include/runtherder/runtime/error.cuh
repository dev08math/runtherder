#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace runtherder::runtime {

inline void check_cuda(cudaError_t err, const char* expr,
                       const char* file, int line) {
    if (err != cudaSuccess) [[unlikely]] {
        std::fprintf(stderr, "CUDA error at %s:%d -- %s\n  %s\n",
                     file, line, cudaGetErrorString(err), expr);
        std::exit(EXIT_FAILURE);
    }
}

[[noreturn]] inline void fail_check(const char* expr, const char* msg,
                                    const char* file, int line) {
    std::fprintf(stderr,
                 "runtherder check failed at %s:%d\n  condition: %s\n  reason:    %s\n",
                 file, line, expr, msg);
    std::exit(EXIT_FAILURE);
}

}  // namespace runtherder::runtime

#define RUNTHERDER_CUDA_CHECK(expr) \
    ::runtherder::runtime::check_cuda((expr), #expr, __FILE__, __LINE__)

#define RUNTHERDER_CUDA_CHECK_LAST() \
    RUNTHERDER_CUDA_CHECK(cudaGetLastError())

#define RUNTHERDER_CHECK(cond, msg)                                            \
    do {                                                                       \
        if (!(cond)) [[unlikely]] {                                            \
            ::runtherder::runtime::fail_check(#cond, (msg), __FILE__, __LINE__);\
        }                                                                      \
    } while (0)
