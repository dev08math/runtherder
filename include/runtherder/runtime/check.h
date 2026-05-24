#pragma once

#include <cstdio>
#include <cstdlib>

namespace runtherder::runtime {

[[noreturn]] inline void fail_check(const char* expr, const char* msg,
                                    const char* file, int line) {
    std::fprintf(stderr,
                 "runtherder check failed at %s:%d\n  condition: %s\n  reason:    %s\n",
                 file, line, expr, msg);
    std::exit(EXIT_FAILURE);
}

}  // namespace runtherder::runtime

#define RUNTHERDER_CHECK(cond, msg)                                            \
    do {                                                                       \
        if (!(cond)) [[unlikely]] {                                            \
            ::runtherder::runtime::fail_check(#cond, (msg), __FILE__, __LINE__);\
        }                                                                      \
    } while (0)
