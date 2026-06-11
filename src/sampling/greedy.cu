#include <runtherder/sampling/greedy.h>

#include <cstddef>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/check.h>
#include <runtherder/device/check.cuh>

namespace runtherder::sampling {

int greedy_argmax(const __nv_bfloat16* logits, int vocab) {
    RUNTHERDER_CHECK(vocab >= 1, "greedy_argmax needs vocab >= 1");

    std::vector<__nv_bfloat16> host(static_cast<std::size_t>(vocab));
    RUNTHERDER_CUDA_CHECK(cudaMemcpy(host.data(), logits,
                                     static_cast<std::size_t>(vocab) * sizeof(__nv_bfloat16),
                                     cudaMemcpyDeviceToHost));

    int   best     = 0;
    float best_val = static_cast<float>(host[0]);
    for (int i = 1; i < vocab; ++i) {
        const float val = static_cast<float>(host[i]);
        if (val > best_val) {
            best     = i;
            best_val = val;
        }
    }
    return best;
}

}  // namespace runtherder::sampling
