#include <runtherder/model/context.h>

#include <cstddef>
#include <span>

#include <cuda_runtime.h>

#include <runtherder/check.h>
#include <runtherder/device/check.cuh>
#include <runtherder/device/memory.cuh>

namespace runtherder::model {

ModelContext::ModelContext(std::size_t scratch_bytes, std::size_t max_batch_tokens)
    : scratch_(scratch_bytes),
      token_ids_(device::make_device_unique<int>(max_batch_tokens)),
      max_batch_tokens_(max_batch_tokens) {
    RUNTHERDER_CHECK(max_batch_tokens_ >= 1, "max_batch_tokens must be >= 1");
}

const int* ModelContext::upload_token_ids(std::span<const int> token_ids) {
    RUNTHERDER_CHECK(token_ids.size() <= max_batch_tokens_,
                     "token count exceeds context capacity");
    RUNTHERDER_CUDA_CHECK(cudaMemcpy(token_ids_.get(), token_ids.data(),
                                     token_ids.size() * sizeof(int),
                                     cudaMemcpyHostToDevice));
    return token_ids_.get();
}

}  // namespace runtherder::model
