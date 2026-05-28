#pragma once

#include <cuda_runtime.h>

#include <string_view>

#include <runtherder/model/sharded_safetensors.h>
#include <runtherder/model/weights.h>

namespace runtherder::model {

/**
 * @brief Upload one tensor's raw bytes from the shard reader into device memory.
 * @pre name exists in sf. A missing name exits via RUNTHERDER_CHECK in the reader.
 * @note The host to device copy is queued on stream. The returned Tensor data is
 *       not valid until stream is synchronized.
 * @note Plain byte copy. shape and dtype carry through from the reader and quant
 *       stays empty.
 */
[[nodiscard]] Tensor upload_tensor(const ShardedSafetensors& sf,
                                   std::string_view name,
                                   cudaStream_t stream);

}  // namespace runtherder::model
