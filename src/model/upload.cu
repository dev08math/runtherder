#include <runtherder/model/upload.cuh>

#include <cstddef>
#include <span>

#include <runtherder/device/memory.cuh>
#include <runtherder/device/check.cuh>

namespace runtherder::model {

Tensor upload_tensor(const ShardedSafetensors& sf,
                     std::string_view name,
                     cudaStream_t stream) {
    const std::span<const std::byte> host_bytes = sf.bytes(name);

    Tensor tensor;
    tensor.shape = sf.shape(name);
    tensor.dtype = sf.dtype(name);
    tensor.data  = device::make_device_unique<std::byte>(host_bytes.size());

    RUNTHERDER_CUDA_CHECK(cudaMemcpyAsync(tensor.data.get(), host_bytes.data(),
                                          host_bytes.size(), cudaMemcpyHostToDevice,
                                          stream));
    return tensor;
}

}  // namespace runtherder::model
