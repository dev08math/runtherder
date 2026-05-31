#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>

#include <runtherder/device/check.cuh>

namespace runtherder::device {

struct CudaFreeDeleter {
    template <typename T>
    void operator()(T* ptr) const noexcept {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

template <typename T>
using DeviceUniquePtr = std::unique_ptr<T[], CudaFreeDeleter>;

/**
 * @brief Owning device allocation of count elements of T, freed via cudaFree.
 * @note count == 0 yields an empty pointer with no allocation.
 */
template <typename T>
[[nodiscard]] inline DeviceUniquePtr<T> make_device_unique(std::size_t count) {
    if (count == 0) {
        return DeviceUniquePtr<T>{};
    }
    T* raw = nullptr;
    RUNTHERDER_CUDA_CHECK(cudaMalloc(&raw, count * sizeof(T)));
    return DeviceUniquePtr<T>{raw};
}

}  // namespace runtherder::device
