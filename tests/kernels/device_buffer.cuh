#pragma once

#include <cstddef>

#include <runtherder/device/check.cuh>
#include <runtherder/device/memory.cuh>

namespace runtherder::device {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;

    explicit DeviceBuffer(std::size_t count)
        : ptr_(make_device_unique<T>(count)), count_(count) {}

    DeviceBuffer(DeviceBuffer&&) noexcept = default;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept = default;

    [[nodiscard]] T* data() noexcept { return ptr_.get(); }
    [[nodiscard]] const T* data() const noexcept { return ptr_.get(); }

    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }
    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

    void zero() {
        if (count_ > 0) {
            RUNTHERDER_CUDA_CHECK(cudaMemset(ptr_.get(), 0, bytes()));
        }
    }

    void copy_from_host(const T* host_src) {
        RUNTHERDER_CUDA_CHECK(
            cudaMemcpy(ptr_.get(), host_src, bytes(), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* host_dst) const {
        RUNTHERDER_CUDA_CHECK(
            cudaMemcpy(host_dst, ptr_.get(), bytes(), cudaMemcpyDeviceToHost));
    }

private:
    DeviceUniquePtr<T> ptr_{};
    std::size_t count_ = 0;
};

}  // namespace runtherder::device
