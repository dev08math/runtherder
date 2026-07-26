#pragma once

#include <cstddef>
#include <string>

#include <cuda_runtime.h>

#include <runtherder/device/check.cuh>

namespace runtherder::device {

struct DeviceProperties {
    std::string name;
    int         compute_major;
    int         compute_minor;
    std::size_t vram_bytes;
    int         memory_clock_khz;
    int         memory_bus_bits;
    double      peak_bandwidth_gbs;
};

/**
 * @brief Reads device capability straight off the driver. Nothing here is
 *        compiled in.
 * @param device  CUDA device ordinal
 */
[[nodiscard]] inline DeviceProperties query_device_properties(int device) {
    cudaDeviceProp prop{};
    RUNTHERDER_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int clock_khz = 0;
    int bus_bits  = 0;
    RUNTHERDER_CUDA_CHECK(
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, device));
    RUNTHERDER_CUDA_CHECK(
        cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, device));

    DeviceProperties out;
    out.name             = prop.name;
    out.compute_major    = prop.major;
    out.compute_minor    = prop.minor;
    out.vram_bytes       = prop.totalGlobalMem;
    out.memory_clock_khz = clock_khz;
    out.memory_bus_bits  = bus_bits;
    out.peak_bandwidth_gbs =
        2.0 * static_cast<double>(clock_khz) * 1.0e3 * static_cast<double>(bus_bits) / 8.0 / 1.0e9;
    return out;
}

}  // namespace runtherder::device
