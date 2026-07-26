// Prints device capability as key=value lines for scripts to read. Nothing it
// reports is a compiled in constant.
//
//   runtherder_devinfo

#include <cstdio>

#include <runtherder/device/properties.cuh>

int main() {
    const runtherder::device::DeviceProperties p =
        runtherder::device::query_device_properties(0);

    std::printf("name=%s\n", p.name.c_str());
    std::printf("compute_cap=%d.%d\n", p.compute_major, p.compute_minor);
    std::printf("vram_mib=%zu\n", p.vram_bytes / (1024 * 1024));
    std::printf("memclk_khz=%d\n", p.memory_clock_khz);
    std::printf("bus_bits=%d\n", p.memory_bus_bits);
    std::printf("peak_gbs=%.1f\n", p.peak_bandwidth_gbs);
    return 0;
}
