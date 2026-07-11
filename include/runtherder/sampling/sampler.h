#pragma once

#include <cstddef>
#include <cstdint>
#include <random>

#include <cuda_bf16.h>

#include <runtherder/device/memory.cuh>

namespace runtherder::sampling {

// Per request knobs. temperature == 0 selects greedy (argmax), ignoring top_p
// and top_k. top_p == 1 disables the nucleus cut, top_k == 0 disables the cap.
struct SamplingParams {
    float temperature = 1.0F;
    float top_p       = 1.0F;
    int   top_k       = 0;
};

class Sampler {
public:
    explicit Sampler(int vocab_size, std::uint64_t seed);

    Sampler(Sampler&&) noexcept            = default;
    Sampler& operator=(Sampler&&) noexcept = default;
    Sampler(const Sampler&)                = delete;
    Sampler& operator=(const Sampler&)     = delete;

    // Advances rng_ only on a stochastic draw, not on greedy (temperature == 0).
    [[nodiscard]] int sample(const __nv_bfloat16* logits, const SamplingParams& params);

private:
    int             vocab_size_;
    std::mt19937_64 rng_;
    // Persistent device pipeline scratch, carved for vocab_size_ once in the ctor.
    device::DeviceUniquePtr<std::byte> scratch_;
    std::size_t                        cub_temp_bytes_ = 0;
};

}  // namespace runtherder::sampling
