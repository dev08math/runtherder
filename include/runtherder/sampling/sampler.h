#pragma once

#include <cstdint>
#include <random>

#include <cuda_bf16.h>

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

    // Advances rng_ only on a stochastic draw, not on greedy (temperature == 0).
    [[nodiscard]] int sample(const __nv_bfloat16* logits, const SamplingParams& params);

private:
    int             vocab_size_;
    std::mt19937_64 rng_;
};

}  // namespace runtherder::sampling
