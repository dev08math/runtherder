#include <runtherder/sampling/sampler.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <runtherder/check.h>
#include <runtherder/device/check.cuh>
#include <runtherder/sampling/greedy.h>

namespace runtherder::sampling {

Sampler::Sampler(int vocab_size, std::uint64_t seed)
    : vocab_size_(vocab_size), rng_(seed) {
    RUNTHERDER_CHECK(vocab_size >= 1, "Sampler needs vocab_size >= 1");
}

int Sampler::sample(const __nv_bfloat16* logits, const SamplingParams& params) {
    if (params.temperature == 0.0F) {
        return greedy_argmax(logits, vocab_size_);
    }

    const std::size_t           n = static_cast<std::size_t>(vocab_size_);
    std::vector<__nv_bfloat16>  host(n);
    RUNTHERDER_CUDA_CHECK(cudaMemcpy(host.data(), logits,
                                     n * sizeof(__nv_bfloat16),
                                     cudaMemcpyDeviceToHost));

    std::vector<float> probs(n);
    const float        inv_t = 1.0F / params.temperature;
    float              max_logit = -INFINITY;
    for (std::size_t i = 0; i < n; ++i) {
        probs[i]  = static_cast<float>(host[i]) * inv_t;
        max_logit = std::max(max_logit, probs[i]);
    }

    float sum = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        probs[i] = std::exp(probs[i] - max_logit);
        sum += probs[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        probs[i] /= sum;
    }

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return probs[a] > probs[b]; });

    std::size_t keep = n;
    if (params.top_k > 0) {
        keep = std::min(keep, static_cast<std::size_t>(params.top_k));
    }
    if (params.top_p < 1.0F) {
        float       cum     = 0.0F;
        std::size_t nucleus = 0;
        for (; nucleus < keep; ++nucleus) {
            cum += probs[order[nucleus]];
            if (cum >= params.top_p) {
                ++nucleus;
                break;
            }
        }
        keep = nucleus;
    }

    float kept_sum = 0.0F;
    for (std::size_t r = 0; r < keep; ++r) {
        kept_sum += probs[order[r]];
    }

    // Draw over the unnormalized survivors. Scaling u by kept_sum is the
    // renormalization, no second pass over probs.
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    const float                           u = dist(rng_) * kept_sum;
    float                                 cum = 0.0F;
    for (std::size_t r = 0; r < keep; ++r) {
        cum += probs[order[r]];
        if (u < cum) {
            return order[r];
        }
    }
    return order[keep - 1];
}

}  // namespace runtherder::sampling
