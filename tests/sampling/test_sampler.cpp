#include <runtherder/sampling/sampler.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {

using runtherder::sampling::Sampler;
using runtherder::sampling::SamplingParams;

int host_argmax_bf16(const std::vector<float>& values) {
    int   best   = 0;
    float best_v = -INFINITY;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float v = static_cast<float>(__float2bfloat16(values[i]));
        if (v > best_v) {
            best_v = v;
            best   = static_cast<int>(i);
        }
    }
    return best;
}

__nv_bfloat16* upload(const std::vector<float>& values) {
    const std::size_t          n = values.size();
    std::vector<__nv_bfloat16> host(n);
    for (std::size_t i = 0; i < n; ++i) {
        host[i] = __float2bfloat16(values[i]);
    }
    __nv_bfloat16* device = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device), n * sizeof(__nv_bfloat16));
    cudaMemcpy(device, host.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    return device;
}

std::vector<double> expected_distribution(const std::vector<float>& logits,
                                          float temperature, int top_k, float top_p) {
    const std::size_t   n = logits.size();
    std::vector<double> p(n);
    double              maxs = -INFINITY;
    for (std::size_t i = 0; i < n; ++i) {
        const float bf = static_cast<float>(__float2bfloat16(logits[i]));
        p[i]           = static_cast<double>(bf) / temperature;
        maxs           = std::max(maxs, p[i]);
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = std::exp(p[i] - maxs);
        sum += p[i];
    }
    for (double& x : p) {
        x /= sum;
    }

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return p[a] > p[b]; });

    std::size_t keep = (top_k > 0 && static_cast<std::size_t>(top_k) < n)
                           ? static_cast<std::size_t>(top_k)
                           : n;
    if (top_p < 1.0F) {
        double      cum = 0.0;
        std::size_t m   = 0;
        for (; m < keep; ++m) {
            cum += p[order[m]];
            if (cum >= top_p) {
                ++m;
                break;
            }
        }
        keep = m;
    }
    if (keep < 1) {
        keep = 1;
    }

    double kept = 0.0;
    for (std::size_t r = 0; r < keep; ++r) {
        kept += p[order[r]];
    }
    std::vector<double> expected(n, 0.0);
    for (std::size_t r = 0; r < keep; ++r) {
        expected[order[r]] = p[order[r]] / kept;
    }
    return expected;
}

}  // namespace

TEST(Sampler, ZeroTemperatureIsGreedy) {
    const std::vector<float> logits = {0.1F, 0.2F, 9.0F, 0.3F};
    __nv_bfloat16*           device = upload(logits);

    Sampler        sampler(static_cast<int>(logits.size()), 1234);
    SamplingParams params;
    params.temperature = 0.0F;

    EXPECT_EQ(sampler.sample(device, params), 2);
    cudaFree(device);
}

TEST(Sampler, TopKOneCollapsesToArgmax) {
    const std::vector<float> logits = {1.0F, 5.0F, 2.0F, 0.5F};
    __nv_bfloat16*           device = upload(logits);

    Sampler        sampler(static_cast<int>(logits.size()), 1234);
    SamplingParams params;
    params.temperature = 1.0F;
    params.top_k       = 1;

    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(sampler.sample(device, params), 1);
    }
    cudaFree(device);
}

TEST(Sampler, DeviceGreedyMatchesHostOracleFuzz) {
    std::mt19937                          rng(2026);
    std::uniform_real_distribution<float> dist(-50.0F, 50.0F);
    const std::vector<int>                vocabs = {1, 2, 4, 127, 1000, 50000, 128256};

    for (const int vocab : vocabs) {
        for (int trial = 0; trial < 5; ++trial) {
            std::vector<float> values(static_cast<std::size_t>(vocab));
            for (float& x : values) {
                x = dist(rng);
            }
            __nv_bfloat16* device = upload(values);

            Sampler        sampler(vocab, 1234);
            SamplingParams params;
            params.temperature = 0.0F;
            const int dev  = sampler.sample(device, params);
            const int host = host_argmax_bf16(values);

            // Compare the winning value not the index: bf16 ties give several valid argmaxes.
            float maxv = -INFINITY;
            for (const float x : values) {
                maxv = std::max(maxv, static_cast<float>(__float2bfloat16(x)));
            }
            const float dev_v  = static_cast<float>(__float2bfloat16(values[dev]));
            const float host_v = static_cast<float>(__float2bfloat16(values[host]));
            EXPECT_EQ(dev_v, maxv) << "vocab=" << vocab << " trial=" << trial;
            EXPECT_EQ(dev_v, host_v) << "vocab=" << vocab << " trial=" << trial;
            cudaFree(device);
        }
    }
}

TEST(Sampler, SameSeedReproducesSequence) {
    const std::vector<float> logits = {1.0F, 1.1F, 0.9F, 1.05F, 0.95F};
    __nv_bfloat16*           device = upload(logits);

    SamplingParams params;

    Sampler a(static_cast<int>(logits.size()), 777);
    Sampler b(static_cast<int>(logits.size()), 777);
    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(a.sample(device, params), b.sample(device, params));
    }
    cudaFree(device);
}

TEST(Sampler, PeakedLogitAlwaysSamplesMode) {
    const std::vector<float> logits = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 20.0F, 0.0F, 0.0F};
    __nv_bfloat16*           device = upload(logits);

    Sampler        sampler(static_cast<int>(logits.size()), 4242);
    SamplingParams params;
    params.top_p = 0.95F;

    for (int i = 0; i < 200; ++i) {
        EXPECT_EQ(sampler.sample(device, params), 5);
    }
    cudaFree(device);
}

TEST(Sampler, MatchesAnalyticDistribution) {
    const std::vector<float> logits = {3.0F, 2.0F, 1.0F, 0.0F, -1.0F, -2.0F};
    const int                vocab  = static_cast<int>(logits.size());
    __nv_bfloat16*           device = upload(logits);

    struct Config {
        float temperature;
        int   top_k;
        float top_p;
    };
    const std::vector<Config> configs = {
        {1.0F, 0, 0.9F},
        {1.0F, 3, 1.0F},
        {0.7F, 0, 1.0F},
    };

    constexpr int    draws = 30000;
    constexpr double tol   = 0.01;

    for (const Config& c : configs) {
        SamplingParams params;
        params.temperature = c.temperature;
        params.top_k       = c.top_k;
        params.top_p       = c.top_p;

        Sampler          sampler(vocab, 20260712);
        std::vector<int> counts(static_cast<std::size_t>(vocab), 0);
        for (int i = 0; i < draws; ++i) {
            counts[static_cast<std::size_t>(sampler.sample(device, params))] += 1;
        }

        const std::vector<double> expected =
            expected_distribution(logits, c.temperature, c.top_k, c.top_p);
        for (int i = 0; i < vocab; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            if (expected[idx] == 0.0) {
                EXPECT_EQ(counts[idx], 0)
                    << "temp=" << c.temperature << " top_k=" << c.top_k
                    << " top_p=" << c.top_p << " index=" << i;
            } else {
                const double freq = static_cast<double>(counts[idx]) / draws;
                EXPECT_NEAR(freq, expected[idx], tol)
                    << "temp=" << c.temperature << " top_k=" << c.top_k
                    << " top_p=" << c.top_p << " index=" << i;
            }
        }
    }
    cudaFree(device);
}
