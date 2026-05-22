#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <runtherder/kernels/rope.cuh>
#include <runtherder/runtime/device_buffer.cuh>
#include <runtherder/runtime/error.cuh>

namespace {

using runtherder::runtime::DeviceBuffer;

std::vector<__nv_bfloat16> make_bf16_random(std::size_t count,
                                            float        scale,
                                            std::uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<__nv_bfloat16>            out(count);
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = __float2bfloat16(dist(rng));
    }
    return out;
}

std::vector<float> to_fp32(const std::vector<__nv_bfloat16>& src) {
    std::vector<float> out(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[i] = __bfloat162float(src[i]);
    }
    return out;
}

// CPU reference: BF16 storage, FP64 trig and rotation arithmetic. Mirrors
// the kernel's half-split layout but with tighter numerics than the device
// path (which uses __powf / __sincosf in FP32). The kernel must still match
// within the BF16-storage tolerance.
void rope_cpu_reference(std::vector<__nv_bfloat16>& tensor,
                        int                          start_pos,
                        int                          seq_len,
                        int                          num_heads,
                        int                          head_dim,
                        float                        theta_base) {
    const int half = head_dim / 2;
    for (int t = 0; t < seq_len; ++t) {
        const int pos = start_pos + t;
        for (int h = 0; h < num_heads; ++h) {
            const std::size_t row_off =
                (static_cast<std::size_t>(t) * num_heads + h) * head_dim;
            for (int i = 0; i < half; ++i) {
                const double exponent =
                    -2.0 * static_cast<double>(i) /
                    static_cast<double>(head_dim);
                const double freq  = std::pow(static_cast<double>(theta_base),
                                              exponent);
                const double theta = static_cast<double>(pos) * freq;
                const double c     = std::cos(theta);
                const double s     = std::sin(theta);

                const double a =
                    __bfloat162float(tensor[row_off + i]);
                const double b =
                    __bfloat162float(tensor[row_off + i + half]);

                tensor[row_off + i] =
                    __float2bfloat16(static_cast<float>(a * c - b * s));
                tensor[row_off + i + half] =
                    __float2bfloat16(static_cast<float>(a * s + b * c));
            }
        }
    }
}

struct CompareResult {
    float max_abs_err;
    float cosine_similarity;
};

CompareResult compare_bf16(const std::vector<__nv_bfloat16>& a,
                           const std::vector<__nv_bfloat16>& b) {
    const std::vector<float> af = to_fp32(a);
    const std::vector<float> bf = to_fp32(b);

    float  max_abs = 0.0f;
    double dot     = 0.0;
    double norm_a  = 0.0;
    double norm_b  = 0.0;
    for (std::size_t i = 0; i < af.size(); ++i) {
        const float diff = std::fabs(af[i] - bf[i]);
        if (diff > max_abs) {
            max_abs = diff;
        }
        dot    += static_cast<double>(af[i]) * static_cast<double>(bf[i]);
        norm_a += static_cast<double>(af[i]) * static_cast<double>(af[i]);
        norm_b += static_cast<double>(bf[i]) * static_cast<double>(bf[i]);
    }
    const double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    const float  cos   = (denom > 0.0)
                             ? static_cast<float>(dot / denom)
                             : 0.0f;
    return {max_abs, cos};
}

constexpr float kAbsTol   = 5e-2f;
constexpr float kCosFloor = 0.9999f;

void run_case(int   seq_len,
              int   num_q_heads,
              int   num_kv_heads,
              int   head_dim,
              float theta_base,
              int   start_pos) {
    const std::size_t q_count =
        static_cast<std::size_t>(seq_len) * num_q_heads  * head_dim;
    const std::size_t k_count =
        static_cast<std::size_t>(seq_len) * num_kv_heads * head_dim;

    auto q_host = make_bf16_random(q_count, 1.0f, 0xC0FFEEu);
    auto k_host = make_bf16_random(k_count, 1.0f, 0xBADBEEFu);

    auto q_expected = q_host;
    auto k_expected = k_host;
    rope_cpu_reference(q_expected, start_pos, seq_len, num_q_heads,
                       head_dim, theta_base);
    rope_cpu_reference(k_expected, start_pos, seq_len, num_kv_heads,
                       head_dim, theta_base);

    DeviceBuffer<__nv_bfloat16> q_dev(q_count);
    DeviceBuffer<__nv_bfloat16> k_dev(k_count);
    q_dev.copy_from_host(q_host.data());
    k_dev.copy_from_host(k_host.data());

    runtherder::kernels::rope_bf16_inplace(
        q_dev.data(), k_dev.data(),
        start_pos, seq_len,
        num_q_heads, num_kv_heads,
        head_dim, theta_base,
        nullptr);
    RUNTHERDER_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> q_out(q_count);
    std::vector<__nv_bfloat16> k_out(k_count);
    q_dev.copy_to_host(q_out.data());
    k_dev.copy_to_host(k_out.data());

    const auto q_cmp = compare_bf16(q_expected, q_out);
    EXPECT_LE(q_cmp.max_abs_err, kAbsTol)
        << "q seq_len=" << seq_len << " head_dim=" << head_dim
        << " start_pos=" << start_pos;
    EXPECT_GE(q_cmp.cosine_similarity, kCosFloor)
        << "q seq_len=" << seq_len << " head_dim=" << head_dim
        << " start_pos=" << start_pos;

    const auto k_cmp = compare_bf16(k_expected, k_out);
    EXPECT_LE(k_cmp.max_abs_err, kAbsTol)
        << "k seq_len=" << seq_len << " head_dim=" << head_dim
        << " start_pos=" << start_pos;
    EXPECT_GE(k_cmp.cosine_similarity, kCosFloor)
        << "k seq_len=" << seq_len << " head_dim=" << head_dim
        << " start_pos=" << start_pos;
}

}  // namespace

TEST(RoPEBF16, MatchesReferenceSmall) {
    run_case(/*seq_len=*/1, /*num_q=*/4, /*num_kv=*/1,
             /*head_dim=*/64, /*theta_base=*/10000.0f, /*start_pos=*/0);
}

TEST(RoPEBF16, MatchesReferenceQwen3Prefill) {
    run_case(8, 32, 8, 128, 1.0e6f, 0);
}

TEST(RoPEBF16, MatchesReferenceQwen3DecodeOffset) {
    run_case(1, 32, 8, 128, 1.0e6f, 1024);
}

TEST(RoPEBF16, MatchesReferenceMaxHeadDim) {
    run_case(2, 8, 2, 256, 1.0e6f, 0);
}

TEST(RoPEBF16, MatchesReferenceBatched) {
    run_case(32, 32, 8, 128, 1.0e6f, 0);
}
