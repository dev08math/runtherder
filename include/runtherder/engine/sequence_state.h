#pragma once

#include <runtherder/check.h>
#include <runtherder/sampling/sampler.h>

namespace runtherder::engine {

struct StopCondition {
    int eos_token;
    int max_new_tokens;
};

/**
 * @brief One in flight generation: decode position with per sequence sampling
 *        and stop criteria.
 * @note Construct only via create(), which exits via RUNTHERDER_CHECK on
 *       prompt_len < 1 or max_new_tokens < 1.
 */
class SequenceState {
public:
    [[nodiscard]] static SequenceState create(int seq_id, int prompt_len,
                                               sampling::SamplingParams sampling,
                                               StopCondition stop) {
        RUNTHERDER_CHECK(prompt_len >= 1, "SequenceState requires a non-empty prompt");
        RUNTHERDER_CHECK(stop.max_new_tokens >= 1, "SequenceState requires max_new_tokens >= 1");
        return SequenceState(seq_id, prompt_len, sampling, stop);
    }

    void advance() noexcept { ++position_; }

    [[nodiscard]] int seq_id() const noexcept { return seq_id_; }
    [[nodiscard]] int position() const noexcept { return position_; }
    [[nodiscard]] int prompt_len() const noexcept { return prompt_len_; }

    [[nodiscard]] const sampling::SamplingParams& sampling() const noexcept { return sampling_; }
    [[nodiscard]] const StopCondition& stop() const noexcept { return stop_; }

    [[nodiscard]] bool reached_max() const noexcept { return position_ >= stop_.max_new_tokens; }
    [[nodiscard]] bool is_eos(int token) const noexcept { return token == stop_.eos_token; }

private:
    SequenceState(int seq_id, int prompt_len, sampling::SamplingParams sampling,
                  StopCondition stop) noexcept
        : seq_id_(seq_id),
          position_(0),
          prompt_len_(prompt_len),
          sampling_(sampling),
          stop_(stop) {}

    int                      seq_id_;
    int                      position_;
    int                      prompt_len_;
    sampling::SamplingParams sampling_;
    StopCondition            stop_;
};

}  // namespace runtherder::engine
