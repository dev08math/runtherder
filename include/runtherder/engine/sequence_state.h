#pragma once

#include <algorithm>
#include <vector>

#include <runtherder/check.h>
#include <runtherder/sampling/sampler.h>

namespace runtherder::engine {

struct StopCondition {
    std::vector<int> eos_tokens;
    int              max_new_tokens;
};

/**
 * @brief Per request state for one in flight generation: decode position,
 *        sampling params, and stop criteria.
 */
class SequenceState {
public:
    /**
     * @param seq_id      id for this sequence, passed to OutputSink::on_token
     * @param prompt_len  number of prompt tokens, >= 1
     * @param sampling    how the next token is chosen: temperature, top_p, top_k
     * @param stop        when to stop: eos token set and max new token cap (>= 1)
     */
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
    [[nodiscard]] bool is_eos(int token) const noexcept {
        return std::find(stop_.eos_tokens.begin(), stop_.eos_tokens.end(), token)
               != stop_.eos_tokens.end();
    }

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
