#pragma once

#include <vector>

namespace runtherder::engine {

/**
* @brief Streaming sink. on_token fires per emitted token, flush once at the end.
*/
class OutputSink {
public:
    virtual ~OutputSink() = default;

    virtual void on_token(int seq_id, int token_id) = 0;
    virtual void flush()                            = 0;
};

class VectorSink final : public OutputSink {
public:
    void on_token([[maybe_unused]] int seq_id, int token_id) override { tokens_.push_back(token_id); }
    void flush() override {}

    [[nodiscard]] const std::vector<int>& tokens() const noexcept { return tokens_; }

private:
    std::vector<int> tokens_;
};

}  // namespace runtherder::engine
