#pragma once

#include <cstddef>

namespace runtherder::model {

struct Config {
    std::size_t hidden_dim;
    std::size_t num_layers;
    std::size_t num_heads;
    std::size_t num_kv_heads;
    std::size_t head_dim;
    std::size_t intermediate_dim;
    std::size_t vocab_size;
    std::size_t max_seq_len;

    float rms_norm_eps;
    float rope_theta;

    bool tie_word_embeddings;
};

}  // namespace runtherder::model
