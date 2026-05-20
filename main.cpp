#include <cstdio>

#include <runtherder/model/weights.h>

int main() {
    runtherder::model::Weights weights{};

    std::printf("Runtherder: bootstrap ok\n");
    std::printf("  layers loaded:    %zu\n", weights.layers.size());
    std::printf("  default dtype:    %u (BF16)\n",
                static_cast<unsigned>(runtherder::model::DType::BF16));
    std::printf("  embedding rank:   %zu\n", weights.token_embedding.shape.size());

    return 0;
}
