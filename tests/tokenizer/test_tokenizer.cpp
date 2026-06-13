#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <runtherder/tokenizer/tokenizer.h>

namespace {

using runtherder::tokenizer::Tokenizer;

[[nodiscard]] std::filesystem::path model_dir() {
    return std::filesystem::path{RUNTHERDER_MODEL_DIR};
}

class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(model_dir() / "tokenizer.json")) {
            GTEST_SKIP() << "tokenizer.json absent under " << model_dir();
        }
    }
};

TEST_F(TokenizerTest, RoundTripsHelloWorld) {
    Tokenizer tok = Tokenizer::load(model_dir());

    const std::string      text = "Hello world";
    const std::vector<int> ids  = tok.encode(text);

    EXPECT_FALSE(ids.empty());

    std::cout << "encode(\"" << text << "\") =";
    for (const int id : ids) {
        std::cout << ' ' << id;
    }
    std::cout << '\n';

    EXPECT_EQ(tok.decode(ids), text);
}

}  // namespace
