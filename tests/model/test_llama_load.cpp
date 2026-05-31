#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <runtherder/device/memory.cuh>
#include <runtherder/model/config.h>
#include <runtherder/model/supported/llama/llama.h>
#include <runtherder/model/supported/llama/qwen3.h>
#include <runtherder/model/upload.cuh>
#include <runtherder/model/weights.h>

namespace {

using runtherder::model::ByName;
using runtherder::model::DType;
using runtherder::model::LlamaConfig;
using runtherder::model::LlamaWeights;
using runtherder::model::Tensor;
using runtherder::model::Uploaded;
namespace qwen3 = runtherder::model::qwen3;

// The Qwen 3 4B config.json verbatim. The untied variant flips one field.
constexpr const char* kQwen3ConfigJson = R"json({
  "architectures": ["Qwen3ForCausalLM"],
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "intermediate_size": 9728,
  "max_position_embeddings": 40960,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "vocab_size": 151936
})json";

// Sentinel device addresses. Spans built on these are never dereferenced, the
// arrange step only moves them around. Distinct values let the alias assertion
// tell a tied lm_head (shares the embedding pointer) from an untied one.
std::byte* const kEmbedPtr  = reinterpret_cast<std::byte*>(0x1000);
std::byte* const kLmHeadPtr = reinterpret_cast<std::byte*>(0x2000);

void write_config(const std::filesystem::path& dir, bool tied) {
    std::string text{kQwen3ConfigJson};
    if (!tied) {
        const std::string from = "\"tie_word_embeddings\": true";
        const auto pos = text.find(from);
        text.replace(pos, from.size(), "\"tie_word_embeddings\": false");
    }
    std::ofstream out(dir / "config.json", std::ios::binary);
    out << text;
}

[[nodiscard]] Tensor make_tensor(std::vector<std::size_t> shape,
                                 std::byte* ptr = nullptr,
                                 std::size_t nbytes = 0) {
    return Tensor{
        std::span<std::byte>(ptr, nbytes),
        std::move(shape),
        DType::BF16,
        std::nullopt,
    };
}

// Every tensor name qwen3::arrange expects, with realistic shapes. arrange
// does not validate shapes against config.
[[nodiscard]] Uploaded build_uploaded(const LlamaConfig& cfg, bool tied) {
    const auto hidden = cfg.base().hidden_dim();
    const auto vocab  = cfg.base().vocab_size();
    const auto heads  = cfg.num_heads();
    const auto kv     = cfg.num_kv_heads();
    const auto hd     = cfg.head_dim();
    const auto inter  = cfg.intermediate_dim();

    ByName m;
    m.emplace("model.embed_tokens.weight",
              make_tensor({vocab, hidden}, kEmbedPtr, vocab * hidden * 2));
    m.emplace("model.norm.weight", make_tensor({hidden}));

    for (std::size_t i = 0; i < cfg.base().num_layers(); ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        m.emplace(p + "input_layernorm.weight",          make_tensor({hidden}));
        m.emplace(p + "self_attn.q_proj.weight",         make_tensor({heads * hd, hidden}));
        m.emplace(p + "self_attn.k_proj.weight",         make_tensor({kv * hd, hidden}));
        m.emplace(p + "self_attn.v_proj.weight",         make_tensor({kv * hd, hidden}));
        m.emplace(p + "self_attn.o_proj.weight",         make_tensor({hidden, heads * hd}));
        m.emplace(p + "self_attn.q_norm.weight",         make_tensor({hd}));
        m.emplace(p + "self_attn.k_norm.weight",         make_tensor({hd}));
        m.emplace(p + "post_attention_layernorm.weight", make_tensor({hidden}));
        m.emplace(p + "mlp.gate_proj.weight",            make_tensor({inter, hidden}));
        m.emplace(p + "mlp.up_proj.weight",              make_tensor({inter, hidden}));
        m.emplace(p + "mlp.down_proj.weight",            make_tensor({hidden, inter}));
    }
    if (!tied) {
        m.emplace("lm_head.weight",
                  make_tensor({vocab, hidden}, kLmHeadPtr, vocab * hidden * 2));
    }

    return Uploaded{runtherder::device::DeviceUniquePtr<std::byte>{}, std::move(m)};
}

class LlamaLoadTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        dir_ = std::filesystem::temp_directory_path() /
               (std::string{"runtherder_"} + info->test_suite_name() + "_" + info->name());
        std::filesystem::create_directories(dir_);
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    std::filesystem::path dir_;
};

TEST_F(LlamaLoadTest, ConfigParsesQwen3Fields) {
    write_config(dir_, /*tied=*/true);
    const LlamaConfig cfg = LlamaConfig::load(dir_);

    EXPECT_EQ(cfg.base().architecture(), runtherder::model::ArchitectureKind::Qwen3);
    EXPECT_EQ(cfg.base().hidden_dim(), 2560u);
    EXPECT_EQ(cfg.base().num_layers(), 36u);
    EXPECT_EQ(cfg.base().vocab_size(), 151936u);
    EXPECT_EQ(cfg.base().max_seq_len(), 40960u);
    EXPECT_TRUE(cfg.base().tie_word_embeddings());

    EXPECT_EQ(cfg.num_heads(), 32u);
    EXPECT_EQ(cfg.num_kv_heads(), 8u);
    EXPECT_EQ(cfg.head_dim(), 128u);
    EXPECT_EQ(cfg.intermediate_dim(), 9728u);
    EXPECT_FLOAT_EQ(cfg.rms_norm_eps(), 1e-6f);
    EXPECT_FLOAT_EQ(cfg.rope_theta(), 1000000.0f);
}

TEST_F(LlamaLoadTest, ArrangeTiedBuildsFullGraph) {
    write_config(dir_, /*tied=*/true);
    const LlamaConfig cfg = LlamaConfig::load(dir_);

    const LlamaWeights w = qwen3::arrange(build_uploaded(cfg, /*tied=*/true), cfg);

    ASSERT_EQ(w.layers().size(), cfg.base().num_layers());
    for (const auto& layer : w.layers()) {
        EXPECT_TRUE(layer.q_norm.has_value());
        EXPECT_TRUE(layer.k_norm.has_value());
    }

    const std::vector<std::size_t> embed_shape{cfg.base().vocab_size(), cfg.base().hidden_dim()};
    EXPECT_EQ(w.token_embedding().shape, embed_shape);

    const std::vector<std::size_t> wq_shape{cfg.num_heads() * cfg.head_dim(), cfg.base().hidden_dim()};
    EXPECT_EQ(w.layers().front().wq.shape, wq_shape);

    // Tied embeddings: lm_head is a fresh Tensor aliasing the embedding span.
    EXPECT_EQ(w.lm_head().data.data(), kEmbedPtr);
    EXPECT_EQ(w.lm_head().data.size(), w.token_embedding().data.size());
    EXPECT_EQ(w.lm_head().shape, w.token_embedding().shape);
}

TEST_F(LlamaLoadTest, ArrangeUntiedUsesSeparateLmHead) {
    write_config(dir_, /*tied=*/false);
    const LlamaConfig cfg = LlamaConfig::load(dir_);

    const LlamaWeights w = qwen3::arrange(build_uploaded(cfg, /*tied=*/false), cfg);

    EXPECT_EQ(w.lm_head().data.data(), kLmHeadPtr);
    EXPECT_NE(w.lm_head().data.data(), w.token_embedding().data.data());
}

TEST_F(LlamaLoadTest, ArrangeMissingTensorAborts) {
    write_config(dir_, /*tied=*/true);
    const LlamaConfig cfg = LlamaConfig::load(dir_);

    Uploaded up = build_uploaded(cfg, /*tied=*/true);
    up.by_name.erase("model.layers.5.self_attn.q_norm.weight");

    EXPECT_EXIT((void)qwen3::arrange(std::move(up), cfg),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                "missing from model");
}

TEST_F(LlamaLoadTest, ArrangeLeftoverTensorAborts) {
    write_config(dir_, /*tied=*/true);
    const LlamaConfig cfg = LlamaConfig::load(dir_);

    Uploaded up = build_uploaded(cfg, /*tied=*/true);
    up.by_name.emplace("model.unexpected.weight", make_tensor({1}));

    EXPECT_EXIT((void)qwen3::arrange(std::move(up), cfg),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                "not consumed");
}

}  // namespace
