#pragma once

#include <cstdint>

namespace runtherder::model {

enum class DType : std::uint8_t {
    BF16,
    FP16,
    INT8,
    INT4_AWQ,
    INT4_GPTQ,
};

[[nodiscard]] constexpr bool is_float_dtype(DType dtype) noexcept {
    switch (dtype) {
        case DType::BF16:
        case DType::FP16:
            return true;
        case DType::INT8:
        case DType::INT4_AWQ:
        case DType::INT4_GPTQ:
            return false;
    }
    return false;
}

}  // namespace runtherder::model
