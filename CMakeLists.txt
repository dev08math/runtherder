cmake_minimum_required(VERSION 3.24)

project(runtherder
    VERSION 0.1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "CUDA LLM inference engine"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(RUNTHERDER_CUDA_ARCH 89 CACHE STRING "Target compute capability, 89 is Ada")
set(CMAKE_CUDA_ARCHITECTURES ${RUNTHERDER_CUDA_ARCH})

get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel)

string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -lineinfo")

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.12.0
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG        c586c52f93f7b060753bd2388eb96a105cb7374d
)
FetchContent_MakeAvailable(tokenizers_cpp)

FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.6.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cli11)

# Jinja for chat templates. Populated but not added: minja is header only and its
# own CMake pulls a test suite and a second nlohmann copy.
FetchContent_Declare(
    minja
    GIT_REPOSITORY https://github.com/google/minja.git
    GIT_TAG        021c2293c187789ef13d56c6cfd89c9b134fd80f
)
FetchContent_Populate(minja)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(runtherder_headers INTERFACE)
target_include_directories(runtherder_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(runtherder_headers INTERFACE cxx_std_20)

add_library(runtherder_warnings INTERFACE)
target_compile_options(runtherder_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills,-warn-double-usage>
    $<$<COMPILE_LANGUAGE:CUDA>:--Wreorder;-Werror=all-warnings>
)

add_library(runtherder_kernels STATIC
    src/kernels/rmsnorm.cu
    src/kernels/rope.cu
    src/kernels/activation.cu
    src/kernels/matmul.cu
    src/kernels/embedding.cu
    src/kernels/attention.cu
    src/kernels/flash_decode.cu
    src/kernels/flash_prefill.cu
    src/kernels/quantization.cu
    src/kernels/bias.cu
)
set_target_properties(runtherder_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(runtherder_kernels
    PUBLIC  runtherder_headers CUDA::cublasLt CUDA::cublas
    PRIVATE runtherder_warnings
)

add_library(runtherder_attention STATIC
    src/attention/naive.cu
    src/attention/adaptive.cu
    src/attention/factory.cu
)
set_target_properties(runtherder_attention PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(runtherder_attention
    PUBLIC  runtherder_headers runtherder_kernels
    PRIVATE runtherder_warnings
)

add_library(runtherder_sampling STATIC
    src/sampling/sampler.cu
)
set_target_properties(runtherder_sampling PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(runtherder_sampling
    PUBLIC  runtherder_headers
    PRIVATE runtherder_warnings
)

add_library(runtherder_tokenizer STATIC
    src/tokenizer/tokenizer.cpp
    src/tokenizer/chat_template.cpp
)
set_target_properties(runtherder_tokenizer PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(runtherder_tokenizer SYSTEM PRIVATE ${minja_SOURCE_DIR}/include)
target_link_libraries(runtherder_tokenizer
    PUBLIC  runtherder_headers tokenizers_cpp
    PRIVATE runtherder_warnings nlohmann_json::nlohmann_json
)

add_library(runtherder_model STATIC
    src/io/mmap_file.cpp
    src/io/json_helpers.cpp
    src/model/safetensors.cpp
    src/model/sharded_safetensors.cpp
    src/model/config.cpp
    src/model/generation_config.cpp
    src/model/upload.cu
    src/model/supported/llama/llama.cu
    src/model/supported/llama/qwen2.cu
    src/model/supported/llama/qwen3.cu
    src/model/supported/llama/llama3.cu
)
set_target_properties(runtherder_model PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(runtherder_model
    PUBLIC  runtherder_headers runtherder_kernels
    PRIVATE nlohmann_json::nlohmann_json runtherder_warnings
)

add_library(runtherder_engine STATIC
    src/engine/context.cu
    src/engine/engine.cu
    src/engine/generator.cu
)
set_target_properties(runtherder_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(runtherder_engine
    PUBLIC  runtherder_headers runtherder_model runtherder_sampling
            runtherder_attention runtherder_kernels
    PRIVATE runtherder_warnings
)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(runtherder main.cpp)
set_target_properties(runtherder PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(runtherder PRIVATE
    runtherder_headers
    runtherder_kernels
    runtherder_model
    runtherder_engine
    runtherder_attention
    runtherder_sampling
    runtherder_tokenizer
    runtherder_warnings
    CLI11::CLI11
)

add_executable(runtherder_devinfo src/device/devinfo.cu)
target_link_libraries(runtherder_devinfo PRIVATE
    runtherder_headers
    runtherder_warnings
)

option(RUNTHERDER_BUILD_TESTS "Build runtherder tests" ON)

if(RUNTHERDER_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
        GIT_SHALLOW    TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)
    FetchContent_MakeAvailable(googletest)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "runtherder ${PROJECT_VERSION}: ${CMAKE_BUILD_TYPE}, sm_${RUNTHERDER_CUDA_ARCH}, tests ${RUNTHERDER_BUILD_TESTS}")
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING
        "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}, not Release. "
        "Host code is unoptimized. Do not publish benchmark numbers from this build.")
endif()
