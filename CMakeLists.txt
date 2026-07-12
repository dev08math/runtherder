cmake_minimum_required(VERSION 3.24)

project(runtherder
    LANGUAGES CXX CUDA
    DESCRIPTION "----"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Ada Lovelace = sm_89.
set(CMAKE_CUDA_ARCHITECTURES 89)

# cublasLt for the GEMM launcher.
find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

add_library(runtherder_headers INTERFACE)
target_include_directories(runtherder_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(runtherder_headers INTERFACE cxx_std_20)

add_library(runtherder_kernels STATIC
    src/kernels/rmsnorm.cu
    src/kernels/rope.cu
    src/kernels/activation.cu
    src/kernels/matmul.cu
    src/kernels/embedding.cu
    src/kernels/attention.cu
    src/kernels/flash_decode.cu
    src/kernels/quantization.cu
)
set_target_properties(runtherder_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(runtherder_kernels PUBLIC runtherder_headers CUDA::cublasLt CUDA::cublas)

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
target_link_libraries(runtherder_attention PUBLIC runtherder_headers runtherder_kernels)

add_library(runtherder_sampling STATIC
    src/sampling/sampler.cu
)
set_target_properties(runtherder_sampling PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(runtherder_sampling PUBLIC runtherder_headers)

add_executable(runtherder main.cpp)
set_target_properties(runtherder PROPERTIES
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(runtherder PRIVATE
    runtherder_headers
    runtherder_kernels
    runtherder_model
    runtherder_engine
    runtherder_attention
    runtherder_sampling
    runtherder_tokenizer
)

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

add_library(runtherder_tokenizer STATIC
    src/tokenizer/tokenizer.cpp
)
set_target_properties(runtherder_tokenizer PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(runtherder_tokenizer
    PUBLIC runtherder_headers tokenizers_cpp
)

add_library(runtherder_model STATIC
    src/io/mmap_file.cpp
    src/io/json_helpers.cpp
    src/model/safetensors.cpp
    src/model/sharded_safetensors.cpp
    src/model/config.cpp
    src/model/upload.cu
    src/model/supported/llama/llama.cu
    src/model/supported/llama/qwen3.cu
    src/model/supported/llama/llama3.cu
)
set_target_properties(runtherder_model PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(runtherder_model
    PUBLIC  runtherder_headers runtherder_kernels
    PRIVATE nlohmann_json::nlohmann_json
)

add_library(runtherder_engine STATIC
    src/engine/context.cu
    src/engine/generator.cu
)
set_target_properties(runtherder_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(runtherder_engine
    PUBLIC runtherder_headers runtherder_model runtherder_sampling runtherder_attention runtherder_kernels
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
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_subdirectory(tests)
endif()

