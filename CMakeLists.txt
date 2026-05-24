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

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(runtherder_headers INTERFACE)
target_include_directories(runtherder_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(runtherder_headers INTERFACE cxx_std_20)

add_library(runtherder_kernels STATIC
    src/kernels/rmsnorm.cu
    src/kernels/rope.cu
    src/kernels/activation.cu
)
set_target_properties(runtherder_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(runtherder_kernels PUBLIC runtherder_headers)

add_executable(runtherder main.cpp)
set_target_properties(runtherder PROPERTIES
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(runtherder PRIVATE
    runtherder_headers
    runtherder_kernels
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

add_library(runtherder_model STATIC
    src/model/safetensors.cpp
    src/model/sharded_safetensors.cpp
)
set_target_properties(runtherder_model PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(runtherder_model
    PUBLIC  runtherder_headers
    PRIVATE nlohmann_json::nlohmann_json
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

