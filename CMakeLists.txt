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

add_executable(runtherder main.cpp)
set_target_properties(runtherder PROPERTIES
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(runtherder PRIVATE runtherder_headers)

