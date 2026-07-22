add_executable(test_rmsnorm test_rmsnorm.cpp)
set_target_properties(test_rmsnorm PROPERTIES
    LANGUAGE CUDA
)
set_source_files_properties(test_rmsnorm.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_rmsnorm PRIVATE
    runtherder_headers
    runtherder_kernels
    GTest::gtest_main
)
gtest_discover_tests(test_rmsnorm)

add_executable(test_rope test_rope.cpp)
set_target_properties(test_rope PROPERTIES
    LANGUAGE CUDA
)
set_source_files_properties(test_rope.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_rope PRIVATE
    runtherder_headers
    runtherder_kernels
    GTest::gtest_main
)
gtest_discover_tests(test_rope)

add_executable(test_activation test_activation.cpp)
set_target_properties(test_activation PROPERTIES
    LANGUAGE CUDA
)
set_source_files_properties(test_activation.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_activation PRIVATE
    runtherder_headers
    runtherder_kernels
    GTest::gtest_main
)
gtest_discover_tests(test_activation)

add_executable(test_matmul test_matmul.cpp)
set_target_properties(test_matmul PROPERTIES
    LANGUAGE CUDA
)
set_source_files_properties(test_matmul.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_matmul PRIVATE
    runtherder_headers
    runtherder_kernels
    GTest::gtest_main
)
gtest_discover_tests(test_matmul)

add_executable(test_attention test_attention.cpp)
set_target_properties(test_attention PROPERTIES
    LANGUAGE CUDA
)
set_source_files_properties(test_attention.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_attention PRIVATE
    runtherder_headers
    runtherder_kernels
    GTest::gtest_main
)
gtest_discover_tests(test_attention)

add_executable(test_quantization test_quantization.cpp)
set_target_properties(test_quantization PROPERTIES
    LANGUAGE CUDA
)
set_source_files_properties(test_quantization.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_quantization PRIVATE
    runtherder_headers
    runtherder_kernels
    GTest::gtest_main
)
gtest_discover_tests(test_quantization)

add_executable(test_flash_decode test_flash_decode.cpp)
set_target_properties(test_flash_decode PROPERTIES
    LANGUAGE CUDA
)
set_source_files_properties(test_flash_decode.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_flash_decode PRIVATE
    runtherder_headers
    runtherder_kernels
    GTest::gtest_main
)
gtest_discover_tests(test_flash_decode)

add_executable(test_flash_prefill test_flash_prefill.cpp)
set_target_properties(test_flash_prefill PROPERTIES
    LANGUAGE CUDA
)
set_source_files_properties(test_flash_prefill.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_flash_prefill PRIVATE
    runtherder_headers
    runtherder_kernels
    GTest::gtest_main
)
gtest_discover_tests(test_flash_prefill)
