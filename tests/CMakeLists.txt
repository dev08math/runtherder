include(GoogleTest)

add_executable(test_rmsnorm test_rmsnorm.cpp)
set_target_properties(test_rmsnorm PROPERTIES
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
)
set_source_files_properties(test_rmsnorm.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_rmsnorm PRIVATE
    runtherder_headers
    runtherder_kernels
    GTest::gtest_main
)
gtest_discover_tests(test_rmsnorm)
