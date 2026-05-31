add_executable(test_llama_load test_llama_load.cpp)
set_target_properties(test_llama_load PROPERTIES
    LANGUAGE CUDA
)
set_source_files_properties(test_llama_load.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_llama_load PRIVATE
    runtherder_headers
    runtherder_model
    GTest::gtest_main
)
gtest_discover_tests(test_llama_load)
