add_executable(test_greedy test_greedy.cpp)
set_target_properties(test_greedy PROPERTIES
    LANGUAGE CUDA
)
set_source_files_properties(test_greedy.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_greedy PRIVATE
    runtherder_headers
    runtherder_sampling
    GTest::gtest_main
)
gtest_discover_tests(test_greedy)
