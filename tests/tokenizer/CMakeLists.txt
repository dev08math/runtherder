add_executable(test_tokenizer test_tokenizer.cpp)
set_target_properties(test_tokenizer PROPERTIES
    LANGUAGE CUDA
)
set_source_files_properties(test_tokenizer.cpp PROPERTIES LANGUAGE CUDA)
target_compile_definitions(test_tokenizer PRIVATE
    RUNTHERDER_MODEL_DIR="${CMAKE_SOURCE_DIR}/models/Qwen3-4B"
)
target_link_libraries(test_tokenizer PRIVATE
    runtherder_headers
    runtherder_tokenizer
    runtherder_model
    GTest::gtest_main
)
gtest_discover_tests(test_tokenizer)
