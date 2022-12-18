add_executable(bitonic_sort_test
    bitonic_sort_float_test.cpp
    bitonic_sort_double_test.cpp
)

target_link_libraries(bitonic_sort_test bitonic_sort GTest::gtest_main)


add_test(bitonic_sort_test bitonic_sort_test)