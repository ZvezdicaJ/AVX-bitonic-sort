add_executable(bitonic_sort_test
    BitonicSortFloatTest.cpp
    BitonicSortDoubleTest.cpp
)

target_link_libraries(bitonic_sort_test bitonic_sort 
    GTest::gtest GTest::gtest_main)


add_test(bitonic_sort_test bitonic_sort_test)