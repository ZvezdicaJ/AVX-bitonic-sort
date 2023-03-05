if(GTest_FOUND)
add_executable(bitonic_sort_test
    BitonicSortIntTest.cpp    
    BitonicSortFloatTest.cpp
    BitonicSortDoubleTest.cpp
)
target_link_libraries(bitonic_sort_test bitonic_sort 
    GTest::gtest GTest::gtest_main)
add_test(bitonic_sort_test bitonic_sort_test)


add_executable(load_sequence_test
    LoadSequenceTest.cpp
)
target_link_libraries(load_sequence_test 
    bitonic_sort
    GTest::gtest 
    GTest::gtest_main)
add_test(load_sequence_test load_sequence_test)
endif()