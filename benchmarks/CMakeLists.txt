if(benchmark_FOUND)
add_executable(benchmarks
    benchmarks.cpp)

target_link_libraries(benchmarks benchmark::benchmark_main bitonic_sort)
endif()