cmake_minimum_required(VERSION 3.19)

project(BITONIC_SORT LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(GTest REQUIRED CONFIG HINTS ${GTEST_PATH})
enable_testing()

find_package(benchmark HINTS ${GBENCHMARK_PATH})

include_directories(src)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(benchmarks)

install(TARGETS benchmarks
        CONFIGURATIONS ${CMAKE_BUILD_TYPE}
)



message("CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
message("CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

