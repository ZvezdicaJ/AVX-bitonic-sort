cmake_minimum_required(VERSION 3.19)

project(BITONIC_SORT LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-mavx2 -std=c++20 -Wno-ignored-attributes")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(GTest REQUIRED CONFIG HINTS ${GTEST_PATH})
enable_testing()

include_directories(src)

add_subdirectory(src)
add_subdirectory(test)
#add_subdirectory(benchmark)
