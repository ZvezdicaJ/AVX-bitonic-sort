cmake_minimum_required(VERSION 3.19)

project(BITONIC_SORT LANGUAGES CXX)

#set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-mavx2 -std=c++20 -Wno-ignored-attributes")

#set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(GTest)
enable_testing()

include_directories(src)

add_subdirectory(src)
add_subdirectory(test)
#add_subdirectory(benchmark)
