cmake_minimum_required(VERSION 3.19)

project(BITONIC_SORT LANGUAGES CXX)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CLANG_COMPILE_OPTIONS -O3 -flto)
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${CLANG_COMPILE_OPTIONS}>")
        add_link_options("$<$<COMPILE_LANGUAGE:CXX>:-flto>")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(GCC_COMPILE_OPTIONS -O3 -flto)
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${GCC_COMPILE_OPTIONS}>")
        add_link_options("$<$<COMPILE_LANGUAGE:CXX>:-flto>")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        # using Intel C++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        #add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/GL>")
        #add_link_options("$<$<COMPILE_LANGUAGE:CXX>:/LTCG>")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(GTest CONFIG HINTS ${GTEST_PATH})
enable_testing()

find_package(benchmark CONFIG HINTS ${GB_PATH})

include_directories(src)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(benchmarks)

if(benchmark_FOUND)
install(TARGETS benchmarks
        CONFIGURATIONS ${CMAKE_BUILD_TYPE}
)
unset(installDir)
endif()

install(TARGETS bitonic_sort
        CONFIGURATIONS ${CMAKE_BUILD_TYPE})

message("CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
message("CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

