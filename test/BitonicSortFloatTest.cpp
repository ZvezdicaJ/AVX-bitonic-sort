#include "test_common.h"

#include <array>
#include <bitonic_sort.hpp>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <immintrin.h>
#include <numeric>
#include <type_definitions.hpp>

namespace bitonic_sort::test {

TEST(bitonic_sort, AVX_SINGLE_REG_FLOAT_RANDOM) {
    for (int i = 0; i < 10; i++) {
        auto randomVec = getRandomVector<float>(8);

        __m256 randomReg = _mm256_loadu_ps(randomVec.data());
        runRegisterSortTest(randomReg);
    }
}

TEST(SORT, BITONIC_AVX_SORT_REG2_FLOAT) {
    {
        __m256 reg0 = _mm256_setr_ps(1, 2, 3, 4, 5, 6, 7, 8);
        __m256 reg1 = _mm256_setr_ps(9, 10, 11, 12, 13, 14, 15, 16);
        runRegisterSortTest(reg0, reg1);
    }
    {
        __m256 reg0 = _mm256_setr_ps(-1, -2, 18, 29, -45, -78, 303, 32);
        __m256 reg1 = _mm256_setr_ps(500, -80, 2, -5, -10, 10, 11, -33);
        runRegisterSortTest(reg0, reg1);
    }
    for (std::uint32_t i = 0; i < 10; i++) {
        std::vector<float> inp0 = getRandomVector<float>(8);
        std::vector<float> inp1 = getRandomVector<float>(8);
        __m256 reg0 = _mm256_loadu_ps(inp0.data());
        __m256 reg1 = _mm256_loadu_ps(inp1.data());
        runRegisterSortTest(reg0, reg1);
    }
}

TEST(SORT, BITONIC_AVX_SORT_4REG_FLOAT) {
    {
        __m256 reg0 = _mm256_setr_ps(1, 2, 3, 4, 5, 6, 7, 8);
        __m256 reg1 = _mm256_setr_ps(9, 10, 11, 12, 13, 14, 15, 16);
        __m256 reg2 = _mm256_setr_ps(17, 18, 19, 20, 21, 22, 23, 24);
        __m256 reg3 = _mm256_setr_ps(25, 26, 27, 28, 29, 30, 31, 32);
        runRegisterSortTest(reg0, reg1, reg2, reg3);
    }
    {
        __m256 reg0 = _mm256_setr_ps(-1, -2, 18, 29, 33, -29, -38, -43);
        __m256 reg1 = _mm256_setr_ps(500, -80, 2, -5, -26, -45, -66, 99);
        __m256 reg2 =
            _mm256_setr_ps(-10, 22, 180, -2900, -3003, -9999, 9999, 0);
        __m256 reg3 = _mm256_setr_ps(38, -120, 25, -17, -8, 8, -99, 99);
        runRegisterSortTest(reg0, reg1, reg2, reg3);
    }

    {
        auto inp0 = getRandomVector<float>(8);
        auto inp1 = getRandomVector<float>(8);
        auto inp2 = getRandomVector<float>(8);
        auto inp3 = getRandomVector<float>(8);

        __m256 reg0 = _mm256_loadu_ps(inp0.data());
        __m256 reg1 = _mm256_loadu_ps(inp1.data());
        __m256 reg2 = _mm256_loadu_ps(inp2.data());
        __m256 reg3 = _mm256_loadu_ps(inp3.data());

        runRegisterSortTest(reg0, reg1, reg2, reg3);
    }
}

TEST(SORT, TEST_2N_SORT_FLOAT_VER) {

    {
        std::vector<float> vec1 = getRandomVector<float>(4);
        std::vector<float> vec2 = vec1;

        bitonic_sort::sort_2n(std::span<float>(vec1));
        std::sort(vec2.begin(), vec2.end());

        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }

    {
        std::vector<float> vec1 = getRandomVector<float>(8);
        std::vector<float> vec2 = vec1;

        bitonic_sort::sort_2n(std::span<float>(vec1));
        std::sort(vec2.begin(), vec2.end());

        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }
}

TEST(SORT, TEST_2N_SORT_FLOAT_VER_TEST2) {

    for (int k = 1; k < 5; k++) {
        for (int j = 1; j < 5; j++) {
            // Shift bits to the left - basically multiplication with 2.
            // 1 << 3:  1000
            std::uint32_t const startAt = 1U << k;
            std::uint32_t const numberToSort = 8192;
            std::uint32_t const size = numberToSort + startAt;

            std::vector<float> vec1 = getRandomVector<float>(size);
            std::vector<float> vec2 = vec1;

            bitonic_sort::sort_2n({vec1.data() + startAt, numberToSort});

            std::sort(vec2.begin() + startAt, vec2.end());

            EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
        }
    }
}

TEST(SORT, TEST_SORT_FLOAT_8n_VECTOR) {

    for (std::uint32_t size = 8; size <= 4000; size += 8) {
        std::vector<float> vec1 = getRandomVector<float>(size);
        std::vector<float> vec2 = vec1;

        bitonic_sort::sort_8n(vec1);
        std::sort(vec2.begin(), vec2.end());

        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }
}

TEST(SORT, TEST_SORT_FLOAT_VECTOR_ALL_CASES) {

    for (std::uint32_t size = 1; size < 200; size += 3) {

        std::vector<float> vec1 = getRandomVector<float>(size);
        std::vector<float> vec2 = vec1;

        bitonic_sort::sort(vec1);
        std::sort(vec2.begin(), vec2.end());
        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }

    for (std::uint32_t size = 17; size < 7000; size += 7) {

        std::vector<float> vec1 = getRandomVector<float>(size);
        std::vector<float> vec2 = vec1;

        bitonic_sort::sort(std::span<float>{vec1.data(), size});

        std::sort(vec2.begin(), vec2.end());

        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }

    constexpr std::uint32_t size = 101;
    std::vector<float> vec1(size);
    std::iota(vec1.rbegin(), vec1.rend(), 1);
    std::vector<float> vec2 = vec1;

    bitonic_sort::sort(std::span<float>{vec1.data(), size});

    std::sort(vec2.begin(), vec2.end());

    EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
}
} // namespace bitonic_sort::test