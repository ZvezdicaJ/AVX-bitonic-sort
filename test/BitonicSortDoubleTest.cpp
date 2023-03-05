#include "TestCommon.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <numeric>
#include <span>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <immintrin.h>

#include <BitonicSort.h>
#include <RandomVectorGenerator.h>

namespace bitonic_sort::test {

using utils::getRandomVector;

TEST(BITONIC_SORT, AVX_SINGLE_REG_DOUBLE_RANDOM) {
    for (int i = 0; i < 10; i++) {
        auto randomVec = getRandomVector<double>(4);

        __m256d randomReg = _mm256_loadu_pd(randomVec.data());
        runRegisterSortTest(randomReg);
    }
}

TEST(SORT, BITONIC_AVX_SORT_REG2_DOUBLE) {
    {
        __m256d reg0 = _mm256_setr_pd(1, 2, 3, 4);
        __m256d reg1 = _mm256_setr_pd(5, 6, 7, 8);
        runRegisterSortTest(reg0, reg1);
    }
    {
        __m256d reg0 = _mm256_setr_pd(-1, -2, 18, 29);
        __m256d reg1 = _mm256_setr_pd(500, -80, 2, -5);
        runRegisterSortTest(reg0, reg1);
    }
    for (std::uint32_t i = 0; i < 10; i++) {
        std::vector<double> inp0 = getRandomVector<double>(4);
        std::vector<double> inp1 = getRandomVector<double>(4);
        __m256d reg0 = _mm256_loadu_pd(inp0.data());
        __m256d reg1 = _mm256_loadu_pd(inp1.data());
        runRegisterSortTest(reg0, reg1);
    }
}

TEST(SORT, BITONIC_AVX_SORT_4REG_DOUBLE) {
    {
        __m256d reg0 = _mm256_setr_pd(1, 2, 3, 4);
        __m256d reg1 = _mm256_setr_pd(5, 6, 7, 8);
        __m256d reg2 = _mm256_setr_pd(9, 10, 11, 12);
        __m256d reg3 = _mm256_setr_pd(13, 14, 15, 16);
        runRegisterSortTest(reg0, reg1, reg2, reg3);
    }
    {
        __m256d reg0 = _mm256_setr_pd(-1, -2, 18, 29);
        __m256d reg1 = _mm256_setr_pd(500, -80, 2, -5);
        __m256d reg2 = _mm256_setr_pd(-10, 22, 180, -2900);
        __m256d reg3 = _mm256_setr_pd(38, -120, 25, -17);
        runRegisterSortTest(reg0, reg1, reg2, reg3);
    }

    {
        std::vector<double> inp0 = getRandomVector<double>(4);
        std::vector<double> inp1 = getRandomVector<double>(4);
        std::vector<double> inp2 = getRandomVector<double>(4);
        std::vector<double> inp3 = getRandomVector<double>(4);

        __m256d reg0 = _mm256_loadu_pd(inp0.data());
        __m256d reg1 = _mm256_loadu_pd(inp1.data());
        __m256d reg2 = _mm256_loadu_pd(inp2.data());
        __m256d reg3 = _mm256_loadu_pd(inp3.data());

        runRegisterSortTest(reg0, reg1, reg2, reg3);
    }
}

TEST(SORT, TEST_2N_SORT_DOUBLE_VER) {

    {
        std::vector<double> vec1 = getRandomVector<double>(4);
        std::vector<double> vec2 = vec1;
        sort_2n(std::span<double>(vec1));
        std::sort(vec2.begin(), vec2.end());
        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }

    {
        std::vector<double> vec1 = getRandomVector<double>(8);
        std::vector<double> vec2 = vec1;
        sort_2n(vec1);
        std::sort(vec2.begin(), vec2.end());
        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }
}

TEST(SORT, TEST_2N_SORT_DOUBLE_VER_TEST2) {

    for (int k = 1; k < 5; k++) {
        for (int j = 1; j < 5; j++) {
            // Shift bits to the left - basically multiplication with 2.
            // 1 << 3:  1000
            std::uint32_t const startAt = 1U << k;
            std::uint32_t const numberToSort = 8192;
            std::uint32_t const size = numberToSort + startAt;

            std::vector<double> vec1 = getRandomVector<double>(size);
            std::vector<double> vec2 = vec1;

            sort_2n({vec1.begin() + startAt, numberToSort});
            std::sort(vec2.begin() + startAt, vec2.end());
            EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
        }
    }
}

TEST(SORT, TEST_SORT_DOUBLE_4n_VECTOR) {

    for (std::uint32_t size = 24; size <= 24; size += 4) {

        std::vector<double> vec1 = getRandomVector<double>(size);
        std::vector<double> vec2 = vec1;

        sort_4n(vec1);
        std::sort(vec2.begin(), vec2.end());

        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }

    {
        constexpr std::uint32_t startAt = 4 * 133;
        constexpr std::uint32_t numberToSort = 4 * 333;
        constexpr std::uint32_t size = numberToSort + startAt;

        std::vector<double> vec1 = getRandomVector<double>(size);
        std::vector<double> vec2 = vec1;

        sort_4n({vec1.begin() + startAt, numberToSort});
        std::sort(vec2.begin() + startAt, vec2.end());

        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }
}

TEST(SORT, TEST_SORT_DOUBLE_VECTOR_ALL_CASES) {

    for (std::uint32_t size = 3; size < 200; size += 1) {

        std::vector<double> vec1 = getRandomVector<double>(size);
        std::vector<double> vec2 = vec1;

        sort(vec1);
        std::sort(vec2.begin(), vec2.end());
        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }

    for (std::uint32_t size = 17; size < 7000; size += 7) {

        std::vector<double> vec1 = getRandomVector<double>(size);
        std::vector<double> vec2 = vec1;

        sort(std::span<double>{vec1.data(), size});

        std::sort(vec2.begin(), vec2.end());

        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }

    constexpr std::uint32_t size = 17;
    std::vector<double> vec1(size);
    std::iota(vec1.rbegin(), vec1.rend(), 1);
    std::vector<double> vec2 = vec1;

    bitonic_sort::sort(std::span<double>{vec1.data(), size});

    std::sort(vec2.begin(), vec2.end());

    EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
}
} // namespace bitonic_sort::test