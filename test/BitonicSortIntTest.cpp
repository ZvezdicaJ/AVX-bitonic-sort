#include "TestCommon.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <immintrin.h>
#include <numeric>

#include <BitonicSort.h>
#include <RandomVectorGenerator.h>

namespace bitonic_sort::test {

using utils::getRandomVector;

TEST(bitonic_sort, AVX_SINGLE_REG_INT_RANDOM) {
    for (int i = 0; i < 10; i++) {
        auto randomVec = getRandomVector<int>(8);
        __m256i randomReg = _mm256_loadu_si256((__m256i *)(randomVec.data()));
        runRegisterSortTest(randomReg);
    }
}

TEST(SORT, BITONIC_AVX_SORT_REG2_INT) {
    {
        __m256i reg0 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        __m256i reg1 = _mm256_setr_epi32(9, 10, 11, 12, 13, 14, 15, 16);

        std::array<int, 8> arr0;
        _mm256_storeu_si256((__m256i *)(arr0.data()), reg0);

        std::array<int, 8> arr1;
        _mm256_storeu_si256((__m256i *)(arr1.data()), reg1);

        runRegisterSortTest(reg0, reg1);
    }
    {
        __m256i reg0 = _mm256_setr_epi32(-1, -2, 18, 29, -45, -78, 303, 32);
        __m256i reg1 = _mm256_setr_epi32(500, -80, 2, -5, -10, 10, 11, -33);
        runRegisterSortTest(reg0, reg1);
    }
    for (std::uint32_t i = 0; i < 10; i++) {
        std::vector<int> inp0 = getRandomVector<int>(8);
        std::vector<int> inp1 = getRandomVector<int>(8);
        __m256i reg0 = _mm256_loadu_si256((__m256i *)(inp0.data()));
        __m256i reg1 = _mm256_loadu_si256((__m256i *)(inp1.data()));
        runRegisterSortTest(reg0, reg1);
    }
}

TEST(SORT, BITONIC_AVX_SORT_4REG_INT) {
    {
        __m256i reg0 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        __m256i reg1 = _mm256_setr_epi32(9, 10, 11, 12, 13, 14, 15, 16);
        __m256i reg2 = _mm256_setr_epi32(17, 18, 19, 20, 21, 22, 23, 24);
        __m256i reg3 = _mm256_setr_epi32(25, 26, 27, 28, 29, 30, 31, 32);
        runRegisterSortTest(reg0, reg1, reg2, reg3);
    }
    {
        __m256i reg0 = _mm256_setr_epi32(-1, -2, 18, 29, 33, -29, -38, -43);
        __m256i reg1 = _mm256_setr_epi32(500, -80, 2, -5, -26, -45, -66, 99);
        __m256i reg2 = _mm256_setr_epi32(-10, 22, 180, -2900, -3003, -9999, 9999, 0);
        __m256i reg3 = _mm256_setr_epi32(38, -120, 25, -17, -8, 8, -99, 99);
        runRegisterSortTest(reg0, reg1, reg2, reg3);
    }

    {
        auto inp0 = getRandomVector<int>(8);
        auto inp1 = getRandomVector<int>(8);
        auto inp2 = getRandomVector<int>(8);
        auto inp3 = getRandomVector<int>(8);

        __m256i reg0 = _mm256_loadu_si256((__m256i *)(inp0.data()));
        __m256i reg1 = _mm256_loadu_si256((__m256i *)(inp1.data()));
        __m256i reg2 = _mm256_loadu_si256((__m256i *)(inp2.data()));
        __m256i reg3 = _mm256_loadu_si256((__m256i *)(inp3.data()));

        runRegisterSortTest(reg0, reg1, reg2, reg3);
    }
}

TEST(SORT, TEST_2N_SORT_INT_VER) {

    {
        std::vector<int> vec1 = getRandomVector<int>(4);
        std::vector<int> vec2 = vec1;

        bitonic_sort::sort_2n(std::span<int>(vec1));
        std::sort(vec2.begin(), vec2.end());

        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }

    {
        std::vector<int> vec1 = getRandomVector<int>(8);
        std::vector<int> vec2 = vec1;

        bitonic_sort::sort_2n(std::span<int>(vec1));
        std::sort(vec2.begin(), vec2.end());

        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }
}

TEST(SORT, TEST_2N_SORT_INT_VER_TEST2) {

    for (int k = 1; k < 5; k++) {
        for (int j = 1; j < 5; j++) {
            std::uint32_t const numberToSort = 8192;
            std::uint32_t const size = numberToSort;

            std::vector<int> vec1 = getRandomVector<int>(size);
            std::vector<int> vec2 = vec1;

            bitonic_sort::sort_2n({vec1.data(), numberToSort});

            std::sort(vec2.begin(), vec2.end());

            EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
        }
    }
}

TEST(SORT, TEST_SORT_INT_8n_VECTOR) {

    for (std::uint32_t size = 8; size <= 400; size += 8) {
        std::vector<int> vec1 = getRandomVector<int>(size);
        std::vector<int> vec2 = vec1;

        bitonic_sort::sort_8n(vec1);
        std::sort(vec2.begin(), vec2.end());

        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }
}

TEST(SORT, TEST_SORT_INT_VECTOR_ALL_CASES) {

    for (std::uint32_t size = 1; size < 200; size += 1) {

        std::vector<int> vec1 = getRandomVector<int>(size);
        std::vector<int> vec2 = vec1;

        bitonic_sort::sort(std::span(vec1));
        std::sort(vec2.begin(), vec2.end());
        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }

    for (std::uint32_t size = 17; size < 7000; size += 7) {

        std::vector<int> vec1 = getRandomVector<int>(size);
        std::vector<int> vec2 = vec1;

        bitonic_sort::sort(std::span<int>{vec1.data(), size});

        std::sort(vec2.begin(), vec2.end());

        EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
    }

    constexpr std::uint32_t size = 101;
    std::vector<int> vec1(size);
    std::iota(vec1.rbegin(), vec1.rend(), 1);
    std::vector<int> vec2 = vec1;
    bitonic_sort::sort(vec1);
    std::sort(vec2.begin(), vec2.end());
    EXPECT_THAT(vec1, ::testing::ContainerEq(vec2));
}

} // namespace bitonic_sort::test