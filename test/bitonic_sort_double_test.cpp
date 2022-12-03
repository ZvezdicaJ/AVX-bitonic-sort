#include "test_common.h"

#include <type_definitions.hpp>
#include <bitonic_sort_double.hpp>
#include <immintrin.h>
#include <random>
#include <chrono>
#include <ctime>
#include <array>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

std::default_random_engine generator(std::time(0));
std::uniform_real_distribution<float> float_dist(-100, 100);
std::uniform_real_distribution<double> double_dist(-100, 100);
std::uniform_int_distribution<int> int_dist(-10000000, 10000000);
std::uniform_int_distribution<int> pos_int_dist(1, 10000000);
auto random_float = std::bind(float_dist, generator);
auto random_double = std::bind(double_dist, generator);
auto random_int = std::bind(int_dist, generator);
auto random_pos_int = std::bind(pos_int_dist, generator);

TEST(BITONIC_SORT, AVX_SINGLE_REG_DOUBLE) {
    std::array registers = {
        _mm256_setr_pd(0, 1, 2, 3),
        _mm256_setr_pd(-10, 3, 6, -12),
        _mm256_setr_pd(5, 3, 6, 48)};

    for(auto &reg : registers) {
        runRegisterSortTest(reg);
    }
}

TEST(BITONIC_SORT, AVX_SINGLE_REG_DOUBLE_RANDOM) {
    for(int i=0; i< 10; i++) {
        double randomVec[] = {
            random_double(), 
            random_double(),
            random_double(),
            random_double()};

        __m256d randomReg = _mm256_loadu_pd(randomVec);
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
    for(std::uint32_t i=0; i<10; i++) {
        std::vector<double> inp0({
            random_double(), 
            random_double(),
            random_double(),
            random_double()});
        std::vector<double> inp1({
            random_double(), 
            random_double(),
            random_double(),
            random_double()});

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
        std::vector<double> inp0({random_double(), random_double(),
                                  random_double(),
                                  random_double()});
        std::vector<double> inp1({random_double(), random_double(),
                                  random_double(),
                                  random_double()});
        std::vector<double> inp2({random_double(), random_double(),
                                  random_double(),
                                  random_double()});
        std::vector<double> inp3({random_double(), random_double(),
                                  random_double(),
                                  random_double()});

        __m256d reg0 = _mm256_loadu_pd(inp0.data());
        __m256d reg1 = _mm256_loadu_pd(inp1.data());
        __m256d reg2 = _mm256_loadu_pd(inp2.data());
        __m256d reg3 = _mm256_loadu_pd(inp3.data());

        runRegisterSortTest(reg0, reg1, reg2, reg3);
    }
}



TEST(SORT, TEST_2N_SORT_DOUBLE_VER) {

    {
        aligned_vector<double> inp0({random_float(), random_float(),
                                     random_float(),
                                     random_float()});
        aligned_vector<double> inp1 = inp0;

        BITONIC_SORT::sort_2n_vector(inp0.data(), 4);
        std::sort(inp1.begin(), inp1.end());

        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(inp1[i], inp0[i]);
        }
    }

    {
        aligned_vector<double> inp0;
        aligned_vector<double> inp1;
        inp0.reserve(8);
        for (int i = 0; i < 8; i++)
            inp0.push_back(random_float());

        inp1 = inp0;
        BITONIC_SORT::sort_2n_vector(inp1.data(), 8);

        std::sort(std::begin(inp0), std::end(inp0));

        for (int i = 0; i < 8; i++) {
            EXPECT_EQ(inp1[i], inp0[i]);
        }
    }
}

TEST(SORT, TEST_2N_SORT_DOUBLE_VER_TEST2) {

    for (int k = 0; k < 5; k++) {
        for (int j = 0; j < 5; j++) {
            unsigned size = 8192;
            size *= (1 << k); //  1 << 2 = (int) Math.pow(2, 2)
            unsigned move_start = 1 << k;
            unsigned move_end = 1 << k;

            aligned_vector<double> inp0;
            aligned_vector<double> inp1;
            size = size + move_start + move_end;

            inp0.reserve(size);

            for (unsigned i = 0; i < size; i++)
                inp0.push_back(random_float());

            inp1 = inp0;

            BITONIC_SORT::sort_2n_vector(inp1.data() + move_start,
                                         size - move_end -
                                             move_start);

            std::sort(std::next(std::begin(inp0), move_start),
                      std::prev(std::end(inp0), move_end));

            for (unsigned i = move_start; i < size - move_end;
                 i++) {
                EXPECT_EQ(inp1[i], inp0[i]);
            }
        }
    }
}



TEST(SORT, TEST_SORT_DOUBLE_4n_VECTOR) {

    for (unsigned size = 4; size <= 2000; size += 4) {
        aligned_vector<double> inp0;
        aligned_vector<double> inp1;
        inp0.reserve(size);
        for (unsigned i = 0; i < size; i++)
            inp0.push_back(random_float());

        inp1 = inp0;
        BITONIC_SORT::sort_4n_vector(inp1.data(), size);

        std::sort(std::begin(inp0), std::end(inp0));

        for (int i = 0; i < size; i++) {
            EXPECT_EQ(inp1[i], inp0[i]);
        }
    }

    for (int i = 0; i < 2; i++) {

        unsigned move_start = (random_pos_int() % 4) * 4;
        unsigned move_end = (random_pos_int() % 4) * 4;
        unsigned size =
            move_start + move_end + 4 * random_pos_int();

        aligned_vector<double> inp0;
        aligned_vector<double> inp1;

        inp0.reserve(size);

        for (unsigned i = 0; i < size; i++) {
            inp0.push_back(random_double());
            // std::cout << i << ": " << inp0[i] << std::endl;
        }

        inp1 = inp0;

        auto p1 = std::next(inp0.begin(), move_start);
        auto p2 = std::prev(inp0.end(), move_end);

        BITONIC_SORT::sort_4n_vector(inp1.data() + move_start,
                                     size - move_end - move_start);
        std::sort(p1, p2);

        for (unsigned i = move_start; i < size - move_end; i++) {
            // std::cout << i << ": " << inp1[i] << "  " << inp0[i]
            //          << std::endl;
            EXPECT_EQ(inp1[i], inp0[i]);
        }
    }
}

TEST(SORT, TEST_SORT_DOUBLE_VECTOR_ALL_CASES) {

    for (unsigned size = 1; size < 1000; size++) {

        aligned_vector<double> inp0;
        aligned_vector<double> inp1;
        inp0.reserve(size);
        for (unsigned i = 0; i < size; i++)
            inp0.push_back(random_double());

        inp1 = inp0;
        BITONIC_SORT::sort_vector(inp1);

        std::sort(inp0.begin(), inp0.end());

        /* for (int i = 0; i < size; i++) {
           std::cout << inp1[i] << std::endl;
           if (mod8(i + 1) == 0)
           std::cout << "\n";
           }*/

        for (int i = 0; i < size; i++) {
            EXPECT_EQ(inp1[i], inp0[i]);
        }
    }

    for (unsigned size = 1000; size < 5000; size++) {

        aligned_vector<double> inp0;
        aligned_vector<double> inp1;
        inp0.reserve(size);
        for (unsigned i = 0; i < size; i++)
            inp0.push_back(random_double());

        inp1 = inp0;

        unsigned move_start = random_pos_int() % 500;
        unsigned move_end = random_pos_int() % 500;

        BITONIC_SORT::sort_vector(std::span<double>{inp1.data() + move_start, size - move_start - move_end});

        std::sort(std::next(inp0.begin(), move_start), std::prev(inp0.end(), move_end));

        for (int i = move_start; i < size - move_end; i++) {
            EXPECT_EQ(inp1[i], inp0[i]);
        }
    }
}