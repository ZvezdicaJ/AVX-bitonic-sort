#ifdef __AVX2__

TEST(MODULO, MOD2) {
    ASSERT_EQ(mod2(5), 1);
    ASSERT_EQ(mod2(2), 0);
    ASSERT_EQ(mod2(1), 1);
    ASSERT_EQ(mod2(0), 0);
}

TEST(MODULO, MOD4) {
    ASSERT_EQ(mod4(5), 1);
    ASSERT_EQ(mod4(2), 2);
    ASSERT_EQ(mod4(1), 1);
    ASSERT_EQ(mod4(0), 0);
    ASSERT_EQ(mod4(9), 1);
    ASSERT_EQ(mod4(7), 3);
}

TEST(MODULO, MOD8) {
    ASSERT_EQ(mod8(5), 5);
    ASSERT_EQ(mod8(2), 2);
    ASSERT_EQ(mod8(1), 1);
    ASSERT_EQ(mod8(0), 0);
    ASSERT_EQ(mod8(25), 1);
    ASSERT_EQ(mod8(23), 7);
}

/*
TEST(MASK, MASK_LOAD) {
    {
        __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 1, 1, 1, 1);
        aligned_vector<float> arr({0, 1, 2, 3, 4, 5, 6, 7});

        __m256 reg = _mm256_maskload_ps(arr.data(), mask);
        float *r = (float *)&reg;
        aligned_vector<float> correct_result({0, 0, 0, 0, 0, 0, 0,
0});

        for (int i = 0; i < 8; i++)
            ASSERT_EQ(r[i], correct_result[i]);
    }
    {
        // high bit of the mask element has to be set
        __m256i mask = _mm256_set_epi32(10, 10, 10, 0xffffffff, 10,
10, 10, 10); print_avx(mask, "mask: "); aligned_vector<float>
arr({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});

        __m256 reg = _mm256_maskload_ps(&arr[0], mask);
        std::cout << "arr: " << arr[0] << " " << arr[1] <<
std::endl; print_avx(reg, "maskload reg: "); float *r = (float
*)&reg; aligned_vector<float> correct_result({0, 1, 2, 3, 4, 5, 6,
7});

        for (int i = 0; i < 8; i++)
            ASSERT_EQ(*(r + i), correct_result[i]);
    }
    }*/

#include "bitonic_sort_int_tests.hpp"
#include "bitonic_sort_float_tests.hpp"
#include "bitonic_sort_double_tests.hpp"

#endif
