#ifdef __AVX2__

TEST(SORT, BITONIC_AVX_REG_SORT_FLOAT_KEY_VALUE) {
    {
        __m256 reg = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i key = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        BITONIC_SORT_KEY_VALUE::bitonic_sort(reg, key);
        float *sol = (float *)&reg;
        float solution[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        int *sol_key = (int *)&key;
        int solution_key[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        for (int i = 0; i < 8; i++) {
            ASSERT_EQ(*(sol + i), solution[i]);
            ASSERT_EQ(*(sol_key + i), solution_key[i]);
        }
    }

    {
        unsigned size = 8;
        std::vector<std::pair<int, float>> pairs;
        std::vector<float> inp0({random_float(), random_float(),
                                 random_float(), random_float(),
                                 random_float(), random_float(),
                                 random_float(), random_float()});
        pairs.reserve(size);
        for (int i = 0; i < 8; i++)
            pairs.push_back(std::pair<int, float>(i, inp0[i]));

        __m256 reg = _mm256_loadu_ps(inp0.data());
        __m256i key = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, float> &a,
                     const std::pair<int, float> &b) {
                      return (a.second) < (b.second);
                  });

        BITONIC_SORT_KEY_VALUE::bitonic_sort(reg, key);

        //        std::cout << pairs << std::endl;
        /*
    for (int i = 0; i < 8; i++) {
        std::cout << *(sol_key + i) << " " << *(sol + i) <<
    std::endl;
        }*/

        float *sol = (float *)&reg;
        int *sol_key = (int *)&key;
        for (int i = 0; i < 8; i++) {
            ASSERT_EQ(pairs[i].first, sol_key[i]);
            ASSERT_EQ(pairs[i].second, sol[i]);
        }
    }
}

TEST(SORT, BITONIC_AVX_SORT_REG2_FLOAT_KEY_VALUE) {
    {
        __m256 reg0 = _mm256_setr_ps(1, 2, 3, 4, 5, 6, 7, 8);
        __m256 reg1 = _mm256_setr_ps(9, 10, 11, 12, 13, 14, 15, 16);
        __m256i key0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i key1 =
            _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
        BITONIC_SORT_KEY_VALUE::bitonic_sort(reg0, reg1, key0,
                                             key1);
        float *s0 = (float *)&reg0;
        float *s1 = (float *)&reg1;

        int *s_key0 = (int *)&key0;
        int *s_key1 = (int *)&key1;

        float sol0[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        float sol1[8] = {9, 10, 11, 12, 13, 14, 15, 16};

        int sol_key0[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        int sol_key1[8] = {8, 9, 10, 11, 12, 13, 14, 15};

        for (int i = 0; i < 8; i++) {

            ASSERT_EQ(*(s0 + i), sol0[i]);
            ASSERT_EQ(*(s1 + i), sol1[i]);

            ASSERT_EQ(*(s_key0 + i), sol_key0[i]);
            ASSERT_EQ(*(s_key1 + i), sol_key1[i]);
        }
    }

    {
        unsigned size = 16;
        std::vector<float> inp;
        inp.reserve(size);
        std::vector<std::pair<int, float>> pairs;
        pairs.reserve(16);

        for (int i = 0; i < size; i++) {
            inp.push_back(random_float());
            pairs.push_back(std::pair<int, float>(i, inp[i]));
        }

        __m256 reg0 = _mm256_loadu_ps(inp.data());
        __m256 reg1 = _mm256_loadu_ps(inp.data() + 8);

        __m256i key0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i key1 =
            _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, float> &a,
                     const std::pair<int, float> &b) {
                      return (a.second) < (b.second);
                  });

        BITONIC_SORT_KEY_VALUE::bitonic_sort(reg0, reg1, key0,
                                             key1);

        float *s0 = (float *)&reg0;
        float *s1 = (float *)&reg1;

        int *s_key0 = (int *)&key0;
        int *s_key1 = (int *)&key1;

        for (int i = 0; i < 8; i++) {
            ASSERT_EQ(*(s_key0 + i), pairs[i].first);
            ASSERT_EQ(*(s_key1 + i), pairs[i + 8].first);

            ASSERT_EQ(*(s0 + i), pairs[i].second);
            ASSERT_EQ(*(s1 + i), pairs[i + 8].second);
        }
    }
}

TEST(SORT, BITONIC_AVX_SORT_REG4_FLOAT_KEY_VALUE) {
    {
        __m256 reg0 = _mm256_setr_ps(1, 2, 3, 4, 5, 6, 7, 8);
        __m256 reg1 = _mm256_setr_ps(9, 10, 11, 12, 13, 14, 15, 16);
        __m256 reg2 =
            _mm256_setr_ps(17, 18, 19, 20, 21, 22, 23, 24);
        __m256 reg3 =
            _mm256_setr_ps(25, 26, 27, 28, 29, 30, 31, 32);

        __m256i key0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i key1 =
            _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
        __m256i key2 =
            _mm256_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23);
        __m256i key3 =
            _mm256_setr_epi32(24, 25, 26, 27, 28, 29, 30, 31);

        BITONIC_SORT_KEY_VALUE::bitonic_sort(
            reg0, reg1, reg2, reg3, key0, key1, key2, key3);
        float *s0 = (float *)&reg0;
        float *s1 = (float *)&reg1;
        float *s2 = (float *)&reg2;
        float *s3 = (float *)&reg3;

        int *s_key0 = (int *)&key0;
        int *s_key1 = (int *)&key1;
        int *s_key2 = (int *)&key2;
        int *s_key3 = (int *)&key3;

        float sol0[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        float sol1[8] = {9, 10, 11, 12, 13, 14, 15, 16};
        float sol2[8] = {17, 18, 19, 20, 21, 22, 23, 24};
        float sol3[8] = {25, 26, 27, 28, 29, 30, 31, 32};

        int sol_key0[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        int sol_key1[8] = {8, 9, 10, 11, 12, 13, 14, 15};
        int sol_key2[8] = {16, 17, 18, 19, 20, 21, 22, 23};
        int sol_key3[8] = {24, 25, 26, 27, 28, 29, 30, 31};

        for (int i = 0; i < 8; i++) {

            ASSERT_EQ(*(s0 + i), sol0[i]);
            ASSERT_EQ(*(s1 + i), sol1[i]);
            ASSERT_EQ(*(s2 + i), sol2[i]);
            ASSERT_EQ(*(s3 + i), sol3[i]);

            ASSERT_EQ(*(s_key0 + i), sol_key0[i]);
            ASSERT_EQ(*(s_key1 + i), sol_key1[i]);
            ASSERT_EQ(*(s_key2 + i), sol_key2[i]);
            ASSERT_EQ(*(s_key3 + i), sol_key3[i]);
        }
    }

    {
        unsigned size = 32;
        std::vector<float> inp;
        inp.reserve(size);
        std::vector<std::pair<int, float>> pairs;
        pairs.reserve(size);

        for (int i = 0; i < size; i++) {
            inp.push_back(random_float());
            pairs.push_back(std::pair<int, float>(i, inp[i]));
        }

        __m256 reg0 = _mm256_loadu_ps(inp.data());
        __m256 reg1 = _mm256_loadu_ps(inp.data() + 8);
        __m256 reg2 = _mm256_loadu_ps(inp.data() + 16);
        __m256 reg3 = _mm256_loadu_ps(inp.data() + 24);

        __m256i key0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i key1 =
            _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
        __m256i key2 =
            _mm256_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23);
        __m256i key3 =
            _mm256_setr_epi32(24, 25, 26, 27, 28, 29, 30, 31);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, float> &a,
                     const std::pair<int, float> &b) {
                      return (a.second) < (b.second);
                  });

        BITONIC_SORT_KEY_VALUE::bitonic_sort(
            reg0, reg1, reg2, reg3, key0, key1, key2, key3);

        float *s0 = (float *)&reg0;
        float *s1 = (float *)&reg1;
        float *s2 = (float *)&reg2;
        float *s3 = (float *)&reg3;

        int *s_key0 = (int *)&key0;
        int *s_key1 = (int *)&key1;
        int *s_key2 = (int *)&key2;
        int *s_key3 = (int *)&key3;

        for (int i = 0; i < 8; i++) {
            ASSERT_EQ(*(s_key0 + i), pairs[i].first);
            ASSERT_EQ(*(s_key1 + i), pairs[i + 8].first);
            ASSERT_EQ(*(s_key2 + i), pairs[i + 16].first);
            ASSERT_EQ(*(s_key3 + i), pairs[i + 24].first);

            ASSERT_EQ(*(s0 + i), pairs[i].second);
            ASSERT_EQ(*(s1 + i), pairs[i + 8].second);
            ASSERT_EQ(*(s2 + i), pairs[i + 16].second);
            ASSERT_EQ(*(s3 + i), pairs[i + 24].second);
        }
    }
}

TEST(SORT, TEST_2N_BITONIC_SORT_KEY_VALUE_FLOAT_VER) {

    {
        unsigned size = 8;
        aligned_vector<float> inp;
        aligned_vector<int> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<int, float>> pairs;
        pairs.reserve(size);

        for (int i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(i);
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(std::pair<int, float>(i, i));
        }

        BITONIC_SORT_KEY_VALUE::sort_2n_key_value(
            inp.data(), keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, float> &a,
                     const std::pair<int, float> &b) {
                      return (a.second) < (b.second);
                  });

        for (int i = 0; i < 7; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
    {
        unsigned size = 64;
        aligned_vector<float> inp;
        aligned_vector<int> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<int, float>> pairs;
        pairs.reserve(size);

        for (int i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(i);
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(std::pair<int, float>(i, i));
        }

        BITONIC_SORT_KEY_VALUE::sort_2n_key_value(
            inp.data(), keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, float> &a,
                     const std::pair<int, float> &b) {
                      return (a.second) < (b.second);
                  });

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
}

TEST(SORT, TEST_8N_BITONIC_SORT_KEY_VALUE_FLOAT_VER) {

    {
        unsigned size = 56;
        aligned_vector<float> inp;
        aligned_vector<int> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<int, float>> pairs;
        pairs.reserve(size);

        for (int i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(i);
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(std::pair<int, float>(i, i));
        }

        BITONIC_SORT_KEY_VALUE::sort_8n_key_value(
            inp.data(), keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, float> &a,
                     const std::pair<int, float> &b) {
                      return (a.second) < (b.second);
                  });

        for (int i = 0; i < 7; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
    {
        unsigned size = 8 * 400;
        aligned_vector<float> inp;
        aligned_vector<int> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<int, float>> pairs;
        pairs.reserve(size);

        for (int i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(i);
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(std::pair<int, float>(i, i));
        }

        BITONIC_SORT_KEY_VALUE::sort_8n_key_value(
            inp.data(), keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, float> &a,
                     const std::pair<int, float> &b) {
                      return (a.second) < (b.second);
                  });

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
}

TEST(SORT, TEST_BITONIC_SORT_KEY_VALUE_FLOAT_ALL_CASES) {

    {
        unsigned size = 33;
        aligned_vector<float> inp;
        aligned_vector<int> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<int, float>> pairs;
        pairs.reserve(size);

        for (int i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(i);
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(std::pair<int, float>(i, i));
        }

        BITONIC_SORT_KEY_VALUE::sort_key_value(inp.data(),
                                               keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, float> &a,
                     const std::pair<int, float> &b) {
                      return (a.second) < (b.second);
                  });

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
    for (int size = 1; size < 3334; size++) {

        aligned_vector<float> inp;
        aligned_vector<int> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<int, float>> pairs;
        pairs.reserve(size);

        for (int i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(i);
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(std::pair<int, float>(i, i));
        }

        BITONIC_SORT_KEY_VALUE::sort_key_value(inp.data(),
                                               keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, float> &a,
                     const std::pair<int, float> &b) {
                      return (a.second) < (b.second);
                  });

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
}

/////////////////////////////////////////////////
////////////////////////////////////////////////

TEST(SORT, BITONIC_AVX_REG_SORT_LONG_DOUBLE_KEY_VALUE) {
    {
        __m256d reg = _mm256_setr_pd(0, 1, 2, 3);
        __m256i key = _mm256_setr_epi64x(0, 1, 2, 3);
        BITONIC_SORT_KEY_VALUE::bitonic_sort(reg, key);
        double *sol = (double *)&reg;
        double solution[4] = {0, 1, 2, 3};
        long int *sol_key = (long *)&key;
        long int solution_key[4] = {0, 1, 2, 3};

        for (int i = 0; i < 4; i++) {
            ASSERT_EQ(*(sol + i), solution[i]);
            ASSERT_EQ(*(sol_key + i), solution_key[i]);
        }
    }

    for (int j = 0; j < 10; j++) {
        unsigned size = 4;
        std::vector<std::pair<long int, double>> pairs;
        std::vector<double> inp0({random_double(), random_double(),
                                  random_double(),
                                  random_double()});
        pairs.reserve(size);
        for (int i = 0; i < size; i++)
            pairs.push_back(
                std::pair<long int, double>(i, inp0[i]));

        __m256d reg = _mm256_loadu_pd(inp0.data());
        __m256i key = _mm256_setr_epi64x(0, 1, 2, 3);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<long int, double> &a,
                     const std::pair<long int, double> &b) {
                      return (a.second) < (b.second);
                  });

        BITONIC_SORT_KEY_VALUE::bitonic_sort(reg, key);

        double *sol = (double *)&reg;
        long int *sol_key = (long int *)&key;
        /*
        for (int i = 0; i < 4; i++) {
            std::cout << pairs[i] << "     " << *(sol_key + i) << "
        "
                      << *(sol + i) << std::endl;
                      }*/

        for (long int i = 0; i < size; i++) {
            ASSERT_EQ(pairs[i].first, sol_key[i]);
            ASSERT_EQ(pairs[i].second, sol[i]);
        }
    }
}

TEST(SORT, BITONIC_AVX_SORT_REG2_LONG_DOUBLE_KEY_VALUE) {
    {
        __m256d reg0 = _mm256_setr_pd(0, 1, 2, 3);
        __m256d reg1 = _mm256_setr_pd(4, 5, 6, 7);
        __m256i key0 = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i key1 = _mm256_setr_epi64x(4, 5, 6, 7);
        BITONIC_SORT_KEY_VALUE::bitonic_sort(reg0, reg1, key0,
                                             key1);
        double *s0 = (double *)&reg0;
        double *s1 = (double *)&reg1;

        long *s_key0 = (long *)&key0;
        long *s_key1 = (long *)&key1;

        double sol0[4] = {0, 1, 2, 3};
        double sol1[4] = {4, 5, 6, 7};

        long sol_key0[4] = {0, 1, 2, 3};
        long sol_key1[4] = {4, 5, 6, 7};

        for (int i = 0; i < 4; i++) {

            ASSERT_EQ(*(s0 + i), sol0[i]);
            ASSERT_EQ(*(s1 + i), sol1[i]);

            ASSERT_EQ(*(s_key0 + i), sol_key0[i]);
            ASSERT_EQ(*(s_key1 + i), sol_key1[i]);
        }
    }

    for (int i = 0; i < 10; i++) {
        unsigned size = 8;
        std::vector<double> inp;
        inp.reserve(size);
        std::vector<std::pair<long, double>> pairs;
        pairs.reserve(8);

        for (long i = 0; i < size; i++) {
            inp.push_back(random_double());
            pairs.push_back(std::pair<long, double>(i, inp[i]));
        }

        __m256d reg0 = _mm256_loadu_pd(inp.data());
        __m256d reg1 = _mm256_loadu_pd(inp.data() + 4);

        __m256i key0 = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i key1 = _mm256_setr_epi64x(4, 5, 6, 7);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<long, double> &a,
                     const std::pair<long, double> &b) {
                      return (a.second) < (b.second);
                  });

        BITONIC_SORT_KEY_VALUE::bitonic_sort(reg0, reg1, key0,
                                             key1);

        double *s0 = (double *)&reg0;
        double *s1 = (double *)&reg1;

        long *s_key0 = (long *)&key0;
        long *s_key1 = (long *)&key1;

        for (long i = 0; i < 4; i++) {
            ASSERT_EQ(*(s_key0 + i), pairs[i].first);
            ASSERT_EQ(*(s_key1 + i), pairs[i + 4].first);

            ASSERT_EQ(*(s0 + i), pairs[i].second);
            ASSERT_EQ(*(s1 + i), pairs[i + 4].second);
        }
    }
}

TEST(SORT, BITONIC_AVX_SORT_REG4_LONG_DOUBLE_KEY_VALUE) {
    {
        __m256d reg0 = _mm256_setr_pd(1, 2, 3, 4);
        __m256d reg1 = _mm256_setr_pd(5, 6, 7, 8);
        __m256d reg2 = _mm256_setr_pd(9, 10, 11, 12);
        __m256d reg3 = _mm256_setr_pd(13, 14, 15, 16);

        __m256i key0 = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i key1 = _mm256_setr_epi64x(4, 5, 6, 7);
        __m256i key2 = _mm256_setr_epi64x(8, 9, 10, 11);
        __m256i key3 = _mm256_setr_epi64x(12, 13, 14, 15);

        BITONIC_SORT_KEY_VALUE::bitonic_sort(
            reg0, reg1, reg2, reg3, key0, key1, key2, key3);
        double *s0 = (double *)&reg0;
        double *s1 = (double *)&reg1;
        double *s2 = (double *)&reg2;
        double *s3 = (double *)&reg3;

        long *s_key0 = (long *)&key0;
        long *s_key1 = (long *)&key1;
        long *s_key2 = (long *)&key2;
        long *s_key3 = (long *)&key3;

        double sol0[4] = {1, 2, 3, 4};
        double sol1[4] = {5, 6, 7, 8};
        double sol2[4] = {9, 10, 11, 12};
        double sol3[4] = {13, 14, 15, 16};

        long sol_key0[4] = {0, 1, 2, 3};
        long sol_key1[4] = {4, 5, 6, 7};
        long sol_key2[4] = {8, 9, 10, 11};
        long sol_key3[4] = {12, 13, 14, 15};

        for (int i = 0; i < 4; i++) {

            ASSERT_EQ(*(s0 + i), sol0[i]);
            ASSERT_EQ(*(s1 + i), sol1[i]);
            ASSERT_EQ(*(s2 + i), sol2[i]);
            ASSERT_EQ(*(s3 + i), sol3[i]);

            ASSERT_EQ(*(s_key0 + i), sol_key0[i]);
            ASSERT_EQ(*(s_key1 + i), sol_key1[i]);
            ASSERT_EQ(*(s_key2 + i), sol_key2[i]);
            ASSERT_EQ(*(s_key3 + i), sol_key3[i]);
        }
    }

    for (int i = 0; i < 1; i++) {
        unsigned size = 16;
        std::vector<double> inp;
        inp.reserve(size);
        std::vector<std::pair<long, double>> pairs;
        pairs.reserve(8);

        for (long i = 0; i < size; i++) {
            inp.push_back(random_double());
            pairs.push_back(std::pair<long, double>(i, inp[i]));
        }

        __m256d reg0 = _mm256_loadu_pd(inp.data());
        __m256d reg1 = _mm256_loadu_pd(inp.data() + 4);
        __m256d reg2 = _mm256_loadu_pd(inp.data() + 8);
        __m256d reg3 = _mm256_loadu_pd(inp.data() + 12);

        __m256i key0 = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i key1 = _mm256_setr_epi64x(4, 5, 6, 7);
        __m256i key2 = _mm256_setr_epi64x(8, 9, 10, 11);
        __m256i key3 = _mm256_setr_epi64x(12, 13, 14, 15);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<long, double> &a,
                     const std::pair<long, double> &b) {
                      return (a.second) < (b.second);
                  });

        BITONIC_SORT_KEY_VALUE::bitonic_sort(
            reg0, reg1, reg2, reg3, key0, key1, key2, key3);

        double *s0 = (double *)&reg0;
        double *s1 = (double *)&reg1;
        double *s2 = (double *)&reg2;
        double *s3 = (double *)&reg3;

        long *s_key0 = (long *)&key0;
        long *s_key1 = (long *)&key1;
        long *s_key2 = (long *)&key2;
        long *s_key3 = (long *)&key3;

        for (long i = 0; i < 4; i++) {
            ASSERT_EQ(*(s_key0 + i), pairs[i].first);
            ASSERT_EQ(*(s_key1 + i), pairs[i + 4].first);
            ASSERT_EQ(*(s_key2 + i), pairs[i + 8].first);
            ASSERT_EQ(*(s_key3 + i), pairs[i + 12].first);

            ASSERT_EQ(*(s0 + i), pairs[i].second);
            ASSERT_EQ(*(s1 + i), pairs[i + 4].second);
            ASSERT_EQ(*(s2 + i), pairs[i + 8].second);
            ASSERT_EQ(*(s3 + i), pairs[i + 12].second);
        }
    }
}

TEST(SORT, TEST_2N_BITONIC_SORT_KEY_VALUE_LONG_DOUBLE_VER) {

    {
        unsigned size = 8;
        aligned_vector<double> inp;
        aligned_vector<long long> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<long long, double>> pairs;
        pairs.reserve(size);

        for (long long i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(i);
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(std::pair<long long, double>(i, i));
        }

        BITONIC_SORT_KEY_VALUE::sort_2n_key_value(
            inp.data(), keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<long long, double> &a,
                     const std::pair<long long, double> &b) {
                      return (a.second) < (b.second);
                  });

        for (int i = 0; i < 7; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
    {
        unsigned size = 64;
        aligned_vector<double> inp;
        aligned_vector<long long> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<long long, double>> pairs;
        pairs.reserve(size);

        for (long long i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(random_double());
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(
                std::pair<long long, double>(i, inp[i]));
        }

        BITONIC_SORT_KEY_VALUE::sort_2n_key_value(
            inp.data(), keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<long long, double> &a,
                     const std::pair<long long, double> &b) {
                      return (a.second) < (b.second);
                  });

        for (long long i = 0; i < size; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
}

TEST(SORT, TEST_4N_BITONIC_SORT_KEY_VALUE_LONG_DOUBLE_VER) {

    {
        unsigned size = 4;
        aligned_vector<double> inp;
        aligned_vector<long long> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<long long, double>> pairs;
        pairs.reserve(size);

        for (long long i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(i);
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(std::pair<long long, double>(i, i));
        }

        BITONIC_SORT_KEY_VALUE::sort_4n_key_value(
            inp.data(), keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<long long, double> &a,
                     const std::pair<long long, double> &b) {
                      return (a.second) < (b.second);
                  });

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
    {
        unsigned size = 384;
        aligned_vector<double> inp;
        aligned_vector<long long> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<long long, double>> pairs;
        pairs.reserve(size);

        for (long long i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(random_double());
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(
                std::pair<long long, double>(i, inp[i]));
        }

        BITONIC_SORT_KEY_VALUE::sort_4n_key_value(
            inp.data(), keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<long long, double> &a,
                     const std::pair<long long, double> &b) {
                      return (a.second) < (b.second);
                  });

        for (long long i = 0; i < size; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
}

TEST(SORT, TEST_BITONIC_SORT_KEY_VALUE_LONG_DOUBLE_VER) {

    for (unsigned size = 1; size < 2000; size++) {
        aligned_vector<double> inp;
        aligned_vector<long long> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<long long, double>> pairs;
        pairs.reserve(size);

        for (long long i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(random_double());
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(
                std::pair<long long, double>(i, inp[i]));
        }

        BITONIC_SORT_KEY_VALUE::sort_key_value(inp.data(),
                                               keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<long long, double> &a,
                     const std::pair<long long, double> &b) {
                      return (a.second) < (b.second);
                  });

        for (long long i = 0; i < size; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
}

/////////////////////////////////////////////////
////////////////////////////////////////////////

TEST(SORT, BITONIC_AVX_REG_SORT_INT_DOUBLE_KEY_VALUE) {
    {
        __m256d reg = _mm256_setr_pd(0, 1, 2, 3);
        __m128i key = _mm_setr_epi32(0, 1, 2, 3);
        BITONIC_SORT_KEY_VALUE::bitonic_sort(reg, key);
        double *sol = (double *)&reg;
        double solution[4] = {0, 1, 2, 3};
        int *sol_key = (int *)&key;
        int solution_key[4] = {0, 1, 2, 3};

        for (int i = 0; i < 4; i++) {
            ASSERT_EQ(*(sol + i), solution[i]);
            ASSERT_EQ(*(sol_key + i), solution_key[i]);
        }
        /*        for (int i = 0; i < 4; i++) {
            std::cout << *(sol_key + i) << *(sol + i) << std::endl;
            }*/
    }

    for (int j = 0; j < 10; j++) {
        unsigned size = 4;
        std::vector<std::pair<int, double>> pairs;
        std::vector<double> inp0({random_double(), random_double(),
                                  random_double(),
                                  random_double()});
        pairs.reserve(size);
        for (int i = 0; i < size; i++)
            pairs.push_back(std::pair<int, double>(i, inp0[i]));

        __m256d reg = _mm256_loadu_pd(inp0.data());
        __m128i key = _mm_setr_epi32(0, 1, 2, 3);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, double> &a,
                     const std::pair<int, double> &b) {
                      return (a.second) < (b.second);
                  });

        BITONIC_SORT_KEY_VALUE::bitonic_sort(reg, key);

        double *sol = (double *)&reg;
        int *sol_key = (int *)&key;
        /*
        for (int i = 0; i < 4; i++) {
            std::cout << pairs[i] << "     " << *(sol_key + i)
        << "
        "
                      << *(sol + i) << std::endl;
                      }*/

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(pairs[i].first, sol_key[i]);
            ASSERT_EQ(pairs[i].second, sol[i]);
        }
    }
}

TEST(SORT, BITONIC_AVX_SORT_REG2_INT_DOUBLE_KEY_VALUE) {
    {
        __m256d reg0 = _mm256_setr_pd(0, 1, 2, 3);
        __m256d reg1 = _mm256_setr_pd(4, 5, 6, 7);
        __m128i key0 = _mm_setr_epi32(0, 1, 2, 3);
        __m128i key1 = _mm_setr_epi32(4, 5, 6, 7);
        BITONIC_SORT_KEY_VALUE::bitonic_sort(reg0, reg1, key0,
                                             key1);
        double *s0 = (double *)&reg0;
        double *s1 = (double *)&reg1;

        int *s_key0 = (int *)&key0;
        int *s_key1 = (int *)&key1;

        double sol0[4] = {0, 1, 2, 3};
        double sol1[4] = {4, 5, 6, 7};

        int sol_key0[4] = {0, 1, 2, 3};
        int sol_key1[4] = {4, 5, 6, 7};

        for (int i = 0; i < 4; i++) {

            ASSERT_EQ(*(s0 + i), sol0[i]);
            ASSERT_EQ(*(s1 + i), sol1[i]);

            ASSERT_EQ(*(s_key0 + i), sol_key0[i]);
            ASSERT_EQ(*(s_key1 + i), sol_key1[i]);
        }
        /* print_sse(key0, "key0: ");
        print_sse(key1, "key1: ");
        print_avx(reg0, "reg0: ");
        print_avx(reg1, "reg1: ");*/
    }

    for (int i = 0; i < 1; i++) {
        unsigned size = 8;
        std::vector<double> inp;
        inp.reserve(size);
        std::vector<std::pair<int, double>> pairs;
        pairs.reserve(8);

        for (int i = 0; i < size; i++) {
            inp.push_back(random_double());
            pairs.push_back(std::pair<int, double>(i, inp[i]));
        }

        __m256d reg0 = _mm256_loadu_pd(inp.data());
        __m256d reg1 = _mm256_loadu_pd(inp.data() + 4);

        __m128i key0 = _mm_setr_epi32(0, 1, 2, 3);
        __m128i key1 = _mm_setr_epi32(4, 5, 6, 7);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, double> &a,
                     const std::pair<int, double> &b) {
                      return (a.second) < (b.second);
                  });

        BITONIC_SORT_KEY_VALUE::bitonic_sort(reg0, reg1, key0,
                                             key1);

        double *s0 = (double *)&reg0;
        double *s1 = (double *)&reg1;

        int *s_key0 = (int *)&key0;
        int *s_key1 = (int *)&key1;

        for (int i = 0; i < 4; i++) {
            ASSERT_EQ(*(s_key0 + i), pairs[i].first);
            ASSERT_EQ(*(s_key1 + i), pairs[i + 4].first);

            ASSERT_EQ(*(s0 + i), pairs[i].second);
            ASSERT_EQ(*(s1 + i), pairs[i + 4].second);
        }
    }
}

TEST(SORT, BITONIC_AVX_SORT_REG4_INT_DOUBLE_KEY_VALUE) {
    {
        __m256d reg0 = _mm256_setr_pd(1, 2, 3, 4);
        __m256d reg1 = _mm256_setr_pd(5, 6, 7, 8);
        __m256d reg2 = _mm256_setr_pd(9, 10, 11, 12);
        __m256d reg3 = _mm256_setr_pd(13, 14, 15, 16);

        __m128i key0 = _mm_setr_epi32(0, 1, 2, 3);
        __m128i key1 = _mm_setr_epi32(4, 5, 6, 7);
        __m128i key2 = _mm_setr_epi32(8, 9, 10, 11);
        __m128i key3 = _mm_setr_epi32(12, 13, 14, 15);

        BITONIC_SORT_KEY_VALUE::bitonic_sort(
            reg0, reg1, reg2, reg3, key0, key1, key2, key3);
        double *s0 = (double *)&reg0;
        double *s1 = (double *)&reg1;
        double *s2 = (double *)&reg2;
        double *s3 = (double *)&reg3;

        int *s_key0 = (int *)&key0;
        int *s_key1 = (int *)&key1;
        int *s_key2 = (int *)&key2;
        int *s_key3 = (int *)&key3;

        double sol0[4] = {1, 2, 3, 4};
        double sol1[4] = {5, 6, 7, 8};
        double sol2[4] = {9, 10, 11, 12};
        double sol3[4] = {13, 14, 15, 16};

        int sol_key0[4] = {0, 1, 2, 3};
        int sol_key1[4] = {4, 5, 6, 7};
        int sol_key2[4] = {8, 9, 10, 11};
        int sol_key3[4] = {12, 13, 14, 15};

        for (int i = 0; i < 4; i++) {

            ASSERT_EQ(*(s0 + i), sol0[i]);
            ASSERT_EQ(*(s1 + i), sol1[i]);
            ASSERT_EQ(*(s2 + i), sol2[i]);
            ASSERT_EQ(*(s3 + i), sol3[i]);

            ASSERT_EQ(*(s_key0 + i), sol_key0[i]);
            ASSERT_EQ(*(s_key1 + i), sol_key1[i]);
            ASSERT_EQ(*(s_key2 + i), sol_key2[i]);
            ASSERT_EQ(*(s_key3 + i), sol_key3[i]);
        }
    }

    for (int i = 0; i < 10; i++) {
        unsigned size = 16;
        std::vector<double> inp;
        inp.reserve(size);
        std::vector<std::pair<int, double>> pairs;
        pairs.reserve(8);

        for (int i = 0; i < size; i++) {
            inp.push_back(random_double());
            pairs.push_back(std::pair<int, double>(i, inp[i]));
        }

        __m256d reg0 = _mm256_loadu_pd(inp.data());
        __m256d reg1 = _mm256_loadu_pd(inp.data() + 4);
        __m256d reg2 = _mm256_loadu_pd(inp.data() + 8);
        __m256d reg3 = _mm256_loadu_pd(inp.data() + 12);

        __m128i key0 = _mm_setr_epi32(0, 1, 2, 3);
        __m128i key1 = _mm_setr_epi32(4, 5, 6, 7);
        __m128i key2 = _mm_setr_epi32(8, 9, 10, 11);
        __m128i key3 = _mm_setr_epi32(12, 13, 14, 15);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, double> &a,
                     const std::pair<int, double> &b) {
                      return (a.second) < (b.second);
                  });

        BITONIC_SORT_KEY_VALUE::bitonic_sort(
            reg0, reg1, reg2, reg3, key0, key1, key2, key3);

        double *s0 = (double *)&reg0;
        double *s1 = (double *)&reg1;
        double *s2 = (double *)&reg2;
        double *s3 = (double *)&reg3;

        int *s_key0 = (int *)&key0;
        int *s_key1 = (int *)&key1;
        int *s_key2 = (int *)&key2;
        int *s_key3 = (int *)&key3;

        for (int i = 0; i < 4; i++) {
            ASSERT_EQ(*(s_key0 + i), pairs[i].first);
            ASSERT_EQ(*(s_key1 + i), pairs[i + 4].first);
            ASSERT_EQ(*(s_key2 + i), pairs[i + 8].first);
            ASSERT_EQ(*(s_key3 + i), pairs[i + 12].first);

            ASSERT_EQ(*(s0 + i), pairs[i].second);
            ASSERT_EQ(*(s1 + i), pairs[i + 4].second);
            ASSERT_EQ(*(s2 + i), pairs[i + 8].second);
            ASSERT_EQ(*(s3 + i), pairs[i + 12].second);
        }
    }
}

TEST(SORT, TEST_2N_BITONIC_SORT_KEY_VALUE_INT_DOUBLE_VER) {

    {
        unsigned size = 8;
        aligned_vector<double> inp;
        aligned_vector<int> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<int, double>> pairs;
        pairs.reserve(size);

        for (int i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(i);
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(std::pair<int, double>(i, i));
        }

        BITONIC_SORT_KEY_VALUE::sort_2n_key_value(
            inp.data(), keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, double> &a,
                     const std::pair<int, double> &b) {
                      return (a.second) < (b.second);
                  });

        for (int i = 0; i < 7; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
    {
        unsigned size = 64;
        aligned_vector<double> inp;
        aligned_vector<int> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<int, double>> pairs;
        pairs.reserve(size);

        for (int i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(random_double());
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(std::pair<int, double>(i, inp[i]));
        }

        BITONIC_SORT_KEY_VALUE::sort_2n_key_value(
            inp.data(), keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, double> &a,
                     const std::pair<int, double> &b) {
                      return (a.second) < (b.second);
                  });

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
}

TEST(SORT, TEST_4N_BITONIC_SORT_KEY_VALUE_INT_DOUBLE_VER) {

    {
        unsigned size = 4;
        aligned_vector<double> inp;
        aligned_vector<int> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<int, double>> pairs;
        pairs.reserve(size);

        for (int i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(i);
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(std::pair<int, double>(i, i));
        }

        BITONIC_SORT_KEY_VALUE::sort_4n_key_value(
            inp.data(), keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, double> &a,
                     const std::pair<int, double> &b) {
                      return (a.second) < (b.second);
                  });

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
    {
        unsigned size = 384;
        aligned_vector<double> inp;
        aligned_vector<int> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<int, double>> pairs;
        pairs.reserve(size);

        for (int i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(random_double());
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(std::pair<int, double>(i, inp[i]));
        }

        BITONIC_SORT_KEY_VALUE::sort_4n_key_value(
            inp.data(), keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, double> &a,
                     const std::pair<int, double> &b) {
                      return (a.second) < (b.second);
                  });

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
}

TEST(SORT, TEST_BITONIC_SORT_KEY_VALUE_INT_DOUBLE_VER) {

    for (unsigned size = 1; size <= 2000; size++) {
        aligned_vector<double> inp;
        aligned_vector<int> keys;

        inp.reserve(size);
        keys.reserve(size);

        std::vector<std::pair<int, double>> pairs;
        pairs.reserve(size);

        for (int i = 0; i < size; i++) {
            // inp.push_back(random_float());
            inp.push_back(random_double());
            keys.push_back(i);
            //            pairs.push_back(std::pair<int, float>(i,
            //            inp[i]));
            pairs.push_back(std::pair<int, double>(i, inp[i]));
        }

        BITONIC_SORT_KEY_VALUE::sort_key_value(inp.data(), keys.data(), size);

        std::sort(std::begin(pairs), std::end(pairs),
                  [](const std::pair<int, double> &a,
                     const std::pair<int, double> &b) {
                      return (a.second) < (b.second);
                  });
        /*
        for (int i = 0; i < size; i++) {
            std::cout << inp[i] << " " << pairs[i].second
                      << std::endl;
                      };*/

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp[i], pairs[i].second);
            ASSERT_EQ(keys[i], pairs[i].first);
        }
    }
}

/*
{
    unsigned size = 32;
    std::vector<double> inp;
    inp.reserve(size);
    std::vector<std::pair<int, double>> pairs;
    pairs.reserve(size);

    for (int i = 0; i < size; i++) {
        inp.push_back(random_double());
        pairs.push_back(std::pair<int, double>(i, inp[i]));
    }

    __m256d reg0 = _mm256_loadu_pd(inp.data());
    __m256d reg1 = _mm256_loadu_pd(inp.data() + 4);
    __m256d reg2 = _mm256_loadu_pd(inp.data() + 8);
    __m256d reg3 = _mm256_loadu_pd(inp.data() + 12);

    __m256i key0 = _mm256_setr_epi64x(0, 1, 2, 3);
    __m256i key1 = _mm256_setr_epi64x(4, 5, 6, 7);
    __m256i key2 = _mm256_setr_epi64x(8, 9, 10, 11);
    __m256i key3 = _mm256_setr_epi64x(12, 13, 14, 15);

    std::sort(std::begin(pairs), std::end(pairs),
              [](const std::pair<int, double> &a,
                 const std::pair<int, double> &b) {
                  return (a.second) < (b.second);
              });

    BITONIC_SORT_KEY_VALUE::bitonic_sort(reg0, reg1, reg2, reg3,
key0, key1, key2, key3);

    double *s0 = (double *)&reg0;
    double *s1 = (double *)&reg1;
    double *s2 = (double *)&reg2;
    double *s3 = (double *)&reg3;

    int *s_key0 = (int *)&key0;
    int *s_key1 = (int *)&key1;
    int *s_key2 = (int *)&key2;
    int *s_key3 = (int *)&key3;

    for (int i = 0; i < 8; i++) {
        ASSERT_EQ(*(s_key0 + i), pairs[i].first);
        ASSERT_EQ(*(s_key1 + i), pairs[i + 4].first);
        ASSERT_EQ(*(s_key2 + i), pairs[i + 8].first);
        ASSERT_EQ(*(s_key3 + i), pairs[i + 12].first);

        ASSERT_EQ(*(s0 + i), pairs[i].second);
        ASSERT_EQ(*(s1 + i), pairs[i + 4].second);
        ASSERT_EQ(*(s2 + i), pairs[i + 8].second);
        ASSERT_EQ(*(s3 + i), pairs[i + 12].second);
    }
    }
    }*/

/*

  TEST(SORT, TEST_IMPROVED_BITONIC_SORT_8N_FLOAT_KEY_VALUE) {
    unsigned size = 4000;
    aligned_vector<float> inp0;
    aligned_vector<float> inp1;
    inp0.reserve(size);
    for (unsigned i = 0; i < size; i++)
        inp0.push_back(random_float());

    inp1 = inp0;

    IMPROVED_BITONIC_SORT::sort_8n_key_value(inp1.data(), 0, size -
1);

    std::sort(std::begin(inp0), std::end(inp0));

    for (unsigned i = 0; i < size; i++) {
        ASSERT_EQ(inp1[i], inp0[i]);
    }
}

TEST(SORT, TEST_IMPROVED_BITONIC_SORT_FLOAT_KEY_VALUE_ALL_CASES) {

    for (unsigned size = 1; size < 4000; size++) {
        aligned_vector<float> inp0;
        aligned_vector<float> inp1;
        inp0.reserve(size);
        for (unsigned i = 0; i < size; i++)
            inp0.push_back(random_float());

        inp1 = inp0;
        IMPROVED_BITONIC_SORT::sort_key_value(inp1, 0, size - 1);

        // std::cout << "end of sort_key_value" << std::endl;
        std::sort(std::begin(inp0), std::end(inp0));

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp1[i], inp0[i]);
        }
    }
}

TEST(SORT, IMPROVED_BITONIC_AVX_SORT_REG_DOUBLE) {
    {
        __m256d reg = _mm256_setr_pd(0, 1, 2, 3);
        BITONIC_SORT::bitonic_sort(reg);
        double *sol = (double *)&reg;
        double solution[4] = {0, 1, 2, 3};
        for (int i = 0; i < 4; i++)
            ASSERT_EQ(*(sol + i), solution[i]);
    }
    {
        __m256d reg2 = _mm256_setr_pd(-10, 3, 6, -12);
        BITONIC_SORT::bitonic_sort(reg2);
        double *sol2 = (double *)&reg2;
        double solution2[4] = {-12, -10, 3, 6};
        for (int i = 0; i < 4; i++)
            ASSERT_EQ(*(sol2 + i), solution2[i]);
    }
    {
        __m256d reg3 = _mm256_setr_pd(5, 3, 6, 48);
        BITONIC_SORT::bitonic_sort(reg3);
        double *sol3 = (double *)&reg3;
        double solution3[4] = {3, 5, 6, 48};
        for (int i = 0; i < 4; i++)
            ASSERT_EQ(*(sol3 + i), solution3[i]);
    }
    {
        std::vector<double> inp0({random_double(), random_double(),
                                  random_double(),
random_double()});

        __m256d reg0 = _mm256_loadu_pd(inp0.data());
        std::sort(std::begin(inp0), std::end(inp0));

        BITONIC_SORT::bitonic_sort(reg0);
        double *s1 = (double *)&reg0;
        // std::cout << "inp: " << std::endl;
        for (int i = 0; i < 4; i++) {
            ASSERT_EQ(*(s1 + i), inp0[i]);
        }
    }
}

TEST(SORT, IMPROVED_BITONIC_AVX_SORT_REG2_DOUBLE) {
    {
        __m256d reg0 = _mm256_setr_pd(1, 2, 3, 4);
        __m256d reg1 = _mm256_setr_pd(5, 6, 7, 8);
        IMPROVED_BITONIC_SORT::bitonic_sort(reg0, reg1);
        double *s1 = (double *)&reg0;
        double *s2 = (double *)&reg1;
        double sol1[4] = {1, 2, 3, 4};
        double sol2[4] = {5, 6, 7, 8};
        for (int i = 0; i < 4; i++) {
            ASSERT_EQ(*(s1 + i), sol1[i]);
            ASSERT_EQ(*(s2 + i), sol2[i]);
        }
    }
    {
        __m256d reg0 = _mm256_setr_pd(-1, -2, 18, 29);
        __m256d reg1 = _mm256_setr_pd(500, -80, 2, -5);
        IMPROVED_BITONIC_SORT::bitonic_sort(reg0, reg1);
        double *s1 = (double *)&reg0;
        double *s2 = (double *)&reg1;
        double sol1[4] = {-80, -5, -2, -1};
        double sol2[4] = {2, 18, 29, 500};
        for (int i = 0; i < 4; i++) {
            ASSERT_EQ(*(s1 + i), sol1[i]);
            ASSERT_EQ(*(s2 + i), sol2[i]);
        }
    }
    {
        std::vector<double> inp0({random_double(), random_double(),
                                  random_double(),
random_double()}); std::vector<double> inp1({random_double(),
random_double(), random_double(), random_double()});

        __m256d reg0 = _mm256_loadu_pd(inp0.data());
        __m256d reg1 = _mm256_loadu_pd(inp1.data());

        inp0.insert(inp0.end(), inp1.begin(), inp1.end());
        std::sort(std::begin(inp0), std::end(inp0));

        IMPROVED_BITONIC_SORT::bitonic_sort(reg0, reg1);
        double *s1 = (double *)&reg0;
        double *s2 = (double *)&reg1;
        // std::cout << "inp: " << std::endl;
        for (int i = 0; i < 4; i++) {
            ASSERT_EQ(*(s1 + i), inp0[i]);
            ASSERT_EQ(*(s2 + i), inp0[i + 4]);
        }
    }
}

TEST(SORT, IMPROVED_BITONIC_AVX_SORT_4REG_DOUBLE) {
    {
        __m256d reg0 = _mm256_setr_pd(1, 2, 3, 4);
        __m256d reg1 = _mm256_setr_pd(5, 6, 7, 8);
        __m256d reg2 = _mm256_setr_pd(9, 10, 11, 12);
        __m256d reg3 = _mm256_setr_pd(13, 14, 15, 16);

        IMPROVED_BITONIC_SORT::bitonic_sort(reg0, reg1, reg2, reg3);

        double *s1 = (double *)&reg0;
        double *s2 = (double *)&reg1;
        double *s3 = (double *)&reg2;
        double *s4 = (double *)&reg3;
        double sol1[4] = {1, 2, 3, 4};
        double sol2[4] = {5, 6, 7, 8};
        double sol3[4] = {9, 10, 11, 12};
        double sol4[4] = {13, 14, 15, 16};
        for (int i = 0; i < 4; i++) {
            ASSERT_EQ(*(s1 + i), sol1[i]);
            ASSERT_EQ(*(s2 + i), sol2[i]);
            ASSERT_EQ(*(s3 + i), sol3[i]);
            ASSERT_EQ(*(s4 + i), sol4[i]);
        }
    }
    {
        __m256d reg0 = _mm256_setr_pd(-1, -2, 18, 29);
        __m256d reg1 = _mm256_setr_pd(500, -80, 2, -5);
        __m256d reg2 = _mm256_setr_pd(-10, 22, 180, -2900);
        __m256d reg3 = _mm256_setr_pd(38, -120, 25, -17);
        IMPROVED_BITONIC_SORT::bitonic_sort(reg0, reg1, reg2, reg3);

        double *s1 = (double *)&reg0;
        double *s2 = (double *)&reg1;
        double *s3 = (double *)&reg2;
        double *s4 = (double *)&reg3;
        std::vector<double> sol({-80, -5, -2, -1, -10, 2, 18, 29,
500, 22, 180, -2900, 38, -120, 25, -17}); std::sort(sol.begin(),
sol.end()); for (int i = 0; i < 4; i++) { ASSERT_EQ(*(s1 + i),
sol[i]); ASSERT_EQ(*(s2 + i), sol[i + 4]); ASSERT_EQ(*(s3 + i),
sol[i + 8]); ASSERT_EQ(*(s4 + i), sol[i + 12]);
        }
    }

    {
        std::vector<double> inp0({random_double(), random_double(),
                                  random_double(),
random_double()}); std::vector<double> inp1({random_double(),
random_double(), random_double(), random_double()});
        std::vector<double> inp2({random_double(), random_double(),
                                  random_double(),
random_double()}); std::vector<double> inp3({random_double(),
random_double(), random_double(), random_double()});

        __m256d reg0 = _mm256_loadu_pd(inp0.data());
        __m256d reg1 = _mm256_loadu_pd(inp1.data());
        __m256d reg2 = _mm256_loadu_pd(inp2.data());
        __m256d reg3 = _mm256_loadu_pd(inp3.data());

        inp0.insert(inp0.end(), inp1.begin(), inp1.end());
        inp0.insert(inp0.end(), inp2.begin(), inp2.end());
        inp0.insert(inp0.end(), inp3.begin(), inp3.end());
        std::sort(std::begin(inp0), std::end(inp0));

        IMPROVED_BITONIC_SORT::bitonic_sort(reg0, reg1, reg2, reg3);
        double *s1 = (double *)&reg0;
        double *s2 = (double *)&reg1;
        double *s3 = (double *)&reg2;
        double *s4 = (double *)&reg3;
        // std::cout << "inp: " << std::endl;
        for (int i = 0; i < 4; i++) {
            ASSERT_EQ(*(s1 + i), inp0[i]);
            ASSERT_EQ(*(s2 + i), inp0[i + 4]);
            ASSERT_EQ(*(s3 + i), inp0[i + 8]);
            ASSERT_EQ(*(s4 + i), inp0[i + 12]);
        }
    }
}

TEST(SORT, TEST_2N_IMPROVED_BITONIC_SORT_DOUBLE_VER) {

    {
        aligned_vector<double> inp0(
            {random_float(), random_float(), random_float(),
random_float()}); aligned_vector<double> inp1 = inp0;

        IMPROVED_BITONIC_SORT::sort_2n_key_value(inp0.data(), 0, 3);
        std::sort(std::begin(inp1), std::end(inp1));

        for (int i = 0; i < 4; i++) {
            ASSERT_EQ(inp1[i], inp0[i]);
        }
    }

    {
        aligned_vector<double> inp0;
        aligned_vector<double> inp1;
        inp0.reserve(8);
        for (int i = 0; i < 8; i++)
            inp0.push_back(random_float());

        inp1 = inp0;
        IMPROVED_BITONIC_SORT::sort_2n_key_value(inp1.data(), 0, 7);

        std::sort(std::begin(inp0), std::end(inp0));

        for (int i = 0; i < 8; i++) {
            ASSERT_EQ(inp1[i], inp0[i]);
        }
    }

    {
        unsigned size = 8192;
        aligned_vector<double> inp0;
        aligned_vector<double> inp1;
        inp0.reserve(size);
        for (unsigned i = 0; i < size; i++)
            inp0.push_back(random_float());

        inp1 = inp0;

        IMPROVED_BITONIC_SORT::sort_2n_key_value(inp1.data(), 0,
size - 1);

        std::sort(std::begin(inp0), std::end(inp0));

        for (unsigned i = 0; i < size; i++) {
            ASSERT_EQ(inp1[i], inp0[i]);
        }
    }
}

TEST(SORT, TEST_IMPROVED_BITONIC_SORT_DOUBLE_4n_KEY_VALUE) {

    for (unsigned size = 4; size <= 2000; size += 4) {
        aligned_vector<double> inp0;
        aligned_vector<double> inp1;
        inp0.reserve(size);
        for (unsigned i = 0; i < size; i++)
            inp0.push_back(random_float());

        inp1 = inp0;
        IMPROVED_BITONIC_SORT::sort_4n_key_value(inp1.data(), 0,
size - 1);

        // std::cout << "end of sort_key_value" << std::endl;
        std::sort(std::begin(inp0), std::end(inp0));

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp1[i], inp0[i]);
        }
    }
}
TEST(SORT, IMPROVED_TEST_SORT_DOUBLE_KEY_VALUE_ALL_CASES) {

    for (unsigned size = 1; size < 5000; size++) {

        aligned_vector<double> inp0;
        aligned_vector<double> inp1;
        inp0.reserve(size);
        for (unsigned i = 0; i < size; i++)
            inp0.push_back(random_double());

        inp1 = inp0;
        IMPROVED_BITONIC_SORT::sort_key_value(inp1, 0, size - 1);

        std::sort(std::begin(inp0), std::end(inp0));

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp1[i], inp0[i]);
        }
    }
}
*/
#endif
