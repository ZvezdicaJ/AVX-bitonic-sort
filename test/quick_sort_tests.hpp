#ifdef __AVX2__

//////////////////////////////////////////////////////////////////////////////////
/////// HYBRID SORT TESTS
/////////////////////////////////////////////////////////////////////////////////
/*
TEST(SORT, TEST_HYBRID_8n_FLOAT_SORT) {
    int size = 16;
    aligned_vector<float> inp0(
        {2, 3, 6, 7, 8, 9, 11, 24, 1, -2, -6, 23, 55, -7, -12, -18});
    aligned_vector<float> inp1;
    inp0.reserve(size);
    // for (unsigned i = 0; i < size; i++)
    //    inp0.push_back(random_double());

    inp1 = inp0;
    HYBRID_SORT::hybrid_sort_8n(inp1, 0, size - 1);

    std::sort(std::begin(inp0), std::end(inp0));

    for (int i = 0; i < size; i++) {
        std::cout << inp1[i] << std::endl;
        if (mod8(i + 1) == 0)
            std::cout << "\n";
    }

    for (int i = 0; i < size; i++) {
        ASSERT_EQ(inp1[i], inp0[i]);
    }
    }*/

//////////////////////////////////////////////////////////////////////////////

TEST(SORT, TEST_HYBRID_8n__FLOAT_SORT) {

    for (int size = 0; size < 8 * 500; size += 8) {
        aligned_vector<float> inp0;
        aligned_vector<float> inp1;
        inp0.reserve(size);
        for (unsigned i = 0; i < size; i++)
            inp0.push_back(random_float());

        inp1 = inp0;
        HYBRID_SORT::hybrid_sort_8n(inp1, 0, size - 1);

        std::sort(std::begin(inp0), std::end(inp0));
        /*
        std::cout << "sorted vector:       correct_result:" << std::endl;
        for (int i = 0; i < size; i++) {
            std::cout << inp1[i] << " " << inp0[i] << std::endl;
            if (mod8(i + 1) == 0)
                std::cout << "\n";
                }*/

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp1[i], inp0[i]);
        }
    }
}

TEST(SORT, TEST_SCALAR_QS_FLOAT_SORT) {

    for (int size = 1; size <= 10000; size += 40) {
        aligned_vector<float> inp0;
        aligned_vector<float> inp1;
        inp0.reserve(size);
        for (unsigned i = 0; i < size; i++)
            inp0.push_back(random_float());

        inp1 = inp0;
        HYBRID_SORT::scalar_QS(inp1.data(), 0, size - 1);

        std::sort(std::begin(inp0), std::end(inp0));
        /*
          std::cout << "sorted vector:       correct_result:" << std::endl;
          for (int i = 0; i < size; i++) {
          std::cout << inp1[i] << " " << inp0[i] << std::endl;
          if (mod8(i + 1) == 0)
          std::cout << "\n";
          }*/

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp1[i], inp0[i]);
        }
    }
}

TEST(SORT, TEST_SIMD_QS_SCALAR_PART_FLOAT_SORT) {

    for (int size = 1; size <= 15; size += 1) {
        aligned_vector<float> inp0;
        aligned_vector<float> inp1;
        inp0.reserve(size);
        for (unsigned i = 0; i < size; i++)
            inp0.push_back(random_float());

        inp1 = inp0;
        HYBRID_SORT::simd_QS(inp1, 0, size - 1);

        std::sort(std::begin(inp0), std::end(inp0));
        /*
          std::cout << "sorted vector:       correct_result:" << std::endl;
          for (int i = 0; i < size; i++) {
          std::cout << inp1[i] << " " << inp0[i] << std::endl;
          if (mod8(i + 1) == 0)
          std::cout << "\n";
          }
        */
        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp1[i], inp0[i]);
        }
    }
}

TEST(SORT, TEST_SIMD_QS_SIMD_PART_FLOAT_SORT) {

    for (int size = 17; size <= 5000; size += 1) {
        aligned_vector<float> inp0;
        aligned_vector<float> inp1;
        inp0.reserve(size);
        for (unsigned i = 0; i < size; i++)
            inp0.push_back(random_float());

        inp1 = inp0;
        HYBRID_SORT::simd_QS(inp1, 0, size - 1);

        std::sort(std::begin(inp0), std::end(inp0));
        /*
        std::cout << "sorted vector:       correct_result:" << std::endl;
        for (int i = 0; i < size; i++) {
            std::cout << inp1[i] << " " << inp0[i] << std::endl;
            if (mod8(i + 1) == 0)
                std::cout << "\n";
        }
        */
        for (int i = 0; i < size; i++) {
            ASSERT_EQ(inp1[i], inp0[i]);
        }
    }
}

#endif
