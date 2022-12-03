////////////////////////////////////////////////////////
////////////VALUE:  DOUBLE  KEY: LONG INT  IMPLEMENTATION
////////////////////////////////////////////////////////

/**
 * @brief The function blends two __m256i vectors based on a
 * provided mask.
 * @param a first vector to be blended
 * @param second vector to be blended
 * @param mask mask according to which to blend two vector. For more
 * into check Intel intrinsics guide - _mm256_blendv_ps function
 */
inline __attribute__((always_inline)) __m256i
_mm256_blendv_epi64(const __m256i &a, const __m256i &b,
                    const __m256i &mask) {
    return _mm256_castpd_si256(_mm256_blendv_pd(
        _mm256_castsi256_pd(a), _mm256_castsi256_pd(b),
        _mm256_castsi256_pd(mask)));
}

/**
 * @brief The overload of _mm256_blendv_epi64 function with mask
 * being of  __m256d type.
 * @param a first vector to be blended
 * @param second vector to be blended
 * @param mask mask according to which to blend two vector. For more
 * info check Intel intrinsics guide - _mm256_blendv_ps function.
 */
inline __attribute__((always_inline)) __m256i
_mm256_blendv_epi64(const __m256i &a, const __m256i &b,
                    const __m256d &mask) {
    return _mm256_castpd_si256(_mm256_blendv_pd(
        _mm256_castsi256_pd(a), _mm256_castsi256_pd(b), mask));
}

/**
 * @brief shuffles a register according to uint8_t mask and compares
 * shuffled register to the original one. The resulting permutation
 * is then applied to the key.
 * @param reg register to be shuffled and compared
 * @param key key vector to be shuffled based on the result of
 * comparison of shuffled float register to the original one.
 * @param mask_epi32 helper mask to blend two vector. For more info
 * check Intel intrinsics guide - _mm256_blendv_ps function.
 * @param mask8 mask according to which float register is
 * shuffled.
 */
inline __attribute__((always_inline)) void
shuffle_and_compare(__m256d &reg, __m256i &key,
                    const __m256i &mask_epi64,
                    const uint8_t mask8) {

    __m256d shuffled_reg = _mm256_shuffle_pd(reg, reg, mask8);

    __m256i shuffled_key = _mm256_castpd_si256(_mm256_shuffle_pd(
        _mm256_castsi256_pd(key), _mm256_castsi256_pd(key), mask8));

    __m256d mask = _mm256_cmp_pd(reg, shuffled_reg, _CMP_LE_OQ);

    mask = _mm256_castsi256_pd(
        _mm256_xor_si256(_mm256_castpd_si256(mask), mask_epi64));

    reg = _mm256_blendv_pd(shuffled_reg, reg, mask);
    key = _mm256_blendv_epi64(shuffled_key, key, mask);
}

/**
 * @brief shuffles a register according to uint8_t mask and compares
 * shuffled register to the original one. The resulting permutation
 * is then applied to the key.
 * @param reg register to be shuffled and compared
 * @param key key vector to be shuffled based on the result of
 * comparison of shuffled float register to the original one.
 * @param mask_epi32 helper mask to blend two vector. For more info
 * check Intel intrinsics guide - _mm256_blendv_ps function.
 * @param uint8_t mask according to which float register is
 * shuffled.
 */
inline __attribute__((always_inline)) void
permute_and_compare(__m256d &reg, __m256i &key,
                    const __m256i &mask_epi64,
                    const uint8_t mask8) {

    __m256d permuted_reg = _mm256_permute4x64_pd(reg, mask8);

    __m256i permuted_key = _mm256_castpd_si256(
        _mm256_permute4x64_pd(_mm256_castsi256_pd(key), mask8));

    __m256d mask = _mm256_cmp_pd(reg, permuted_reg, _CMP_LE_OQ);

    mask = _mm256_castsi256_pd(
        _mm256_xor_si256(_mm256_castpd_si256(mask), mask_epi64));

    reg = _mm256_blendv_pd(permuted_reg, reg, mask);
    key = _mm256_blendv_epi64(permuted_key, key, mask);
}

/**
 * @brief The function reverses a register according  and
 * compares reversed register to the original one. The resulting
 * permutation is then applied to the key.
 * @param reg register to be shuffled and compared
 * @param key key vector to be shuffled based on the result of
 * comparison of shuffled float register to the original one.
 * @param mask_epi32 helper mask to blend two vector. For more info
 * check Intel intrinsics guide - _mm256_blendv_ps function.
 */
inline __attribute__((always_inline)) void
reverse_and_compare(__m256d &reg, __m256i &key,
                    const __m256i &mask_epi64) {

    __m256d reversed_reg = _mm256_permute4x64_pd(reg, 0b00011011);
    __m256i reversed_key =
        _mm256_permute4x64_epi64(key, 0b00011011);

    __m256d mask = _mm256_cmp_pd(reg, reversed_reg, _CMP_LE_OQ);

    mask = _mm256_castsi256_pd(
        _mm256_xor_si256(_mm256_castpd_si256(mask), mask_epi64));

    reg = _mm256_blendv_pd(reversed_reg, reg, mask);
    key = _mm256_blendv_epi64(reversed_key, key, mask);
}

/**
 * @brief The function accepts a single __m256d vector and a single
 * __m256i. It sorts the __m256d vector and permutes __m256i vector
 * corresponding to permutations in __m256d vector.
 * @param reg register to be sorted
 */
inline void bitonic_sort(__m256d &reg, __m256i &key) {
    shuffle_and_compare(reg, key, double_masks256.mask1, 0b0101);
    reverse_and_compare(reg, key, double_masks256.mask2);
    shuffle_and_compare(reg, key, double_masks256.mask1, 0b0101);
}

/**
 * @brief The function reverses one of the input registers and
 * compares the reversed register to the other supplied register.
 * The resulting permutation is then applied to the key1 and key2
 * pair.
 * @param reg0 register to be shuffled and compared to be reversed
 * @param reg1 register bo be compared to register reg0
 * @param key1 register of keys to be sorted
 * @param key2 register of keys to be sorted
 */
inline __attribute__((always_inline)) void
reverse_and_compare(__m256d &reg0, __m256d &reg1, __m256i &key0,
                    __m256i &key1) {

    __m256d reversed_reg = _mm256_permute4x64_pd(reg0, 0b00011011);
    __m256i reversed_key =
        _mm256_permute4x64_epi64(key0, 0b00011011);

    __m256d mask = _mm256_cmp_pd(reg1, reversed_reg, _CMP_LE_OQ);

    reg0 = _mm256_blendv_pd(reversed_reg, reg1, mask);
    reg1 = _mm256_blendv_pd(reg1, reversed_reg, mask);

    key0 = _mm256_blendv_epi64(reversed_key, key1,
                               _mm256_castpd_si256(mask));
    key1 = _mm256_blendv_epi64(key1, reversed_key,
                               _mm256_castpd_si256(mask));
}

/**
 * @brief The function accepts two __m256d vectors and sorts them.
 * @param reg1 upper vector of numbers - at the end it contains
 * larger values, the largest value is in the upper half of register
 * [255:192]
 *
 * @param reg0 lower vector of numbers - at the end it contains
 * smaller values. The smallest value is in the lowest half of
 * register - at [63:0]
 * @param key0 vector of integer indices corresponding to reg0
 * vector
 * @param key1 vector of integer indices corresponding to reg1
 * vector
 */
inline __attribute__((always_inline)) void
bitonic_sort(__m256d &reg0, __m256d &reg1, __m256i &key0,
             __m256i &key1) {
    bitonic_sort(reg1, key1); // sort first register
    bitonic_sort(reg0, key0); // sort second register

    reverse_and_compare(reg0, reg1, key0, key1);
    permute_and_compare(reg1, key1, double_masks256.mask2,
                        0b01001110);
    shuffle_and_compare(reg1, key1, double_masks256.mask1, 0b0101);

    permute_and_compare(reg0, key0, double_masks256.mask2,
                        0b01001110);
    shuffle_and_compare(reg0, key0, double_masks256.mask1, 0b0101);
    return;
}

/**
 * @brief The function reverses one of the input registers and
 * compares the reversed register to the other supplied register.
 * The resulting permutation is then applied to the key1 and key2
 * pair.
 * @param reg0 register to be shuffled and compared to be reversed
 * @param reg1 register bo be compared to register reg0
 * @param key1 register of keys to be sorted
 * @param key2 register of keys to be sorted
 */
inline __attribute__((always_inline)) void compare(__m256d &reg0,
                                                   __m256d &reg1,
                                                   __m256i &key0,
                                                   __m256i &key1) {

    __m256d mask = _mm256_cmp_pd(reg1, reg0, _CMP_LE_OQ);

    __m256d tmp = _mm256_blendv_pd(reg0, reg1, mask);
    reg1 = _mm256_blendv_pd(reg1, reg0, mask);
    reg0 = tmp;
    __m256i tmp_key =
        _mm256_blendv_epi64(key0, key1, _mm256_castpd_si256(mask));
    key1 =
        _mm256_blendv_epi64(key1, key0, _mm256_castpd_si256(mask));
    key0 = tmp_key;
}

/**
 * @brief The function accepts two __m256d vectors and sorts them.
 * @param reg1 upper vector of numbers - at the end it contains
 * larger values, the largest value is in the upper half of register
 * [255:192]
 *
 * @param reg0 lower vector of numbers - at the end it contains
 * smaller values. The smallest value is in the lowest half of
 * register - at [63:0]
 *
 */
inline void bitonic_sort(__m256d &reg0, __m256d &reg1,
                         __m256d &reg2, __m256d &reg3,
                         __m256i &key0, __m256i &key1,
                         __m256i &key2, __m256i &key3) {
    bitonic_sort(reg0, key0); // sort first register
    bitonic_sort(reg1, key1); // sort second register
    bitonic_sort(reg2, key2); // sort third register
    bitonic_sort(reg3, key3); // sort fourth register

    bitonic_sort(reg0, reg1, key0, key1); // sort third register
    bitonic_sort(reg2, reg3, key2, key3); // sort fourth register

    reverse_and_compare(reg0, reg3, key0, key3);
    reverse_and_compare(reg1, reg2, key1, key2);

    compare(reg0, reg1, key0, key1);
    compare(reg2, reg3, key2, key3);

    permute_and_compare(reg0, key0, double_masks256.mask2,
                        0b01001110);
    permute_and_compare(reg1, key1, double_masks256.mask2,
                        0b01001110);
    permute_and_compare(reg2, key2, double_masks256.mask2,
                        0b01001110);
    permute_and_compare(reg3, key3, double_masks256.mask2,
                        0b01001110);

    // shuffle neighbour numbers

    shuffle_and_compare(reg0, key0, double_masks256.mask1, 0b0101);
    shuffle_and_compare(reg1, key1, double_masks256.mask1, 0b0101);
    shuffle_and_compare(reg2, key2, double_masks256.mask1, 0b0101);
    shuffle_and_compare(reg3, key3, double_masks256.mask1, 0b0101);

    return;
}

//////////////////////////////////////////////////////////////////
// 2^N KEY:VALUE  DOUBLE:LONG LONG SORTING ALGORITHM

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the array to be sorted
 * @param keys pointer to the array of integers to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 */
inline void compare_full_length(double *arr, long long *keys,
                                unsigned start, unsigned end) {
    unsigned length = end - start + 1;
    unsigned half = length / 2;
    for (unsigned i = 0; i < half; i += 4) {
        double *p1 = arr + start + i;
        double *p2 = arr + end - 3 - i;
        long long *k1 = keys + start + i;
        long long *k2 = keys + end - 3 - i;

        __m256d vec1 = _mm256_loadu_pd(p1);
        __m256d vec2 = _mm256_loadu_pd(p2);
        __m256i key1 = _mm256_loadu_si256((__m256i *)k1);
        __m256i key2 = _mm256_loadu_si256((__m256i *)k2);

        reverse_and_compare(vec1, vec2, key1, key2);

        _mm256_storeu_pd(p1, vec1);
        _mm256_storeu_pd(p2, vec2);
        _mm256_storeu_si256((__m256i *)k1, key1);
        _mm256_storeu_si256((__m256i *)k2, key2);
    }
}

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the array to be sorted
 * @param keys pointer to the array of integers to be permuted
 * according to permutation in vector of floats
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param depth a parameter to follow the depth of recursion
 */
inline void lane_crossing_compare(double *arr, long long *keys,
                                  unsigned start, unsigned end,
                                  unsigned depth) {

    unsigned length = end - start + 1;

    if (length == 4) {
        __m256d reg = _mm256_loadu_pd(arr + start);
        __m256i key = _mm256_loadu_si256((__m256i *)(keys + start));
        // this is the ending case do single vector permutations

        permute_and_compare(reg, key, double_masks256.mask2,
                            0b01001110);

        shuffle_and_compare(reg, key, double_masks256.mask1,
                            0b0101);
        _mm256_storeu_pd(arr + start, reg);
        _mm256_storeu_si256((__m256i *)(keys + start), key);

        return;
    }
    double *p = arr + start;
    long long *k = keys + start;
    for (unsigned i = 0; i < length / 2; i += 4) {
        {
            double *p1 = p + i;
            double *p2 = p + length / 2 + i;
            long long *k1 = k + i;
            long long *k2 = k + length / 2 + i;

            __m256d reg0 = _mm256_loadu_pd(p1); // i-ti od začetka
            __m256d reg1 =
                _mm256_loadu_pd(p2); // ta je prvi čez polovico

            __m256i key0 =
                _mm256_loadu_si256((__m256i *)k1); // i-ti od začetka
            __m256i key1 = _mm256_loadu_si256(
                (__m256i *)k2); // ta je prvi čez polovico

            compare(reg0, reg1, key0, key1);

            _mm256_storeu_pd(p1, reg0);
            _mm256_storeu_pd(p2, reg1);

            _mm256_storeu_si256((__m256i *)k1, key0);
            _mm256_storeu_si256((__m256i *)k2, key1);
        }
    }
    lane_crossing_compare(arr, keys, start, (start + end) / 2,
                          depth + 1);
    lane_crossing_compare(arr, keys, (start + end) / 2 + 1, end,
                          depth + 1);
};

/**
 * @brief The function sorts an array with 2^n elements
 * @param array pointer to the start of float vector
 * @param keys pointer to the start of the vector of integers to be
 * permuted according to permutation in array
 * @param start index of the first number to be sorted
 * @param end index of the last number to be sorted
 * @details end-start+1 should be 2^n
 */
inline void sort_2n_key_value(double *array, long long *keys,
                              unsigned num_to_sort) {

    unsigned end =
        num_to_sort - 1; // number of double numbers to sort
    unsigned vec_count = num_to_sort / 4;
    assert(
        (num_to_sort >= 0 && !(num_to_sort & (num_to_sort - 1))) &&
        "The array to be sorted is not the power of 2!");

    //__m256d *arr = avx_vec.data();
    if (num_to_sort == 4) {

        __m256d vec = _mm256_loadu_pd(array);
        __m256i key = _mm256_loadu_si256((__m256i *)keys);

        bitonic_sort(vec, key);
        _mm256_storeu_pd(array, vec);
        _mm256_storeu_si256((__m256i *)keys, key);

    } else if (num_to_sort == 8) {

        __m256d vec1 = _mm256_loadu_pd(array);
        __m256d vec2 = _mm256_loadu_pd(array + 4);

        __m256i key1 = _mm256_loadu_si256((__m256i *)keys);
        __m256i key2 = _mm256_loadu_si256((__m256i *)(keys + 4));

        bitonic_sort(vec1, vec2, key1, key2);

        _mm256_storeu_pd(array, vec1);
        _mm256_storeu_pd(array + 4, vec2);

        _mm256_storeu_si256((__m256i *)keys, key1);
        _mm256_storeu_si256((__m256i *)(keys + 4), key2);

    } else if (num_to_sort == 16) {

        __m256d vec1 = _mm256_loadu_pd(array);
        __m256d vec2 = _mm256_loadu_pd(array + 4);
        __m256d vec3 = _mm256_loadu_pd(array + 8);
        __m256d vec4 = _mm256_loadu_pd(array + 12);

        __m256i key1 = _mm256_loadu_si256((__m256i *)keys);
        __m256i key2 = _mm256_loadu_si256((__m256i *)(keys + 4));
        __m256i key3 = _mm256_loadu_si256((__m256i *)(keys + 8));
        __m256i key4 = _mm256_loadu_si256((__m256i *)(keys + 12));

        bitonic_sort(vec1, vec2, vec3, vec4, key1, key2, key3,
                     key4);

        _mm256_storeu_pd(array, vec1);
        _mm256_storeu_pd(array + 4, vec2);
        _mm256_storeu_pd(array + 8, vec3);
        _mm256_storeu_pd(array + 12, vec4);

        _mm256_storeu_si256((__m256i *)keys, key1);
        _mm256_storeu_si256((__m256i *)(keys + 4), key2);
        _mm256_storeu_si256((__m256i *)(keys + 8), key3);
        _mm256_storeu_si256((__m256i *)(keys + 12), key4);

    } else if (num_to_sort >= 32) {
        for (unsigned i = 0; i < end; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(array + i);
            __m256i key1 = _mm256_loadu_si256((__m256i *)(keys + i));
            bitonic_sort(vec1, key1);
            _mm256_storeu_pd(array + i, vec1);
            _mm256_storeu_si256((__m256i *)(keys + i), key1);
        }

        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        for (unsigned len = 8; len <= num_to_sort; len *= 2) {
            // inner loop goes over all subdivisions
            for (unsigned n = 0; n < num_to_sort; n += len) {
                compare_full_length(array, keys, n, n + len - 1);
                lane_crossing_compare(array, keys, n, n + len - 1,
                                      0);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////
// KEY:VALUE LONG LONG:DOUBLE SORTING ALGORITHM, the array must
// contain 4n elements

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the start of vector of doubles to be sorted
 * @param keys pointer to the start of vector of long ints to be
 * permuted according to permutations in arr vector of doubles
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param last_index last index of array to be sorted, maximum is
 * array size - 1
 */
inline void compare_full_length(double *arr, long long *keys,
                                unsigned start, unsigned end,
                                unsigned last_index) {

    unsigned length = end - start + 1;
    unsigned half = length / 2; // half je index prvega cez polovico
    for (int i = half - 4; i >= 0; i -= 4) {
        if (UNLIKELY(end - 3 - i > last_index))
            break;
        double *p1 = arr + start + i;
        double *p2 = arr + end - 3 - i;

        long long *k1 = keys + start + i;
        long long *k2 = keys + end - 3 - i;

        {
            __m256d vec1 = _mm256_loadu_pd(p1);
            __m256d vec2 = _mm256_loadu_pd(p2);
            __m256i key1 = _mm256_loadu_si256((__m256i *)k1);
            __m256i key2 = _mm256_loadu_si256((__m256i *)k2);

            reverse_and_compare(vec1, vec2, key1, key2);
            vec1 = _mm256_permute4x64_pd(vec1, 0b00011011);
            key1 = _mm256_permute4x64_epi64(key1, 0b00011011);

            _mm256_storeu_pd(p1, vec1);
            _mm256_storeu_pd(p2, vec2);
            _mm256_storeu_si256((__m256i *)k1, key1);
            _mm256_storeu_si256((__m256i *)k2, key2);
        }
    }
}

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the array of doubles to be sorted
 * @param keys pointer to the array of doubles to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param last_index last index of array to be sorted, maximum is
 * array size - 1
 * @param depth a parameter to follow the depth of recursion
 */
inline void lane_crossing_compare(double *arr, long long *keys,
                                  unsigned start, unsigned end,
                                  unsigned last_index,
                                  unsigned depth) {
    if (start > last_index) {
        return;
    }
    unsigned length = end - start + 1;
    if (length == 4) {
        // this is the ending case do single vector permutations
        __m256d reg = _mm256_loadu_pd(arr + start);
        __m256i key = _mm256_loadu_si256((__m256i *)(keys + start));
        // this is the ending case do single vector permutations

        permute_and_compare(reg, key, double_masks256.mask2,
                            0b01001110);

        shuffle_and_compare(reg, key, double_masks256.mask1,
                            0b0101);
        _mm256_storeu_pd(arr + start, reg);
        _mm256_storeu_si256((__m256i *)(keys + start), key);

        return;
    }
    double *p = arr + start;
    long long *k = keys + start;
    for (unsigned i = 0; i < length / 2; i += 4) {
        if (start + length / 2 + i > last_index)
            break;
        {
            double *p1 = p + i;
            double *p2 = p + length / 2 + i;
            long long *k1 = k + i;
            long long *k2 = k + length / 2 + i;
            __m256i key0 = _mm256_loadu_si256((__m256i *)k1);
            __m256i key1 = _mm256_loadu_si256((__m256i *)k2);
            __m256d reg0 = _mm256_loadu_pd(p1); // i-ti od začetka
            // ta je prvi čez polovico
            // register 2 vsebuje min vrednosti
            __m256d reg1 = _mm256_loadu_pd(p2);

            compare(reg0, reg1, key0, key1);

            _mm256_storeu_pd(p1, reg0);
            _mm256_storeu_pd(p2, reg1);
            _mm256_storeu_si256((__m256i *)(k1), key0);
            _mm256_storeu_si256((__m256i *)(k2), key1);
        }
    }
    lane_crossing_compare(arr, keys, start, (start + end) / 2,
                          last_index, depth + 1);
    lane_crossing_compare(arr, keys, (start + end) / 2 + 1, end,
                          last_index, depth + 1);
};

/**
 * @brief The function sorts an array with 4*n elements
 * @brief The function permutes an array with 4*n elements according
 * to the permutation in array
 * @param array pointer to the start of the array
 * @param start index of the first number to be sorted
 * @param end index of the last number to be sorted
 * @details end-start+1 should be 8*n
 */
inline void sort_4n_key_value(double *array, long long *keys,
                              unsigned num_to_sort) {
    int pow2 = (int)std::ceil(std::log2f(num_to_sort));
    int imaginary_length = (int)std::pow(2, pow2);
    unsigned end = num_to_sort - 1;
    unsigned last_index = end - 3; // last index to be loaded
    assert((num_to_sort >= 0 && (mod4(num_to_sort) == 0)) &&
           "The array to be sorted is not a multiple of 4!");

    if (num_to_sort == 4) {
        __m256d vec = _mm256_loadu_pd(array);
        __m256i key = _mm256_loadu_si256((__m256i *)keys);
        bitonic_sort(vec, key);
        _mm256_storeu_pd(array, vec);
        _mm256_storeu_si256((__m256i *)(keys), key);

    } else if (num_to_sort == 8) {
        __m256d vec1 = _mm256_loadu_pd(array);
        __m256d vec2 = _mm256_loadu_pd(array + 4);
        __m256i key1 = _mm256_loadu_si256((__m256i *)keys);
        __m256i key2 = _mm256_loadu_si256((__m256i *)(keys + 4));

        bitonic_sort(vec1, vec2, key1, key2);

        _mm256_storeu_pd(array, vec1);
        _mm256_storeu_pd(array + 4, vec2);
        _mm256_storeu_si256((__m256i *)(keys), key1);
        _mm256_storeu_si256((__m256i *)(keys + 4), key2);

    } else if (num_to_sort == 16) {
        __m256d vec1 = _mm256_loadu_pd(array);
        __m256d vec2 = _mm256_loadu_pd(array + 4);
        __m256d vec3 = _mm256_loadu_pd(array + 8);
        __m256d vec4 = _mm256_loadu_pd(array + 12);

        __m256i key1 = _mm256_loadu_si256((__m256i *)keys);
        __m256i key2 = _mm256_loadu_si256((__m256i *)(keys + 4));
        __m256i key3 = _mm256_loadu_si256((__m256i *)(keys + 8));
        __m256i key4 = _mm256_loadu_si256((__m256i *)(keys + 12));

        bitonic_sort(vec1, vec2, vec3, vec4, key1, key2, key3,
                     key4);

        _mm256_storeu_pd(array, vec1);
        _mm256_storeu_pd(array + 4, vec2);
        _mm256_storeu_pd(array + 8, vec3);
        _mm256_storeu_pd(array + 12, vec4);

        _mm256_storeu_si256((__m256i *)(keys), key1);
        _mm256_storeu_si256((__m256i *)(keys + 4), key2);
        _mm256_storeu_si256((__m256i *)(keys + 8), key3);
        _mm256_storeu_si256((__m256i *)(keys + 12), key4);

    } else {
        for (unsigned i = 0; i < end; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(array + i);
            __m256i key1 = _mm256_loadu_si256((__m256i *)(keys + i));

            bitonic_sort(vec1, key1);

            _mm256_storeu_pd(array + i, vec1);
            _mm256_storeu_si256((__m256i *)(keys + i), key1);
        }

        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        for (unsigned len = 8; len <= imaginary_length; len *= 2) {
            // std::cout << "len: " << len << std::endl;
            // inner loop goes over all subdivisions
            for (unsigned n = 0; n < imaginary_length; n += len) {
                compare_full_length(array, keys, n, n + len - 1,
                                    last_index);
                lane_crossing_compare(array, keys, n, n + len - 1,
                                      last_index, 0);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////
/// KEY-VALUE : LONG - DOUBLE SORT  FOR ARRAYS OF ARBITRARY LENGTH
////////////////////////////////////////////////////////////////////

inline void maskload(int diff, double *p2, long long *k2,
                     __m256i &mask, __m256d &reg1, __m256i &key1) {
    switch (diff) {
    case 0: {
        mask = double_load_masks256.mask1;
        reg1 = _mm256_maskload_pd(p2, mask);
        key1 = _mm256_maskload_epi64(k2, mask);

        reg1 = _mm256_blend_pd(m256d_infinity, reg1, 0b0001);
    } break;
    case 1: {
        mask = double_load_masks256.mask2;
        reg1 = _mm256_maskload_pd(p2, mask);
        key1 = _mm256_maskload_epi64(k2, mask);

        reg1 = _mm256_blend_pd(m256d_infinity, reg1, 0b0011);
    } break;
    case 2: {
        mask = double_load_masks256.mask3;
        reg1 = _mm256_maskload_pd(p2, mask);
        key1 = _mm256_maskload_epi64(k2, mask);

        reg1 = _mm256_blend_pd(m256d_infinity, reg1, 0b0111);
    } break;
    };
}

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the float array to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 */
inline void compare_full_length_all_cases(double *arr,
                                          long long *keys,
                                          unsigned start,
                                          unsigned end,
                                          unsigned last_index) {
    // std::cout << "start: " << start << "end: " << end
    //          << " last index: " << last_index << std::endl;
    unsigned length = end - start + 1;
    unsigned half = length / 2; // half je index prvega cez polovico
    for (int i = half - 4; i >= 0; i -= 4) {
        // index of last vector to load
        int last_vec_to_load = end - 3 - i;
        int diff = last_index - last_vec_to_load;
        if (diff < 0)
            return;
        // define pointers to the start of vectors to load
        double *p1 = arr + start + i;
        double *p2 = arr + last_vec_to_load;
        long long *k1 = keys + start + i;
        long long *k2 = keys + last_vec_to_load;
        __m256d vec2;
        __m256i mask, key2;
        if (diff < 3)
            maskload(diff, p2, k2, mask, vec2, key2);
        else {
            vec2 = _mm256_loadu_pd(p2);
            key2 = _mm256_loadu_si256((__m256i *)k2);
        }
        __m256d vec1 = _mm256_loadu_pd(p1);
        __m256i key1 = _mm256_loadu_si256((__m256i *)k1);

        reverse_and_compare(vec1, vec2, key1, key2);
        vec1 = _mm256_permute4x64_pd(vec1, 0b00011011);
        key1 = _mm256_permute4x64_epi64(key1, 0b00011011);

        _mm256_storeu_pd(p1, vec1);
        _mm256_storeu_si256((__m256i *)k1, key1);

        if (diff < 3) {
            _mm256_maskstore_pd(p2, mask, vec2);
            _mm256_maskstore_epi64(k2, mask, key2);
        } else {
            _mm256_storeu_pd(p2, vec2);
            _mm256_storeu_si256((__m256i *)k2, key2);
        }
    }
}

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the array to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param depth a parameter to follow the depth of recursion
 */
inline void lane_crossing_compare_all_cases(
    double *arr, long long *keys, unsigned start, unsigned end,
    unsigned last_index, unsigned depth) {
    if (start > last_index) {
        return;
    }
    unsigned length = end - start + 1;
    if (length == 4) {
        int diff = last_index - start;
        if (diff < 1)
            return;
        __m256d reg;
        __m256i load_store_mask, key;
        if (UNLIKELY(diff < 3))
            maskload(diff, arr + start, keys + start,
                     load_store_mask, reg, key);
        else {
            reg = _mm256_loadu_pd(arr + start);
            key = _mm256_loadu_si256((__m256i *)(keys + start));
        }

        permute_and_compare(reg, key, double_masks256.mask2,
                            0b01001110);

        shuffle_and_compare(reg, key, double_masks256.mask1,
                            0b0101);

        if (UNLIKELY(diff < 3)) {
            _mm256_maskstore_pd(arr + start, load_store_mask, reg);
            _mm256_maskstore_epi64(keys + start, load_store_mask,
                                   key);
        } else {
            _mm256_storeu_pd(arr + start, reg);
            _mm256_storeu_si256((__m256i *)(keys + start), key);
        }
        return;
    }

    double *p = arr + start;
    long long *k = keys + start;
    for (unsigned i = 0; i < length / 2; i += 4) {
        int diff = last_index - (start + length / 2 + i);
        if (diff < 0)
            break;
        __m256d reg1;
        __m256i mask, key1;
        double *p2 = p + length / 2 + i;
        long long *k2 = k + length / 2 + i;
        if (diff < 3)
            maskload(diff, p2, k2, mask, reg1, key1);
        else {
            reg1 = _mm256_loadu_pd(p2);
            key1 = _mm256_loadu_si256((__m256i *)k2);
        }
        double *p1 = p + i;
        long long *k1 = k + i;
        __m256d reg0 = _mm256_loadu_pd(p1); // i-ti od začetka
        __m256i key0 = _mm256_loadu_si256((__m256i *)k1);
        // register 2 vsebuje min vrednosti

        compare(reg0, reg1, key0, key1);

        _mm256_storeu_pd(p1, reg0);
        _mm256_storeu_si256((__m256i *)k1, key0);

        if (diff < 3) {
            _mm256_maskstore_pd(p2, mask, reg1);
            _mm256_maskstore_epi64(k2, mask, key1);
        } else {
            _mm256_storeu_pd(p2, reg1);
            _mm256_storeu_si256((__m256i *)k2, key1);
        }
    }

    lane_crossing_compare_all_cases(
        arr, keys, start, (start + end) / 2, last_index, depth + 1);

    lane_crossing_compare_all_cases(arr, keys,
                                    (start + end) / 2 + 1, end,
                                    last_index, depth + 1);
};

/**
 * @brief The function sorts an array with 2^n elements
 * @param array pointer to the start of the array
 * @param start index of the first number to be sorted
 * @param end index of the last number to be sorted
 * @details end - start + 1 should be 2^n
 */
inline void sort_key_value(double *array, long long *keys,
                           unsigned num_to_sort) {
    unsigned end = num_to_sort - 1;
    if (num_to_sort <= 1)
        return;
    else if (num_to_sort < 4) {
        int reminder = mod4(end);
        double *p = array + end - reminder;
        long long *k = keys + end - reminder;
        __m256d reg1;
        __m256i mask, key1;
        maskload(reminder, p, k, mask, reg1, key1);
        bitonic_sort(reg1, key1);
        _mm256_maskstore_pd(array + end - reminder, mask, reg1);
        _mm256_maskstore_epi64(keys + end - reminder, mask, key1);
    } else if (!(num_to_sort & (num_to_sort - 1)))
        sort_2n_key_value(array, keys, num_to_sort);
    else if (mod4(num_to_sort) == 0)
        sort_4n_key_value(array, keys, num_to_sort);
    else {
        int pow2 = (int)std::ceil(std::log2f(end + 1));
        int imaginary_length = (int)std::pow(2, pow2);
        unsigned end = num_to_sort - 1;
        unsigned last_index = end;

        for (unsigned i = 0; i <= end - 3; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(array + i);
            __m256i key1 = _mm256_loadu_si256((__m256i *)(keys + i));
            bitonic_sort(vec1, key1);
            _mm256_storeu_pd(array + i, vec1);
            _mm256_storeu_si256((__m256i *)(keys + i), key1);
        }
        ///////////////////////////////// load the partial one
        int reminder = mod4(end);
        double *p = array + end - reminder;
        long long *k = keys + end - reminder;
        __m256d reg1;
        __m256i mask, key1;
        maskload(reminder, p, k, mask, reg1, key1);
        bitonic_sort(reg1, key1);
        _mm256_maskstore_pd(array + end - reminder, mask, reg1);
        _mm256_maskstore_epi64(keys + end - reminder, mask, key1);

        ///////////////////////////////////////////////////////

        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        // 8 and less has already been done above
        // for (unsigned len = 16; len <= imaginary_length; len *=
        // 2) {
        for (unsigned len = 8; len <= imaginary_length; len *= 2) {
            // inner loop goes over all subdivisions
            for (unsigned n = 0; n < imaginary_length; n += len) {
                compare_full_length_all_cases(
                    array, keys, n, n + len - 1, last_index);
                lane_crossing_compare_all_cases(
                    array, keys, n, n + len - 1, last_index, 0);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
// DOUBLE:INT ARRAY:KEY SORTING ALGORITHM FOR ARRAYS OF
// ARBITRARY LENGTH

/**
 * @brief The overload of _mm256_blendv_epi64 function with mask
 * being of  __m256d type.
 * @param a first vector to be blended
 * @param second vector to be blended
 * @param mask mask according to which to blend two vector. For more
 * info check Intel intrinsics guide - _mm256_blendv_ps function.
 */
inline __attribute__((always_inline)) __m128i
_mm_blendv_epi32(const __m128i &a, const __m128i &b,
                 const __m256d &mask) {
    return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(a),
                                          _mm_castsi128_ps(b),
                                          _mm256_cvtpd_ps(mask)));
}

/**
 * @brief shuffles a register according to uint8_t mask and
 * compares shuffled register to the original one. The resulting
 * permutation is then applied to the key.
 * @param reg register to be shuffled and compared
 * @param key key vector to be shuffled based on the result of
 * comparison of shuffled float register to the original one.
 * @param mask_epi32 helper mask to blend two vector. For more
 * info check Intel intrinsics guide - _mm256_blendv_ps
 * function.
 * @param mask8 mask according to which float register is
 * shuffled.
 */
inline __attribute__((always_inline)) void
shuffle_and_compare(__m256d &reg, __m128i &key,
                    const __m256i &mask_epi64,
                    const uint8_t mask256, const uint8_t mask128) {

    __m256d shuffled_reg = _mm256_shuffle_pd(reg, reg, mask256);

    __m128i shuffled_key = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(key), _mm_castsi128_ps(key), mask128));

    __m256d mask = _mm256_cmp_pd(reg, shuffled_reg, _CMP_LE_OQ);

    mask = _mm256_castsi256_pd(
        _mm256_xor_si256(_mm256_castpd_si256(mask), mask_epi64));

    reg = _mm256_blendv_pd(shuffled_reg, reg, mask);
    key = _mm_blendv_epi32(shuffled_key, key, mask);
}

/**
 * @brief shuffles a register according to uint8_t mask and compares
 * shuffled register to the original one. The resulting permutation
 * is then applied to the key.
 * @param reg register to be shuffled and compared
 * @param key key vector to be shuffled based on the result of
 * comparison of shuffled float register to the original one.
 * @param mask_epi32 helper mask to blend two vector. For more info
 * check Intel intrinsics guide - _mm256_blendv_ps function.
 * @param uint8_t mask according to which float register is
 * shuffled.
 */
inline __attribute__((always_inline)) void
permute_and_compare(__m256d &reg, __m128i &key,
                    const __m256i &mask_epi64,
                    const uint8_t mask8) {

    __m256d permuted_reg = _mm256_permute4x64_pd(reg, mask8);
    __m128i permuted_key = _mm_castps_si128(
        _mm_permute_ps(_mm_castsi128_ps(key), mask8));

    __m256d mask = _mm256_cmp_pd(reg, permuted_reg, _CMP_LE_OQ);

    mask = _mm256_castsi256_pd(
        _mm256_xor_si256(_mm256_castpd_si256(mask), mask_epi64));

    reg = _mm256_blendv_pd(permuted_reg, reg, mask);
    key = _mm_blendv_epi32(permuted_key, key, mask);
}

/**
 * @brief The function reverses a register according  and
 * compares reversed register to the original one. The resulting
 * permutation is then applied to the key.
 * @param reg register to be shuffled and compared
 * @param key key vector to be shuffled based on the result of
 * comparison of shuffled float register to the original one.
 * @param mask_epi32 helper mask to blend two vector. For more info
 * check Intel intrinsics guide - _mm256_blendv_ps function.
 */
inline __attribute__((always_inline)) void
reverse_and_compare(__m256d &reg, __m128i &key,
                    const __m256i &mask_epi64) {

    __m256d reversed_reg = _mm256_permute4x64_pd(reg, 0b00011011);
    __m128i reversed_key = _mm_castps_si128(
        _mm_permute_ps(_mm_castsi128_ps(key), 0b00011011));

    __m256d mask = _mm256_cmp_pd(reg, reversed_reg, _CMP_LE_OQ);

    mask = _mm256_castsi256_pd(
        _mm256_xor_si256(_mm256_castpd_si256(mask), mask_epi64));

    reg = _mm256_blendv_pd(reversed_reg, reg, mask);
    key = _mm_blendv_epi32(reversed_key, key, mask);
}

/**
 * @brief The function accepts a single __m256d vector and a single
 * __m256i. It sorts the __m256d vector and permutes __m256i vector
 * corresponding to permutations in __m256d vector.
 * @param reg register to be sorted
 */
inline void bitonic_sort(__m256d &reg, __m128i &key) {

    shuffle_and_compare(reg, key, double_masks256.mask1, 0b0101,
                        0b10110001);

    reverse_and_compare(reg, key, double_masks256.mask2);
    shuffle_and_compare(reg, key, double_masks256.mask1, 0b0101,
                        0b10110001);
}

/**
 * @brief The function reverses one of the input registers and
 * compares the reversed register to the other supplied register.
 * The resulting permutation is then applied to the key1 and key2
 * pair.
 * @param reg0 register to be shuffled and compared to be reversed
 * @param reg1 register bo be compared to register reg0
 * @param key1 register of keys to be sorted
 * @param key2 register of keys to be sorted
 */
inline __attribute__((always_inline)) void
reverse_and_compare(__m256d &reg0, __m256d &reg1, __m128i &key0,
                    __m128i &key1) {

    __m256d reversed_reg = _mm256_permute4x64_pd(reg0, 0b00011011);
    __m128i reversed_key = _mm_castps_si128(
        _mm_permute_ps(_mm_castsi128_ps(key0), 0b00011011));

    __m256d mask = _mm256_cmp_pd(reg1, reversed_reg, _CMP_LE_OQ);

    reg0 = _mm256_blendv_pd(reversed_reg, reg1, mask);
    reg1 = _mm256_blendv_pd(reg1, reversed_reg, mask);

    key0 = _mm_blendv_epi32(reversed_key, key1, mask);
    key1 = _mm_blendv_epi32(key1, reversed_key, mask);
}

/**
 * @brief The function accepts two __m256d vectors and sorts them.
 * @param reg1 upper vector of numbers - at the end it contains
 * larger values, the largest value is in the upper half of register
 * [255:192]
 *
 * @param reg0 lower vector of numbers - at the end it contains
 * smaller values. The smallest value is in the lowest half of
 * register - at [63:0]
 * @param key0 vector of integer indices corresponding to reg0
 * vector
 * @param key1 vector of integer indices corresponding to reg1
 * vector
 */
inline __attribute__((always_inline)) void
bitonic_sort(__m256d &reg0, __m256d &reg1, __m128i &key0,
             __m128i &key1) {
    bitonic_sort(reg1, key1); // sort first register
    bitonic_sort(reg0, key0); // sort second register

    reverse_and_compare(reg0, reg1, key0, key1);
    permute_and_compare(reg1, key1, double_masks256.mask2,
                        0b01001110);

    shuffle_and_compare(reg1, key1, double_masks256.mask1, 0b0101,
                        0b10110001);

    permute_and_compare(reg0, key0, double_masks256.mask2,
                        0b01001110);
    shuffle_and_compare(reg0, key0, double_masks256.mask1, 0b0101,
                        0b10110001);
    return;
}

/**
 * @brief The function reverses one of the input registers and
 * compares the reversed register to the other supplied register.
 * The resulting permutation is then applied to the key1 and key2
 * pair.
 * @param reg0 register to be shuffled and compared to be reversed
 * @param reg1 register bo be compared to register reg0
 * @param key1 register of keys to be sorted
 * @param key2 register of keys to be sorted
 */
inline __attribute__((always_inline)) void compare(__m256d &reg0,
                                                   __m256d &reg1,
                                                   __m128i &key0,
                                                   __m128i &key1) {

    __m256d mask = _mm256_cmp_pd(reg1, reg0, _CMP_LE_OQ);

    __m256d tmp = _mm256_blendv_pd(reg0, reg1, mask);
    reg1 = _mm256_blendv_pd(reg1, reg0, mask);
    reg0 = tmp;
    __m128i tmp_key = _mm_blendv_epi32(key0, key1, mask);
    key1 = _mm_blendv_epi32(key1, key0, mask);
    key0 = tmp_key;
}

/**
 * @brief The function accepts two __m256d vectors and sorts them.
 * @param reg1 upper vector of numbers - at the end it contains
 * larger values, the largest value is in the upper half of register
 * [255:192]
 *
 * @param reg0 lower vector of numbers - at the end it contains
 * smaller values. The smallest value is in the lowest half of
 * register - at [63:0]
 *
 */
inline void bitonic_sort(__m256d &reg0, __m256d &reg1,
                         __m256d &reg2, __m256d &reg3,
                         __m128i &key0, __m128i &key1,
                         __m128i &key2, __m128i &key3) {
    bitonic_sort(reg0, key0); // sort first register
    bitonic_sort(reg1, key1); // sort second register
    bitonic_sort(reg2, key2); // sort third register
    bitonic_sort(reg3, key3); // sort fourth register

    bitonic_sort(reg0, reg1, key0, key1); // sort third register
    bitonic_sort(reg2, reg3, key2, key3); // sort fourth register

    reverse_and_compare(reg0, reg3, key0, key3);
    reverse_and_compare(reg1, reg2, key1, key2);

    compare(reg0, reg1, key0, key1);
    compare(reg2, reg3, key2, key3);

    permute_and_compare(reg0, key0, double_masks256.mask2,
                        0b01001110);
    permute_and_compare(reg1, key1, double_masks256.mask2,
                        0b01001110);
    permute_and_compare(reg2, key2, double_masks256.mask2,
                        0b01001110);
    permute_and_compare(reg3, key3, double_masks256.mask2,
                        0b01001110);

    // shuffle neighbour numbers

    shuffle_and_compare(reg0, key0, double_masks256.mask1, 0b0101,
                        0b10110001);
    shuffle_and_compare(reg1, key1, double_masks256.mask1, 0b0101,
                        0b10110001);
    shuffle_and_compare(reg2, key2, double_masks256.mask1, 0b0101,
                        0b10110001);
    shuffle_and_compare(reg3, key3, double_masks256.mask1, 0b0101,
                        0b10110001);

    return;
}

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the array to be sorted
 * @param keys pointer to the array of integers to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 */
inline void compare_full_length(double *arr, int *keys,
                                unsigned start, unsigned end) {
    unsigned length = end - start + 1;
    unsigned half = length / 2;
    for (unsigned i = 0; i < half; i += 4) {
        double *p1 = arr + start + i;
        double *p2 = arr + end - 3 - i;
        int *k1 = keys + start + i;
        int *k2 = keys + end - 3 - i;

        __m256d vec1 = _mm256_loadu_pd(p1);
        __m256d vec2 = _mm256_loadu_pd(p2);
        __m128i key1 = _mm_load_si128((__m128i *)k1);
        __m128i key2 = _mm_load_si128((__m128i *)k2);

        reverse_and_compare(vec1, vec2, key1, key2);

        _mm256_storeu_pd(p1, vec1);
        _mm256_storeu_pd(p2, vec2);
        _mm_store_si128((__m128i *)k1, key1);
        _mm_store_si128((__m128i *)k2, key2);
    }
}

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the array to be sorted
 * @param keys pointer to the array of integers to be permuted
 * according to permutation in vector of floats
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param depth a parameter to follow the depth of recursion
 */
inline void lane_crossing_compare(double *arr, int *keys,
                                  unsigned start, unsigned end,
                                  unsigned depth) {

    unsigned length = end - start + 1;

    if (length == 4) {
        __m256d reg = _mm256_loadu_pd(arr + start);
        __m128i key = _mm_load_si128((__m128i *)(keys + start));
        // this is the ending case do single vector permutations

        permute_and_compare(reg, key, double_masks256.mask2,
                            0b01001110);

        shuffle_and_compare(reg, key, double_masks256.mask1, 0b0101,
                            0b10110001);
        _mm256_storeu_pd(arr + start, reg);
        _mm_store_si128((__m128i *)(keys + start), key);

        return;
    }
    double *p = arr + start;
    int *k = keys + start;
    for (unsigned i = 0; i < length / 2; i += 4) {
        {
            double *p1 = p + i;
            double *p2 = p + length / 2 + i;
            int *k1 = k + i;
            int *k2 = k + length / 2 + i;

            __m256d reg0 = _mm256_loadu_pd(p1); // i-ti od začetka
            __m256d reg1 =
                _mm256_loadu_pd(p2); // ta je prvi čez polovico

            __m128i key0 =
                _mm_load_si128((__m128i *)k1); // i-ti od začetka
            __m128i key1 = _mm_load_si128(
                (__m128i *)k2); // ta je prvi čez polovico

            compare(reg0, reg1, key0, key1);

            _mm256_storeu_pd(p1, reg0);
            _mm256_storeu_pd(p2, reg1);

            _mm_store_si128((__m128i *)k1, key0);
            _mm_store_si128((__m128i *)k2, key1);
        }
    }
    lane_crossing_compare(arr, keys, start, (start + end) / 2,
                          depth + 1);
    lane_crossing_compare(arr, keys, (start + end) / 2 + 1, end,
                          depth + 1);
};

/**
 * @brief The function sorts an array with 2^n elements
 * @param array pointer to the start of float vector
 * @param keys pointer to the start of the vector of integers to be
 * permuted according to permutation in array
 * @param start index of the first number to be sorted
 * @param end index of the last number to be sorted
 * @details end-start+1 should be 2^n
 */
inline void sort_2n_key_value(double *array, int *keys,
                              unsigned num_to_sort) {

    unsigned end =
        num_to_sort - 1; // number of double numbers to sort
    unsigned vec_count = num_to_sort / 4;
    assert(
        (num_to_sort >= 0 && !(num_to_sort & (num_to_sort - 1))) &&
        "The array to be sorted is not the power of 2!");

    //__m256d *arr = avx_vec.data();
    if (num_to_sort == 4) {

        __m256d vec = _mm256_loadu_pd(array);
        __m128i key = _mm_load_si128((__m128i *)keys);

        bitonic_sort(vec, key);
        _mm256_storeu_pd(array, vec);
        _mm_store_si128((__m128i *)keys, key);

    } else if (num_to_sort == 8) {

        __m256d vec1 = _mm256_loadu_pd(array);
        __m256d vec2 = _mm256_loadu_pd(array + 4);

        __m128i key1 = _mm_load_si128((__m128i *)keys);
        __m128i key2 = _mm_load_si128((__m128i *)(keys + 4));

        bitonic_sort(vec1, vec2, key1, key2);

        _mm256_storeu_pd(array, vec1);
        _mm256_storeu_pd(array + 4, vec2);

        _mm_store_si128((__m128i *)keys, key1);
        _mm_store_si128((__m128i *)(keys + 4), key2);

    } else if (num_to_sort == 16) {

        __m256d vec1 = _mm256_loadu_pd(array);
        __m256d vec2 = _mm256_loadu_pd(array + 4);
        __m256d vec3 = _mm256_loadu_pd(array + 8);
        __m256d vec4 = _mm256_loadu_pd(array + 12);

        __m128i key1 = _mm_load_si128((__m128i *)keys);
        __m128i key2 = _mm_load_si128((__m128i *)(keys + 4));
        __m128i key3 = _mm_load_si128((__m128i *)(keys + 8));
        __m128i key4 = _mm_load_si128((__m128i *)(keys + 12));

        bitonic_sort(vec1, vec2, vec3, vec4, key1, key2, key3,
                     key4);

        _mm256_storeu_pd(array, vec1);
        _mm256_storeu_pd(array + 4, vec2);
        _mm256_storeu_pd(array + 8, vec3);
        _mm256_storeu_pd(array + 12, vec4);

        _mm_store_si128((__m128i *)keys, key1);
        _mm_store_si128((__m128i *)(keys + 4), key2);
        _mm_store_si128((__m128i *)(keys + 8), key3);
        _mm_store_si128((__m128i *)(keys + 12), key4);

    } else if (num_to_sort >= 32) {
        for (unsigned i = 0; i < end; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(array + i);
            __m128i key1 = _mm_load_si128((__m128i *)(keys + i));
            bitonic_sort(vec1, key1);
            _mm256_storeu_pd(array + i, vec1);
            _mm_store_si128((__m128i *)(keys + i), key1);
        }

        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        for (unsigned len = 8; len <= num_to_sort; len *= 2) {
            // inner loop goes over all subdivisions
            for (unsigned n = 0; n < num_to_sort; n += len) {
                compare_full_length(array, keys, n, n + len - 1);
                lane_crossing_compare(array, keys, n, n + len - 1,
                                      0);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////
// KEY:VALUE INT:DOUBLE SORTING ALGORITHM, the array must
// contain 4n elements

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the start of vector of doubles to be sorted
 * @param keys pointer to the start of vector of long ints to be
 * permuted according to permutations in arr vector of doubles
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param last_index last index of array to be sorted, maximum is
 * array size - 1
 */
inline void compare_full_length(double *arr, int *keys,
                                unsigned start, unsigned end,
                                unsigned last_index) {

    unsigned length = end - start + 1;
    unsigned half = length / 2; // half je index prvega cez polovico
    for (int i = half - 4; i >= 0; i -= 4) {
        if (UNLIKELY(end - 3 - i > last_index))
            break;
        double *p1 = arr + start + i;
        double *p2 = arr + end - 3 - i;

        int *k1 = keys + start + i;
        int *k2 = keys + end - 3 - i;

        {
            __m256d vec1 = _mm256_loadu_pd(p1);
            __m256d vec2 = _mm256_loadu_pd(p2);
            __m128i key1 = _mm_load_si128((__m128i *)k1);
            __m128i key2 = _mm_load_si128((__m128i *)k2);

            reverse_and_compare(vec1, vec2, key1, key2);
            vec1 = _mm256_permute4x64_pd(vec1, 0b00011011);
            key1 = _mm_castps_si128(
                _mm_permute_ps(_mm_castsi128_ps(key1), 0b00011011));

            _mm256_storeu_pd(p1, vec1);
            _mm256_storeu_pd(p2, vec2);
            _mm_store_si128((__m128i *)k1, key1);
            _mm_store_si128((__m128i *)k2, key2);
        }
    }
}

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the array of doubles to be sorted
 * @param keys pointer to the array of doubles to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param last_index last index of array to be sorted, maximum is
 * array size - 1
 * @param depth a parameter to follow the depth of recursion
 */
inline void lane_crossing_compare(double *arr, int *keys,
                                  unsigned start, unsigned end,
                                  unsigned last_index,
                                  unsigned depth) {
    if (start > last_index) {
        return;
    }
    unsigned length = end - start + 1;
    if (length == 4) {
        // this is the ending case do single vector permutations
        __m256d reg = _mm256_loadu_pd(arr + start);
        __m128i key = _mm_load_si128((__m128i *)(keys + start));
        // this is the ending case do single vector permutations

        permute_and_compare(reg, key, double_masks256.mask2,
                            0b01001110);

        shuffle_and_compare(reg, key, double_masks256.mask1, 0b0101,
                            0b10110001);
        _mm256_storeu_pd(arr + start, reg);
        _mm_store_si128((__m128i *)(keys + start), key);

        return;
    }
    double *p = arr + start;
    int *k = keys + start;
    for (unsigned i = 0; i < length / 2; i += 4) {
        if (start + length / 2 + i > last_index)
            break;
        {
            double *p1 = p + i;
            double *p2 = p + length / 2 + i;
            int *k1 = k + i;
            int *k2 = k + length / 2 + i;
            __m128i key0 = _mm_load_si128((__m128i *)k1);
            __m128i key1 = _mm_load_si128((__m128i *)k2);
            __m256d reg0 = _mm256_loadu_pd(p1); // i-ti od začetka
            // ta je prvi čez polovico
            // register 2 vsebuje min vrednosti
            __m256d reg1 = _mm256_loadu_pd(p2);

            compare(reg0, reg1, key0, key1);

            _mm256_storeu_pd(p1, reg0);
            _mm256_storeu_pd(p2, reg1);
            _mm_store_si128((__m128i *)(k1), key0);
            _mm_store_si128((__m128i *)(k2), key1);
        }
    }
    lane_crossing_compare(arr, keys, start, (start + end) / 2,
                          last_index, depth + 1);
    lane_crossing_compare(arr, keys, (start + end) / 2 + 1, end,
                          last_index, depth + 1);
};

/**
 * @brief The function sorts an array with 4*n elements
 * @brief The function permutes an array with 4*n elements according
 * to the permutation in array
 * @param array pointer to the start of the array
 * @param start index of the first number to be sorted
 * @param end index of the last number to be sorted
 * @details end-start+1 should be 8*n
 */
inline void sort_4n_key_value(double *array, int *keys,
                              unsigned num_to_sort) {

    int pow2 = (int)std::ceil(std::log2f(num_to_sort));
    int imaginary_length = (int)std::pow(2, pow2);
    unsigned end = num_to_sort - 1;
    unsigned last_index = end - 3; // last index to be loaded
    assert((num_to_sort >= 0 && (mod4(num_to_sort) == 0)) &&
           "The array to be sorted is not a multiple of 4!");

    if (num_to_sort == 4) {
        __m256d vec = _mm256_loadu_pd(array);
        __m128i key = _mm_load_si128((__m128i *)keys);
        bitonic_sort(vec, key);
        _mm256_storeu_pd(array, vec);
        _mm_store_si128((__m128i *)(keys), key);

    } else if (num_to_sort == 8) {
        __m256d vec1 = _mm256_loadu_pd(array);
        __m256d vec2 = _mm256_loadu_pd(array + 4);
        __m128i key1 = _mm_load_si128((__m128i *)keys);
        __m128i key2 = _mm_load_si128((__m128i *)(keys + 4));

        bitonic_sort(vec1, vec2, key1, key2);

        _mm256_storeu_pd(array, vec1);
        _mm256_storeu_pd(array + 4, vec2);
        _mm_store_si128((__m128i *)(keys), key1);
        _mm_store_si128((__m128i *)(keys + 4), key2);

    } else if (num_to_sort == 16) {
        __m256d vec1 = _mm256_loadu_pd(array);
        __m256d vec2 = _mm256_loadu_pd(array + 4);
        __m256d vec3 = _mm256_loadu_pd(array + 8);
        __m256d vec4 = _mm256_loadu_pd(array + 12);

        __m128i key1 = _mm_load_si128((__m128i *)keys);
        __m128i key2 = _mm_load_si128((__m128i *)(keys + 4));
        __m128i key3 = _mm_load_si128((__m128i *)(keys + 8));
        __m128i key4 = _mm_load_si128((__m128i *)(keys + 12));

        bitonic_sort(vec1, vec2, vec3, vec4, key1, key2, key3,
                     key4);

        _mm256_storeu_pd(array, vec1);
        _mm256_storeu_pd(array + 4, vec2);
        _mm256_storeu_pd(array + 8, vec3);
        _mm256_storeu_pd(array + 12, vec4);

        _mm_store_si128((__m128i *)(keys), key1);
        _mm_store_si128((__m128i *)(keys + 4), key2);
        _mm_store_si128((__m128i *)(keys + 8), key3);
        _mm_store_si128((__m128i *)(keys + 12), key4);

    } else {
        for (unsigned i = 0; i < end; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(array + i);
            __m128i key1 = _mm_load_si128((__m128i *)(keys + i));

            bitonic_sort(vec1, key1);

            _mm256_storeu_pd(array + i, vec1);
            _mm_store_si128((__m128i *)(keys + i), key1);
        }

        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        for (unsigned len = 8; len <= imaginary_length; len *= 2) {
            // std::cout << "len: " << len << std::endl;
            // inner loop goes over all subdivisions
            for (unsigned n = 0; n < imaginary_length; n += len) {
                compare_full_length(array, keys, n, n + len - 1,
                                    last_index);
                lane_crossing_compare(array, keys, n, n + len - 1,
                                      last_index, 0);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////
// DOUBLE:INT  ARRAY:KEY SORTING ALGORITHM FOR ARRAYS OF ARBITRARY
// LENGTH
//
inline void maskload(int diff, double *p, int *k, __m256i &mask,
                     __m128i &key_mask, __m256d &reg1,
                     __m128i &key1) {
    switch (diff) {
    case 0: {
        mask = double_load_masks256.mask1;
        key_mask = double_key_load_masks128.mask1;
        reg1 = _mm256_maskload_pd(p, mask);
        key1 = _mm_maskload_epi32(k, key_mask);
        reg1 = _mm256_blend_pd(m256d_infinity, reg1, 0b0001);
    } break;
    case 1: {
        mask = double_load_masks256.mask2;
        key_mask = double_key_load_masks128.mask2;
        reg1 = _mm256_maskload_pd(p, mask);
        key1 = _mm_maskload_epi32(k, key_mask);
        reg1 = _mm256_blend_pd(m256d_infinity, reg1, 0b0011);
    } break;
    case 2: {
        mask = double_load_masks256.mask3;
        key_mask = double_key_load_masks128.mask3;
        reg1 = _mm256_maskload_pd(p, mask);
        key1 = _mm_maskload_epi32(k, key_mask);
        reg1 = _mm256_blend_pd(m256d_infinity, reg1, 0b0111);
    } break;
    };
}

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the float array to be sorted
 * @param keys pointer to the array of long ints to be permuted
 * according to arr permutations
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param last_index index of the last element of array to be sorted
 */
inline void compare_full_length_all_cases(double *arr, int *keys,
                                          unsigned start,
                                          unsigned end,
                                          unsigned last_index) {
    unsigned length = end - start + 1;
    unsigned half = length / 2; // half je index prvega cez polovico
    for (int i = half - 4; i >= 0; i -= 4) {
        // index of last vector to load
        int last_vec_to_load = end - 3 - i;
        int diff = last_index - last_vec_to_load;
        if (diff < 0)
            return;
        // define pointers to the start of vectors to load
        double *p1 = arr + start + i;
        double *p2 = arr + last_vec_to_load;
        int *k1 = keys + start + i;
        int *k2 = keys + last_vec_to_load;

        __m256d vec2;
        __m256i mask;
        __m128i key_load_store_mask;
        __m128i key2;
        if (diff < 3) {
            maskload(diff, p2, k2, mask, key_load_store_mask, vec2,
                     key2);

        } else {
            vec2 = _mm256_loadu_pd(p2);
            key2 = _mm_load_si128((__m128i *)k2);
        }
        __m256d vec1 = _mm256_loadu_pd(p1);
        __m128i key1 = _mm_load_si128((__m128i *)k1);
        // reverse one of registers register reg0

        reverse_and_compare(vec1, vec2, key1, key2);

        vec1 = _mm256_permute4x64_pd(vec1, 0b00011011);
        key1 = _mm_castps_si128(
            _mm_permute_ps(_mm_castsi128_ps(key1), 0b00011011));

        _mm256_storeu_pd(p1, vec1);
        _mm_store_si128((__m128i *)k1, key1);
        if (diff <= 2) {
            _mm256_maskstore_pd(p2, mask, vec2);
            _mm_maskstore_epi32(k2, key_load_store_mask, key2);
        } else {
            _mm256_storeu_pd(p2, vec2);
            _mm_store_si128((__m128i *)k2, key2);
        }
    }
}

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the array to be sorted
 * @param keys pointer to the array of integers to be permuted using
 * the same permutation as arr
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param depth a parameter to follow the depth of recursion
 */
inline void lane_crossing_compare_all_cases(double *arr, int *keys,
                                            unsigned start,
                                            unsigned end,
                                            unsigned last_index,
                                            unsigned depth) {
    if (start > last_index) {
        return;
    }
    unsigned length = end - start + 1;
    if (length == 4) {
        int diff = last_index - start;
        if (diff < 1)
            return;
        __m256d reg;
        __m256i load_store_mask;
        __m128i key_load_store_mask;
        __m128i key;
        if (diff < 3)
            maskload(diff, arr + start, keys + start,
                     load_store_mask, key_load_store_mask, reg,
                     key);
        else {
            reg = _mm256_loadu_pd(arr + start);
            key = _mm_load_si128((__m128i *)(keys + start));
        }
        // else

        permute_and_compare(reg, key, double_masks256.mask2,
                            0b01001110);

        shuffle_and_compare(reg, key, double_masks256.mask1, 0b0101,
                            0b10110001);
        if (diff < 3) {
            _mm256_maskstore_pd(arr + start, load_store_mask, reg);
            _mm_maskstore_epi32((keys + start), key_load_store_mask,
                                key);
        } else {
            _mm256_storeu_pd(arr + start, reg);
            _mm_store_si128((__m128i *)(keys + start), key);
        }
        return;
    }

    double *p = arr + start;
    int *k = keys + start;
    for (unsigned i = 0; i < length / 2; i += 4) {
        int diff = last_index - (start + length / 2 + i);
        if (diff < 0)
            break;
        __m256d reg1;
        __m256i mask;
        __m128i key_load_store_mask;
        __m128i key1;
        double *p2 = p + length / 2 + i;
        int *k2 = k + length / 2 + i;
        if (diff < 3)
            maskload(diff, p2, k2, mask, key_load_store_mask, reg1,
                     key1);
        else {
            reg1 = _mm256_loadu_pd(p2);
            key1 = _mm_load_si128((__m128i *)k2);
        }
        double *p1 = p + i;
        int *k1 = k + i;
        __m256d reg0 = _mm256_loadu_pd(p1);
        __m128i key0 = _mm_load_si128((__m128i *)k1);

        compare(reg0, reg1, key0, key1);

        _mm256_storeu_pd(p1, reg0);
        _mm_store_si128((__m128i *)k1, key0);

        if (diff < 3) {
            _mm256_maskstore_pd(p2, mask, reg1);
            _mm_maskstore_epi32(k2, key_load_store_mask, key1);
        } else {
            _mm256_storeu_pd(p2, reg1);
            _mm_store_si128((__m128i *)k2, key1);
        }
    }
    lane_crossing_compare_all_cases(
        arr, keys, start, (start + end) / 2, last_index, depth + 1);
    lane_crossing_compare_all_cases(arr, keys,
                                    (start + end) / 2 + 1, end,
                                    last_index, depth + 1);
};

/**
 * @brief The function sorts an array with 2^n elements
 * @param array pointer to the start of the array
 * @param start index of the first number to be sorted
 * @param end index of the last number to be sorted
 * @details end - start + 1 should be 2^n
 */
inline void sort_key_value(double *array, int *keys,
                           unsigned num_to_sort) {
    unsigned end = num_to_sort - 1;
    if (num_to_sort <= 1)
        return;
    else if (num_to_sort < 4) {
        int reminder = mod4(end);
        double *p = array + end - reminder;
        int *k = keys + end - reminder;
        __m256d reg1;
        __m256i mask;
        __m128i key_load_store_mask;
        __m128i key1;

        maskload(reminder, p, k, mask, key_load_store_mask, reg1,
                 key1);

        bitonic_sort(reg1, key1);
        _mm256_maskstore_pd(array + end - reminder, mask,
                            reg1);
        _mm_maskstore_epi32(keys + end - reminder,
                            _mm256_castsi256_si128(mask), key1);
    } else if (!(num_to_sort & (num_to_sort - 1)))
        sort_2n_key_value(array, keys, num_to_sort);
    else if (mod4(num_to_sort) == 0)
        sort_4n_key_value(array, keys, num_to_sort);
    else {
        int pow2 = (int)std::ceil(std::log2f(end + 1));
        int imaginary_length = (int)std::pow(2, pow2);

        unsigned last_index = end;

        for (unsigned i = 0; i <= end - 3; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(array + i);
            __m128i key1 =
                _mm_load_si128((__m128i *)(keys + i));
            bitonic_sort(vec1, key1);
            _mm256_storeu_pd(array + i, vec1);
            _mm_store_si128((__m128i *)(keys + i), key1);
        }
        ///////////////////////////////// load the partial one
        int reminder = mod4(end);
        double *p = array + end - reminder;
        int *k = keys + end - reminder;
        __m256d reg1;
        __m256i mask;
        __m128i key_load_store_mask;
        __m128i key1;
        maskload(reminder, p, k, mask, key_load_store_mask, reg1,
                 key1);
        bitonic_sort(reg1, key1);
        _mm256_maskstore_pd(array + end - reminder, mask,
                            reg1);
        _mm_maskstore_epi32(keys + end - reminder,
                            key_load_store_mask, key1);

        ///////////////////////////////////////////////////////

        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        // 8 and less has already been done above
        // for (unsigned len = 16; len <= imaginary_length; len *=
        // 2) {
        for (unsigned len = 8; len <= imaginary_length; len *= 2) {
            // inner loop goes over all subdivisions
            for (unsigned n = 0; n < imaginary_length; n += len) {
                compare_full_length_all_cases(
                    array, keys, n, n + len - 1,
                    last_index);
                lane_crossing_compare_all_cases(
                    array, keys, n, n + len - 1,
                    last_index, 0);
            }
        }
    }
}
