#include "bitonic_sort.hpp"
#include "type_definitions.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <span>
#include <vector>

namespace bitonic_sort {

inline void maskload(int diff, float *p2, __m256i &mask, __m256 &reg1) {
    switch (diff) {
    case 0:
        {
            mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, LOAD);
            reg1 = _mm256_maskload_ps(p2, mask);
            __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            reg1 = _mm256_blend_ps(infinity, reg1, 0b00000001);
        }
        break;
    case 1:
        {
            mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, LOAD, LOAD);
            reg1 = _mm256_maskload_ps(p2, mask);
            __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            reg1 = _mm256_blend_ps(infinity, reg1, 0b00000011);
        }
        break;
    case 2:
        {
            mask = _mm256_set_epi32(0, 0, 0, 0, 0, LOAD, LOAD, LOAD);
            reg1 = _mm256_maskload_ps(p2, mask);
            __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            reg1 = _mm256_blend_ps(infinity, reg1, 0b00000111);
        }
        break;
    case 3:
        {
            mask = _mm256_set_epi32(0, 0, 0, 0, LOAD, LOAD, LOAD, LOAD);
            reg1 = _mm256_maskload_ps(p2, mask);
            __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            reg1 = _mm256_blend_ps(infinity, reg1, 0b00001111);
        }
        break;
    case 4:
        {
            mask = _mm256_set_epi32(0, 0, 0, LOAD, LOAD, LOAD, LOAD, LOAD);
            reg1 = _mm256_maskload_ps(p2, mask);
            __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            reg1 = _mm256_blend_ps(infinity, reg1, 0b00011111);
        }
        break;
    case 5:
        {
            mask = _mm256_set_epi32(0, 0, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD);
            reg1 = _mm256_maskload_ps(p2, mask);
            __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            reg1 = _mm256_blend_ps(infinity, reg1, 0b00111111);
        }
        break;
    case 6:
        {
            mask = _mm256_set_epi32(0, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD);
            reg1 = _mm256_maskload_ps(p2, mask);
            __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            reg1 = _mm256_blend_ps(infinity, reg1, 0b01111111);
        }
        break;
    };
}

template <std::uint8_t mask1, std::uint8_t mask2>
void shuffle_and_compare(__m256 &reg) {
    __m256 shuffled_reg = _mm256_shuffle_ps(reg, reg, mask1);
    __m256 max = _mm256_max_ps(reg, shuffled_reg);
    __m256 min = _mm256_min_ps(reg, shuffled_reg);
    reg = _mm256_blend_ps(max, min, mask2);
}

void reverse_and_compare(__m256 &reg) {
    __m256 reversed_halves = _mm256_permute2f128_ps(reg, reg, 0b00000001);
    __m256 reversed = _mm256_shuffle_ps(reversed_halves, reversed_halves, 0b00011011);
    __m256 max = _mm256_max_ps(reg, reversed);
    __m256 min = _mm256_min_ps(reg, reversed);
    reg = _mm256_blend_ps(max, min, 0b00001111);
}

void reverse_and_compare(__m256 &reg0, __m256 &reg1) {
    // reverse one of registers register reg0
    __m256 reversed_halves = _mm256_permute2f128_ps(reg0, reg0, 0b00000001);
    __m256 reversed = _mm256_shuffle_ps(reversed_halves, reversed_halves, 0b00011011);
    // register 2 vsebuje min vrednosti
    reg0 = _mm256_min_ps(reg1, reversed);
    // register 1 vsebuje max vrednosti
    reg1 = _mm256_max_ps(reg1, reversed);
}

void reverse_halves_and_compare(__m256 &reg0) {
    // tu narediš *----* *----* *----* *----*
    __m256 reversed_halves = _mm256_permute2f128_ps(reg0, reg0, 0b00000001);
    __m256 max = _mm256_max_ps(reg0, reversed_halves);
    __m256 min = _mm256_min_ps(reg0, reversed_halves);
    reg0 = _mm256_blend_ps(max, min, 0b00001111);
}

void compare(__m256 &reg0, __m256 &reg1) {
    {
        __m256 max = _mm256_max_ps(reg0, reg1);
        __m256 min = _mm256_min_ps(reg0, reg1);
        reg0 = min;
        reg1 = max;
    }
}

void sort(__m256 &reg) {
    //    shuffle_and_compare(reg, 0b10110001, 0b10101010);
    shuffle_and_compare<0b10110001, 0b01010101>(reg);
    shuffle_and_compare<0b00011011, 0b00110011>(reg);
    shuffle_and_compare<0b10110001, 0b01010101>(reg);
    reverse_and_compare(reg);
    shuffle_and_compare<0b01001110, 0b00110011>(reg);
    shuffle_and_compare<0b10110001, 0b01010101>(reg);
}

void sort(__m256 &reg0, __m256 &reg1) {
    sort(reg1); // sort first register
    sort(reg0); // sort second register
    reverse_and_compare(reg0, reg1);
    reverse_halves_and_compare(reg0);
    reverse_halves_and_compare(reg1);
    shuffle_and_compare<0b01001110, 0b00110011>(reg0);
    shuffle_and_compare<0b10110001, 0b01010101>(reg0);
    shuffle_and_compare<0b01001110, 0b00110011>(reg1);
    shuffle_and_compare<0b10110001, 0b01010101>(reg1);
    return;
}

/**
 * @brief The function accepts two already sorted  __m256 vectors
 * and sorts them.
 * @param reg1 upper vector of numbers - at the end it contains
 * larger values, the largest value is in the upper half of register
 * [255:192]
 *
 * @param reg0 lower vector of numbers - at the end it contains
 * smaller values. The smallest value is in the lowest half of
 * register - at [63:0]
 *
 */

inline void bitonic_merge(__m256 &reg0, __m256 &reg1) {
    reverse_and_compare(reg0, reg1);
    reverse_halves_and_compare(reg0);
    reverse_halves_and_compare(reg1);
    shuffle_and_compare<0b01001110, 0b00110011>(reg0);
    shuffle_and_compare<0b10110001, 0b01010101>(reg0);
    shuffle_and_compare<0b01001110, 0b00110011>(reg1);
    shuffle_and_compare<0b10110001, 0b01010101>(reg1);
    return;
}

/**
 * @brief The function accepts four unsorted  __m256 vectors and
 * sorts them.
 * @param reg3 upper vector of numbers - at the end it contains
 * larger values, the largest value is in the upper half of register
 * [255:192]
 * @param reg2
 * @param reg1
 * @param reg0 lower vector of numbers - at the end it contains
 * smaller values. The smallest value is in the lowest half of
 * register - at [63:0]
 *
 */
void sort(__m256 &reg0, __m256 &reg1, __m256 &reg2, __m256 &reg3) {

    // sort each quarter
    sort(reg0);
    sort(reg1);
    sort(reg2);
    sort(reg3);
    // sort each half
    sort(reg0, reg1);
    sort(reg2, reg3);

    reverse_and_compare(reg0, reg3);
    reverse_and_compare(reg1, reg2);
    // sort full width
    compare(reg1, reg3);
    compare(reg0, reg2);
    compare(reg0, reg1);
    compare(reg2, reg3);

    reverse_halves_and_compare(reg0);
    reverse_halves_and_compare(reg1);
    reverse_halves_and_compare(reg2);
    reverse_halves_and_compare(reg3);

    shuffle_and_compare<0b01001110, 0b00110011>(reg0);
    shuffle_and_compare<0b01001110, 0b00110011>(reg1);
    shuffle_and_compare<0b01001110, 0b00110011>(reg2);
    shuffle_and_compare<0b01001110, 0b00110011>(reg3);

    shuffle_and_compare<0b10110001, 0b01010101>(reg0);
    shuffle_and_compare<0b10110001, 0b01010101>(reg1);
    shuffle_and_compare<0b10110001, 0b01010101>(reg2);
    shuffle_and_compare<0b10110001, 0b01010101>(reg3);
}

/**
 * @brief The function accepts four sorted  __m256 vectors and sorts
 * them.
 * @param reg3 upper vector of numbers - at the end it contains
 * larger values, the largest value is in the upper half of register
 * [255:192]
 * @param reg2
 * @param reg1
 * @param reg0 lower vector of numbers - at the end it contains
 * smaller values. The smallest value is in the lowest half of
 * register - at [63:0]
 *
 */
inline void bitonic_merge(__m256 &reg0, __m256 &reg1, __m256 &reg2, __m256 &reg3) {
    reverse_and_compare(reg0, reg3);
    reverse_and_compare(reg1, reg2);
    // sort full width
    compare(reg1, reg3);
    compare(reg0, reg2);

    compare(reg0, reg1);
    compare(reg2, reg3);

    reverse_halves_and_compare(reg0);
    reverse_halves_and_compare(reg1);
    reverse_halves_and_compare(reg2);
    reverse_halves_and_compare(reg3);

    shuffle_and_compare<0b01001110, 0b00110011>(reg0);
    shuffle_and_compare<0b01001110, 0b00110011>(reg1);
    shuffle_and_compare<0b01001110, 0b00110011>(reg2);
    shuffle_and_compare<0b01001110, 0b00110011>(reg3);

    shuffle_and_compare<0b10110001, 0b01010101>(reg0);
    shuffle_and_compare<0b10110001, 0b01010101>(reg1);
    shuffle_and_compare<0b10110001, 0b01010101>(reg2);
    shuffle_and_compare<0b10110001, 0b01010101>(reg3);
}

//////////////////////////////////////////////////////////////////
// 2^N FLOAT ARRAY SORTING ALGORITHM

// float implementation
/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the float array to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 */
inline void compare_full_length(float *arr, unsigned start, unsigned end) {
    unsigned length = end - start + 1;
    unsigned half = length / 2;
    for (int i = 0; i < half; i += 8) {
        float *p1 = arr + start + i;
        float *p2 = arr + end - 7 - i;
        { // reverse lover half and compare to upper half
            __m256 vec1 = _mm256_loadu_ps(p1);
            __m256 vec2 = _mm256_loadu_ps(p2);
            reverse_and_compare(vec1, vec2);
            _mm256_storeu_ps(p1, vec1);
            _mm256_storeu_ps(p2, vec2);
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
inline void lane_crossing_compare(float *arr, unsigned start, unsigned end, unsigned depth) {
    unsigned length = end - start + 1;
    if (length == 8) {
        __m256 reg = _mm256_loadu_ps(arr + start);
        // this is the ending case do single vector permutations
        reverse_halves_and_compare(reg);
        shuffle_and_compare<0b01001110, 0b00110011>(reg);
        shuffle_and_compare<0b10110001, 0b01010101>(reg);
        _mm256_storeu_ps(arr + start, reg);
        return;
    }
    float *p = arr + start;
    for (unsigned i = 0; i < length / 2; i += 8) {
        {
            float *p1 = p + i;
            float *p2 = p + length / 2 + i;
            __m256 reg0 = _mm256_loadu_ps(p1); // i-ti od začetka
            __m256 reg1 = _mm256_loadu_ps(p2); // ta je prvi čez polovico
            // register 2 vsebuje min vrednosti
            __m256 min = _mm256_min_ps(reg1, reg0);
            // register 1 vsebuje max vrednosti
            reg1 = _mm256_max_ps(reg1, reg0);
            reg0 = min;
            _mm256_storeu_ps(p1, reg0);
            _mm256_storeu_ps(p2, reg1);
        }
    }
    lane_crossing_compare(arr, start, (start + end) / 2, depth + 1);
    lane_crossing_compare(arr, (start + end) / 2 + 1, end, depth + 1);
};

/**
 * @brief The function sorts an array with 8*n elements
 * @param array pointer to the first element to be sorted
 * @param num_to_sort number of elements to be sorted
 * @details num_to_sort should be multiple of 8. If it's not,
 * then add additional elements - infinity, so that it will be
 * large enough
 */
void sort_2n(std::span<float> span) {

    float *array = span.data();
    unsigned num_to_sort = span.size();

    unsigned vec_count = num_to_sort / 8;

    assert((num_to_sort >= 0 && !(num_to_sort & (num_to_sort - 1))) &&
           "The array to be sorted is not the power of 2!");

    //__m256d *arr = avx_vec.data();
    if (num_to_sort < 8) {
        __m256 reg1;
        __m256i mask;
        maskload(num_to_sort - 1, array, mask, reg1);
        sort(reg1);
        _mm256_maskstore_ps(array, mask, reg1);
    } else if (num_to_sort == 8) {

        __m256 vec = _mm256_loadu_ps(array);
        sort(vec);
        _mm256_storeu_ps(array, vec);

    } else if (num_to_sort == 16) {

        __m256 vec1 = _mm256_loadu_ps(array);
        __m256 vec2 = _mm256_loadu_ps(array + 8);
        sort(vec1, vec2);
        _mm256_storeu_ps(array, vec1);
        _mm256_storeu_ps(array + 8, vec2);

    } else if (num_to_sort == 32) {

        __m256 vec1 = _mm256_loadu_ps(array);
        __m256 vec2 = _mm256_loadu_ps(array + 8);
        __m256 vec3 = _mm256_loadu_ps(array + 16);
        __m256 vec4 = _mm256_loadu_ps(array + 24);
        sort(vec1, vec2, vec3, vec4);
        _mm256_storeu_ps(array, vec1);
        _mm256_storeu_ps(array + 8, vec2);
        _mm256_storeu_ps(array + 16, vec3);
        _mm256_storeu_ps(array + 24, vec4);

    } else if (num_to_sort >= 64) {
        for (unsigned i = 0; i < num_to_sort; i += 8) {
            __m256 vec1 = _mm256_loadu_ps(array + i);
            sort(vec1);
            _mm256_storeu_ps(array + i, vec1);
        }

        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        for (unsigned len = 16; len <= num_to_sort; len *= 2) {
            // std::cout << "len: " << len << std::endl;
            // inner loop goes over all subdivisions
            for (unsigned n = 0; n < num_to_sort; n += len) {
                compare_full_length(array, n, n + len - 1);
                lane_crossing_compare(array, n, n + len - 1, 0);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////
// FLOAT ARRAY SORTING ALGORITHM, the array must contain 8n elements

// float implementation
/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the float array to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param last_index last index of array to be sorted, maximum is
 * array size - 1
 */
inline void compare_full_length(float *arr, unsigned start, unsigned end,
                                unsigned last_index) {
    unsigned length = end - start + 1;
    unsigned half = length / 2; // half je index prvega cez polovico
    for (int i = half - 8; i >= 0; i -= 8) {
        if (end - 7 - i > last_index)
            break;
        float *p1 = arr + start + i;
        float *p2 = arr + end - 7 - i;

        { // reverse lover half and compare to upper half
            __m256 vec1 = _mm256_loadu_ps(p1);
            __m256 vec2 = _mm256_loadu_ps(p2);
            __m256 reversed_halves = _mm256_permute2f128_ps(vec1, vec1, 0b00000001);
            __m256 reversed = _mm256_shuffle_ps(reversed_halves, reversed_halves, 0b00011011);
            vec1 = _mm256_min_ps(reversed, vec2);
            vec2 = _mm256_max_ps(reversed, vec2);
            reversed_halves = _mm256_permute2f128_ps(vec1, vec1, 0b00000001);
            vec1 = _mm256_shuffle_ps(reversed_halves, reversed_halves, 0b00011011);
            _mm256_storeu_ps(p1, vec1);
            _mm256_storeu_ps(p2, vec2);
        }
    }
}

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the array to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param last_index last index of array to be sorted, maximum is
 * array size - 1
 * @param depth a parameter to follow the depth of recursion
 */
inline void lane_crossing_compare(float *arr, unsigned start, unsigned end,
                                  unsigned last_index, unsigned depth) {

    if (start > last_index) {
        // std::cout << "aborted start index: " << start <<
        // std::endl;
        return;
    }
    unsigned length = end - start + 1;
    if (length == 8) {
        __m256 reg = _mm256_loadu_ps(arr + start);
        // this is the ending case do single vector permutations
        reverse_halves_and_compare(reg);
        shuffle_and_compare<0b01001110, 0b00110011>(reg);
        shuffle_and_compare<0b10110001, 0b01010101>(reg);
        _mm256_storeu_ps(arr + start, reg);
        return;
    }
    float *p = arr + start;
    // for (unsigned i = 0; i < length / 2; i += 8) {
    for (unsigned i = 0; i < length / 2; i += 8) {
        if (start + length / 2 + i > last_index)
            break;
        {
            float *p1 = p + i;
            float *p2 = p + length / 2 + i;
            __m256 reg0 = _mm256_loadu_ps(p1); // i-ti od začetka
            __m256 reg1 = _mm256_loadu_ps(p2); // ta je prvi čez polovico
            // register 2 vsebuje min vrednosti
            __m256 min = _mm256_min_ps(reg1, reg0);
            // register 1 vsebuje max vrednosti
            reg1 = _mm256_max_ps(reg1, reg0);
            reg0 = min;
            _mm256_storeu_ps(p1, reg0);
            _mm256_storeu_ps(p2, reg1);
        }
    }
    lane_crossing_compare(arr, start, (start + end) / 2, last_index, depth + 1);
    lane_crossing_compare(arr, (start + end) / 2 + 1, end, last_index, depth + 1);
};

/**
 * @brief The function sorts an array with 2^n elements
 * @param array pointer to the first element to be sorted
 * @param num_to_sort number of elements to sort
 * @details num_to_sort should be multiple of 8.
 */
void sort_8n(std::span<float> span) {

    float *array = span.data();
    unsigned num_to_sort = span.size();

    //    unsigned full_length = end - start + 1; // number of
    //    double numbers to sort
    unsigned end = num_to_sort - 1;
    int pow2 = (int)std::ceil(std::log2f(num_to_sort));
    int imaginary_length = (int)std::pow(2, pow2);

    unsigned last_index = end - 7; // last index to be loaded
    // std::cout << "imaginary_length: " << imaginary_length <<
    // std::endl;
    unsigned vec_count = num_to_sort / 8;
    assert((num_to_sort >= 0 && (num_to_sort % 8) == 0) &&
           "The array to be sorted is not a multiples of 8!");

    if (num_to_sort == 8) {
        __m256 vec = _mm256_loadu_ps(array);
        sort(vec);
        _mm256_storeu_ps(array, vec);
    } else if (num_to_sort == 16) {
        __m256 vec1 = _mm256_loadu_ps(array);
        __m256 vec2 = _mm256_loadu_ps(array + 8);
        sort(vec1, vec2);
        _mm256_storeu_ps(array, vec1);
        _mm256_storeu_ps(array + 8, vec2);

    } else if (num_to_sort == 32) {
        __m256 vec1 = _mm256_loadu_ps(array);
        __m256 vec2 = _mm256_loadu_ps(array + 8);
        __m256 vec3 = _mm256_loadu_ps(array + 16);
        __m256 vec4 = _mm256_loadu_ps(array + 24);
        sort(vec1, vec2, vec3, vec4);
        _mm256_storeu_ps(array, vec1);
        _mm256_storeu_ps(array + 8, vec2);
        _mm256_storeu_ps(array + 16, vec3);
        _mm256_storeu_ps(array + 24, vec4);

    } else {
        for (unsigned i = 0; i < end; i += 8) {
            __m256 vec1 = _mm256_loadu_ps(array + i);
            sort(vec1);
            _mm256_storeu_ps(array + i, vec1);
        }
        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        for (unsigned len = 16; len <= imaginary_length; len *= 2) {
            // std::cout << "len: " << len << std::endl;
            // inner loop goes over all subdivisions
            for (unsigned n = 0; n < imaginary_length; n += len) {
                compare_full_length(array, n, n + len - 1, last_index);
                lane_crossing_compare(array, n, n + len - 1, last_index, 0);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////
// FLOAT ARRAY SORTING ALGORITHM FOR ARRAYS OF ARBITRARY LENGTH, the
// array must contain 8n elements float implementation

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the float array to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 */
inline void compare_full_length_all_cases(float *arr, unsigned start, unsigned end,
                                          unsigned last_index) {

    unsigned length = end - start + 1;
    unsigned half = length / 2; // half je index prvega cez polovico
    for (int i = half - 8; i >= 0; i -= 8) {
        // index of last vector to load
        int last_vec_to_load = end - 7 - i;
        int diff = last_index - last_vec_to_load;
        if (UNLIKELY(diff < 0))
            return;
        // define pointers to the start of vectors to load
        float *p1 = arr + start + i;
        float *p2 = arr + last_vec_to_load;
        __m256 vec2;
        __m256i mask;
        if (UNLIKELY(diff < 7))
            maskload(diff, p2, mask, vec2);
        else
            vec2 = _mm256_loadu_ps(p2);
        { // reverse lover half and compare to upper half
            __m256 vec1 = _mm256_loadu_ps(p1);
            __m256 reversed_halves = _mm256_permute2f128_ps(vec1, vec1, 0b00000001);
            __m256 reversed = _mm256_shuffle_ps(reversed_halves, reversed_halves, 0b00011011);
            vec1 = _mm256_min_ps(reversed, vec2);
            vec2 = _mm256_max_ps(reversed, vec2);
            reversed_halves = _mm256_permute2f128_ps(vec1, vec1, 0b00000001);
            vec1 = _mm256_shuffle_ps(reversed_halves, reversed_halves, 0b00011011);
            _mm256_storeu_ps(p1, vec1);
            if (UNLIKELY(diff <= 6))
                _mm256_maskstore_ps(p2, mask, vec2);
            else
                _mm256_storeu_ps(p2, vec2);
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
void lane_crossing_compare_all_cases(float *arr, unsigned start, unsigned end,
                                     unsigned last_index, unsigned depth) {
    if (start > last_index) {
        return;
    }
    unsigned length = end - start + 1;
    if (length == 8) {
        int diff = last_index - start;
        if (diff < 1)
            return;
        __m256 reg;
        __m256i mask;
        if (UNLIKELY(diff < 7))
            maskload(diff, arr + start, mask, reg);
        else
            reg = _mm256_loadu_ps(arr + start);

        reverse_halves_and_compare(reg);
        shuffle_and_compare<0b01001110, 0b00110011>(reg);
        shuffle_and_compare<0b10110001, 0b01010101>(reg);
        if (diff < 7)
            _mm256_maskstore_ps(arr + start, mask, reg);
        else
            _mm256_storeu_ps(arr + start, reg);

        return;
    }

    float *p = arr + start;
    for (unsigned i = 0; i < length / 2; i += 8) {
        int diff = last_index - (start + length / 2 + i);
        if (UNLIKELY(diff < 0))
            break;
        __m256 reg1;
        __m256i mask;
        float *p2 = p + length / 2 + i;
        if (UNLIKELY(diff < 7))
            maskload(diff, p2, mask, reg1);
        else
            reg1 = _mm256_loadu_ps(p2);
        float *p1 = p + i;
        __m256 reg0 = _mm256_loadu_ps(p1); // i-ti od začetka
        // register 2 vsebuje min vrednosti
        __m256 min = _mm256_min_ps(reg1, reg0);
        // register 1 vsebuje max vrednosti
        reg1 = _mm256_max_ps(reg1, reg0);
        reg0 = min;
        _mm256_storeu_ps(p1, reg0);
        if (UNLIKELY(diff < 7))
            _mm256_maskstore_ps(p2, mask, reg1);
        else
            _mm256_storeu_ps(p2, reg1);
    }
    lane_crossing_compare_all_cases(arr, start, (start + end) / 2, last_index, depth + 1);
    lane_crossing_compare_all_cases(arr, (start + end) / 2 + 1, end, last_index, depth + 1);
};

/**
 * @brief The function sorts an array with arbitrary number of elements
 * @param array pointer to the first element to be sorted
 * @param num_to_sort number of elements to be sorted
 */
void sort(std::span<float> span) {

    float *array = span.data();
    unsigned num_to_sort = span.size();
    unsigned end = num_to_sort - 1;
    if (num_to_sort <= 1)
        return;
    else if (num_to_sort < 8) {
        __m256i mask;
        __m256 reg;
        maskload(num_to_sort - 1, array, mask, reg);
        sort(reg);
        _mm256_maskstore_ps(array, mask, reg);
    } else if (!(num_to_sort & (num_to_sort - 1)))
        sort_2n(span);
    else if (mod8(num_to_sort) == 0)
        sort_8n(span);
    else if (num_to_sort < 16) {
        __m256i mask;
        __m256 reg1, reg2;
        reg1 = _mm256_loadu_ps(array);
        maskload(num_to_sort - 9, array + 8, mask, reg2);
        sort(reg1, reg2);
        _mm256_storeu_ps(array, reg1);
        _mm256_maskstore_ps(array + 8, mask, reg2);
    } else {
        int pow2 = (int)std::ceil(std::log2f(num_to_sort));
        int imaginary_length = (int)std::pow(2, pow2);
        unsigned last_index = end;

        for (unsigned i = 0; i <= end - 7; i += 8) {
            __m256 vec1 = _mm256_loadu_ps(array + i);
            sort(vec1);
            _mm256_storeu_ps(array + i, vec1);
        }
        ///////////////////////////////// load the partial one
        int reminder = mod8(end);
        float *p = array + end - reminder;
        __m256 reg1;
        __m256i mask;
        maskload(reminder, p, mask, reg1);
        sort(reg1);
        _mm256_maskstore_ps(array + end - reminder, mask, reg1);

        ///////////////////////////////////////////////////////

        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        // 8 and less has already been done above
        // for (unsigned len = 16; len <= imaginary_length; len *=
        // 2) {
        for (unsigned len = 16; len <= imaginary_length; len *= 2) {
            // std::cout << "len: " << len << std::endl;
            // inner loop goes over all subdivisions
            for (unsigned n = 0; n < imaginary_length; n += len) {
                compare_full_length_all_cases(array, n, n + len - 1, last_index);
                lane_crossing_compare_all_cases(array, n, n + len - 1, last_index, 0);
            }
        }
    }
}

} // namespace bitonic_sort
