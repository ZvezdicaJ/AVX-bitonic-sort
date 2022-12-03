#pragma once

#include "type_definitions.hpp"
#include "bitonic_sort.hpp"

#include <cassert>
#include <cstdint>
#include <vector>
#include <cmath>
#include <span>
#include <immintrin.h>


namespace BITONIC_SORT {

template <uint8_t mask>
void shuffle_and_compare(__m256d &reg) {
    __m256d shuffled_reg = _mm256_shuffle_pd(reg, reg, mask);
    __m256d max = _mm256_max_pd(reg, shuffled_reg);
    __m256d min = _mm256_min_pd(reg, shuffled_reg);
    reg = _mm256_unpacklo_pd(min, max);
}

inline __attribute__((always_inline)) void
reverse_and_compare(__m256d &reg) {
    __m256d reversed_reg = _mm256_permute4x64_pd(reg, 0b00011011);
    __m256d max = _mm256_max_pd(reg, reversed_reg);
    __m256d min = _mm256_min_pd(reg, reversed_reg);
    // max mora biti pri 256
    reg = _mm256_blend_pd(max, min, 0b0011);
}

inline __attribute__((always_inline)) void
reverse_and_compare(__m256d &reg0, __m256d &reg1) {
    __m256d reverse0 = _mm256_permute4x64_pd(reg0, 0b00011011);
    // register 2 vsebuje min vrednosti
    reg0 = _mm256_min_pd(reg1, reverse0);
    // register 1 vsebuje max vrednosti
    reg1 = _mm256_max_pd(reg1, reverse0);
}

template<int mask>
void permute_and_compare(__m256d &reg) {
    __m256d shuffled_reg = _mm256_permute4x64_pd(reg, mask);
    __m256d max = _mm256_max_pd(reg, shuffled_reg);
    __m256d min = _mm256_min_pd(reg, shuffled_reg);
    reg = _mm256_blend_pd(max, min, 0b0011);
}

inline __attribute__((always_inline)) void compare(__m256d &reg0,
                                                   __m256d &reg1) {
    __m256d min = _mm256_min_pd(reg0, reg1);
    __m256d max = _mm256_max_pd(reg0, reg1);
    reg0 = min;
    reg1 = max;
}

/**
 *@brief The function accepts a single __m256d vector and sorts
 *it.
 * @param reg register to be sorted
 */
inline void bitonic_sort(__m256d &reg) {
    shuffle_and_compare<0b0101>(reg);
    reverse_and_compare(reg);
    shuffle_and_compare<0b0101>(reg);
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
inline void bitonic_sort(__m256d &reg0, __m256d &reg1) {
    bitonic_sort(reg1); // sort first register
    bitonic_sort(reg0); // sort second register
    reverse_and_compare(reg0, reg1);
    permute_and_compare<0b01001110>(reg1);
    shuffle_and_compare<0b0101>(reg1);
    permute_and_compare<0b01001110>(reg0);
    shuffle_and_compare<0b0101>(reg0);
    return;
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
                         __m256d &reg2, __m256d &reg3) {
    bitonic_sort(reg0);       // sort first register
    bitonic_sort(reg1);       // sort second register
    bitonic_sort(reg2);       // sort third register
    bitonic_sort(reg3);       // sort fourth register
    bitonic_sort(reg0, reg1); // sort third register
    bitonic_sort(reg2, reg3); // sort fourth register
    reverse_and_compare(reg0, reg3);
    reverse_and_compare(reg1, reg2);

    compare(reg1, reg3);
    compare(reg0, reg2);

    compare(reg0, reg1);
    compare(reg2, reg3);

    permute_and_compare<0b01001110>(reg0);
    permute_and_compare<0b01001110>(reg1);
    permute_and_compare<0b01001110>(reg2);
    permute_and_compare<0b01001110>(reg3);

    shuffle_and_compare<0b0101>(reg0);
    shuffle_and_compare<0b0101>(reg1);
    shuffle_and_compare<0b0101>(reg2);
    shuffle_and_compare<0b0101>(reg3);

    return;
}

//////////////////////////////////////////////////////////////////
// 2^N DOUBLE ARRAY SORTING ALGORITHM

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the array to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 */
inline void compare_full_length(double *arr, unsigned start,
                                unsigned end) {
    unsigned length = end - start + 1;
    unsigned half = length / 2;
    for (unsigned i = 0; i < half; i += 4) {
        double *p1 = arr + start + i;
        double *p2 = arr + end - 3 - i;
        __m256d vec1 = _mm256_loadu_pd(p1);
        __m256d vec2 = _mm256_loadu_pd(p2);
        reverse_and_compare(vec1, vec2);
        _mm256_storeu_pd(p1, vec1);
        _mm256_storeu_pd(p2, vec2);
    }
}

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the array to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param depth a parameter to follow the depth of recursion
 */
inline void lane_crossing_compare(double *arr, unsigned start,
                                  unsigned end, unsigned depth) {

    unsigned length = end - start + 1;
    if (length == 4) {
        __m256d reg = _mm256_loadu_pd(arr + start);
        permute_and_compare<0b01001110>(reg);
        shuffle_and_compare<0b0101>(reg);
        _mm256_storeu_pd(arr + start, reg);
        return;
    }
    double *p = arr + start;
    for (unsigned i = 0; i < length / 2; i += 4) {
        {
            double *p1 = p + i;
            double *p2 = p + length / 2 + i;

            __m256d reg0 = _mm256_loadu_pd(p1); // i-ti od začetka
            __m256d reg1 =
                _mm256_loadu_pd(p2); // ta je prvi čez polovico
            // register 2 vsebuje min vrednosti
            __m256d min = _mm256_min_pd(reg1, reg0);
            // register 1 vsebuje max vrednosti
            reg1 = _mm256_max_pd(reg1, reg0);
            reg0 = min;
            // print_avx(max, "max: ");
            // print_avx(min, "min: ");
            _mm256_storeu_pd(p1, reg0);
            _mm256_storeu_pd(p2, reg1);
        }
    }
    lane_crossing_compare(arr, start, (start + end) / 2, depth + 1);
    lane_crossing_compare(arr, (start + end) / 2 + 1, end, depth + 1);
};

/**
 * @brief The function sorts an array with 2^n elements
 * @param array pointer to the start of the array
 * @param start index of the first number to be sorted
 * @param end index of the last number to be sorted
 * @details end-start+1 should be 2^n
 */
inline void sort_2n_vector(double *array, unsigned num_to_sort) {

    unsigned end = num_to_sort - 1;
    unsigned vec_count = num_to_sort / 4;
    assert(
        (num_to_sort >= 0 && !(num_to_sort & (num_to_sort - 1))) &&
        "The array to be sorted is not the power of 2!");

    //__m256d *arr = avx_vec.data();
    if (num_to_sort == 4) {

        __m256d vec = _mm256_loadu_pd(array);
        bitonic_sort(vec);
        _mm256_storeu_pd(array, vec);

    } else if (num_to_sort == 8) {

        __m256d vec1 = _mm256_loadu_pd(array);
        __m256d vec2 = _mm256_loadu_pd(array + 4);
        bitonic_sort(vec1, vec2);
        _mm256_storeu_pd(array, vec1);
        _mm256_storeu_pd(array + 4, vec2);

    } else if (num_to_sort == 16) {

        __m256d vec1 = _mm256_loadu_pd(array);
        __m256d vec2 = _mm256_loadu_pd(array + 4);
        __m256d vec3 = _mm256_loadu_pd(array + 8);
        __m256d vec4 = _mm256_loadu_pd(array + 12);
        bitonic_sort(vec1, vec2, vec3, vec4);
        _mm256_storeu_pd(array, vec1);
        _mm256_storeu_pd(array + 4, vec2);
        _mm256_storeu_pd(array + 8, vec3);
        _mm256_storeu_pd(array + 12, vec4);

    } else if (num_to_sort >= 32) {
        for (unsigned i = 0; i < end; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(array + i);
            bitonic_sort(vec1);
            _mm256_storeu_pd(array + i, vec1);
        }

        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        for (unsigned len = 8; len <= num_to_sort; len *= 2) {
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
// 4N DOUBLE ARRAY SORTING ALGORITHM --- the array must contain 8n
// elements

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the double array to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param last_index last index of array to be sorted, maximum is
 * array size - 1
 */
inline void compare_full_length(double *arr, unsigned start,
                                unsigned end, unsigned last_index) {

    unsigned length = end - start + 1;
    unsigned half = length / 2; // half je index prvega cez polovico
    for (int i = half - 4; i >= 0; i -= 4) {
        if (UNLIKELY(end - 3 - i > last_index))
            break;
        double *p1 = arr + start + i;
        double *p2 = arr + end - 3 - i;
        {
            __m256d vec1 = _mm256_loadu_pd(p1);
            __m256d vec2 = _mm256_loadu_pd(p2);
            // reverse one of registers register reg0
            __m256d reverse =
                _mm256_permute4x64_pd(vec1, 0b00011011);
            // register 2 vsebuje min vrednosti
            vec1 = _mm256_min_pd(vec2, reverse);
            vec1 = _mm256_permute4x64_pd(vec1, 0b00011011);
            // register 1 vsebuje max vrednosti
            vec2 = _mm256_max_pd(vec2, reverse);
            _mm256_storeu_pd(p1, vec1);
            _mm256_storeu_pd(p2, vec2);
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
inline void lane_crossing_compare(double *arr, unsigned start,
                                  unsigned end, unsigned last_index,
                                  unsigned depth) {

    if (start > last_index) {
        return;
    }
    unsigned length = end - start + 1;
    if (length == 4) {
        // this is the ending case do single vector permutations
        __m256d reg = _mm256_loadu_pd(arr + start);

        permute_and_compare<0b01001110>(reg);
        shuffle_and_compare<0b0101>(reg);
        _mm256_storeu_pd(arr + start, reg);

        return;
    }
    double *p = arr + start;
    for (unsigned i = 0; i < length / 2; i += 4) {
        if (start + length / 2 + i > last_index)
            break;
        {
            double *p1 = p + i;
            double *p2 = p + length / 2 + i;
            __m256d reg0 = _mm256_loadu_pd(p1); // i-ti od začetka
            __m256d reg1 =
                _mm256_loadu_pd(p2); // ta je prvi čez polovico
            // register 2 vsebuje min vrednosti
            __m256d min = _mm256_min_pd(reg1, reg0);
            // register 1 vsebuje max vrednosti
            reg1 = _mm256_max_pd(reg1, reg0);
            reg0 = min;
            _mm256_storeu_pd(p1, reg0);
            _mm256_storeu_pd(p2, reg1);
        }
    }
    lane_crossing_compare(arr, start, (start + end) / 2, last_index,
                          depth + 1);
    lane_crossing_compare(arr, (start + end) / 2 + 1, end,
                          last_index, depth + 1);
};

/**
 * @brief The function sorts an array with 8*n elements
 * @param array pointer to the start of the array
 * @param start index of the first number to be sorted
 * @param end index of the last number to be sorted
 * @details end-start+1 should be 8*n
 */
inline void sort_4n_vector(double *array, unsigned num_to_sort) {

    unsigned end = num_to_sort - 1;
    int pow2 = (int)std::ceil(std::log2f(num_to_sort));
    int imaginary_length = (int)std::pow(2, pow2);
    unsigned last_index = end - 3; // last index to be loaded
    assert((num_to_sort >= 0 && (mod4(num_to_sort) == 0)) &&
           "The array to be sorted is not a multiple of 4!");

    if (num_to_sort == 4) {
        __m256d vec = _mm256_loadu_pd(array);
        bitonic_sort(vec);
        _mm256_storeu_pd(array, vec);
    } else if (num_to_sort == 8) {
        __m256d vec1 = _mm256_loadu_pd(array);
        __m256d vec2 = _mm256_loadu_pd(array + 4);
        bitonic_sort(vec1, vec2);
        _mm256_storeu_pd(array, vec1);
        _mm256_storeu_pd(array + 4, vec2);

    } else if (num_to_sort == 16) {
        __m256d vec1 = _mm256_loadu_pd(array);
        __m256d vec2 = _mm256_loadu_pd(array + 4);
        __m256d vec3 = _mm256_loadu_pd(array + 8);
        __m256d vec4 = _mm256_loadu_pd(array + 12);
        bitonic_sort(vec1, vec2, vec3, vec4);
        _mm256_storeu_pd(array, vec1);
        _mm256_storeu_pd(array + 4, vec2);
        _mm256_storeu_pd(array + 8, vec3);
        _mm256_storeu_pd(array + 12, vec4);

    } else {
        for (unsigned i = 0; i < end; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(array + i);
            bitonic_sort(vec1);
            _mm256_storeu_pd(array + i, vec1);
        }

        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        for (unsigned len = 8; len <= imaginary_length; len *= 2) {
            // std::cout << "len: " << len << std::endl;
            // inner loop goes over all subdivisions
            for (unsigned n = 0; n < imaginary_length; n += len) {
                compare_full_length(array, n, n + len - 1,
                                    last_index);
                lane_crossing_compare(array, n, n + len - 1,
                                      last_index, 0);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////
// DOUBLE ARRAY SORTING ALGORITHM FOR ARRAYS OF ARBITRARY LENGTH,
// the array must contain 8n elements float implementation

inline void maskload(int diff, double *p2, __m256i &mask,
                     __m256d &reg1) {
    switch (diff) {
    case 0: {
        mask = _mm256_set_epi64x(0, 0, 0, LOAD);
        reg1 = _mm256_maskload_pd(p2, mask);
        __m256d infinity =
            _mm256_set1_pd(std::numeric_limits<double>::infinity());
        reg1 = _mm256_blend_pd(infinity, reg1, 0b0001);
    } break;
    case 1: {
        mask = _mm256_set_epi64x(0, 0, LOAD, LOAD);
        reg1 = _mm256_maskload_pd(p2, mask);
        __m256d infinity =
            _mm256_set1_pd(std::numeric_limits<double>::infinity());
        reg1 = _mm256_blend_pd(infinity, reg1, 0b0011);
    } break;
    case 2: {
        mask = _mm256_set_epi64x(0, LOAD, LOAD, LOAD);
        reg1 = _mm256_maskload_pd(p2, mask);
        __m256d infinity =
            _mm256_set1_pd(std::numeric_limits<double>::infinity());
        reg1 = _mm256_blend_pd(infinity, reg1, 0b0111);
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
        __m256d vec2;
        __m256i mask;
        if (diff < 3)
            maskload(diff, p2, mask, vec2);
        else
            vec2 = _mm256_loadu_pd(p2);

        __m256d vec1 = _mm256_loadu_pd(p1);
        reverse_and_compare(vec1, vec2);
        vec1 = _mm256_permute4x64_pd(vec1, 0b00011011);
        _mm256_storeu_pd(p1, vec1);
        if (diff <= 2)
            _mm256_maskstore_pd(p2, mask, vec2);
        else
            _mm256_storeu_pd(p2, vec2);
    }
}

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the array to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param depth a parameter to follow the depth of recursion
 */
inline void lane_crossing_compare_all_cases(double *arr,
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
        __m256i mask;
        if (diff < 3)
            maskload(diff, arr + start, mask, reg);
        else
            reg = _mm256_loadu_pd(arr + start);
        permute_and_compare<0b01001110>(reg);
        shuffle_and_compare<0b0101>(reg);
        if (UNLIKELY(diff < 3))
            _mm256_maskstore_pd(arr + start, mask, reg);
        else
            _mm256_storeu_pd(arr + start, reg);

        return;
    }

    double *p = arr + start;
    for (unsigned i = 0; i < length / 2; i += 4) {
        int diff = last_index - (start + length / 2 + i);
        if (diff < 0)
            break;
        __m256d reg1;
        __m256i mask;
        double *p2 = p + length / 2 + i;
        if (diff < 3)
            maskload(diff, p2, mask, reg1);
        else
            reg1 = _mm256_loadu_pd(p2);

        double *p1 = p + i;
        __m256d reg0 = _mm256_loadu_pd(p1); // i-ti od začetka
        // register 2 vsebuje min vrednosti
        __m256d min = _mm256_min_pd(reg1, reg0);
        // register 1 vsebuje max vrednosti
        reg1 = _mm256_max_pd(reg1, reg0);
        reg0 = min;
        _mm256_storeu_pd(p1, reg0);
        if (diff < 3)
            _mm256_maskstore_pd(p2, mask, reg1);
        else
            _mm256_storeu_pd(p2, reg1);
    }
    lane_crossing_compare_all_cases(arr, start, (start + end) / 2,
                                    last_index, depth + 1);
    lane_crossing_compare_all_cases(arr, (start + end) / 2 + 1, end,
                                    last_index, depth + 1);
};

/**
 * @brief The function sorts an array with 2^n elements
 * @param array pointer to the start of the array
 * @param start index of the first number to be sorted
 * @param end index of the last number to be sorted
 * @details end - start + 1 should be 2^n
 */
inline void sort_vector(std::span<double> span) {
    auto array = span.data();
    auto num_to_sort = span.size();

    unsigned end = num_to_sort - 1;
    if (num_to_sort <= 1)
        return;
    else if (num_to_sort < 4) {
        int reminder = mod4(end);
        double *p = array + end - reminder;
        __m256d reg1;
        __m256i mask;
        maskload(reminder, p, mask, reg1);
        bitonic_sort(reg1);
        _mm256_maskstore_pd(array + end - reminder, mask, reg1);
    } else if (!(num_to_sort & (num_to_sort - 1)))
        sort_2n_vector(array, num_to_sort);
    else if (mod4(num_to_sort) == 0)
        sort_4n_vector(array, num_to_sort);
    else {
        int pow2 = (int)std::ceil(std::log2f(end + 1));
        int imaginary_length = (int)std::pow(2, pow2);
        unsigned last_index = end;

        for (unsigned i = 0; i <= end - 3; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(array + i);
            bitonic_sort(vec1);
            _mm256_storeu_pd(array + i, vec1);
        }
        ///////////////////////////////// load the partial one
        int reminder = mod4(end);
        double *p = array + end - reminder;
        __m256d reg1;
        __m256i mask;
        maskload(reminder, p, mask, reg1);
        bitonic_sort(reg1);
        _mm256_maskstore_pd(array + end - reminder, mask, reg1);

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
                compare_full_length_all_cases(array, n, n + len - 1,
                                              last_index);
                lane_crossing_compare_all_cases(
                    array, n, n + len - 1, last_index, 0);
            }
        }
    }
}

} // namespace BITONIC_SORT
