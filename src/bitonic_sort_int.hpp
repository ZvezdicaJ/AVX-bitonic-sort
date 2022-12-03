#include "type_definitions.hpp"

#include <cassert>
#include <cstdint>
#include <vector>
#include <cmath>
#include <span>

namespace BITONIC_SORT {

#define shuffle_and_compare(reg, mask1, mask2) \
    __m256i shuffled_reg = _mm256_shuffle_epi32(reg, int(mask1)); \
    __m256i max = _mm256_max_epi32(reg, shuffled_reg); \
    __m256i min = _mm256_min_epi32(reg, shuffled_reg); \
    reg = _mm256_blend_epi32(max, min, int(mask2)); \

inline __attribute__((always_inline)) void
reverse_and_compare(__m256i &reg) {
    __m256i reversed_halves =
        _mm256_permute2f128_si256(reg, reg, 0b00000001);
    __m256i reversed =
        _mm256_shuffle_epi32(reversed_halves, 0b00011011);
    __m256i max = _mm256_max_epi32(reg, reversed);
    __m256i min = _mm256_min_epi32(reg, reversed);
    reg = _mm256_blend_epi32(max, min, 0b00001111);
}

/**
 *@brief The function accepts a single __m256i vector and sorts it.
 * @param reg register to be sorted
 */
void bitonic_sort(__m256i &reg) {
    //    shuffle_and_compare(reg, 0b10110001, 0b10101010);
    shuffle_and_compare(reg, 0b10110001, 0b01010101);
    shuffle_and_compare(reg, 0b00011011, 0b00110011);
    shuffle_and_compare(reg, 0b10110001, 0b01010101);
    reverse_and_compare(reg);
    shuffle_and_compare(reg, 0b01001110, 0b00110011);
    shuffle_and_compare(reg, 0b10110001, 0b01010101);
}

inline __attribute__((always_inline)) void
reverse_and_compare(__m256i &reg0, __m256i &reg1) {
    // reverse one of registers register reg0
    __m256i reversed_halves =
        _mm256_permute2f128_si256(reg0, reg0, 0b00000001);
    __m256i reversed =
        _mm256_shuffle_epi32(reversed_halves, 0b00011011);
    // register 2 vsebuje min vrednosti
    reg0 = _mm256_min_epi32(reg1, reversed);
    // register 1 vsebuje max vrednosti
    reg1 = _mm256_max_epi32(reg1, reversed);
}

inline __attribute__((always_inline)) void
reverse_halves_and_compare(__m256i &reg0) {
    // tu narediš *----* *----* *----* *----*
    __m256i reversed_halves =
        _mm256_permute2f128_si256(reg0, reg0, 0b00000001);
    __m256i max = _mm256_max_epi32(reg0, reversed_halves);
    __m256i min = _mm256_min_epi32(reg0, reversed_halves);
    reg0 = _mm256_blend_epi32(max, min, 0b00001111);
}

inline __attribute__((always_inline)) void compare(__m256i &reg0,
                                                   __m256i &reg1) {
    {
        __m256i max = _mm256_max_epi32(reg0, reg1);
        __m256i min = _mm256_min_epi32(reg0, reg1);
        reg0 = min;
        reg1 = max;
    }
}

/**
 * @brief The function accepts two unsorted  __m256i vectors and
 * sorts them.
 * @param reg1 upper vector of numbers - at the end it contains
 * larger values, the largest value is in the upper half of register
 * [255:192]
 *
 * @param reg0 lower vector of numbers - at the end it contains
 * smaller values. The smallest value is in the lowest half of
 * register - at [63:0]
 *
 */

inline void bitonic_sort(__m256i &reg0, __m256i &reg1) {
    bitonic_sort(reg1); // sort first register
    bitonic_sort(reg0); // sort second register
    reverse_and_compare(reg0, reg1);
    reverse_halves_and_compare(reg0);
    reverse_halves_and_compare(reg1);
    shuffle_and_compare(reg0, 0b01001110, 0b00110011);
    shuffle_and_compare(reg0, 0b10110001, 0b01010101);
    shuffle_and_compare(reg1, 0b01001110, 0b00110011);
    shuffle_and_compare(reg1, 0b10110001, 0b01010101);
    return;
}
/**
 * @brief The function accepts two already sorted  __m256i vectors
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

inline void bitonic_merge(__m256i &reg0, __m256i &reg1) {
    reverse_and_compare(reg0, reg1);
    reverse_halves_and_compare(reg0);
    reverse_halves_and_compare(reg1);
    shuffle_and_compare(reg0, 0b01001110, 0b00110011);
    shuffle_and_compare(reg0, 0b10110001, 0b01010101);
    shuffle_and_compare(reg1, 0b01001110, 0b00110011);
    shuffle_and_compare(reg1, 0b10110001, 0b01010101);
    return;
}

/**
 * @brief The function accepts four unsorted  __m256i vectors and
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
inline void bitonic_sort(__m256i &reg0, __m256i &reg1,
                         __m256i &reg2, __m256i &reg3) {

    // sort each quarter
    bitonic_sort(reg0);
    bitonic_sort(reg1);
    bitonic_sort(reg2);
    bitonic_sort(reg3);
    // sort each half
    bitonic_sort(reg0, reg1);
    bitonic_sort(reg2, reg3);

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

    shuffle_and_compare(reg0, 0b01001110, 0b00110011);
    shuffle_and_compare(reg1, 0b01001110, 0b00110011);
    shuffle_and_compare(reg2, 0b01001110, 0b00110011);
    shuffle_and_compare(reg3, 0b01001110, 0b00110011);

    shuffle_and_compare(reg0, 0b10110001, 0b01010101);
    shuffle_and_compare(reg1, 0b10110001, 0b01010101);
    shuffle_and_compare(reg2, 0b10110001, 0b01010101);
    shuffle_and_compare(reg3, 0b10110001, 0b01010101);
}

/**
 * @brief The function accepts four sorted  __m256i vectors and
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
inline void bitonic_merge(__m256i &reg0, __m256i &reg1,
                          __m256i &reg2, __m256i &reg3) {
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

    shuffle_and_compare(reg0, 0b01001110, 0b00110011);
    shuffle_and_compare(reg1, 0b01001110, 0b00110011);
    shuffle_and_compare(reg2, 0b01001110, 0b00110011);
    shuffle_and_compare(reg3, 0b01001110, 0b00110011);

    shuffle_and_compare(reg0, 0b10110001, 0b01010101);
    shuffle_and_compare(reg1, 0b10110001, 0b01010101);
    shuffle_and_compare(reg2, 0b10110001, 0b01010101);
    shuffle_and_compare(reg3, 0b10110001, 0b01010101);
}

//////////////////////////////////////////////////////////////////
// 2^N INT ARRAY SORTING ALGORITHM

// int implementation
/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the int array to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 */
inline void compare_full_length(int *arr, unsigned start,
                                unsigned end) {
    unsigned length = end - start + 1;
    unsigned half = length / 2;
    for (int i = 0; i < half; i += 8) {
        int *p1 = arr + start + i;
        int *p2 = arr + end - 7 - i;
        { // reverse lover half and compare to upper half
            __m256i vec1 = _mm256_loadu_si256((__m256i *)p1);
            __m256i vec2 = _mm256_loadu_si256((__m256i *)p2);
            reverse_and_compare(vec1, vec2);
            _mm256_storeu_si256((__m256i *)p1, vec1);
            _mm256_storeu_si256((__m256i *)p2, vec2);
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
inline void lane_crossing_compare(int *arr, unsigned start,
                                  unsigned end, unsigned depth) {
    unsigned length = end - start + 1;
    if (length == 8) {
        __m256i reg = _mm256_loadu_si256((__m256i *)(arr + start));
        // this is the ending case do single vector permutations
        reverse_halves_and_compare(reg);
        shuffle_and_compare(reg, 0b01001110, 0b00110011);
        shuffle_and_compare(reg, 0b10110001, 0b01010101);
        _mm256_storeu_si256((__m256i *)(arr + start), reg);
        return;
    }
    int *p = arr + start;
    for (unsigned i = 0; i < length / 2; i += 8) {
        {
            int *p1 = p + i;
            int *p2 = p + length / 2 + i;
            __m256i reg0 = _mm256_loadu_si256(
                (__m256i *)p1); // i-ti od začetka
            __m256i reg1 = _mm256_loadu_si256(
                (__m256i *)p2); // ta je prvi čez polovico
            // register 2 vsebuje min vrednosti
            __m256i min = _mm256_min_epi32(reg1, reg0);
            // register 1 vsebuje max vrednosti
            reg1 = _mm256_max_epi32(reg1, reg0);
            reg0 = min;
            _mm256_storeu_si256((__m256i *)p1, reg0);
            _mm256_storeu_si256((__m256i *)p2, reg1);
        }
    }
    lane_crossing_compare(arr, start, (start + end) / 2, depth + 1);
    lane_crossing_compare(arr, (start + end) / 2 + 1, end,
                          depth + 1);
};

/**
 * @brief The function sorts an array with 8*n elements
 * @param array pointer to the first element to be sorted
 * @param num_to_sort number of elements to sort
 * @details num_to_sort should be 2^n. If it's not,
 * then add additional elements - like infinity, so that it will be
 * large enough
 */
inline void sort_2n_vector(int *array, unsigned num_to_sort) {

    unsigned vec_count = num_to_sort / 8;

    assert(
        (num_to_sort >= 0 && !(num_to_sort & (num_to_sort - 1))) &&
        "The array to be sorted is not the power of 2!");

    //__m256id *arr = avx_vec.data();
    if (num_to_sort == 8) {

        __m256i vec = _mm256_loadu_si256((__m256i *)array);
        bitonic_sort(vec);
        _mm256_storeu_si256((__m256i *)array, vec);

    } else if (num_to_sort == 16) {

        __m256i vec1 = _mm256_loadu_si256((__m256i *)array);
        __m256i vec2 = _mm256_loadu_si256((__m256i *)(array + 8));
        bitonic_sort(vec1, vec2);
        _mm256_storeu_si256((__m256i *)array, vec1);
        _mm256_storeu_si256((__m256i *)(array + 8), vec2);

    } else if (num_to_sort == 32) {

        __m256i vec1 = _mm256_loadu_si256((__m256i *)array);
        __m256i vec2 = _mm256_loadu_si256((__m256i *)(array + 8));
        __m256i vec3 = _mm256_loadu_si256((__m256i *)(array + 16));
        __m256i vec4 = _mm256_loadu_si256((__m256i *)(array + 24));
        bitonic_sort(vec1, vec2, vec3, vec4);
        _mm256_storeu_si256((__m256i *)array, vec1);
        _mm256_storeu_si256((__m256i *)(array + 8), vec2);
        _mm256_storeu_si256((__m256i *)(array + 16), vec3);
        _mm256_storeu_si256((__m256i *)(array + 24), vec4);

    } else if (num_to_sort >= 64) {
        for (unsigned i = 0; i < num_to_sort; i += 8) {
            __m256i vec1 =
                _mm256_loadu_si256((__m256i *)(array + i));
            bitonic_sort(vec1);
            _mm256_storeu_si256((__m256i *)(array + i), vec1);
        }

        // outer loop
        // len is number of ints in length to be compared
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
// INT ARRAY SORTING ALGORITHM, the array must contain 8n elements

// int implementation
/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the int array to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param last_index last index of array to be sorted, maximum is
 * array size - 1
 */
inline void compare_full_length(int *arr, unsigned start,
                                unsigned end, unsigned last_index) {
    unsigned length = end - start + 1;
    unsigned half = length / 2; // half je index prvega cez polovico
    for (int i = half - 8; i >= 0; i -= 8) {
        if (end - 7 - i > last_index)
            break;
        int *p1 = arr + start + i;
        int *p2 = arr + end - 7 - i;

        { // reverse lover half and compare to upper half
            __m256i vec1 = _mm256_loadu_si256((__m256i *)p1);
            __m256i vec2 = _mm256_loadu_si256((__m256i *)p2);
            __m256i reversed_halves =
                _mm256_permute2f128_si256(vec1, vec1, 0b00000001);
            __m256i reversed =
                _mm256_shuffle_epi32(reversed_halves, 0b00011011);
            vec1 = _mm256_min_epi32(reversed, vec2);
            vec2 = _mm256_max_epi32(reversed, vec2);
            reversed_halves =
                _mm256_permute2f128_si256(vec1, vec1, 0b00000001);
            vec1 =
                _mm256_shuffle_epi32(reversed_halves, 0b00011011);
            _mm256_storeu_si256((__m256i *)p1, vec1);
            _mm256_storeu_si256((__m256i *)p2, vec2);
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
inline void lane_crossing_compare(int *arr, unsigned start,
                                  unsigned end, unsigned last_index,
                                  unsigned depth) {

    if (start > last_index) {
        // std::cout << "aborted start index: " << start <<
        // std::endl;
        return;
    }
    unsigned length = end - start + 1;
    if (length == 8) {
        __m256i reg = _mm256_loadu_si256((__m256i *)(arr + start));
        // this is the ending case do single vector permutations
        reverse_halves_and_compare(reg);
        shuffle_and_compare(reg, 0b01001110, 0b00110011);
        shuffle_and_compare(reg, 0b10110001, 0b01010101);
        _mm256_storeu_si256((__m256i *)(arr + start), reg);
        return;
    }
    int *p = arr + start;
    // for (unsigned i = 0; i < length / 2; i += 8) {
    for (unsigned i = 0; i < length / 2; i += 8) {
        if (start + length / 2 + i > last_index)
            break;
        {
            int *p1 = p + i;
            int *p2 = p + length / 2 + i;
            __m256i reg0 = _mm256_loadu_si256(
                (__m256i *)p1); // i-ti od začetka
            __m256i reg1 = _mm256_loadu_si256(
                (__m256i *)p2); // ta je prvi čez polovico
            // register 2 vsebuje min vrednosti
            __m256i min = _mm256_min_epi32(reg1, reg0);
            // register 1 vsebuje max vrednosti
            reg1 = _mm256_max_epi32(reg1, reg0);
            reg0 = min;
            _mm256_storeu_si256((__m256i *)p1, reg0);
            _mm256_storeu_si256((__m256i *)p2, reg1);
        }
    }
    lane_crossing_compare(arr, start, (start + end) / 2, last_index,
                          depth + 1);
    lane_crossing_compare(arr, (start + end) / 2 + 1, end,
                          last_index, depth + 1);
};

/**
 * @brief The function sorts an array with 2^n elements
 * @param array pointer to the first element to be sorted
 * @param num_to_sort number of elements to be sorted
 * @details num_to_sort should be multiple of 8.
 */
inline void sort_8n_vector(int *array, unsigned num_to_sort) {

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
        __m256i vec = _mm256_loadu_si256((__m256i *)array);
        bitonic_sort(vec);
        _mm256_storeu_si256((__m256i *)array, vec);
    } else if (num_to_sort == 16) {
        __m256i vec1 = _mm256_loadu_si256((__m256i *)array);
        __m256i vec2 = _mm256_loadu_si256((__m256i *)(array + 8));
        bitonic_sort(vec1, vec2);
        _mm256_storeu_si256((__m256i *)array, vec1);
        _mm256_storeu_si256((__m256i *)(array + 8), vec2);

    } else if (num_to_sort == 32) {
        __m256i vec1 = _mm256_loadu_si256((__m256i *)array);
        __m256i vec2 = _mm256_loadu_si256((__m256i *)(array + 8));
        __m256i vec3 = _mm256_loadu_si256((__m256i *)(array + 16));
        __m256i vec4 = _mm256_loadu_si256((__m256i *)(array + 24));
        bitonic_sort(vec1, vec2, vec3, vec4);
        _mm256_storeu_si256((__m256i *)array, vec1);
        _mm256_storeu_si256((__m256i *)(array + 8), vec2);
        _mm256_storeu_si256((__m256i *)(array + 16), vec3);
        _mm256_storeu_si256((__m256i *)(array + 24), vec4);

    } else {
        for (unsigned i = 0; i < end; i += 8) {
            __m256i vec1 =
                _mm256_loadu_si256((__m256i *)(array + i));
            bitonic_sort(vec1);
            _mm256_storeu_si256((__m256i *)(array + i), vec1);
        }
        // outer loop
        // len is number of ints in length to be compared
        // each step increases this length by factor of 2.
        for (unsigned len = 16; len <= imaginary_length; len *= 2) {
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
// INT ARRAY SORTING ALGORITHM FOR ARRAYS OF ARBITRARY LENGTH, the
// array must contain 8n elements int implementation

inline void maskload(int toLoad, int *p2, __m256i &mask,
                     __m256i &reg1) {
    switch (toLoad) {
    case 0: {
        mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, LOAD);
        reg1 = _mm256_maskload_epi32(p2, mask);
        __m256i infinity =
            _mm256_set1_epi32(std::numeric_limits<int>::max());
        reg1 = _mm256_blend_epi32(infinity, reg1, 0b00000001);
    } break;
    case 1: {
        mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, LOAD, LOAD);
        reg1 = _mm256_maskload_epi32(p2, mask);
        __m256i infinity =
            _mm256_set1_epi32(std::numeric_limits<int>::max());
        reg1 = _mm256_blend_epi32(infinity, reg1, 0b00000011);
    } break;
    case 2: {
        mask = _mm256_set_epi32(0, 0, 0, 0, 0, LOAD, LOAD, LOAD);
        reg1 = _mm256_maskload_epi32(p2, mask);
        __m256i infinity =
            _mm256_set1_epi32(std::numeric_limits<int>::max());
        reg1 = _mm256_blend_epi32(infinity, reg1, 0b00000111);
    } break;
    case 3: {
        mask = _mm256_set_epi32(0, 0, 0, 0, LOAD, LOAD, LOAD, LOAD);
        reg1 = _mm256_maskload_epi32(p2, mask);
        __m256i infinity =
            _mm256_set1_epi32(std::numeric_limits<int>::max());
        reg1 = _mm256_blend_epi32(infinity, reg1, 0b00001111);
    } break;
    case 4: {
        mask =
            _mm256_set_epi32(0, 0, 0, LOAD, LOAD, LOAD, LOAD, LOAD);
        reg1 = _mm256_maskload_epi32(p2, mask);
        __m256i infinity =
            _mm256_set1_epi32(std::numeric_limits<int>::max());
        reg1 = _mm256_blend_epi32(infinity, reg1, 0b00011111);
    } break;
    case 5: {
        mask = _mm256_set_epi32(0, 0, LOAD, LOAD, LOAD, LOAD, LOAD,
                                LOAD);
        reg1 = _mm256_maskload_epi32(p2, mask);
        __m256i infinity =
            _mm256_set1_epi32(std::numeric_limits<int>::max());
        reg1 = _mm256_blend_epi32(infinity, reg1, 0b00111111);
    } break;
    case 6: {
        mask = _mm256_set_epi32(0, LOAD, LOAD, LOAD, LOAD, LOAD,
                                LOAD, LOAD);
        reg1 = _mm256_maskload_epi32(p2, mask);
        __m256i infinity =
            _mm256_set1_epi32(std::numeric_limits<int>::max());
        reg1 = _mm256_blend_epi32(infinity, reg1, 0b01111111);
    } break;
    };
}

/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the int array to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 */
inline void compare_full_length_all_cases(int *arr, unsigned start,
                                          unsigned end,
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
        int *p1 = arr + start + i;
        int *p2 = arr + last_vec_to_load;
        __m256i vec2;
        __m256i mask;
        if (UNLIKELY(diff < 7))
            maskload(diff, p2, mask, vec2);
        else
            vec2 = _mm256_loadu_si256((__m256i *)p2);
        { // reverse lover half and compare to upper half
            __m256i vec1 = _mm256_loadu_si256((__m256i *)p1);
            __m256i reversed_halves =
                _mm256_permute2f128_si256(vec1, vec1, 0b00000001);
            __m256i reversed =
                _mm256_shuffle_epi32(reversed_halves, 0b00011011);
            vec1 = _mm256_min_epi32(reversed, vec2);
            vec2 = _mm256_max_epi32(reversed, vec2);
            reversed_halves =
                _mm256_permute2f128_si256(vec1, vec1, 0b00000001);
            vec1 =
                _mm256_shuffle_epi32(reversed_halves, 0b00011011);
            _mm256_storeu_si256((__m256i *)p1, vec1);
            if (UNLIKELY(diff <= 6))
                _mm256_maskstore_epi32(p2, mask, vec2);
            else
                _mm256_storeu_si256((__m256i *)p2, vec2);
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
inline void lane_crossing_compare_all_cases(int *arr,
                                            unsigned start,
                                            unsigned end,
                                            unsigned last_index,
                                            unsigned depth) {
    if (start > last_index) {
        return;
    }
    unsigned length = end - start + 1;
    if (length == 8) {
        int diff = last_index - start;
        if (diff < 1)
            return;
        __m256i reg;
        __m256i mask;
        if (UNLIKELY(diff < 7))
            maskload(diff, arr + start, mask, reg);
        else
            reg = _mm256_loadu_si256((__m256i *)(arr + start));

        reverse_halves_and_compare(reg);
        shuffle_and_compare(reg, 0b01001110, 0b00110011);
        shuffle_and_compare(reg, 0b10110001, 0b01010101);
        if (diff < 7)
            _mm256_maskstore_epi32(arr + start, mask, reg);
        else
            _mm256_storeu_si256((__m256i *)(arr + start), reg);

        return;
    }

    int *p = arr + start;
    for (unsigned i = 0; i < length / 2; i += 8) {
        int diff = last_index - (start + length / 2 + i);
        if (UNLIKELY(diff < 0))
            break;
        __m256i reg1;
        __m256i mask;
        int *p2 = p + length / 2 + i;
        if (UNLIKELY(diff < 7))
            maskload(diff, p2, mask, reg1);
        else
            reg1 = _mm256_loadu_si256((__m256i *)p2);
        int *p1 = p + i;
        __m256i reg0 =
            _mm256_loadu_si256((__m256i *)p1); // i-ti od začetka
        // register 2 vsebuje min vrednosti
        __m256i min = _mm256_min_epi32(reg1, reg0);
        // register 1 vsebuje max vrednosti
        reg1 = _mm256_max_epi32(reg1, reg0);
        reg0 = min;
        _mm256_storeu_si256((__m256i *)p1, reg0);
        if (UNLIKELY(diff < 7))
            _mm256_maskstore_epi32(p2, mask, reg1);
        else
            _mm256_storeu_si256((__m256i *)p2, reg1);
    }
    lane_crossing_compare_all_cases(arr, start, (start + end) / 2,
                                    last_index, depth + 1);
    lane_crossing_compare_all_cases(arr, (start + end) / 2 + 1, end,
                                    last_index, depth + 1);
};

/**
 * @brief The function sorts an array with arbitrary number of elements
 * @param array pointer to the first element to be sorted
 * @param num_to_sort number of elements to be sorted
 */
inline void sort_vector(std::span<int> span) {

    auto size = span.size();
    auto array = span.data();
    unsigned end = size - 1;

    if (size <= 1) {
        return;
    } else if (size < 8) {
        __m256i mask;
        __m256i reg;
        maskload(size - 1, array, mask, reg);
        bitonic_sort(reg);
        _mm256_maskstore_epi32(array, mask, reg);
    } else if (!(size & (size - 1))) {
        sort_2n_vector(array, size);
    } else if (mod8(size) == 0) {
        sort_8n_vector(array, size);
    } else if (size < 16) {
        __m256i mask;
        __m256i reg1, reg2;
        reg1 = _mm256_loadu_si256((__m256i *)array);
        maskload(size - 9, array + 8, mask, reg2);

        bitonic_sort(reg1, reg2);

        _mm256_storeu_si256((__m256i *)array, reg1);
        _mm256_maskstore_epi32(array + 8, mask, reg2);
    } else {
        int pow2 = (int)std::ceil(std::log2f(size));
        int imaginary_length = (int)std::pow(2, pow2);
        unsigned last_index = end;

        for (unsigned i = 0; i <= end - 7; i += 8) {
            __m256i vec1 =
                _mm256_loadu_si256((__m256i *)(array + i));
            bitonic_sort(vec1);
            _mm256_storeu_si256((__m256i *)(array + i), vec1);
        }
        ///////////////////////////////// load the partial one
        int reminder = mod8(end);
        int *p = array + end - reminder;
        __m256i reg1;
        __m256i mask;
        maskload(reminder, p, mask, reg1);
        bitonic_sort(reg1);
        _mm256_maskstore_epi32(array + end - reminder, mask, reg1);

        ///////////////////////////////////////////////////////

        // outer loop
        // len is number of ints in length to be compared
        // each step increases this length by factor of 2.
        // 8 and less has already been done above
        // for (unsigned len = 16; len <= imaginary_length; len *=
        // 2) {
        for (unsigned len = 16; len <= imaginary_length; len *= 2) {
            // std::cout << "len: " << len << std::endl;
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
