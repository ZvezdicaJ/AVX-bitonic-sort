#include "bitonic_sort.hpp"
#include "type_definitions.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <span>
#include <vector>
#include <fstream>

namespace bitonic_sort {
namespace {

struct alignas(32) RegMask {
    __m256 reg;
    __m256i mask;
};

struct InternalSortParams {
    std::span<float> span;
    std::size_t firstIdx;
    std::size_t lastIdx;
};

RegMask maskload(std::span<float const> const &span) {
    __m256 reg;
    __m256i mask;
    auto p = span.data();
    switch (span.size()) {
    case 1:
        {
            mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, LOAD);
            reg = _mm256_maskload_ps(p, mask);
            __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            reg = _mm256_blend_ps(infinity, reg, 0b00000001);
        }
        break;
    case 2:
        {
            mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, LOAD, LOAD);
            reg = _mm256_maskload_ps(p, mask);
            __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            reg = _mm256_blend_ps(infinity, reg, 0b00000011);
        }
        break;
    case 3:
        {
            mask = _mm256_set_epi32(0, 0, 0, 0, 0, LOAD, LOAD, LOAD);
            reg = _mm256_maskload_ps(p, mask);
            __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            reg = _mm256_blend_ps(infinity, reg, 0b00000111);
        }
        break;
    case 4:
        {
            mask = _mm256_set_epi32(0, 0, 0, 0, LOAD, LOAD, LOAD, LOAD);
            reg = _mm256_maskload_ps(p, mask);
            __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            reg = _mm256_blend_ps(infinity, reg, 0b00001111);
        }
        break;
    case 5:
        {
            mask = _mm256_set_epi32(0, 0, 0, LOAD, LOAD, LOAD, LOAD, LOAD);
            reg = _mm256_maskload_ps(p, mask);
            __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            reg = _mm256_blend_ps(infinity, reg, 0b00011111);
        }
        break;
    case 6:
        {
            mask = _mm256_set_epi32(0, 0, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD);
            reg = _mm256_maskload_ps(p, mask);
            __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            reg = _mm256_blend_ps(infinity, reg, 0b00111111);
        }
        break;
    case 7:
        {
            mask = _mm256_set_epi32(0, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD);
            reg = _mm256_maskload_ps(p, mask);
            __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            reg = _mm256_blend_ps(infinity, reg, 0b01111111);
        }
        break;
    case 8:
        {
            mask = _mm256_set_epi32(LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD);
            reg = _mm256_loadu_ps(p);
        }
        break;
    default:
        throw std::runtime_error("Invalid number of floats to load.");
    };
    return RegMask{std::move(reg), std::move(mask)};
}

template <std::uint8_t mask1, std::uint8_t mask2>
void shuffleAndCompare(__m256 &reg) {
    __m256 shuffled_reg = _mm256_shuffle_ps(reg, reg, mask1);
    __m256 max = _mm256_max_ps(reg, shuffled_reg);
    __m256 min = _mm256_min_ps(reg, shuffled_reg);
    reg = _mm256_blend_ps(max, min, mask2);
}

__m256 reverseRegister(__m256 const &reg) {
    // [1,2,3,4,5,6,7,8] -> [5, 6, 7, 8, 1, 2, 3, 4]
    auto reversed = _mm256_permute2f128_ps(reg, reg, 0b00000001);
    // [5, 6, 7, 8, 1, 2, 3, 4] -> [8, 7, 6, 5, 4, 3, 2, 1]
    return _mm256_shuffle_ps(reversed, reversed, 0b00011011);
}

void reverseAndCompare(__m256 &reg) {
    __m256 reversed = reverseRegister(reg);
    __m256 max = _mm256_max_ps(reg, reversed);
    __m256 min = _mm256_min_ps(reg, reversed);
    reg = _mm256_blend_ps(max, min, 0b00001111);
}

void reverseAndCompare(__m256 &reg0, __m256 &reg1) {
    __m256 reversed = reverseRegister(reg0);
    // register 2 vsebuje min vrednosti
    reg0 = _mm256_min_ps(reg1, reversed);
    // register 1 vsebuje max vrednosti
    reg1 = _mm256_max_ps(reg1, reversed);
}

void reverseHalvesAndCompare(__m256 &reg0) {
    // tu narediš *----* *----* *----* *----*
    __m256 reversed_halves = _mm256_permute2f128_ps(reg0, reg0, 0b00000001);
    __m256 max = _mm256_max_ps(reg0, reversed_halves);
    __m256 min = _mm256_min_ps(reg0, reversed_halves);
    reg0 = _mm256_blend_ps(max, min, 0b00001111);
}

void computeMinMax(__m256 &reg0, __m256 &reg1) {
    {
        __m256 max = _mm256_max_ps(reg0, reg1);
        __m256 min = _mm256_min_ps(reg0, reg1);
        reg0 = min;
        reg1 = max;
    }
}

inline void compareFullLength2N(InternalSortParams const& params) {

    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    assert(lastIdx > firstIdx);
    std::size_t length = lastIdx - firstIdx + 1;
    std::size_t half = length / 2;
    for (std::size_t i = 0; i < half; i += 8) {
        float *p1 = params.span.data() + firstIdx + i;
        float *p2 = params.span.data() + lastIdx - 7 - i;
        { // reverse lover half and compare to upper half
            __m256 vec1 = _mm256_loadu_ps(p1);
            __m256 vec2 = _mm256_loadu_ps(p2);
            reverseAndCompare(vec1, vec2);
            _mm256_storeu_ps(p1, vec1);
            _mm256_storeu_ps(p2, vec2);
        }
    }
}

inline void laneCrossingCompare2N(InternalSortParams const &params, std::uint32_t depth) {
    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t length = lastIdx - firstIdx + 1;
    if (length == 8) {
        __m256 reg = _mm256_loadu_ps(params.span.data() + firstIdx);
        // this is the ending case do single vector permutations
        reverseHalvesAndCompare(reg);
        shuffleAndCompare<0b01001110, 0b00110011>(reg);
        shuffleAndCompare<0b10110001, 0b01010101>(reg);
        _mm256_storeu_ps(params.span.data() + firstIdx, reg);
        return;
    }
    float *p = params.span.data() + firstIdx;
    for (std::size_t i = 0; i < length / 2; i += 8) {
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
    laneCrossingCompare2N({params.span, firstIdx, (firstIdx + lastIdx) / 2}, depth + 1);
    laneCrossingCompare2N({params.span, (firstIdx + lastIdx) / 2 + 1, lastIdx}, depth + 1);
};

inline void compareFullLength8N(InternalSortParams const &params) {
    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const &maxIdx = params.span.size() - 1;

    std::size_t length = lastIdx - firstIdx + 1;
    std::size_t patHalfIdx = length / 2; // half je index prvega cez polovico
    for (std::int32_t toLoadIdx = patHalfIdx - 8; toLoadIdx >= 0; toLoadIdx -= 8) {
        if (lastIdx - 7 - toLoadIdx > maxIdx)
            break;
        float *p1 = params.span.data() + firstIdx + toLoadIdx;
        float *p2 = params.span.data() + lastIdx - 7 - toLoadIdx;

        __m256 vec1 = _mm256_loadu_ps(p1);
        __m256 vec2 = _mm256_loadu_ps(p2);
        auto reversed = reverseRegister(vec1);
        vec1 = _mm256_min_ps(reversed, vec2);
        vec2 = _mm256_max_ps(reversed, vec2);
        vec1 = reverseRegister(vec1);
        _mm256_storeu_ps(p1, vec1);
        _mm256_storeu_ps(p2, vec2);
    }
}

inline void laneCrossingCompare8N(InternalSortParams const &params, std::uint32_t depth) {

    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    assert(params.span.size() > 0);
    std::size_t const &maxIdx = params.span.size() - 1;

    if (firstIdx > maxIdx) {
        return;
    }
    std::size_t length = lastIdx - firstIdx + 1;
    if (length == 8) {
        __m256 reg = _mm256_loadu_ps(params.span.data() + firstIdx);
        // this is the ending case do single vector permutations
        reverseHalvesAndCompare(reg);
        shuffleAndCompare<0b01001110, 0b00110011>(reg);
        shuffleAndCompare<0b10110001, 0b01010101>(reg);
        _mm256_storeu_ps(params.span.data() + firstIdx, reg);
        return;
    }
    float *p = params.span.data() + firstIdx;
    // for (unsigned i = 0; i < length / 2; i += 8) {
    for (std::size_t i = 0; i < length / 2; i += 8) {
        if (firstIdx + length / 2 + i > maxIdx)
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
    laneCrossingCompare8N({params.span, firstIdx, (firstIdx + lastIdx) / 2}, depth + 1);
    laneCrossingCompare8N({params.span, (firstIdx + lastIdx) / 2 + 1, lastIdx},  depth + 1);
};

void sortLessThan8(std::span<float> &span) {
    auto [reg, mask] = maskload(span);
    sort(reg);
    _mm256_maskstore_ps(span.data(), mask, reg);
}

} // namespace

void sort(__m256 &reg) {
    //    shuffleAndCompare(reg, 0b10110001, 0b10101010);
    shuffleAndCompare<0b10110001, 0b01010101>(reg);
    shuffleAndCompare<0b00011011, 0b00110011>(reg);
    shuffleAndCompare<0b10110001, 0b01010101>(reg);
    reverseAndCompare(reg);
    shuffleAndCompare<0b01001110, 0b00110011>(reg);
    shuffleAndCompare<0b10110001, 0b01010101>(reg);
}

void sort(__m256 &reg0, __m256 &reg1) {
    sort(reg1); // sort first register
    sort(reg0); // sort second register
    reverseAndCompare(reg0, reg1);
    reverseHalvesAndCompare(reg0);
    reverseHalvesAndCompare(reg1);
    shuffleAndCompare<0b01001110, 0b00110011>(reg0);
    shuffleAndCompare<0b10110001, 0b01010101>(reg0);
    shuffleAndCompare<0b01001110, 0b00110011>(reg1);
    shuffleAndCompare<0b10110001, 0b01010101>(reg1);
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
    reverseAndCompare(reg0, reg1);
    reverseHalvesAndCompare(reg0);
    reverseHalvesAndCompare(reg1);
    shuffleAndCompare<0b01001110, 0b00110011>(reg0);
    shuffleAndCompare<0b10110001, 0b01010101>(reg0);
    shuffleAndCompare<0b01001110, 0b00110011>(reg1);
    shuffleAndCompare<0b10110001, 0b01010101>(reg1);
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

    reverseAndCompare(reg0, reg3);
    reverseAndCompare(reg1, reg2);
    // sort full width
    computeMinMax(reg1, reg3);
    computeMinMax(reg0, reg2);
    computeMinMax(reg0, reg1);
    computeMinMax(reg2, reg3);

    reverseHalvesAndCompare(reg0);
    reverseHalvesAndCompare(reg1);
    reverseHalvesAndCompare(reg2);
    reverseHalvesAndCompare(reg3);

    shuffleAndCompare<0b01001110, 0b00110011>(reg0);
    shuffleAndCompare<0b01001110, 0b00110011>(reg1);
    shuffleAndCompare<0b01001110, 0b00110011>(reg2);
    shuffleAndCompare<0b01001110, 0b00110011>(reg3);

    shuffleAndCompare<0b10110001, 0b01010101>(reg0);
    shuffleAndCompare<0b10110001, 0b01010101>(reg1);
    shuffleAndCompare<0b10110001, 0b01010101>(reg2);
    shuffleAndCompare<0b10110001, 0b01010101>(reg3);
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
    reverseAndCompare(reg0, reg3);
    reverseAndCompare(reg1, reg2);
    // sort full width
    computeMinMax(reg1, reg3);
    computeMinMax(reg0, reg2);

    computeMinMax(reg0, reg1);
    computeMinMax(reg2, reg3);

    reverseHalvesAndCompare(reg0);
    reverseHalvesAndCompare(reg1);
    reverseHalvesAndCompare(reg2);
    reverseHalvesAndCompare(reg3);

    shuffleAndCompare<0b01001110, 0b00110011>(reg0);
    shuffleAndCompare<0b01001110, 0b00110011>(reg1);
    shuffleAndCompare<0b01001110, 0b00110011>(reg2);
    shuffleAndCompare<0b01001110, 0b00110011>(reg3);

    shuffleAndCompare<0b10110001, 0b01010101>(reg0);
    shuffleAndCompare<0b10110001, 0b01010101>(reg1);
    shuffleAndCompare<0b10110001, 0b01010101>(reg2);
    shuffleAndCompare<0b10110001, 0b01010101>(reg3);
}

//////////////////////////////////////////////////////////////////
// 2^N FLOAT ARRAY SORTING ALGORITHM
/**
 * @brief The function sorts an array with 2^n elements
 * @param array pointer to the first element to be sorted
 * @param num_to_sort number of elements to be sorted
 * @details num_to_sort should be multiple of 8. If it's not,
 * then add additional elements - infinity, so that it will be
 * large enough
 */
void sort_2n(std::span<float> span) {

    float *array = span.data();
    std::uint32_t num_to_sort = span.size();

    assert((num_to_sort >= 0 && !(num_to_sort & (num_to_sort - 1))) &&
           "The array to be sorted does not have length 2^n, where n is some integer.!");

    if (num_to_sort < 8) {
        auto [reg, mask] = maskload(span);
        sort(reg);
        _mm256_maskstore_ps(array, mask, reg);
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
        for (std::uint32_t i = 0; i < num_to_sort; i += 8) {
            __m256 vec1 = _mm256_loadu_ps(array + i);
            sort(vec1);
            _mm256_storeu_ps(array + i, vec1);
        }

        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        for (std::uint32_t len = 16; len <= num_to_sort; len *= 2) {
            // std::cout << "len: " << len << std::endl;
            // inner loop goes over all subdivisions
            for (std::uint32_t n = 0; n < num_to_sort; n += len) {
                compareFullLength2N({span, n, n + len - 1});
                laneCrossingCompare2N({span, n, n + len - 1}, 0);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////
// FLOAT ARRAY SORTING ALGORITHM, the array must contain 8n elements

/**
 * @brief The function sorts an array with 2^n elements
 * @param array pointer to the first element to be sorted
 * @param num_to_sort number of elements to sort
 * @details num_to_sort should be multiple of 8.
 */
void sort_8n(std::span<float> span) {

    float *p = span.data();
    std::size_t num_to_sort = span.size();

    //    unsigned full_length = end - start + 1; // number of
    //    double numbers to sort
    std::size_t end = num_to_sort - 1;
    int pow2 = (int)std::ceil(std::log2f(num_to_sort));
    int imaginary_length = (int)std::pow(2, pow2);

    assert((num_to_sort % 8) == 0 &&
           "The array to be sorted does not have the size that is a multiple of 8!");

    if (num_to_sort == 8) {
        __m256 vec = _mm256_loadu_ps(p);
        sort(vec);
        _mm256_storeu_ps(p, vec);
    } else if (num_to_sort == 16) {
        auto p2 = span.data() + 8;
        __m256 vec1 = _mm256_loadu_ps(p);
        __m256 vec2 = _mm256_loadu_ps(p2);
        sort(vec1, vec2);
        _mm256_storeu_ps(p, vec1);
        _mm256_storeu_ps(p2, vec2);
    } else if (num_to_sort == 32) {
        __m256 vec1 = _mm256_loadu_ps(p);
        __m256 vec2 = _mm256_loadu_ps(p + 8);
        __m256 vec3 = _mm256_loadu_ps(p + 16);
        __m256 vec4 = _mm256_loadu_ps(p + 24);
        sort(vec1, vec2, vec3, vec4);
        _mm256_storeu_ps(p, vec1);
        _mm256_storeu_ps(p + 8, vec2);
        _mm256_storeu_ps(p + 16, vec3);
        _mm256_storeu_ps(p + 24, vec4);
    } else {
        for (std::size_t i = 0; i < end; i += 8) {
            __m256 vec1 = _mm256_loadu_ps(p + i);
            sort(vec1);
            _mm256_storeu_ps(p + i, vec1);
        }
        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        for (std::size_t len = 16; len <= imaginary_length; len *= 2) {
            // inner loop goes over all subdivisions
            for (std::size_t n = 0; n < imaginary_length; n += len) {
                compareFullLength8N({span, n, n + len - 1});
                laneCrossingCompare8N({span, n, n + len - 1}, 0);
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
inline void compareFullLength(InternalSortParams const &params) {

    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const &maxIdx = params.span.size() - 1;

    assert(lastIdx >= firstIdx);
    std::size_t length = lastIdx - firstIdx + 1;
    std::size_t half = (length + 1) / 2; // half je index prvega cez polovico
    for (std::int32_t i = half - 8; i >= 0; i -= 8) {
        std::int32_t last_vec_to_load = lastIdx - 7 - i;
        std::int32_t diff = maxIdx - last_vec_to_load;
        if (diff < 0)
            return;
        // define pointers to the start of vectors to load
        float *p1 = params.span.data() + firstIdx + i;
        float *p2 = params.span.data() + last_vec_to_load;
        auto [vec2, mask] = [&]() {
            if (diff < 7)
                return maskload(std::span<float const>{p2, std::size_t(diff + 1)});
            else
                return RegMask{_mm256_loadu_ps(p2), __m256i{}};
        }();
        {
            __m256 vec1 = _mm256_loadu_ps(p1);
            __m256 reversed_halves = _mm256_permute2f128_ps(vec1, vec1, 0b00000001);
            __m256 reversed = _mm256_shuffle_ps(reversed_halves, reversed_halves, 0b00011011);
            vec1 = _mm256_min_ps(reversed, vec2);
            vec2 = _mm256_max_ps(reversed, vec2);
            reversed_halves = _mm256_permute2f128_ps(vec1, vec1, 0b00000001);
            vec1 = _mm256_shuffle_ps(reversed_halves, reversed_halves, 0b00011011);
            _mm256_storeu_ps(p1, vec1);
            if (diff <= 6)
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
void laneCrossingCompare(InternalSortParams const &params, std::uint32_t depth) {

    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const &maxIdx = params.span.size() - 1;

    if (firstIdx > maxIdx) {
        return;
    }
    std::size_t length = lastIdx - firstIdx + 1;
    if (length == 8) {
        std::int64_t diff = maxIdx - firstIdx;
        if (diff < 1) {
            return;
        }

        auto p = params.span.data() + firstIdx;
        auto [reg, mask] = [&]() {
            if (diff < 7)
                return maskload(std::span<float const>{p, std::size_t(diff + 1)});
            else
                return RegMask{_mm256_loadu_ps(p), __m256i{}};
        }();

        reverseHalvesAndCompare(reg);
        shuffleAndCompare<0b01001110, 0b00110011>(reg);
        shuffleAndCompare<0b10110001, 0b01010101>(reg);
        if (diff < 7)
            _mm256_maskstore_ps(p, mask, reg);
        else
            _mm256_storeu_ps(p, reg);
        return;
    }

    auto p = params.span.data() + firstIdx;
    // This should be (length + 1) / 2, but not adding 1 does not change anything since
    // the length is always a multiple of 8.
    assert(length % 8 == 0);
    std::size_t halfLength = length / 2;

    for (std::uint32_t i = 0; i < halfLength; i += 8) {
        std::int32_t diff = maxIdx - (firstIdx + halfLength + i);
        if (diff < 0)
            break;

        float *p2 = p + halfLength + i;
        auto [reg1, mask] = [&]() {
            if (diff < 7)
                return maskload(std::span<float const>{p2, std::size_t(diff + 1)});
            else
                return RegMask{_mm256_loadu_ps(p2), __m256i{}};
        }();

        float *p1 = p + i;
        __m256 reg0 = _mm256_loadu_ps(p1); // i-ti od začetka
        // register 2 vsebuje min vrednosti
        __m256 min = _mm256_min_ps(reg1, reg0);
        // register 1 vsebuje max vrednosti
        reg1 = _mm256_max_ps(reg1, reg0);
        reg0 = min;
        _mm256_storeu_ps(p1, reg0);
        if (diff < 7)
            _mm256_maskstore_ps(p2, mask, reg1);
        else
            _mm256_storeu_ps(p2, reg1);
    }

    laneCrossingCompare({params.span, firstIdx, (firstIdx + lastIdx) / 2}, depth + 1);
    laneCrossingCompare({params.span, (firstIdx + lastIdx) / 2 + 1, lastIdx}, depth + 1);
};


/** @brief compared vectors from top and bottom of array and then
 * gradually compare inner vectors.
 * @param arr pointer to the array to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 * @param depth a parameter to follow the depth of recursion
 */
void laneCrossingCompareNew(InternalSortParams const &params, std::uint32_t depth) {


    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const &maxIdx = params.span.size() - 1;

    if (firstIdx > maxIdx) {
        return;
    }
    std::size_t length = lastIdx - firstIdx + 1;
    if (length == 8) {
        int diff = maxIdx - firstIdx;
        if (diff < 1) {
            return;
        }

        auto p = params.span.data() + firstIdx;
        auto [reg, mask] = [&]() {
            if (diff < 7)
                return maskload(std::span<float const>{p, std::size_t(diff + 1)});
            else
                return RegMask{_mm256_loadu_ps(p), __m256i{}};
        }();

        reverseHalvesAndCompare(reg);
        shuffleAndCompare<0b01001110, 0b00110011>(reg);
        shuffleAndCompare<0b10110001, 0b01010101>(reg);
        if (diff < 7)
            _mm256_maskstore_ps(p, mask, reg);
        else
            _mm256_storeu_ps(p, reg);
        return;
    }

    auto minMax = [](__m256 &reg0, __m256 &reg1) {
        __m256 min = _mm256_min_ps(reg1, reg0);
        reg1 = _mm256_max_ps(reg1, reg0);
        reg0 = min;
    };


    // This should be (length + 1) / 2, but not adding 1 does not change anything since
    // the length is always a multiple of 8.
    assert(length % 8 == 0);
    std::size_t halfLength = (length) / 2;
    std::uint32_t globalSecondHalfFirstIdx = firstIdx + halfLength;

    std::size_t numPairsToCompare = [&]() -> std::size_t {
        if(maxIdx >= lastIdx) {
            return halfLength;
        }
        assert(lastIdx > maxIdx);
        auto segmentExcess = lastIdx - maxIdx;
        if(segmentExcess >= halfLength){
            return 0;
        }
        return halfLength - segmentExcess;
    }();

    float *p1 = params.span.data() + firstIdx;
    float *p2 = params.span.data() + globalSecondHalfFirstIdx;

    std::uint32_t numPairsCompared = 7;
    for (; numPairsCompared < numPairsToCompare; numPairsCompared+=8) {
        __m256 reg1 = _mm256_loadu_ps(p1);
        __m256 reg2 = _mm256_loadu_ps(p2);
        minMax(reg1, reg2);
        _mm256_storeu_ps(p1, reg1);
        _mm256_storeu_ps(p2, reg2);
        p1 += 8;
        p2 += 8;
    }

    std::uint32_t leftToLoad = numPairsToCompare - (numPairsCompared - 7);
    if(leftToLoad != 0) {
        __m256 reg1 = _mm256_loadu_ps(p1);
        auto [reg2, mask] = maskload({p2, leftToLoad});
        minMax(reg1, reg2);
        _mm256_storeu_ps(p1, reg1);
        _mm256_maskstore_ps(p2, mask, reg2);
    }
    
    laneCrossingCompareNew({params.span, firstIdx, (firstIdx + lastIdx) / 2}, depth + 1);
    laneCrossingCompareNew({params.span, (firstIdx + lastIdx) / 2 + 1, lastIdx}, depth + 1);
};

/**
 * @brief The function sorts an array with arbitrary number of elements
 * @param array pointer to the first element to be sorted
 * @param num_to_sort number of elements to be sorted
 */
void sort(std::span<float> span) {

        float *array = span.data();
    std::size_t numToSort = span.size();
    unsigned end = numToSort - 1;
    if (numToSort <= 1)
        return;
    else if (numToSort < 8) {
        auto [reg, mask] = maskload(span);
        sort(reg);
        _mm256_maskstore_ps(array, mask, reg);
    } else if (!(numToSort & (numToSort - 1)))
        sort_2n(span);
    else if (mod8(numToSort) == 0)
        sort_8n(span);
    else if (numToSort < 16) {
        __m256 reg1;
        reg1 = _mm256_loadu_ps(array);
        auto [reg2, mask] = maskload({array + 8, numToSort - 8});
        sort(reg1, reg2);
        _mm256_storeu_ps(array, reg1);
        _mm256_maskstore_ps(array + 8, mask, reg2);
    } else {
        std::size_t log2 = std::size_t(std::ceil(std::log2f(numToSort)));
        std::size_t maxSegmentLength = std::size_t(std::pow(2, log2));

        for (unsigned i = 0; i <= end - 7; i += 8) {
            __m256 vec1 = _mm256_loadu_ps(array + i);
            sort(vec1);
            _mm256_storeu_ps(array + i, vec1);
        }
        ///////////////////////////////// load the partial one
        std::int32_t reminder = mod8(span.size());
        float *p = array + numToSort - reminder;
        auto [reg1, mask] = maskload({p, std::size_t(reminder)});
        sort(reg1);
        _mm256_maskstore_ps(p, mask, reg1);

        ///////////////////////////////////////////////////////
        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        // 8 and less has already been done above
        // for (unsigned len = 16; len <= imaginary_length; len *=
        // 2)
        std::size_t segmentLength = 16;
        for (std::uint32_t i = 0; i <= log2 - 4; i++) {
            for (std::size_t n = 0; n < maxSegmentLength; n += segmentLength) {
                compareFullLength({span, n, n + segmentLength - 1});
                laneCrossingCompareNew({span, n, n + segmentLength - 1}, 0U);
            }
            segmentLength *= 2;
        }
    }
}

} // namespace bitonic_sort
