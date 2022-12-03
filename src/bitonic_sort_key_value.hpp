#pragma once
#ifdef __AVX2__
#include <bitset>

#ifndef __BITONIC_SORT_KEY_VALUE__
#define __BITONIC_SORT_KEY_VALUE__

std::ostream &operator<<(std::ostream &out,
                         std::vector<std::pair<int, float>> &vec) {

    for (auto &a : vec)
        std::cout << a << std::endl;
    return out;
}

namespace BITONIC_SORT_KEY_VALUE {

static const int LOAD = 0xffffffff;
static const int STORE = 0xffffffff;

struct double_masks256_ {
    __m256i mask1 =
        _mm256_set_epi64x(0xffffffffffffffff, 0x0000000000000000,
                          0xffffffffffffffff, 0x0000000000000000);

    __m256i mask2 =
        _mm256_set_epi64x(0xffffffffffffffff, 0xffffffffffffffff,
                          0x0000000000000000, 0x0000000000000000);
};

struct double_load_masks256_ {
    __m256i mask1 = _mm256_set_epi64x(0, 0, 0, LOAD);
    __m256i mask2 = _mm256_set_epi64x(0, 0, LOAD, LOAD);
    __m256i mask3 = _mm256_set_epi64x(0, LOAD, LOAD, LOAD);
};

struct double_key_load_masks128_ {
    __m128i mask1 = _mm_set_epi32(0, 0, 0, LOAD);
    __m128i mask2 = _mm_set_epi32(0, 0, LOAD, LOAD);
    __m128i mask3 = _mm_set_epi32(0, LOAD, LOAD, LOAD);
};

static const double_masks256_ double_masks256;
static const double_load_masks256_ double_load_masks256;
static const double_key_load_masks128_ double_key_load_masks128;

struct float_masks256_ {
    __m256i mask1 = _mm256_set_epi32(
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000,
        0x00000000, 0x00000000, 0x00000000);

    __m256i mask2 = _mm256_set_epi32(
        0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0xffffffff,
        0xffffffff, 0x00000000, 0x00000000);

    __m256i mask3 = _mm256_set_epi32(
        0xffffffff, 0x00000000, 0xffffffff, 0x00000000, 0xffffffff,
        0x00000000, 0xffffffff, 0x00000000);
};

struct float_load_masks256_ {
    __m256i mask1 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, LOAD);
    __m256i mask2 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, LOAD, LOAD);
    __m256i mask3 =
        _mm256_set_epi32(0, 0, 0, 0, 0, LOAD, LOAD, LOAD);

    __m256i mask4 =
        _mm256_set_epi32(0, 0, 0, 0, LOAD, LOAD, LOAD, LOAD);
    __m256i mask5 =
        _mm256_set_epi32(0, 0, 0, LOAD, LOAD, LOAD, LOAD, LOAD);
    __m256i mask6 =
        _mm256_set_epi32(0, 0, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD);

    __m256i mask7 = _mm256_set_epi32(0, LOAD, LOAD, LOAD, LOAD,
                                     LOAD, LOAD, LOAD);
};

static const float_masks256_ float_masks256;
static const float_load_masks256_ float_load_masks256;
const __m256 m256_infinity =
    _mm256_set1_ps(std::numeric_limits<float>::infinity());
const __m256d m256d_infinity =
    _mm256_set1_pd(std::numeric_limits<double>::infinity());

#include "bitonic_sort_key_value_float.hpp"
#include "bitonic_sort_key_value_double.hpp"

} // namespace BITONIC_SORT_KEY_VALUE
#endif
#endif
