#pragma once

#ifdef __AVX2__
#include <bitset>
#include <immintrin.h>
#include <span>

inline int mod2(const int number) {
    return number & 0b00000001;
}

inline int mod4(const int number) {
    return number & 0b00000011;
}

inline int mod8(const int number) {
    return number & 0b00000111;
}

namespace bitonic_sort {

static constexpr int LOAD = 0xffffffff;
static constexpr int STORE = 0xffffffff;


void sort(__m256 &reg);
void sort(__m256 &reg0, __m256 &reg1) ;
void sort(__m256 &reg0, __m256 &reg1, __m256 &reg2, __m256 &reg3);
void sort_2n(std::span<float> span);
void sort_8n(std::span<float> span);
void sort(std::span<float> span);

void sort(__m256d &reg);
void sort(__m256d &reg0, __m256d &reg1);
void sort(__m256d &reg0, __m256d &reg1, __m256d &reg2, __m256d &reg3);
void sort_2n(std::span<double> span);
void sort_4n(std::span<double> span);
void sort(std::span<double> span);

} // namespace BITONIC_SORT

#endif
