#pragma once


#include <bitset>
#include <span>

#include <immintrin.h>

inline std::int32_t mod2(std::int32_t number) { return number & 0b00000001; }

inline std::int32_t mod4(std::int32_t number) { return number & 0b00000011; }

inline std::int32_t mod8(std::int32_t number) { return number & 0b00000111; }

namespace bitonic_sort {

static constexpr int LOAD = 0xffffffff;
static constexpr int STORE = 0xffffffff;

void sort(__m256 &reg);
void sort(__m256 &reg0, __m256 &reg1);
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

} // namespace bitonic_sort


