#pragma once

#ifdef __AVX2__
#include <bitset>
#include <immintrin.h>

inline __attribute__((always_inline)) int mod2(const int number) {
    return number & 0b00000001;
}

inline __attribute__((always_inline)) int mod4(const int number) {
    return number & 0b00000011;
}

inline __attribute__((always_inline)) int mod8(const int number) {
    return number & 0b00000111;
}

namespace BITONIC_SORT {

static constexpr int LOAD = 0xffffffff;
static constexpr int STORE = 0xffffffff;

} // namespace BITONIC_SORT

#endif
