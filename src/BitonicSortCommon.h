#pragma once

#include <array>
#include <cstdint>
#include <ctime>
#include <functional>
#include <random>

#include <immintrin.h>

namespace bitonic_sort::utils {

template <typename T>
struct SimdReg;

template <>
struct SimdReg<__m256d> {
    using ElementType = double;
    using IntType = std::int64_t;
    using RegType = __m256d;
    constexpr static std::size_t SimdSize = sizeof(RegType) / sizeof(ElementType);
    constexpr static auto storeReg = _mm256_storeu_pd;
    constexpr static auto loadReg = _mm256_loadu_pd;

    constexpr static auto setMask = _mm256_set_epi64x;
    constexpr static auto maskload = _mm256_maskload_pd;
    constexpr static auto set1 = _mm256_set1_pd;

    template <int imm8>
    static inline __m256d blend(__m256d a, __m256d b) {
        return _mm256_blend_pd(a, b, imm8);
    };
};

template <>
struct SimdReg<__m256> {
    using ElementType = float;
    using IntType = std::int32_t;
    using RegType = __m256;
    constexpr static std::size_t SimdSize = sizeof(RegType) / sizeof(ElementType);
    constexpr static auto storeReg = _mm256_storeu_ps;
    constexpr static auto loadReg = _mm256_loadu_ps;

    constexpr static auto setMask = _mm256_set_epi32;
    constexpr static auto maskload = _mm256_maskload_ps;
    constexpr static auto set1 = _mm256_set1_ps;

    template <std::uint32_t imm8>
    static inline __m256 blend(__m256 a, __m256 b) {
        return _mm256_blend_ps(a, b, imm8);
    };
};

template <>
struct SimdReg<__m256i> {
    using ElementType = std::int32_t;
    using IntType = std::int32_t;
    using RegType = __m256i;
    constexpr static std::size_t SimdSize = sizeof(RegType) / sizeof(ElementType);
    constexpr static auto storeReg = _mm256_storeu_si256;
    constexpr static auto loadReg = _mm256_loadu_si256;

    constexpr static auto setMask = _mm256_set_epi32;
    constexpr static auto maskload = _mm256_maskload_epi32;
    constexpr static auto set1 = _mm256_set1_epi32;

    template <int imm8>
    static inline __m256i blend(__m256i a, __m256i b) {
        return _mm256_blend_epi32(a, b, imm8);
    };
};

template <typename Type>
using SimdElement_t = typename SimdReg<Type>::ElementType;

template <typename Type>
using SimdReg_t = typename SimdReg<Type>::RegType;

template <typename Type>
inline constexpr static std::size_t SimdSize = SimdReg<Type>::SimdSize;

template <typename RegType>
struct alignas(32) RegMask {
    RegType reg;
    __m256i mask;
};

} // namespace bitonic_sort::utils