#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <random>

#include <immintrin.h>

namespace bitonic_sort::utils {

inline static std::default_random_engine generator(std::time(0));
inline static std::uniform_real_distribution<float> floatDist(-100, 100);
inline static std::uniform_real_distribution<double> doubleDist(-100, 100);
inline static std::uniform_int_distribution<int> intDist(-10000000, 10000000);

inline auto randomNumber = []<typename T>() -> T {
    if constexpr (std::is_same_v<T, double>) {
        static auto generate = std::bind(doubleDist, generator);
        return generate();
    } else if constexpr (std::is_same_v<T, float>) {
        static auto generate = std::bind(floatDist, generator);
        return generate();
    } else if constexpr (std::is_same_v<T, int>) {
        static auto generate = std::bind(intDist, generator);
        return generate();
    } else {
        static_assert(!sizeof(T), "Not implemented for selected type.");
    }
};

template <typename T>
std::vector<T> getRandomVector(std::size_t size) {
    std::vector<T> vec;
    vec.reserve(size);
    for (std::size_t i = 0; i < size; i++) {
        vec.push_back(randomNumber.template operator()<T>());
    }
    return vec;
}

template <typename T>
struct SimdReg;

template <>
struct SimdReg<__m256d> {
    using type = double;
    using RegType = __m256d;
    constexpr static std::size_t SimdSize = sizeof(RegType) / sizeof(type);
    constexpr static auto storeReg = _mm256_storeu_pd;
};

template <>
struct SimdReg<__m256> {
    using type = float;
    using RegType = __m256;
    constexpr static std::size_t SimdSize = sizeof(RegType) / sizeof(type);
    constexpr static auto storeReg = _mm256_storeu_ps;
};

template <>
struct SimdReg<__m256i> {
    using type = std::int32_t;
    using RegType = __m256i;
    constexpr static std::size_t SimdSize = sizeof(RegType) / sizeof(type);
    constexpr static auto storeReg = _mm256_storeu_si256;
};

template <typename Type>
using SimdReg_t = typename SimdReg<Type>::type;

template <typename Type>
inline constexpr static std::size_t SimdSize = SimdReg<Type>::SimdSize;

}