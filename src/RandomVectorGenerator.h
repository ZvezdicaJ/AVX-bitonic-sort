#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <random>

namespace bitonic_sort::utils {

inline static std::default_random_engine generator(std::time(0));
inline static std::uniform_real_distribution<float> floatDist(-100, 100);
inline static std::uniform_real_distribution<double> doubleDist(-100, 100);
inline static std::uniform_int_distribution<int> intDist(-1000, 1000);

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

}