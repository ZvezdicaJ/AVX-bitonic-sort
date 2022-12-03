#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdint>
#include <array>

#include <immintrin.h>

#include <bitonic_sort_double.hpp>

template<typename T>
struct SimdReg;

template<>
struct SimdReg<__m256d> {
    using type = double;
    using RegType = __m256d;
    constexpr static std::size_t SimdSize = sizeof(RegType) / sizeof(type);
    constexpr static auto storeReg = _mm256_storeu_pd;
};

template<>
struct SimdReg<__m256> {
    using type = float;
    using RegType = __m256;
    constexpr static std::size_t SimdSize = sizeof(RegType) / sizeof(type);
    constexpr static auto storeReg = _mm256_storeu_ps;
};

template<>
struct SimdReg<__m256i> {
    using type = std::int32_t;
    using RegType = __m256i;
    constexpr static std::size_t SimdSize = sizeof(RegType) / sizeof(type);
    constexpr static auto storeReg = _mm256_storeu_si256;
};


template<typename Type>
using SimdReg_t = SimdReg<Type>::type;

template<typename Type>
inline constexpr static std::size_t SimdSize = SimdReg<Type>::SimdSize;

template<typename RegType, typename... Regs>
static auto regToArray(RegType const &reg, Regs const&... regs) {

    static_assert((std::is_same_v<Regs, RegType> && ...));
    constexpr auto RegsCount = 1 + sizeof...(Regs);
    using T = SimdReg_t<RegType>;

    std::array<T, SimdSize<RegType> * RegsCount> toReturn;

    SimdReg<RegType>::storeReg(toReturn.data(), reg);
    std::uint32_t i=1;
    auto storeIth = [&] (auto const& r) {
        SimdReg<RegType>::storeReg(toReturn.data() + i * SimdSize<RegType>, r);
        i++;
    };
    (storeIth(regs), ...);

    return toReturn;
}

template<typename Reg, typename... Regs>
void runRegisterSortTest(Reg& reg, Regs&... regs) {

    using BaseType = std::decay_t<Reg>;
    static_assert((std::is_same_v<std::decay_t<Regs>, BaseType> && ...));
    constexpr auto elemCount = SimdSize<BaseType>;
    using T = SimdReg_t<BaseType>;

    constexpr auto RegCount = sizeof...(Regs); 
    static_assert(RegCount == 0 || RegCount == 1 || RegCount == 3, "Bitonic sort only supports 1, 2 or 4 register sorting.");
   
    auto regArraySolution = regToArray(reg, regs...);
    std::sort(regArraySolution.begin(), regArraySolution.end());

    BITONIC_SORT::bitonic_sort(reg, regs...);
    auto regArrayToCheck = regToArray(reg, regs...);

    EXPECT_THAT(regArrayToCheck, ::testing::ContainerEq(regArraySolution));
}