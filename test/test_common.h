#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <functional>
#include <random>

#include <immintrin.h>

#include <BitonicSort.h>
#include <BitonicSortCommon.h>

namespace bitonic_sort::test {

template <typename RegType, typename... Regs>
static auto regToArray(RegType const &reg, Regs const &...regs) {

    static_assert((std::is_same_v<Regs, RegType> && ...));
    constexpr auto RegsCount = 1 + sizeof...(Regs);
    using T = utils::SimdElement_t<RegType>;
    constexpr auto regSize = utils::SimdSize<RegType>;

    std::array<T, regSize * RegsCount> toReturn;

    utils::SimdReg<RegType>::storeReg(toReturn.data(), reg);
    std::uint32_t i = 1;
    auto storeIth = [&](auto const &r) {
        utils::SimdReg<RegType>::storeReg(toReturn.data() + i * regSize, r);
        i++;
    };
    (storeIth(regs), ...);

    return toReturn;
}

template <typename Reg, typename... Regs>
void runRegisterSortTest(Reg &reg, Regs &...regs) {

    using BaseType = std::decay_t<Reg>;
    static_assert((std::is_same_v<std::decay_t<Regs>, BaseType> && ...));
    constexpr auto regSize = utils::SimdSize<BaseType>;
    using T = utils::SimdReg_t<BaseType>;

    constexpr auto RegCount = sizeof...(Regs);
    static_assert(RegCount == 0 || RegCount == 1 || RegCount == 3,
                  "Bitonic sort only supports 1, 2 or 4 register sorting.");

    auto regArraySolution = regToArray(reg, regs...);
    std::sort(regArraySolution.begin(), regArraySolution.end());

    bitonic_sort::sort(reg, regs...);
    auto regArrayToCheck = regToArray(reg, regs...);

    EXPECT_THAT(regArrayToCheck, ::testing::ContainerEq(regArraySolution));
}

} // namespace bitonic_sort::test