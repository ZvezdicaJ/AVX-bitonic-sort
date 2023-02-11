#pragma once

#include <array>
#include <cstdint>
#include <immintrin.h>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "BitonicSort.h"
#include "BitonicSortCommon.h"

namespace bitonic_sort::utils {

template <std::uint32_t N>
consteval std::uint32_t computeBlendMask() {
    return (1 << (N - 1)) + computeBlendMask<N - 1>();
}

template <>
consteval std::uint32_t computeBlendMask<0>() {
    return 0;
}

// General template for load sequence.
template <std::int64_t... Vals>
struct LoadSequence {
    constexpr std::size_t size() { return sizeof...(Vals); }
};

// General template for load sequence helper.
template <std::size_t N, std::size_t ToLoad, std::int64_t... Vals>
struct MakeLoadSequenceHelper {};

// N tells how many elements are still to be added to load sequence. This specializations adds
// a 0 to the sequence.
template <std::size_t N, std::size_t ToLoad, std::int64_t... Vals>
    requires(N > ToLoad)
struct MakeLoadSequenceHelper<N, ToLoad, Vals...> {
    using type =
        typename MakeLoadSequenceHelper<N - 1, ToLoad, Vals..., std::int64_t(0)>::type;
};

// This specializations adds a LOAD to the sequence.
template <std::size_t N, std::size_t ToLoad, std::int64_t... Vals>
    requires(N <= ToLoad)
struct MakeLoadSequenceHelper<N, ToLoad, Vals...> {
    using type =
        typename MakeLoadSequenceHelper<N - 1, ToLoad, Vals..., std::int64_t(LOAD)>::type;
};

template <std::size_t ToLoad, std::int64_t... Vals>
struct MakeLoadSequenceHelper<0, ToLoad, Vals...> {
    using type = LoadSequence<Vals...>;
};

template <std::size_t N, std::size_t ToLoad>
using MakeLoadSequence = typename MakeLoadSequenceHelper<N, ToLoad>::type;

template <typename RegType>
struct MaskLoader {
    using T = typename SimdReg<RegType>::ElementType;
    using IntT = typename SimdReg<RegType>::IntType;
    constexpr static auto N = SimdSize<RegType>;

    template <std::uint32_t LoadCount, std::int64_t... Ints>
        requires(LoadCount <= SimdSize<RegType>)
    RegMask<RegType> maskloadHelper(std::span<T> span, LoadSequence<Ints...>) {
        if (span.size() == LoadCount) {
            std::vector<std::int64_t> vec{Ints...};
            auto mask = SimdReg<RegType>::setMask(Ints...);
            auto reg = SimdReg<RegType>::maskload(span.data(), mask);
            auto infinity = SimdReg<RegType>::set1(std::numeric_limits<T>::infinity());

            auto constexpr blendMask = computeBlendMask<LoadCount>();
            reg = SimdReg<RegType>::template blend<blendMask>(infinity, reg);
            return {reg, mask};
        }
        return maskloadHelper<LoadCount + 1>(
            span, MakeLoadSequence<SimdSize<RegType>, LoadCount + 1>());
    }

    template <std::uint32_t LoadCount, std::int64_t... Ints>
        requires(LoadCount > SimdSize<RegType>)
    RegMask<RegType> maskloadHelper(std::span<T> span, LoadSequence<Ints...>) {
        throw std::runtime_error("Attempt to load more elements than SIMD vector can hold.");
    }

    RegMask<RegType> load(std::span<T> span) {
        if (span.size() == 0)
            return RegMask<RegType>{};
        return maskloadHelper<1>(span, MakeLoadSequence<SimdSize<RegType>, 1>());
    }
};

} // namespace bitonic_sort::utils
