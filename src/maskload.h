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

template <std::size_t ElementCount, std::size_t ToLoad>
using MakeLoadSequence = typename MakeLoadSequenceHelper<ElementCount, ToLoad>::type;

template <typename RegType>
RegMask<RegType> maskload(std::span<typename SimdReg<RegType>::ElementType const> const);

extern template RegMask<__m256> maskload<__m256>(std::span<float const> const);
extern template RegMask<__m256d> maskload<__m256d>(std::span<double const> const);
extern template RegMask<__m256i> maskload<__m256i>(std::span<int const> const);

} // namespace bitonic_sort::utils
