#include "maskload.h"

#include <array>
#include <cstdint>
#include <immintrin.h>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "BitonicSort.h"
#include "BitonicSortCommon.h"

namespace bitonic_sort::utils {
namespace {

template <typename RegType>
using T = typename SimdReg<RegType>::ElementType;

template <typename RegType>
using IntT = typename SimdReg<RegType>::IntType;

template <typename RegType>
constexpr static auto N = SimdSize<RegType>;

template <typename RegType, std::uint32_t LoadCount, std::int64_t... Ints>
    requires(LoadCount > SimdSize<RegType>)
static RegMask<RegType> maskloadHelper(std::span<T<RegType> const> const span,
                                       LoadSequence<Ints...>) {
    throw std::runtime_error("Attempt to load more elements than SIMD vector can hold.");
}

template <typename RegType, std::uint32_t LoadCount, std::int64_t... Ints>
    requires(LoadCount <= SimdSize<RegType>)
static RegMask<RegType> maskloadHelper(std::span<T<RegType> const> const span,
                                       LoadSequence<Ints...>) {
    if (span.size() == LoadCount) {
        std::vector<std::int64_t> vec{Ints...};
        auto mask = SimdReg<RegType>::setMask(Ints...);
        auto reg = SimdReg<RegType>::maskload(span.data(), mask);
        auto infinity = SimdReg<RegType>::set1(std::numeric_limits<T<RegType>>::infinity());

        auto constexpr blendMask = computeBlendMask<LoadCount>();
        reg = SimdReg<RegType>::template blend<blendMask>(infinity, reg);
        return {reg, mask};
    }
    return maskloadHelper<RegType, LoadCount + 1>(
        span, MakeLoadSequence<SimdSize<RegType>, LoadCount + 1>());
}

} // namespace

template <typename RegType>
RegMask<RegType> maskload(std::span<T<RegType> const> const span) {
    if (span.size() == 0)
        return RegMask<RegType>{};
    return maskloadHelper<RegType, 1>(span, MakeLoadSequence<SimdSize<RegType>, 1>());
}

template RegMask<__m256> maskload<__m256>(std::span<float const> const);
template RegMask<__m256d> maskload<__m256d>(std::span<double const> const);
template RegMask<__m256i> maskload<__m256i>(std::span<int const> const);

} // namespace bitonic_sort::utils