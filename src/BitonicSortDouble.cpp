#include "BitonicSort.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <span>

#include <BitonicSortCommon.h>
#include <BitonicSortPrivate.h>
#include <TypeDefinitions.h>
#include <maskload.h>

namespace bitonic_sort {
namespace {

using RegMask = utils::RegMask<__m256d>;
constexpr auto maskload = utils::maskload<__m256d>;

/*RegMask maskload(std::span<double const> const &span) {
    __m256d reg;
    __m256i mask;
    auto p = span.data();
    switch (span.size()) {
    case 1: {
        mask = _mm256_set_epi64x(0, 0, 0, LOAD);
        reg = _mm256_maskload_pd(p, mask);
        __m256d infinity = _mm256_set1_pd(std::numeric_limits<double>::infinity());
        reg = _mm256_blend_pd(infinity, reg, 0b0001);
    } break;
    case 2: {
        mask = _mm256_set_epi64x(0, 0, LOAD, LOAD);
        reg = _mm256_maskload_pd(p, mask);
        __m256d infinity = _mm256_set1_pd(std::numeric_limits<double>::infinity());
        reg = _mm256_blend_pd(infinity, reg, 0b0011);
    } break;
    case 3: {
        mask = _mm256_set_epi64x(0, LOAD, LOAD, LOAD);
        reg = _mm256_maskload_pd(p, mask);
        __m256d infinity = _mm256_set1_pd(std::numeric_limits<double>::infinity());
        reg = _mm256_blend_pd(infinity, reg, 0b0111);
    } break;
    case 4: {
        mask = _mm256_set_epi64x(LOAD, LOAD, LOAD, LOAD);
        reg = _mm256_loadu_pd(p);
    } break;
    default:
        throw std::runtime_error("Invalid number of floats to load.");
    };
    return RegMask{std::move(reg), std::move(mask)};
}*/

template <uint8_t mask>
void shuffle_and_compare(__m256d &reg) {
    __m256d shuffled_reg = _mm256_shuffle_pd(reg, reg, mask);
    __m256d max = _mm256_max_pd(reg, shuffled_reg);
    __m256d min = _mm256_min_pd(reg, shuffled_reg);
    reg = _mm256_unpacklo_pd(min, max);
}

void reverse_and_compare(__m256d &reg) {
    __m256d reversed_reg = _mm256_permute4x64_pd(reg, 0b00011011);
    __m256d max = _mm256_max_pd(reg, reversed_reg);
    __m256d min = _mm256_min_pd(reg, reversed_reg);
    // max mora biti pri 256
    reg = _mm256_blend_pd(max, min, 0b0011);
}

void reverse_and_compare(__m256d &reg0, __m256d &reg1) {
    __m256d reverse0 = _mm256_permute4x64_pd(reg0, 0b00011011);
    // register 2 vsebuje min vrednosti
    reg0 = _mm256_min_pd(reg1, reverse0);
    // register 1 vsebuje max vrednosti
    reg1 = _mm256_max_pd(reg1, reverse0);
}

template <int mask>
void permute_and_compare(__m256d &reg) {
    __m256d shuffled_reg = _mm256_permute4x64_pd(reg, mask);
    __m256d max = _mm256_max_pd(reg, shuffled_reg);
    __m256d min = _mm256_min_pd(reg, shuffled_reg);
    reg = _mm256_blend_pd(max, min, 0b0011);
}

void compare(__m256d &reg0, __m256d &reg1) {
    __m256d min = _mm256_min_pd(reg0, reg1);
    __m256d max = _mm256_max_pd(reg0, reg1);
    reg0 = min;
    reg1 = max;
}

void computeMinMax(__m256d &reg0, __m256d &reg1) {
    {
        __m256d min = _mm256_min_pd(reg0, reg1);
        reg1 = _mm256_max_pd(reg0, reg1);
        reg0 = min;
    }
}

void compareFullLength_2n(InternalSortParams<double> params) {

    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const length = lastIdx - firstIdx + 1;

    assert(length % 2 == 0);
    std::size_t half = length / 2;

    double *p1 = params.span.data() + firstIdx;
    double *p2 = p1 + length - 4;
    for (unsigned i = 0; i < half; i += 4) {
        __m256d vec1 = _mm256_loadu_pd(p1);
        __m256d vec2 = _mm256_loadu_pd(p2);
        reverse_and_compare(vec1, vec2);
        _mm256_storeu_pd(p1, vec1);
        _mm256_storeu_pd(p2, vec2);
        p1 += 4;
        p2 -= 4;
    }
}

void laneCrossingCompare_2n(InternalSortParams<double> params, std::uint32_t depth) {

    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const length = lastIdx - firstIdx + 1;

    if (length == 4) {
        __m256d reg = _mm256_loadu_pd(params.span.data() + firstIdx);
        permute_and_compare<0b01001110>(reg);
        shuffle_and_compare<0b0101>(reg);
        _mm256_storeu_pd(params.span.data() + firstIdx, reg);
        return;
    }
    double *p = params.span.data() + firstIdx;
    for (std::size_t i = 0; i < length / 2; i += 4) {
        {
            double *p1 = p + i;
            double *p2 = p + length / 2 + i;

            __m256d reg0 = _mm256_loadu_pd(p1); // i-ti od začetka
            __m256d reg1 = _mm256_loadu_pd(p2); // ta je prvi čez polovico
            // register 2 vsebuje min vrednosti
            __m256d min = _mm256_min_pd(reg1, reg0);
            // register 1 vsebuje max vrednosti
            reg1 = _mm256_max_pd(reg1, reg0);
            reg0 = min;
            // print_avx(max, "max: ");
            // print_avx(min, "min: ");
            _mm256_storeu_pd(p1, reg0);
            _mm256_storeu_pd(p2, reg1);
        }
    }
    laneCrossingCompare_2n({params.span, firstIdx, (firstIdx + lastIdx) / 2}, depth + 1);
    laneCrossingCompare_2n({params.span, (firstIdx + lastIdx) / 2 + 1, lastIdx}, depth + 1);
};

void compareFullLength_4n(InternalSortParams<double> const &params) {

    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const &maxIdx = params.span.size() - 1;

    std::size_t const length = lastIdx - firstIdx + 1;
    std::size_t pastHalfIdx = length / 2; // half je index prvega cez polovico
    for (std::int32_t toLoadIdx = pastHalfIdx - 4; toLoadIdx >= 0; toLoadIdx -= 4) {
        if ((lastIdx - 3 - toLoadIdx > maxIdx))
            break;
        double *p1 = params.span.data() + firstIdx + toLoadIdx;
        double *p2 = params.span.data() + lastIdx - 3 - toLoadIdx;

        assert(firstIdx + toLoadIdx + 3 <= maxIdx);
        assert(lastIdx - toLoadIdx <= maxIdx);

        __m256d vec1 = _mm256_loadu_pd(p1);
        __m256d vec2 = _mm256_loadu_pd(p2);
        // reverse one of registers register reg0
        __m256d reverse = _mm256_permute4x64_pd(vec1, 0b00011011);
        // register 2 vsebuje min vrednosti
        vec1 = _mm256_min_pd(vec2, reverse);
        vec1 = _mm256_permute4x64_pd(vec1, 0b00011011);
        // register 1 vsebuje max vrednosti
        vec2 = _mm256_max_pd(vec2, reverse);
        _mm256_storeu_pd(p1, vec1);
        _mm256_storeu_pd(p2, vec2);
    }
}

void laneCrossingCompare_4n(InternalSortParams<double> const &params, std::uint32_t depth) {

    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const &maxIdx = params.span.size() - 1;

    if (firstIdx > maxIdx) {
        return;
    }
    std::size_t const length = lastIdx - firstIdx + 1;
    auto p = params.span.data() + firstIdx;

    if (length == 4) {
        __m256d reg = _mm256_loadu_pd(p);
        permute_and_compare<0b01001110>(reg);
        shuffle_and_compare<0b0101>(reg);
        _mm256_storeu_pd(p, reg);
        return;
    }

    for (std::size_t i = 0; i < length / 2; i += 4) {
        if (firstIdx + length / 2 + i > maxIdx)
            break;
        {
            double *p1 = p + i;
            double *p2 = p + length / 2 + i;
            auto maxWriteIdx = firstIdx + length / 2 + i + 3;
            assert(maxWriteIdx <= maxIdx);
            __m256d reg0 = _mm256_loadu_pd(p1); // i-ti od začetka
            __m256d reg1 = _mm256_loadu_pd(p2); // ta je prvi čez polovico
            // register 2 vsebuje min vrednosti
            __m256d min = _mm256_min_pd(reg1, reg0);
            // register 1 vsebuje max vrednosti
            reg1 = _mm256_max_pd(reg1, reg0);
            reg0 = min;
            _mm256_storeu_pd(p1, reg0);
            _mm256_storeu_pd(p2, reg1);
        }
    }
    laneCrossingCompare_4n({params.span, firstIdx, (firstIdx + lastIdx) / 2}, depth + 1);
    laneCrossingCompare_4n({params.span, (firstIdx + lastIdx) / 2 + 1, lastIdx}, depth + 1);
};

// void compareFullLength(double *arr, unsigned start, unsigned end,
//                       unsigned last_index) {

void compareFullLength(InternalSortParams<double> const &params) {

    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const &maxIdx = params.span.size() - 1;
    auto p = params.span.data();

    assert(lastIdx >= firstIdx);
    std::size_t length = lastIdx - firstIdx + 1;
    std::size_t half = length / 2;
    for (int i = half - 4; i >= 0; i -= 4) {
        std::size_t last_vec_to_load = lastIdx - 3 - i;
        if (maxIdx < last_vec_to_load) {
            return;
        }
        std::size_t diff = maxIdx - last_vec_to_load;

        double *p1 = p + firstIdx + i;
        double *p2 = p + last_vec_to_load;

        auto [vec2, mask] = [&]() {
            if (diff < 3) {
                return maskload(std::span{p2, diff + 1});
            }
            return RegMask{_mm256_loadu_pd(p2), __m256i{}};
        }();

        __m256d vec1 = _mm256_loadu_pd(p1);
        reverse_and_compare(vec1, vec2);
        vec1 = _mm256_permute4x64_pd(vec1, 0b00011011);
        _mm256_storeu_pd(p1, vec1);
        if (diff <= 2)
            _mm256_maskstore_pd(p2, mask, vec2);
        else
            _mm256_storeu_pd(p2, vec2);
    }
}

void laneCrossingCompare(InternalSortParams<double> const &params, std::uint32_t depth) {

    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const &maxIdx = params.span.size() - 1;
    if (firstIdx > maxIdx) {
        return;
    }

    auto p = params.span.data();
    auto p1 = p + firstIdx;
    std::size_t length = lastIdx - firstIdx + 1;
    if (length == 4) {
        assert(maxIdx >= firstIdx);
        std::size_t diff = maxIdx - firstIdx;
        if (diff < 1) {
            return;
        }

        auto [reg, mask] = [&]() {
            if (diff < 3) {
                return maskload(std::span{p1, diff + 1});
            }
            return RegMask{_mm256_loadu_pd(p1), __m256i{}};
        }();

        permute_and_compare<0b01001110>(reg);
        shuffle_and_compare<0b0101>(reg);
        if (diff < 3)
            _mm256_maskstore_pd(p1, mask, reg);
        else
            _mm256_storeu_pd(p1, reg);

        return;
    }

    // This should be (length + 1) / 2, but not adding 1 does not change
    // anything since the length is always a multiple of 4.
    assert(length % 4 == 0);
    std::size_t halfLength = length / 2;

    for (std::size_t i = 0; i < halfLength; i += 4) {

        std::size_t loadIndex = firstIdx + halfLength + i;
        if (maxIdx < loadIndex) {
            break;
        }
        std::size_t diff = maxIdx - loadIndex;

        double *p2 = p + loadIndex;
        auto [reg1, mask] = [&]() {
            if (diff < 3) {
                return maskload(std::span{p2, diff + 1});
            }
            return RegMask{_mm256_loadu_pd(p2), __m256i{}};
        }();

        __m256d reg0 = _mm256_loadu_pd(p1); // i-ti od začetka
        // register 2 vsebuje min vrednosti
        __m256d min = _mm256_min_pd(reg1, reg0);
        // register 1 vsebuje max vrednosti
        reg1 = _mm256_max_pd(reg1, reg0);
        reg0 = min;
        _mm256_storeu_pd(p1, reg0);
        if (diff < 3)
            _mm256_maskstore_pd(p2, mask, reg1);
        else
            _mm256_storeu_pd(p2, reg1);
        p1 += 4;
    }
    laneCrossingCompare({params.span, firstIdx, (firstIdx + lastIdx) / 2}, depth + 1);
    laneCrossingCompare({params.span, (firstIdx + lastIdx) / 2 + 1, lastIdx}, depth + 1);
};

void laneCrossingCompareNew(InternalSortParams<double> const &params, std::uint32_t depth) {

    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const &maxIdx = params.span.size() - 1;
    if (firstIdx > maxIdx) {
        return;
    }

    auto p = params.span.data();
    auto p1 = p + firstIdx;
    std::size_t length = lastIdx - firstIdx + 1;
    if (length == 4) {
        assert(maxIdx >= firstIdx);
        std::size_t diff = maxIdx - firstIdx;
        if (diff < 1) {
            return;
        }

        auto [reg, mask] = [&]() {
            if (diff < 3) {
                return maskload(std::span{p1, diff + 1});
            }
            return RegMask{_mm256_loadu_pd(p1), __m256i{}};
        }();

        permute_and_compare<0b01001110>(reg);
        shuffle_and_compare<0b0101>(reg);
        if (diff < 3)
            _mm256_maskstore_pd(p1, mask, reg);
        else
            _mm256_storeu_pd(p1, reg);

        return;
    }

    // This should be (length + 1) / 2, but not adding 1 does not change
    // anything since the length is always a multiple of 8.
    assert(length % 4 == 0);
    std::size_t halfLength = (length) / 2;
    std::uint32_t globalSecondHalfFirstIdx = firstIdx + halfLength;

    std::size_t numPairsToCompare = [&]() -> std::size_t {
        if (maxIdx >= lastIdx) {
            return halfLength;
        }
        assert(lastIdx > maxIdx);
        auto segmentExcess = lastIdx - maxIdx;
        if (segmentExcess >= halfLength) {
            return 0;
        }
        return halfLength - segmentExcess;
    }();

    auto *p2 = params.span.data() + globalSecondHalfFirstIdx;

    std::uint32_t numPairsCompared = 3;
    for (; numPairsCompared < numPairsToCompare; numPairsCompared += 4) {
        __m256d reg1 = _mm256_loadu_pd(p1);
        __m256d reg2 = _mm256_loadu_pd(p2);
        computeMinMax(reg1, reg2);
        _mm256_storeu_pd(p1, reg1);
        _mm256_storeu_pd(p2, reg2);
        p1 += 4;
        p2 += 4;
    }

    std::uint32_t leftToLoad = numPairsToCompare - (numPairsCompared - 3);
    if (leftToLoad != 0) {
        __m256d reg1 = _mm256_loadu_pd(p1);
        auto [reg2, mask] = maskload({p2, leftToLoad});
        computeMinMax(reg1, reg2);
        _mm256_storeu_pd(p1, reg1);
        _mm256_maskstore_pd(p2, mask, reg2);
    }

    laneCrossingCompareNew({params.span, firstIdx, (firstIdx + lastIdx) / 2}, depth + 1);
    laneCrossingCompareNew({params.span, (firstIdx + lastIdx) / 2 + 1, lastIdx}, depth + 1);
};

} // namespace

/**
 *@brief The function accepts a single __m256d vector and sorts
 *it.
 * @param reg register to be sorted
 */
void sort(__m256d &reg) {
    shuffle_and_compare<0b0101>(reg);
    reverse_and_compare(reg);
    shuffle_and_compare<0b0101>(reg);
}

/**
 * @brief The function accepts two __m256d vectors and sorts them.
 * @param reg1 upper vector of numbers - at the end it contains
 * larger values, the largest value is in the upper half of register
 * [255:192]
 *
 * @param reg0 lower vector of numbers - at the end it contains
 * smaller values. The smallest value is in the lowest half of
 * register - at [63:0]
 *
 */
void sort(__m256d &reg0, __m256d &reg1) {
    sort(reg1); // sort first register
    sort(reg0); // sort second register
    reverse_and_compare(reg0, reg1);
    permute_and_compare<0b01001110>(reg1);
    shuffle_and_compare<0b0101>(reg1);
    permute_and_compare<0b01001110>(reg0);
    shuffle_and_compare<0b0101>(reg0);
}

/**
 * @brief The function accepts two __m256d vectors and sorts them.
 * @param reg1 upper vector of numbers - at the end it contains
 * larger values, the largest value is in the upper half of register
 * [255:192]
 *
 * @param reg0 lower vector of numbers - at the end it contains
 * smaller values. The smallest value is in the lowest half of
 * register - at [63:0]
 *
 */
void sort(__m256d &reg0, __m256d &reg1, __m256d &reg2, __m256d &reg3) {
    sort(reg0);       // sort first register
    sort(reg1);       // sort second register
    sort(reg2);       // sort third register
    sort(reg3);       // sort fourth register
    sort(reg0, reg1); // sort third register
    sort(reg2, reg3); // sort fourth register
    reverse_and_compare(reg0, reg3);
    reverse_and_compare(reg1, reg2);

    compare(reg1, reg3);
    compare(reg0, reg2);

    compare(reg0, reg1);
    compare(reg2, reg3);

    permute_and_compare<0b01001110>(reg0);
    permute_and_compare<0b01001110>(reg1);
    permute_and_compare<0b01001110>(reg2);
    permute_and_compare<0b01001110>(reg3);

    shuffle_and_compare<0b0101>(reg0);
    shuffle_and_compare<0b0101>(reg1);
    shuffle_and_compare<0b0101>(reg2);
    shuffle_and_compare<0b0101>(reg3);
}

void sort_2n(std::span<double> span) {
    auto p = span.data();
    auto numToSort = span.size();

    unsigned end = numToSort - 1;
    assert(!(numToSort & (numToSort - 1)) &&
           "The array to be sorted does not have length 2^n, where n is some "
           "integer.!");

    //__m256d *arr = avx_vec.data();
    if (numToSort == 4) {
        __m256d vec = _mm256_loadu_pd(p);
        sort(vec);
        _mm256_storeu_pd(p, vec);

    } else if (numToSort == 8) {
        __m256d vec1 = _mm256_loadu_pd(p);
        __m256d vec2 = _mm256_loadu_pd(p + 4);
        sort(vec1, vec2);
        _mm256_storeu_pd(p, vec1);
        _mm256_storeu_pd(p + 4, vec2);

    } else if (numToSort == 16) {
        __m256d vec1 = _mm256_loadu_pd(p);
        __m256d vec2 = _mm256_loadu_pd(p + 4);
        __m256d vec3 = _mm256_loadu_pd(p + 8);
        __m256d vec4 = _mm256_loadu_pd(p + 12);
        sort(vec1, vec2, vec3, vec4);
        _mm256_storeu_pd(p, vec1);
        _mm256_storeu_pd(p + 4, vec2);
        _mm256_storeu_pd(p + 8, vec3);
        _mm256_storeu_pd(p + 12, vec4);

    } else if (numToSort >= 32) {
        for (unsigned i = 0; i < end; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(p + i);
            sort(vec1);
            _mm256_storeu_pd(p + i, vec1);
        }

        for (unsigned len = 8; len <= numToSort; len *= 2) {
            // Inner loop goes over all subdivisions.
            for (unsigned n = 0; n < numToSort; n += len) {
                InternalSortParams<double> const params{span, n, n + len - 1};
                compareFullLength_2n(params);
                laneCrossingCompare_2n(params, 0);
            }
        }
    }
}

void sort_4n(std::span<double> span) {
    auto p = span.data();
    auto numberToSort = span.size();

    std::size_t pow2 = (std::size_t)std::ceil(std::log2f(numberToSort));
    std::size_t imaginary_length = (int)std::pow(2, pow2);

    assert(mod4(numberToSort) == 0 && "The array to be sorted is not a multiple of 4!");

    if (numberToSort == 4) {
        __m256d vec = _mm256_loadu_pd(p);
        sort(vec);
        _mm256_storeu_pd(p, vec);
    } else if (numberToSort == 8) {
        __m256d vec1 = _mm256_loadu_pd(p);
        __m256d vec2 = _mm256_loadu_pd(p + 4);
        sort(vec1, vec2);
        _mm256_storeu_pd(p, vec1);
        _mm256_storeu_pd(p + 4, vec2);

    } else if (numberToSort == 16) {
        __m256d vec1 = _mm256_loadu_pd(p);
        __m256d vec2 = _mm256_loadu_pd(p + 4);
        __m256d vec3 = _mm256_loadu_pd(p + 8);
        __m256d vec4 = _mm256_loadu_pd(p + 12);
        sort(vec1, vec2, vec3, vec4);
        _mm256_storeu_pd(p, vec1);
        _mm256_storeu_pd(p + 4, vec2);
        _mm256_storeu_pd(p + 8, vec3);
        _mm256_storeu_pd(p + 12, vec4);

    } else {
        for (unsigned i = 0; i < span.size(); i += 4) {
            __m256d vec1 = _mm256_loadu_pd(p + i);
            sort(vec1);
            _mm256_storeu_pd(p + i, vec1);
        }

        // outer loop
        // len is number of floats in length to be compared
        // each step increases this length by factor of 2.
        for (unsigned len = 8; len <= imaginary_length; len *= 2) {
            // inner loop goes over all subdivisions
            for (unsigned n = 0; n < imaginary_length; n += len) {
                InternalSortParams<double> const params{span, n, n + len - 1};
                compareFullLength_4n(params);
                laneCrossingCompare_4n(params, 0);
            }
        }
    }
}

void sort(std::span<double> span) {
    if (span.size() <= 1)
        return;

    auto array = span.data();
    auto numberToSort = span.size();
    std::size_t maxIdx = numberToSort - 1;

    if (numberToSort < 4) {
        auto [reg1, mask] = maskload(std::span{array, numberToSort});
        sort(reg1);
        _mm256_maskstore_pd(array, mask, reg1);
    } else if (!(numberToSort & (numberToSort - 1)))
        sort_2n(span);
    else if (mod4(numberToSort) == 0)
        sort_4n(span);
    else {
        std::size_t log2 = std::size_t(std::ceil(std::log2f(numberToSort)));
        std::size_t maxSegmentLength = std::size_t(std::pow(2, log2));

        for (unsigned i = 0; i <= maxIdx - 3; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(array + i);
            sort(vec1);
            _mm256_storeu_pd(array + i, vec1);
        }
        ///////////////////////////////// load the partial one
        std::size_t reminder = mod4(span.size());
        if (reminder != 0) {
            auto idxToLoad = span.size() - reminder;
            auto p = array + idxToLoad;
            auto [reg1, mask] = maskload({p, reminder});
            sort(reg1);
            _mm256_maskstore_pd(array + idxToLoad, mask, reg1);
        }

        std::size_t segmentLength = 8;
        for (std::uint32_t i = 0; i <= log2 - 3; i++) {
            for (std::size_t n = 0; n < maxSegmentLength; n += segmentLength) {
                InternalSortParams<double> const params{span, n, n + segmentLength - 1};
                compareFullLength(params);
                laneCrossingCompareNew(params, 0U);
            }
            segmentLength *= 2;
        }
    }
}

} // namespace bitonic_sort
