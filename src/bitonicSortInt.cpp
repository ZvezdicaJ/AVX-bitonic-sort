#include "BitonicSort.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <span>
#include <vector>

#include <BitonicSortCommon.h>
#include <BitonicSortPrivate.h>
#include <TypeDefinitions.h>
#include <maskload.h>

namespace bitonic_sort {
namespace {

using RegMask = utils::RegMask<__m256i>;

RegMask maskload(std::span<const int> const span) {
    __m256i reg;
    __m256i mask;
    switch (span.size()) {
    case 1: {
        mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, LOAD);
        reg = _mm256_maskload_epi32(span.data(), mask);
        __m256i infinity = _mm256_set1_epi32(std::numeric_limits<int>::max());
        reg = _mm256_blend_epi32(infinity, reg, 0b00000001);
    } break;
    case 2: {
        mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, LOAD, LOAD);
        reg = _mm256_maskload_epi32(span.data(), mask);
        __m256i infinity = _mm256_set1_epi32(std::numeric_limits<int>::max());
        reg = _mm256_blend_epi32(infinity, reg, 0b00000011);
    } break;
    case 3: {
        mask = _mm256_set_epi32(0, 0, 0, 0, 0, LOAD, LOAD, LOAD);
        reg = _mm256_maskload_epi32(span.data(), mask);
        __m256i infinity = _mm256_set1_epi32(std::numeric_limits<int>::max());
        reg = _mm256_blend_epi32(infinity, reg, 0b00000111);
    } break;
    case 4: {
        mask = _mm256_set_epi32(0, 0, 0, 0, LOAD, LOAD, LOAD, LOAD);
        reg = _mm256_maskload_epi32(span.data(), mask);
        __m256i infinity = _mm256_set1_epi32(std::numeric_limits<int>::max());
        reg = _mm256_blend_epi32(infinity, reg, 0b00001111);
    } break;
    case 5: {
        mask = _mm256_set_epi32(0, 0, 0, LOAD, LOAD, LOAD, LOAD, LOAD);
        reg = _mm256_maskload_epi32(span.data(), mask);
        __m256i infinity = _mm256_set1_epi32(std::numeric_limits<int>::max());
        reg = _mm256_blend_epi32(infinity, reg, 0b00011111);
    } break;
    case 6: {
        mask = _mm256_set_epi32(0, 0, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD);
        reg = _mm256_maskload_epi32(span.data(), mask);
        __m256i infinity = _mm256_set1_epi32(std::numeric_limits<int>::max());
        reg = _mm256_blend_epi32(infinity, reg, 0b00111111);
    } break;
    case 7: {
        mask = _mm256_set_epi32(0, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD);
        reg = _mm256_maskload_epi32(span.data(), mask);
        __m256i infinity = _mm256_set1_epi32(std::numeric_limits<int>::max());
        reg = _mm256_blend_epi32(infinity, reg, 0b01111111);
    } break;
    case 8: {
        mask = _mm256_set_epi32(LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD);
        reg = _mm256_loadu_si256((__m256i *)span.data());
    } break;
    default:
        throw std::runtime_error("Invalid number of ints to load.");
    };
    return RegMask{std::move(reg), std::move(mask)};
}

template <std::uint8_t mask1, std::uint8_t mask2>
void shuffleAndCompare(__m256i &reg) {
    __m256i shuffled_reg = _mm256_shuffle_epi32(reg, mask1);
    __m256i max = _mm256_max_epi32(reg, shuffled_reg);
    __m256i min = _mm256_min_epi32(reg, shuffled_reg);
    reg = _mm256_blend_epi32(max, min, mask2);
}
void reverseAndCompare(__m256i &reg) {
    __m256i reversed_halves = _mm256_permute2f128_si256(reg, reg, 0b00000001);
    __m256i reversed = _mm256_shuffle_epi32(reversed_halves, 0b00011011);
    __m256i max = _mm256_max_epi32(reg, reversed);
    __m256i min = _mm256_min_epi32(reg, reversed);
    reg = _mm256_blend_epi32(max, min, 0b00001111);
}

void reverseAndCompare(__m256i &reg0, __m256i &reg1) {
    // reverse one of registers register reg0
    __m256i reversed_halves = _mm256_permute2f128_si256(reg0, reg0, 0b00000001);
    __m256i reversed = _mm256_shuffle_epi32(reversed_halves, 0b00011011);
    // register 2 vsebuje min vrednosti
    reg0 = _mm256_min_epi32(reg1, reversed);
    // register 1 vsebuje max vrednosti
    reg1 = _mm256_max_epi32(reg1, reversed);
}

void reverseHalvesAndCompare(__m256i &reg0) {
    // tu narediš *----* *----* *----* *----*
    __m256i reversed_halves = _mm256_permute2f128_si256(reg0, reg0, 0b00000001);
    __m256i max = _mm256_max_epi32(reg0, reversed_halves);
    __m256i min = _mm256_min_epi32(reg0, reversed_halves);
    reg0 = _mm256_blend_epi32(max, min, 0b00001111);
}

void compare(__m256i &reg0, __m256i &reg1) {
    {
        __m256i max = _mm256_max_epi32(reg0, reg1);
        __m256i min = _mm256_min_epi32(reg0, reg1);
        reg0 = min;
        reg1 = max;
    }
}
} // namespace

void sort(__m256i &reg) {
    //    shuffle_and_compare(reg, 0b10110001, 0b10101010);
    shuffleAndCompare<0b10110001, 0b01010101>(reg);
    shuffleAndCompare<0b00011011, 0b00110011>(reg);
    shuffleAndCompare<0b10110001, 0b01010101>(reg);
    reverseAndCompare(reg);
    shuffleAndCompare<0b01001110, 0b00110011>(reg);
    shuffleAndCompare<0b10110001, 0b01010101>(reg);
}

inline void sort(__m256i &reg0, __m256i &reg1) {
    sort(reg1); // sort first register
    sort(reg0); // sort second register
    reverseAndCompare(reg0, reg1);
    reverseHalvesAndCompare(reg0);
    reverseHalvesAndCompare(reg1);
    shuffleAndCompare<0b01001110, 0b00110011>(reg0);
    shuffleAndCompare<0b10110001, 0b01010101>(reg0);
    shuffleAndCompare<0b01001110, 0b00110011>(reg1);
    shuffleAndCompare<0b10110001, 0b01010101>(reg1);
}

void bitonic_merge(__m256i &reg0, __m256i &reg1) {
    reverseAndCompare(reg0, reg1);
    reverseHalvesAndCompare(reg0);
    reverseHalvesAndCompare(reg1);
    shuffleAndCompare<0b01001110, 0b00110011>(reg0);
    shuffleAndCompare<0b10110001, 0b01010101>(reg0);
    shuffleAndCompare<0b01001110, 0b00110011>(reg1);
    shuffleAndCompare<0b10110001, 0b01010101>(reg1);
}

void sort(__m256i &reg0, __m256i &reg1, __m256i &reg2, __m256i &reg3) {

    // sort each quarter
    sort(reg0);
    sort(reg1);
    sort(reg2);
    sort(reg3);
    // sort each half
    sort(reg0, reg1);
    sort(reg2, reg3);

    reverseAndCompare(reg0, reg3);
    reverseAndCompare(reg1, reg2);
    // sort full width
    compare(reg1, reg3);
    compare(reg0, reg2);
    compare(reg0, reg1);
    compare(reg2, reg3);

    reverseHalvesAndCompare(reg0);
    reverseHalvesAndCompare(reg1);
    reverseHalvesAndCompare(reg2);
    reverseHalvesAndCompare(reg3);

    shuffleAndCompare<0b01001110, 0b00110011>(reg0);
    shuffleAndCompare<0b01001110, 0b00110011>(reg1);
    shuffleAndCompare<0b01001110, 0b00110011>(reg2);
    shuffleAndCompare<0b01001110, 0b00110011>(reg3);

    shuffleAndCompare<0b10110001, 0b01010101>(reg0);
    shuffleAndCompare<0b10110001, 0b01010101>(reg1);
    shuffleAndCompare<0b10110001, 0b01010101>(reg2);
    shuffleAndCompare<0b10110001, 0b01010101>(reg3);
}

void bitonic_merge(__m256i &reg0, __m256i &reg1, __m256i &reg2, __m256i &reg3) {
    reverseAndCompare(reg0, reg3);
    reverseAndCompare(reg1, reg2);
    // sort full width
    compare(reg1, reg3);
    compare(reg0, reg2);

    compare(reg0, reg1);
    compare(reg2, reg3);

    reverseHalvesAndCompare(reg0);
    reverseHalvesAndCompare(reg1);
    reverseHalvesAndCompare(reg2);
    reverseHalvesAndCompare(reg3);

    shuffleAndCompare<0b01001110, 0b00110011>(reg0);
    shuffleAndCompare<0b01001110, 0b00110011>(reg1);
    shuffleAndCompare<0b01001110, 0b00110011>(reg2);
    shuffleAndCompare<0b01001110, 0b00110011>(reg3);

    shuffleAndCompare<0b10110001, 0b01010101>(reg0);
    shuffleAndCompare<0b10110001, 0b01010101>(reg1);
    shuffleAndCompare<0b10110001, 0b01010101>(reg2);
    shuffleAndCompare<0b10110001, 0b01010101>(reg3);
}

void compareFullLength2N(InternalSortParams<int> const &params) {
    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    assert(lastIdx > firstIdx);
    std::size_t length = lastIdx - firstIdx + 1;
    std::size_t half = length / 2;
    for (std::size_t i = 0; i < half; i += 8) {
        int *p1 = params.span.data() + firstIdx + i;
        int *p2 = params.span.data() + lastIdx - 7 - i;
        { // reverse lover half and compare to upper half
            __m256i vec1 = _mm256_loadu_si256((__m256i *)p1);
            __m256i vec2 = _mm256_loadu_si256((__m256i *)p2);
            reverseAndCompare(vec1, vec2);
            _mm256_storeu_si256((__m256i *)p1, vec1);
            _mm256_storeu_si256((__m256i *)p2, vec2);
        }
    }
}

void laneCrossingCompare2N(InternalSortParams<int> const &params, std::uint32_t depth) {
    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t length = lastIdx - firstIdx + 1;

    auto p = params.span.data() + firstIdx;
    if (length == 8) {
        __m256i reg = _mm256_loadu_si256((__m256i *)(p));
        // this is the ending case do single vector permutations
        reverseHalvesAndCompare(reg);
        shuffleAndCompare<0b01001110, 0b00110011>(reg);
        shuffleAndCompare<0b10110001, 0b01010101>(reg);
        _mm256_storeu_si256((__m256i *)(p), reg);
        return;
    }

    for (std::size_t i = 0; i < length / 2; i += 8) {
        {
            int *p1 = p + i;
            int *p2 = p + length / 2 + i;
            __m256i reg0 = _mm256_loadu_si256((__m256i *)p1); // i-ti od začetka
            __m256i reg1 = _mm256_loadu_si256((__m256i *)p2); // ta je prvi čez polovico
            // register 2 vsebuje min vrednosti
            __m256i min = _mm256_min_epi32(reg1, reg0);
            // register 1 vsebuje max vrednosti
            reg1 = _mm256_max_epi32(reg1, reg0);
            reg0 = min;
            _mm256_storeu_si256((__m256i *)p1, reg0);
            _mm256_storeu_si256((__m256i *)p2, reg1);
        }
    }
    laneCrossingCompare2N({params.span, firstIdx, (firstIdx + lastIdx) / 2}, depth + 1);
    laneCrossingCompare2N({params.span, (firstIdx + lastIdx) / 2 + 1, lastIdx}, depth + 1);
};

void sort_2n(std::span<int> span) {

    auto p = span.data();
    std::uint32_t numToSort = span.size();

    assert(!(numToSort & (numToSort - 1)) &&
           "The array to be sorted does not have length 2^n, where n is some "
           "integer. Got: numToSort = !");

    if (numToSort < 8) {
        auto [reg, mask] = maskload(span);
        sort(reg);
        _mm256_maskstore_epi32(p, mask, reg);
    } else if (numToSort == 8) {
        __m256i vec = _mm256_loadu_si256((__m256i *)p);
        sort(vec);
        _mm256_storeu_si256((__m256i *)p, vec);
    } else if (numToSort == 16) {
        __m256i vec1 = _mm256_loadu_si256((__m256i *)p);
        __m256i vec2 = _mm256_loadu_si256((__m256i *)(p + 8));
        sort(vec1, vec2);
        _mm256_storeu_si256((__m256i *)p, vec1);
        _mm256_storeu_si256((__m256i *)(p + 8), vec2);
    } else if (numToSort == 32) {
        __m256i vec1 = _mm256_loadu_si256((__m256i *)p);
        __m256i vec2 = _mm256_loadu_si256((__m256i *)(p + 8));
        __m256i vec3 = _mm256_loadu_si256((__m256i *)(p + 16));
        __m256i vec4 = _mm256_loadu_si256((__m256i *)(p + 24));
        sort(vec1, vec2, vec3, vec4);
        _mm256_storeu_si256((__m256i *)p, vec1);
        _mm256_storeu_si256((__m256i *)(p + 8), vec2);
        _mm256_storeu_si256((__m256i *)(p + 16), vec3);
        _mm256_storeu_si256((__m256i *)(p + 24), vec4);

    } else if (numToSort >= 64) {
        for (std::size_t i = 0; i < numToSort; i += 8) {
            __m256i vec1 = _mm256_loadu_si256((__m256i *)(p + i));
            sort(vec1);
            _mm256_storeu_si256((__m256i *)(p + i), vec1);
        }

        for (std::size_t len = 16; len <= numToSort; len *= 2) {
            for (std::size_t n = 0; n < numToSort; n += len) {
                InternalSortParams<int> const params{span, n, n + len - 1};
                compareFullLength2N(params);
                laneCrossingCompare2N(params, 0U);
            }
        }
    }
}

void compareFullLength8N(InternalSortParams<int> const &params) {
    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const &maxIdx = params.span.size() - 1;

    std::size_t length = firstIdx - lastIdx + 1;
    std::size_t half = length / 2; // half je index prvega cez polovico
    for (std::int32_t toLoadIdx = half - 8; toLoadIdx >= 0; toLoadIdx -= 8) {
        if (lastIdx - 7 - toLoadIdx > maxIdx)
            break;
        int *p1 = params.span.data() + firstIdx + toLoadIdx;
        int *p2 = params.span.data() + lastIdx - 7 - toLoadIdx;

        { // reverse lover half and compare to upper half
            __m256i vec1 = _mm256_loadu_si256((__m256i *)p1);
            __m256i vec2 = _mm256_loadu_si256((__m256i *)p2);
            __m256i reversed_halves = _mm256_permute2f128_si256(vec1, vec1, 0b00000001);
            __m256i reversed = _mm256_shuffle_epi32(reversed_halves, 0b00011011);
            vec1 = _mm256_min_epi32(reversed, vec2);
            vec2 = _mm256_max_epi32(reversed, vec2);
            reversed_halves = _mm256_permute2f128_si256(vec1, vec1, 0b00000001);
            vec1 = _mm256_shuffle_epi32(reversed_halves, 0b00011011);
            _mm256_storeu_si256((__m256i *)p1, vec1);
            _mm256_storeu_si256((__m256i *)p2, vec2);
        }
    }
}

inline void laneCrossingCompare8N(InternalSortParams<int> const &params, std::uint32_t depth) {

    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const &maxIdx = params.span.size() - 1;

    if (firstIdx > maxIdx) {
        return;
    }
    std::size_t const length = lastIdx - firstIdx + 1;
    int *p = params.span.data() + firstIdx;

    if (length == 8) {
        __m256i reg = _mm256_loadu_si256((__m256i *)(p));
        // this is the ending case do single vector permutations
        reverseHalvesAndCompare(reg);
        shuffleAndCompare<0b01001110, 0b00110011>(reg);
        shuffleAndCompare<0b10110001, 0b01010101>(reg);
        _mm256_storeu_si256((__m256i *)(p), reg);
        return;
    }

    for (std::size_t i = 0; i < length / 2; i += 8) {
        if (firstIdx + length / 2 + i > maxIdx)
            break;
        {
            int *p1 = p + i;
            int *p2 = p + length / 2 + i;
            __m256i reg0 = _mm256_loadu_si256((__m256i *)p1); // i-ti od začetka
            __m256i reg1 = _mm256_loadu_si256((__m256i *)p2); // ta je prvi čez polovico
            // register 2 vsebuje min vrednosti
            __m256i min = _mm256_min_epi32(reg1, reg0);
            // register 1 vsebuje max vrednosti
            reg1 = _mm256_max_epi32(reg1, reg0);
            reg0 = min;
            _mm256_storeu_si256((__m256i *)p1, reg0);
            _mm256_storeu_si256((__m256i *)p2, reg1);
        }
    }
    laneCrossingCompare8N({params.span, firstIdx, (firstIdx + lastIdx) / 2}, depth + 1);
    laneCrossingCompare8N({params.span, (firstIdx + lastIdx) / 2 + 1, lastIdx}, depth + 1);
};

inline void sort_8n(std::span<int> span) {

    int *p = span.data();
    std::size_t numToSort = span.size();

    assert(mod8(numToSort) == 0 && "The array to be sorted does not have the "
                                   "size that is a multiple of 8!");

    std::size_t vec_count = numToSort / 8;
    assert((numToSort >= 0 && (numToSort % 8) == 0) &&
           "The array to be sorted is not a multiples of 8!");

    if (numToSort == 8) {
        __m256i vec = _mm256_loadu_si256((__m256i *)p);
        sort(vec);
        _mm256_storeu_si256((__m256i *)p, vec);
    } else if (numToSort == 16) {
        __m256i vec1 = _mm256_loadu_si256((__m256i *)p);
        __m256i vec2 = _mm256_loadu_si256((__m256i *)(p + 8));
        sort(vec1, vec2);
        _mm256_storeu_si256((__m256i *)p, vec1);
        _mm256_storeu_si256((__m256i *)(p + 8), vec2);
    } else if (numToSort == 32) {
        __m256i vec1 = _mm256_loadu_si256((__m256i *)p);
        __m256i vec2 = _mm256_loadu_si256((__m256i *)(p + 8));
        __m256i vec3 = _mm256_loadu_si256((__m256i *)(p + 16));
        __m256i vec4 = _mm256_loadu_si256((__m256i *)(p + 24));
        sort(vec1, vec2, vec3, vec4);
        _mm256_storeu_si256((__m256i *)p, vec1);
        _mm256_storeu_si256((__m256i *)(p + 8), vec2);
        _mm256_storeu_si256((__m256i *)(p + 16), vec3);
        _mm256_storeu_si256((__m256i *)(p + 24), vec4);
    } else {
        for (std::size_t i = 0; i < span.size(); i += 8) {
            __m256i vec1 = _mm256_loadu_si256((__m256i *)(p + i));
            sort(vec1);
            _mm256_storeu_si256((__m256i *)(p + i), vec1);
        }

        std::size_t log2 = std::size_t(std::ceil(std::log2f(numToSort)));
        std::size_t maxSegmentLength = std::size_t(std::pow(2, log2));
        std::size_t segmentLength = 16;
        for (std::uint32_t i = 0; i <= log2 - 4; i++) {
            for (std::size_t n = 0; n < maxSegmentLength; n += segmentLength) {
                InternalSortParams<int> const params{span, n, n + segmentLength - 1};
                compareFullLength8N(params);
                laneCrossingCompare8N(params, 0U);
            }
        }
    }
}

inline void compareFullLength(InternalSortParams<int> params) {

    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const &maxIdx = params.span.size() - 1;

    std::size_t length = firstIdx - lastIdx + 1;
    std::size_t half = length / 2; // half je index prvega cez polovico
    for (int i = half - 8; i >= 0; i -= 8) {
        // index of last vector to load
        int last_vec_to_load = lastIdx - 7 - i;
        int diff = maxIdx - last_vec_to_load;
        if (diff <= 0) {
            return;
        }
        // define pointers to the start of vectors to load
        int *p1 = params.span.data() + firstIdx + i;
        int *p2 = params.span.data() + last_vec_to_load;
        __m256i vec2, mask;
        if (diff <= 8)
            auto [vec2, mask] = maskload({p2, std::uint32_t(diff + 1)});
        else
            vec2 = _mm256_loadu_si256((__m256i *)p2);
        { // reverse lover half and compare to upper half
            __m256i vec1 = _mm256_loadu_si256((__m256i *)p1);
            __m256i reversed_halves = _mm256_permute2f128_si256(vec1, vec1, 0b00000001);
            __m256i reversed = _mm256_shuffle_epi32(reversed_halves, 0b00011011);
            vec1 = _mm256_min_epi32(reversed, vec2);
            vec2 = _mm256_max_epi32(reversed, vec2);
            reversed_halves = _mm256_permute2f128_si256(vec1, vec1, 0b00000001);
            vec1 = _mm256_shuffle_epi32(reversed_halves, 0b00011011);
            _mm256_storeu_si256((__m256i *)p1, vec1);
            if (diff < 8)
                _mm256_maskstore_epi32(p2, mask, vec2);
            else
                _mm256_storeu_si256((__m256i *)p2, vec2);
        }
    }
}

inline void laneCrossingCompare(InternalSortParams<int> const &params, std::uint32_t depth) {
    std::size_t const &firstIdx = params.firstIdx;
    std::size_t const &lastIdx = params.lastIdx;
    std::size_t const &maxIdx = params.span.size() - 1;
    std::size_t length = lastIdx - firstIdx + 1;

    if (firstIdx > maxIdx) {
        return;
    }

    auto p = params.span.data();
    auto p1 = p + firstIdx;
    if (length == 8) {
        std::int64_t diff = maxIdx - firstIdx;
        if (diff < 1) {
            return;
        }

        auto [reg, mask] = [&]() {
            if (diff < 7)
                return maskload(std::span<int const>{p1, std::size_t(diff + 1)});
            else
                return RegMask{_mm256_loadu_si256((__m256i *)(p1)), __m256i{}};
        }();

        reverseHalvesAndCompare(reg);
        shuffleAndCompare<0b01001110, 0b00110011>(reg);
        shuffleAndCompare<0b10110001, 0b01010101>(reg);
        if (diff < 7)
            _mm256_maskstore_epi32(p1, mask, reg);
        else
            _mm256_storeu_si256((__m256i *)(p1), reg);

        return;
    }

    // This should be (length + 1) / 2, but not adding 1 does not change
    // anything since the length is always a multiple of 8.
    assert(length % 8 == 0);
    std::size_t halfLength = length / 2;

    for (std::size_t i = 0; i < length / 2; i += 8) {

        std::size_t loadIndex = firstIdx + halfLength + i;
        if (maxIdx < loadIndex) {
            break;
        }
        std::int32_t diff = maxIdx - loadIndex;

        int *p2 = p + loadIndex;
        auto [reg1, mask] = [&]() {
            if (diff < 7)
                return maskload(std::span<int const>{p2, std::size_t(diff + 1)});
            else
                return RegMask{_mm256_loadu_si256((__m256i *)p2), __m256i{}};
        }();

        __m256i reg0 = _mm256_loadu_si256((__m256i *)p1); // i-ti od začetka
        // register 2 vsebuje min vrednosti
        __m256i min = _mm256_min_epi32(reg1, reg0);
        // register 1 vsebuje max vrednosti
        reg1 = _mm256_max_epi32(reg1, reg0);
        reg0 = min;
        _mm256_storeu_si256((__m256i *)p1, reg0);
        if (diff < 7)
            _mm256_maskstore_epi32(p2, mask, reg1);
        else
            _mm256_storeu_si256((__m256i *)p2, reg1);
        p1 += 8;
    }

    laneCrossingCompare({params.span, firstIdx, (firstIdx + lastIdx) / 2}, depth + 1);
    laneCrossingCompare({params.span, (firstIdx + lastIdx) / 2 + 1, lastIdx}, depth + 1);
};

void sort(std::span<int> span) {

    auto p = span.data();
    std::size_t const numToSort = span.size();
    std::size_t const end = numToSort - 1;

    if (numToSort <= 1) {
        return;
    } else if (numToSort < 8) {
        auto [reg, mask] = maskload(span);
        sort(reg);
        _mm256_maskstore_epi32(p, mask, reg);
    } else if (!(numToSort & (numToSort - 1))) {
        sort_2n({p, numToSort});
    } else if (mod8(numToSort) == 0) {
        sort_8n({p, numToSort});
    } else if (numToSort < 16) {
        __m256i reg1 = _mm256_loadu_si256((__m256i *)p);
        auto [reg2, mask] = maskload({p + 8, numToSort - 8});
        sort(reg1, reg2);
        _mm256_storeu_si256((__m256i *)p, reg1);
        _mm256_maskstore_epi32(p + 8, mask, reg2);
    } else {
        std::size_t log2 = std::size_t(std::ceil(std::log2f(numToSort)));
        std::size_t maxSegmentLength = std::size_t(std::pow(2, log2));

        for (std::size_t i = 0; i <= end - 7; i += 8) {
            __m256i vec1 = _mm256_loadu_si256((__m256i *)(p + i));
            sort(vec1);
            _mm256_storeu_si256((__m256i *)(p + i), vec1);
        }

        std::int32_t reminder = mod8(span.size());
        if (reminder != 0) {
            int *p = p + numToSort - reminder;
            auto [reg1, mask] = maskload({p, std::size_t(reminder)});
            sort(reg1);
            _mm256_maskstore_epi32(p, mask, reg1);
        }

        std::size_t segmentLength = 16;
        for (std::uint32_t i = 0; i <= log2 - 4; i++) {
            for (std::size_t n = 0; n < maxSegmentLength; n += segmentLength) {
                InternalSortParams<int> const params{span, n, n + segmentLength - 1};
                compareFullLength(params);
                laneCrossingCompare(params, 0U);
            }
            segmentLength *= 2;
        }
    }
}

} // namespace bitonic_sort
