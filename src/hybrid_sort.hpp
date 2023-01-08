#pragma once
#include "bitonic_sort.hpp"
#include "auxiliary_functions.hpp"
#include <bitset>
/* This file contains implementation of hybrid sort (mixture of quick sort and
 * bitonic sort algorithms).
 */

namespace HYBRID_SORT {

static const int LOAD = 0xffffffff;
static const int STORE = 0xffffffff;
//////////////////////////////////////////
//////// A scalar implementation of Quick sort from
/// https://www.geeksforgeeks.org/quick-sort/
//////////////////////////////////////////

// A utility function to swap two elements
/** @brief The function just swaps to values based on the pointers provided
 * @param a pointer to the first element
 * @param b pointer to the second element
 */
template <typename T>
inline void swap(T *a, T *b) {
    T t = *a;
    *a = *b;
    *b = t;
}

/** @brief Partition function for scalar quick sort algorithm
 * @details This function takes last element as pivot, places
 * the pivot element at its correct position in sorted
 * array, and places all smaller (smaller than pivot)
 * to left of pivot and all greater elements to right
 * of pivot.
 * @param arr pointer to an array of floats
 * @param low first element to be sorted
 * @param high last element to be sorted
 */
template <typename T>
T scalar_partition(T *arr, int low, int high) {
    // pivot is an element around which we partition the given array
    // it can be chosen arbitrarily
    // we partition the given array in two parts:
    // first part: elements bigger than pivot
    // second part: elements smaller than pivot
    T pivot = arr[high];
    // set the starting index one less than the first element of array
    // before the first write it will get incremented by 1
    int i = (low - 1); // Index of smaller element

    // start at the beginning of array
    // check if an element is smaller than the pivot
    // if yes put into the part of array containing smaller elements
    // move that element that was previously there to upper part
    // you are yet to check the upper part - as j increments
    for (int j = low; j <= high - 1; j++) {
        // If current element is smaller than or
        // equal to pivot
        if (arr[j] <= pivot) {
            i++; // increment index of smaller element
            swap(arr + i, arr + j);
        }
    }
    swap(arr + i + 1, arr + high);
    // return pivot index
    return (i + 1);
}

/** @brief The main function that implements scalar QuickSort
 * @param arr pointer to an array to be sorted
 * @param low  first index to be sorted
 * @param high last index to be sorted
 */
template <typename T>
inline void scalar_QS(T *arr, int low, int high) {
    if (low < high) {
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = scalar_partition(arr, low, high);

        // Separately sort elements before
        // partition and after partition
        scalar_QS(arr, low, pi - 1);
        scalar_QS(arr, pi + 1, high);
    }
}
//////////////////////////////////////////////////////////////////
//////////////// End of scalar Quick sort
///////////////////////////////////////////////////////////////////

void print_aligned_vector(aligned_vector<float> vec, std::string str) {
    std::cout << str << " " << std::endl;
    for (int i = 0; i < vec.size(); i++) {
        std::cout << vec[i] << std::endl;
        if (mod8(i + 1) == 0)
            std::cout << "\n";
    }
}
///////////////////////////////////////////////////////
///////////  This is just a try at my own merge  - which does not work in place
///////////////////////////////////////////////////////

/**
 * @brief The function accepts two already sorted  __m256 vectors
 * and sorts them.
 * @param reg1 upper vector of numbers - at the end it contains
 * larger values, the largest value is in the upper half of register
 * [255:192]
 *
 * @param reg0 lower vector of numbers - at the end it contains
 * smaller values. The smallest value is in the lowest half of
 * register - at [63:0]
 *
 */

inline void bitonic_merge(__m256 &reg0, __m256 &reg1) {
    reverseAndCompare(reg0, reg1);
    reverseHalvesAndCompare(reg0);
    reverseHalvesAndCompare(reg1);
    shuffleAndCompare<0b01001110, 0b00110011>(reg0);
    shuffleAndCompare<0b10110001, 0b01010101>(reg0);
    shuffleAndCompare<0b01001110, 0b00110011>(reg1);
    shuffleAndCompare<0b10110001, 0b01010101>(reg1);
    return;
}

/**
 * @brief The function accepts four sorted  __m256 vectors and sorts
 * them.
 * @param reg3 upper vector of numbers - at the end it contains
 * larger values, the largest value is in the upper half of register
 * [255:192]
 * @param reg2
 * @param reg1
 * @param reg0 lower vector of numbers - at the end it contains
 * smaller values. The smallest value is in the lowest half of
 * register - at [63:0]
 *
 */
inline void bitonic_merge(__m256 &reg0, __m256 &reg1, __m256 &reg2, __m256 &reg3) {
    reverseAndCompare(reg0, reg3);
    reverseAndCompare(reg1, reg2);
    // sort full width
    computeMinMax(reg1, reg3);
    computeMinMax(reg0, reg2);

    computeMinMax(reg0, reg1);
    computeMinMax(reg2, reg3);

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



/**
 * @brief function merges ordered arrays
 * @param arrayA pointer to the start of the array
 * @param startA index of the first element/vector to be loaded
 * @param endA index of the last vector to be loaded
 * @param endB index of the last vector to be loaded
 */
inline void bitonic_merge_8n(float *array, int startA, int endA, int endB) {

    aligned_vector<float> temp_vec;
    temp_vec.reserve(endB - startA + 8);
    float *p = temp_vec.data();

    __m256 reg1 = _mm256_load_ps(array + startA);
    __m256 reg2 = _mm256_load_ps(array + endA + 8);
    // below indices already account for above loads
    int indexB = endA + 16;
    int indexA = startA + 8;
    // int i = startA;
    int i = 0;
    BITONIC_SORT::bitonic_merge(reg1, reg2);
    // _mm256_store_ps(array + i, reg1);
    _mm256_store_ps(p + i, reg1);
    i += 8;

    while (indexB <= endB && indexA <= endA) {

        if (*(array + indexA) < *(array + indexB)) {
            reg1 = _mm256_load_ps(array + indexA);
            indexA += 8;
        } else {
            reg1 = _mm256_load_ps(array + indexB);
            indexB += 8;
        }
        BITONIC_SORT::bitonic_merge(reg1, reg2);
        //_mm256_store_ps(array + i, reg1);
        _mm256_store_ps(p + i, reg1);
        i += 8;
    }

    while (indexA <= endA) {
        reg1 = _mm256_load_ps(array + indexA);
        indexA += 8;
        BITONIC_SORT::bitonic_merge(reg1, reg2);
        //_mm256_store_ps(array + i, reg1);
        _mm256_store_ps(p + i, reg1);
        i += 8;
    }

    while (indexB <= endB) {

        reg1 = _mm256_load_ps(array + indexB);
        indexB += 8;
        BITONIC_SORT::bitonic_merge(reg1, reg2);
        //_mm256_store_ps(array + i, reg1);
        _mm256_store_ps(p + i, reg1);
        i += 8;
    }
    //_mm256_store_ps(array + i, reg2);
    _mm256_store_ps(p + i, reg2);

    __m256i *pi = (__m256i *)p;
    __m256i *arr = (__m256i *)(array + startA);
    unsigned num_vec = (endB - startA + 8) / 8;

    for (int i = 0; i < num_vec; i++) {
        __m256i temp = _mm256_stream_load_si256(pi);
        _mm256_stream_si256(arr, temp);
        pi++;
        arr++;
    }
}

/**
 *@brief Hybrid sort function which sorts arrays containing 8n elements
 * @param vec vector to be sorted
 * @param start index of first element to be sorted
 * @param end index of the last element to be sorted
 */
inline void hybrid_sort_8n(aligned_vector<float> &vec, int start, int end) {
    unsigned size = end - start + 1;
    unsigned num_registers = size / 8;
    float *p = vec.data();
    for (int i = start; i < end; i += 8) {
        __m256 reg = _mm256_load_ps(p + i);
        BITONIC_SORT::bitonic_sort(reg);
        _mm256_store_ps(p + i, reg);
    }

    int pow2 = (int)std::ceil(std::log2f(end + 1)) - 1;
    int imaginary_half = (int)std::pow(2, pow2);
    int imaginary_length = 2 * imaginary_half;

    // width is max width of each of arrays to be merged
    for (int width = 8; width <= imaginary_half; width *= 2) {
        //  std::cout << "\n\nWIDTH: " << width << "\n" << std::endl;
        // if (width >= 16)
        //    break;
        for (int start = 0; start <= end - 7 - width; start += 2 * width)

            // startA - index of first A vector to be loaded
            // endA - index of the last A vector to be loaded
            // endB - index of the last B vector to be loaded
            bitonic_merge_8n(vec.data(), start, start + width - 8,
                             std::min(start + 2 * width - 8, end - 7));
    }
    //
}

//////////////////////////////////////////////////////////////
// HYBRID_SORT FROM PAPER: A Novel Hybrid Quicksort Algorithm Vectorized using
// AVX-512 on Intel Skylake
///////////////////////////////////////////////////////

/** @brief Function compresses 256 vector towards the lower half (lower
 * elements) based on a provided mask.
 * @detail more details on
 * https://stackoverflow.com/questions/36932240/avx2-what-is-the-most-efficient-way-to-pack-left-based-on-a-mask
 * @param src 256 bit register to be compressed
 * @param mask  mask according to which the register is compressed
 */
inline __m256 compress256(__m256 src, unsigned int mask /* from movmskps */) {
    uint64_t expanded_mask =
        _pdep_u64(mask, 0x0101010101010101); // unpack each bit to a byte
    expanded_mask *= 0xFF; // mask |= mask<<1 | mask<<2 | ... | mask<<7;
    // ABC... -> AAAAAAAABBBBBBBBCCCCCCCC...: replicate each bit to fill its
    // byte

    const uint64_t identity_indices =
        0x0706050403020100; // the identity shuffle for vpermps, packed to one
                            // index per byte
    uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

    __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
    __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);

    return _mm256_permutevar8x32_ps(src, shufmask);
}

#ifndef __AVX512__
/** @brief The function compresses and stores 256 bit register.
 * @param p pointer to an array of floats
 * @param mask_u32 32 bit mask according to which the register is compressed
 * @param vec 256 bit register to compress
 */
inline void _mm256_compresstoreu_ps(float *p, unsigned int mask_u32,
                                    __m256 vec) {

    __m256 compressed = compress256(vec, mask_u32);
    unsigned count = _mm_popcnt_u32(mask_u32);

    switch (count) {
    case 1: {
        __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, STORE);
        _mm256_maskstore_ps(p, mask, compressed);
    } break;
    case 2: {
        __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, STORE, STORE);
        _mm256_maskstore_ps(p, mask, compressed);
    } break;
    case 3: {
        __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, STORE, STORE, STORE);
        _mm256_maskstore_ps(p, mask, compressed);
    } break;
    case 4: {
        __m256i mask = _mm256_set_epi32(0, 0, 0, 0, STORE, STORE, STORE, STORE);
        _mm256_maskstore_ps(p, mask, compressed);
    } break;
    case 5: {
        __m256i mask =
            _mm256_set_epi32(0, 0, 0, STORE, STORE, STORE, STORE, STORE);
        _mm256_maskstore_ps(p, mask, compressed);
    } break;
    case 6: {
        __m256i mask =
            _mm256_set_epi32(0, 0, STORE, STORE, STORE, STORE, STORE, STORE);
        _mm256_maskstore_ps(p, mask, compressed);
    } break;
    case 7: {
        __m256i mask = _mm256_set_epi32(0, STORE, STORE, STORE, STORE, STORE,
                                        STORE, STORE);
        _mm256_maskstore_ps(p, mask, compressed);
    } break;
    }
}
#endif

/**
 * @brief This function performs simd partition for quick sort algorithm
 * @details The function is an avx2 implementation of an algorithm from the
 * paper: A Novel Hybrid Quicksort Algorithm Vectorized using AVX-512 on
 * Intel Skylake; Author: Berenger Bramas
 * @param array Pointer to the start of whole array
 * @param left First index to be sorted
 * @param right Last index to be sorted - which is also pivot
 *
 */
inline unsigned simd_partition(float *array, unsigned left, unsigned right) {
    // std::cout << "left:  " << left << "  right:  " << right << std::endl;
    unsigned array_end = right;
    int length = right - left + 1;

    static const unsigned S = 8;

    float pivot = array[right];
    __m256 pivotvec = _mm256_set1_ps(pivot);

    __m256 left_val = _mm256_loadu_ps(array + left);
    int left_w = left;
    left += 8;

    int right_w = right + 1;
    right -= 7;
    __m256 right_val = _mm256_loadu_ps(array + right);

    while (left + S <= right) {
        // free_left indicates how many indices are free to write on the left
        // end of array
        const int free_left = left - left_w;
        // free_right indicates how many indices are free to write on the right
        // side of array
        const int free_right = right_w - right;

        __m256 val;
        // on which side we load new vector depends on the number of free
        // indices
        if (free_left <= free_right) {
            val = _mm256_loadu_ps(array + left);
            left += S;
        } else {
            right -= S;
            val = _mm256_loadu_ps(array + right);
        }
        __m256 mask_vec = _mm256_cmp_ps(val, pivotvec, _CMP_LE_OQ);
        unsigned mask = _mm256_movemask_ps(mask_vec);
        unsigned num_bits_set = _mm_popcnt_u32(mask);

        if (0 < num_bits_set && num_bits_set < 8) {
            _mm256_compresstoreu_ps(array + left_w, mask, val);
            left_w += num_bits_set;
            unsigned mask_high = (~mask) & 0x00ff;
            right_w -= (S - num_bits_set);
            _mm256_compresstoreu_ps(array + right_w, (~mask) & 0x00ff, val);
        } else if (num_bits_set == 8) {
            _mm256_storeu_ps(array + left_w, val);
            left_w += 8;
        } else if (num_bits_set == 0) {
            right_w -= S;
            _mm256_storeu_ps(array + right_w, val);
        }
    }

    if (left != right) {
        unsigned remaining = right - left;

        __m256 val = _mm256_loadu_ps(array + left);
        left = right;
        __m256 mask_vec = _mm256_cmp_ps(val, pivotvec, _CMP_LE_OQ);
        unsigned int mask = _mm256_movemask_ps(mask_vec);
        unsigned int mask_low = mask & ~(0xFF << remaining);
        unsigned int mask_high = (~mask_low & ~(0xFF << remaining)) & 0xFF;
        unsigned num_bits_low = _mm_popcnt_u32(mask_low);
        unsigned num_bits_high = _mm_popcnt_u32(mask_high);

        if (0 <= num_bits_low && num_bits_low < 8) {
            _mm256_compresstoreu_ps(array + left_w, mask_low, val);
            left_w += num_bits_low;
            right_w -= num_bits_high;
            _mm256_compresstoreu_ps(array + right_w, mask_high, val);

        } else if (num_bits_low == 8) {
            _mm256_storeu_ps(array + left_w, val);
            left_w += 8;
        } else if (num_bits_low == 8) {
            right_w -= 8;
            _mm256_storeu_ps(array + right_w, val);
        }
    }
    {
        __m256 mask_vec = _mm256_cmp_ps(left_val, pivotvec, _CMP_LE_OQ);
        unsigned mask = _mm256_movemask_ps(mask_vec);
        unsigned num_bits_set = _mm_popcnt_u32(mask);

        if (0 < num_bits_set && num_bits_set < 8) {
            _mm256_compresstoreu_ps(array + left_w, mask, left_val);
            left_w += num_bits_set;
            right_w -= (S - num_bits_set);
            _mm256_compresstoreu_ps(array + right_w, (~mask) & 0x00ff,
                                    left_val);
        } else if (num_bits_set == 8) {
            _mm256_storeu_ps(array + left_w, left_val);
            left_w += 8;
        } else if (num_bits_set == 0) {
            right_w -= S;
            _mm256_storeu_ps(array + right_w, left_val);
        }
    }
    {
        __m256 mask_vec = _mm256_cmp_ps(right_val, pivotvec, _CMP_LE_OQ);
        unsigned mask = _mm256_movemask_ps(mask_vec);
        unsigned num_bits_set = _mm_popcnt_u32(mask);

        if (0 < num_bits_set && num_bits_set < 8) {
            _mm256_compresstoreu_ps(array + left_w, mask, right_val);
            left_w += num_bits_set;
            right_w -= (S - num_bits_set);
            _mm256_compresstoreu_ps(array + right_w, (~mask) & 0x00FF,
                                    right_val);
        } else if (num_bits_set == 8) {
            _mm256_storeu_ps(array + left_w, right_val);
            left_w += 8;
        } else if (num_bits_set == 0) {
            right_w -= 8;
            _mm256_storeu_ps(array + right_w, right_val);
        }
    }
    return left_w - 1;
}

/**
 * @brief Helper function for sorting vector.
 * @details Sorting algorithm is based simd quick sort algorithm presented in
 * HYBRID_SORT FROM PAPER: A Novel Hybrid Quicksort Algorithm Vectorized using
 * AVX-512 on Intel Skylake
 * @param array pointer to data to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 */
inline void simd_QS_helper(float *array, unsigned start, unsigned end) {
    int length = end - start + 1;
    if (UNLIKELY(length <= 1))
        return;
    if (LIKELY(length > 16)) {
        unsigned partition_bound = simd_partition(array, start, end);
        simd_QS_helper(array, start, partition_bound - 1);
        simd_QS_helper(array, partition_bound + 1, end);
    } else if (UNLIKELY(length == 16)) {
        __m256 vec1 = _mm256_loadu_ps(array + start);
        __m256 vec2 = _mm256_loadu_ps(array + start + 8);
        BITONIC_SORT::bitonic_sort(vec1, vec2);
        _mm256_storeu_ps(array + start, vec1);
        _mm256_storeu_ps(array + start + 8, vec2);
    } else if (LIKELY(8 < length)) {
        __m256 vec1 = _mm256_loadu_ps(array + start);

        int reminder = mod8(length);

        float *p = array + start + 8;
        __m256 vec2;
        __m256i mask;
        BITONIC_SORT::maskload(reminder - 1, p, mask, vec2);
        BITONIC_SORT::bitonic_sort(vec1, vec2);

        _mm256_storeu_ps(array + start, vec1);
        _mm256_maskstore_ps(array + start + 8, mask, vec2);

        //        quickSort(array, start, end);
    } else if (UNLIKELY(length == 8)) {
        __m256 reg = _mm256_loadu_ps(array + start);
        BITONIC_SORT::bitonic_sort(reg);
        _mm256_storeu_ps(array + start, reg);

    } else if (1 < length && length < 8) {
        __m256 reg;
        int diff = end - start;
        __m256i mask;
        BITONIC_SORT::maskload(diff, array + start, mask, reg);
        BITONIC_SORT::bitonic_sort(reg);
        _mm256_maskstore_ps(array + start, mask, reg);
    }
}

/**
 * @brief The function sorts aligned vector of floats.
 * @details Sorting algorithm is based simd quick sort algorithm presented in
 * HYBRID_SORT FROM PAPER: A Novel Hybrid Quicksort Algorithm Vectorized using
 * AVX-512 on Intel Skylake
 * @param vec vector to be sorted
 * @param start index of the first element to be sorted
 * @param end index of the last element to be sorted
 */
inline void simd_QS(aligned_vector<float> &vec, unsigned start, unsigned end) {
    simd_QS_helper(vec.data(), start, end);
}

} // namespace HYBRID_SORT
