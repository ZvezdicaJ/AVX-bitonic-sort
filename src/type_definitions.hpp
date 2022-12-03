#pragma once
#ifndef __ALIGNED_VECTOR__

#include <vector>
#include <cstdlib>
#include <memory>
#include <limits>
#include <stdexcept>

#include <AlignedAllocator.h>

/*template <typename T>
using aligned_vector =
    std::vector<T, boost::alignment::aligned_allocator<T, sizeof(T)>>;
*/
// it's better to just use 32 byte alignment because in case of large elements a
// lot of space might be sued - like in case of aligned_vector<__m256d>
//template <typename T>
//using aligned_vector = std::vector<T>;
template <typename T>
using aligned_vector =
    std::vector<T, AlignedAllocator<T, 32>>;  //AlignedAllocator<T, 32>>;
#define __ALIGNED_VECTOR__ 1
#endif

// hints for compiler optimization
// this optimizes the machine code in such a way that it's less likely that
// instruction cache is going to be trashed
#define LIKELY(condition) __builtin_expect(static_cast<bool>(condition), 1)
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)


/*avx2 compare codes*/
/* Ordered vs Unordered has to do with whether the comparison is
 * true if one of the operands contains a NaN. Signaling (S) vs
 * non-signaling (Q for quiet?) will determine whether an exception
 * is raised if an operand contains a NaN.
 */
#ifndef __INTEL_COMPILER
#define _CMP_EQ_OQ 0x00 /* Equal (ordered, non-signaling)  */
#define _CMP_LT_OS 0x01 /* Less-than (ordered, signaling)  */
#define _CMP_LE_OS                                                 \
    0x02 /* Less-than-or-equal (ordered, signaling)  */
#define _CMP_UNORD_Q 0x03 /* Unordered (non-signaling)  */
#define _CMP_NEQ_UQ                                                \
    0x04 /* Not-equal (unordered, non-signaling)                   \
          */
#define _CMP_NLT_US                                                \
    0x05 /* Not-less-than (unordered, signaling)                   \
          */
#define _CMP_NLE_US                                                \
    0x06 /* Not-less-than-or-equal (unordered, signaling)  */
#define _CMP_ORD_Q 0x07 /* Ordered (nonsignaling)   */
#define _CMP_EQ_UQ 0x08 /* Equal (unordered, non-signaling)  */
#define _CMP_NGE_US                                                \
    0x09 /* Not-greater-than-or-equal (unord, signaling)  */
#define _CMP_NGT_US                                                \
    0x0a /* Not-greater-than (unordered, signaling)  */
#define _CMP_FALSE_OQ 0x0b /* False (ordered, non-signaling)  */
#define _CMP_NEQ_OQ 0x0c   /* Not-equal (ordered, non-signaling)  */
#define _CMP_GE_OS                                                 \
    0x0d /* Greater-than-or-equal (ordered, signaling)  */
#define _CMP_GT_OS 0x0e   /* Greater-than (ordered, signaling)  */
#define _CMP_TRUE_UQ 0x0f /* True (unordered, non-signaling)  */
#define _CMP_EQ_OS 0x10   /* Equal (ordered, signaling)  */
#define _CMP_LT_OQ 0x11   /* Less-than (ordered, non-signaling)  */
#define _CMP_LE_OQ                                                 \
    0x12 /* Less-than-or-equal (ordered, non-signaling)  */
#define _CMP_UNORD_S 0x13 /* Unordered (signaling)  */
#define _CMP_NEQ_US 0x14  /* Not-equal (unordered, signaling)  */
#define _CMP_NLT_UQ                                                \
    0x15 /* Not-less-than (unordered, non-signaling)  */
#define _CMP_NLE_UQ                                                \
    0x16 /* Not-less-than-or-equal (unord, non-signaling)  */
#define _CMP_ORD_S 0x17 /* Ordered (signaling)  */
#define _CMP_EQ_US 0x18 /* Equal (unordered, signaling)  */
#define _CMP_NGE_UQ                                                \
    0x19 /* Not-greater-than-or-equal (unord, non-sign)  */
#define _CMP_NGT_UQ                                                \
    0x1a /* Not-greater-than (unordered, non-signaling)  */
#define _CMP_FALSE_OS 0x1b /* False (ordered, signaling)  */
#define _CMP_NEQ_OS 0x1c   /* Not-equal (ordered, signaling)  */
#define _CMP_GE_OQ                                                 \
    0x1d /* Greater-than-or-equal (ordered, non-signaling)  */
#define _CMP_GT_OQ                                                 \
    0x1e                  /* Greater-than (ordered, non-signaling) \
                           */
#define _CMP_TRUE_US 0x1f /* True (unordered, signaling)  */
#endif
