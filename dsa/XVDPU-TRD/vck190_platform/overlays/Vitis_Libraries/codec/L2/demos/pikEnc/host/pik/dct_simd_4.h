// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_DCT_SIMD_4_H_
#define PIK_DCT_SIMD_4_H_

#include "pik/block.h"
#include "pik/compiler_specific.h"
#include "pik/dct_simd_any.h"
#include "pik/simd/simd.h"

#if (SIMD_TARGET_VALUE != SIMD_AVX2) && (SIMD_TARGET_VALUE != SIMD_NONE)

namespace pik {

// DCT building blocks that require SIMD vector length to be 4, e.g. SSE4.
static_assert(BlockDesc<8>().N == 4, "Wrong vector size, must be 4");

template <class From, class To>
static SIMD_ATTR PIK_INLINE void TransposeBlock8_V4(const From& from,
                                                    const To& to) {
  const auto p0L = from.Load(0, 0);
  const auto p0H = from.Load(0, 4);
  const auto p1L = from.Load(1, 0);
  const auto p1H = from.Load(1, 4);
  const auto p2L = from.Load(2, 0);
  const auto p2H = from.Load(2, 4);
  const auto p3L = from.Load(3, 0);
  const auto p3H = from.Load(3, 4);
  const auto p4L = from.Load(4, 0);
  const auto p4H = from.Load(4, 4);
  const auto p5L = from.Load(5, 0);
  const auto p5H = from.Load(5, 4);
  const auto p6L = from.Load(6, 0);
  const auto p6H = from.Load(6, 4);
  const auto p7L = from.Load(7, 0);
  const auto p7H = from.Load(7, 4);

  const auto q0L = interleave_lo(p0L, p2L);
  const auto q0H = interleave_lo(p0H, p2H);
  const auto q1L = interleave_lo(p1L, p3L);
  const auto q1H = interleave_lo(p1H, p3H);
  const auto q2L = interleave_hi(p0L, p2L);
  const auto q2H = interleave_hi(p0H, p2H);
  const auto q3L = interleave_hi(p1L, p3L);
  const auto q3H = interleave_hi(p1H, p3H);
  const auto q4L = interleave_lo(p4L, p6L);
  const auto q4H = interleave_lo(p4H, p6H);
  const auto q5L = interleave_lo(p5L, p7L);
  const auto q5H = interleave_lo(p5H, p7H);
  const auto q6L = interleave_hi(p4L, p6L);
  const auto q6H = interleave_hi(p4H, p6H);
  const auto q7L = interleave_hi(p5L, p7L);
  const auto q7H = interleave_hi(p5H, p7H);

  const auto r0L = interleave_lo(q0L, q1L);
  const auto r0H = interleave_lo(q0H, q1H);
  const auto r1L = interleave_hi(q0L, q1L);
  const auto r1H = interleave_hi(q0H, q1H);
  const auto r2L = interleave_lo(q2L, q3L);
  const auto r2H = interleave_lo(q2H, q3H);
  const auto r3L = interleave_hi(q2L, q3L);
  const auto r3H = interleave_hi(q2H, q3H);
  const auto r4L = interleave_lo(q4L, q5L);
  const auto r4H = interleave_lo(q4H, q5H);
  const auto r5L = interleave_hi(q4L, q5L);
  const auto r5H = interleave_hi(q4H, q5H);
  const auto r6L = interleave_lo(q6L, q7L);
  const auto r6H = interleave_lo(q6H, q7H);
  const auto r7L = interleave_hi(q6L, q7L);
  const auto r7H = interleave_hi(q6H, q7H);

  to.Store(r0L, 0, 0);
  to.Store(r4L, 0, 4);
  to.Store(r1L, 1, 0);
  to.Store(r5L, 1, 4);
  to.Store(r2L, 2, 0);
  to.Store(r6L, 2, 4);
  to.Store(r3L, 3, 0);
  to.Store(r7L, 3, 4);
  to.Store(r0H, 4, 0);
  to.Store(r4H, 4, 4);
  to.Store(r1H, 5, 0);
  to.Store(r5H, 5, 4);
  to.Store(r2H, 6, 0);
  to.Store(r6H, 6, 4);
  to.Store(r3H, 7, 0);
  to.Store(r7H, 7, 4);
}

template <class From>
static SIMD_ATTR PIK_INLINE float ComputeScaledDC8_V4(const From& from) {
  const auto p0L = from.Load(0, 0);
  const auto p0H = from.Load(0, 4);
  const auto p1L = from.Load(1, 0);
  const auto p1H = from.Load(1, 4);
  const auto p2L = from.Load(2, 0);
  const auto p2H = from.Load(2, 4);
  const auto p3L = from.Load(3, 0);
  const auto p3H = from.Load(3, 4);
  const auto p4L = from.Load(4, 0);
  const auto p4H = from.Load(4, 4);
  const auto p5L = from.Load(5, 0);
  const auto p5H = from.Load(5, 4);
  const auto p6L = from.Load(6, 0);
  const auto p6H = from.Load(6, 4);
  const auto p7L = from.Load(7, 0);
  const auto p7H = from.Load(7, 4);

  const auto q0 = p0L + p0H;
  const auto q1 = p1L + p1H;
  const auto q2 = p2L + p2H;
  const auto q3 = p3L + p3H;
  const auto q4 = p4L + p4H;
  const auto q5 = p5L + p5H;
  const auto q6 = p6L + p6H;
  const auto q7 = p7L + p7H;

  const auto r0 = q0 + q1;
  const auto r2 = q2 + q3;
  const auto r4 = q4 + q5;
  const auto r6 = q6 + q7;

  const auto s0 = r0 + r2;
  const auto s4 = r4 + r6;

  const auto sum = ext::sum_of_lanes(s0 + s4);

  return get_part(SIMD_PART(float, 1)(), sum);
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ComputeTransposedScaledDCT8_V4(const From& from,
                                                         const To& to) {
  // TODO(user): it is possible to avoid using temporary array,
  // after generalizing "To" to be bi-directional; all sub-transforms could
  // be performed "in-place".
  SIMD_ALIGN float block[8 * 8];
  ColumnDCT8(from, ToBlock<8>(block));
  TransposeBlock8_V4(FromBlock<8>(block), ToBlock<8>(block));
  ColumnDCT8(FromBlock<8>(block), to);
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ComputeTransposedScaledIDCT8_V4(const From& from,
                                                          const To& to) {
  // TODO(user): it is possible to avoid using temporary array,
  // after generalizing "To" to be bi-directional; all sub-transforms could
  // be performed "in-place".
  SIMD_ALIGN float block[8 * 8];
  ColumnIDCT8(from, ToBlock<8>(block));
  TransposeBlock8_V4(FromBlock<8>(block), ToBlock<8>(block));
  ColumnIDCT8(FromBlock<8>(block), to);
}

}  // namespace pik

#endif  // SIMD_TARGET_VALUE

#endif  // THIRD_PARTY_DCT_SIMD_4_H_
