// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_DCT_SIMD_8_H_
#define PIK_DCT_SIMD_8_H_

#include "pik/block.h"
#include "pik/compiler_specific.h"
#include "pik/dct_simd_any.h"
#include "pik/simd/simd.h"

#if SIMD_TARGET_VALUE == SIMD_AVX2

namespace pik {

// DCT building blocks that require SIMD vector length to be 8, e.g. AVX2.
static_assert(BlockDesc<8>().N == 8, "Wrong vector size, must be 8");

// Each vector holds one row of the input/output block.
template <class V>
SIMD_ATTR PIK_INLINE void TransposeBlock8_V8(V& i0, V& i1, V& i2, V& i3, V& i4,
                                             V& i5, V& i6, V& i7) {
  // Surprisingly, this straightforward implementation (24 cycles on port5) is
  // faster than load128+insert and load_dup128+concat_hi_lo+blend.
  const auto q0 = interleave_lo(i0, i2);
  const auto q1 = interleave_lo(i1, i3);
  const auto q2 = interleave_hi(i0, i2);
  const auto q3 = interleave_hi(i1, i3);
  const auto q4 = interleave_lo(i4, i6);
  const auto q5 = interleave_lo(i5, i7);
  const auto q6 = interleave_hi(i4, i6);
  const auto q7 = interleave_hi(i5, i7);

  const auto r0 = interleave_lo(q0, q1);
  const auto r1 = interleave_hi(q0, q1);
  const auto r2 = interleave_lo(q2, q3);
  const auto r3 = interleave_hi(q2, q3);
  const auto r4 = interleave_lo(q4, q5);
  const auto r5 = interleave_hi(q4, q5);
  const auto r6 = interleave_lo(q6, q7);
  const auto r7 = interleave_hi(q6, q7);

  i0 = concat_lo_lo(r4, r0);
  i1 = concat_lo_lo(r5, r1);
  i2 = concat_lo_lo(r6, r2);
  i3 = concat_lo_lo(r7, r3);
  i4 = concat_hi_hi(r4, r0);
  i5 = concat_hi_hi(r5, r1);
  i6 = concat_hi_hi(r6, r2);
  i7 = concat_hi_hi(r7, r3);
}

template <class From>
static SIMD_ATTR PIK_INLINE float ComputeScaledDC8_V8(const From& from) {
  const auto q0 = from.Load(0, 0);
  const auto q1 = from.Load(1, 0);
  const auto q2 = from.Load(2, 0);
  const auto q3 = from.Load(3, 0);
  const auto q4 = from.Load(4, 0);
  const auto q5 = from.Load(5, 0);
  const auto q6 = from.Load(6, 0);
  const auto q7 = from.Load(7, 0);

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
SIMD_ATTR PIK_INLINE void TransposeBlock8_V8(const From& from, const To& to) {
  auto i0 = from.Load(0, 0);
  auto i1 = from.Load(1, 0);
  auto i2 = from.Load(2, 0);
  auto i3 = from.Load(3, 0);
  auto i4 = from.Load(4, 0);
  auto i5 = from.Load(5, 0);
  auto i6 = from.Load(6, 0);
  auto i7 = from.Load(7, 0);
  TransposeBlock8_V8(i0, i1, i2, i3, i4, i5, i6, i7);
  to.Store(i0, 0, 0);
  to.Store(i1, 1, 0);
  to.Store(i2, 2, 0);
  to.Store(i3, 3, 0);
  to.Store(i4, 4, 0);
  to.Store(i5, 5, 0);
  to.Store(i6, 6, 0);
  to.Store(i7, 7, 0);
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ComputeTransposedScaledDCT8_V8(const From& from,
                                                         const To& to) {
  const BlockDesc<8> d;

  const float c1234_lanes[4] = {
      0.707106781186548f,  // 1 / sqrt(2)
      0.382683432365090f,  // cos(3 * pi / 8)
      1.30656296487638f,   // 1 / (2 * cos(3 * pi / 8))
      0.541196100146197f   // sqrt(2) * cos(3 * pi / 8)
  };
  const auto c1234 = load_dup128(d, c1234_lanes);
  const auto k1 = set1(d, 1.0f);

  auto i0 = from.template LoadPart<8>(0, 0);
  auto i7 = from.template LoadPart<8>(7, 0);
  auto t00 = i0 + i7;           // 2 (faster than fadd)
  auto t01 = fsub(i0, k1, i7);  // 4
  SIMD_FENCE;

  auto i3 = from.template LoadPart<8>(3, 0);
  auto i4 = from.template LoadPart<8>(4, 0);
  auto t02 = i3 + i4;
  auto t03 = fsub(i3, k1, i4);  // 1
  SIMD_FENCE;

  auto i2 = from.template LoadPart<8>(2, 0);
  auto i5 = from.template LoadPart<8>(5, 0);
  auto t04 = i2 + i5;  // 1
  auto t05 = fsub(i2, k1, i5);
  SIMD_FENCE;

  auto i1 = from.template LoadPart<8>(1, 0);
  auto i6 = from.template LoadPart<8>(6, 0);
  auto t06 = i1 + i6;  // !
  SIMD_FENCE;

  auto t07 = i1 - i6;
  auto t09 = fsub(t00, k1, t02);
  const auto c4 = broadcast<3>(c1234);

  auto t11 = t06 - t04;           // !
  auto t08 = fadd(t00, k1, t02);  // 2
  const auto c3 = broadcast<2>(c1234);

  auto t14 = t05 + t03;
  auto t10 = fadd(t06, k1, t04);  // 1; dep-1

  auto t13 = t01 + t07;  // limits odd d
  const auto c1 = broadcast<0>(c1234);

  auto t15 = t11 + t09;  // !
  const auto c2 = broadcast<1>(c1234);

  auto t12 = t07 + t05;  // !
  auto ct14 = c4 * t14;

  auto t16 = t14 - t13;  // 1
  auto ct13 = c3 * t13;

  auto d0 = fadd(t08, k1, t10);
  auto d2 = mul_add(c1, t15, t09);

  auto t21 = nmul_add(c1, t12, t01);  // 2

  auto d6 = nmul_add(c1, t15, t09);
  auto t20 = mul_add(c1, t12, t01);  // 2

  auto t23 = mul_add(c2, t16, ct14);

  auto d4 = t08 - t10;
  auto t22 = mul_add(c2, t16, ct13);  // !

  const auto q0 = interleave_lo(d0, d2);

  const auto q2 = interleave_hi(d0, d2);

  const auto q4 = interleave_lo(d4, d6);

  auto d3 = t21 - t23;
  const auto q6 = interleave_hi(d4, d6);

  auto d1 = t20 + t22;
  const auto q1 = interleave_lo(d1, d3);

  const auto r0 = interleave_lo(q0, q1);
  const auto r1 = interleave_hi(q0, q1);

  auto d7 = t20 - t22;
  const auto q3 = interleave_hi(d1, d3);
  const auto r2 = interleave_lo(q2, q3);
  const auto r3 = interleave_hi(q2, q3);

  auto d5 = t21 + t23;
  const auto q5 = interleave_lo(d5, d7);
  const auto r4 = interleave_lo(q4, q5);
  const auto r5 = interleave_hi(q4, q5);

  const auto q7 = interleave_hi(d5, d7);
  const auto r6 = interleave_lo(q6, q7);
  const auto r7 = interleave_hi(q6, q7);

  // Second column-DCT after transpose
  i0 = concat_lo_lo(r4, r0);
  i7 = concat_hi_hi(r7, r3);
  t01 = i0 - i7;           // 1
  t00 = fadd(i0, k1, i7);  // 2

  i1 = concat_lo_lo(r5, r1);
  i6 = concat_hi_hi(r6, r2);
  t07 = i1 - i6;           // !
  t06 = fadd(i1, k1, i6);  // 2

  i3 = concat_lo_lo(r7, r3);
  i4 = concat_hi_hi(r4, r0);
  t03 = i3 - i4;           // 1
  t02 = fadd(i3, k1, i4);  // !

  i2 = concat_lo_lo(r6, r2);
  i5 = concat_hi_hi(r5, r1);
  t05 = i2 - i5;

  t13 = t01 + t07;  // 1

  t04 = i2 + i5;

  t14 = t05 + t03;
  t12 = fadd(t07, k1, t05);  // 2

  t09 = fsub(t00, k1, t02);
  ct13 = c3 * t13;  // 1

  t11 = t06 - t04;  // 1
  t10 = fadd(t06, k1, t04);

  t16 = t14 - t13;  // !
  ct14 = c4 * t14;

  t08 = t00 + t02;

  t20 = mul_add(c1, t12, t01);  // 1

  t15 = t11 + t09;
  t22 = mul_add(c2, t16, ct13);

  i0 = t08 + t10;

  t21 = nmul_add(c1, t12, t01);
  t23 = mul_add(c2, t16, ct14);

  i4 = t08 - t10;
  i2 = mul_add(c1, t15, t09);

  i6 = nmul_add(c1, t15, t09);
  to.template StorePart<8>(i0, 0, 0);
  SIMD_FENCE;

  i1 = t20 + t22;

  i7 = t20 - t22;
  to.template StorePart<8>(i2, 2, 0);
  to.template StorePart<8>(i4, 4, 0);
  SIMD_FENCE;

  i3 = t21 - t23;
  to.template StorePart<8>(i1, 1, 0);
  SIMD_FENCE;

  i5 = t21 + t23;
  to.template StorePart<8>(i6, 6, 0);
  to.template StorePart<8>(i7, 7, 0);
  to.template StorePart<8>(i3, 3, 0);
  to.template StorePart<8>(i5, 5, 0);
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ComputeTransposedScaledIDCT8_V8(const From& from,
                                                          const To& to) {
  const BlockDesc<8> d;

  const float k1_lanes[4] = {SIMD_REP4(1.0f)};
  const auto k1 = load_dup128(d, k1_lanes);
  const float c1234_lanes[4] = {
      1.41421356237310f,  // sqrt(2)
      2.61312592975275f,  // 1 / cos(3 * pi / 8)
      0.76536686473018f,  // 2 * cos(3 * pi / 8)
      1.08239220029239f   // 2 * sqrt(2) * cos(3 * pi / 8)
  };
  const auto c1234 = load_dup128(d, c1234_lanes);
  SIMD_FENCE;

  // Finish d5,d7 and d0,d2 first so we can overlap more port5 (shuffles) with
  // other computations; they have a shorter dependency chain than d13/46.

  auto i1 = from.Load(1, 0);
  auto i7 = from.Load(7, 0);
  auto t05 = i7 - i1;           // !
  auto t04 = fadd(i7, k1, i1);  // 1

  auto i3 = from.Load(3, 0);
  auto i5 = from.Load(5, 0);
  auto t07 = i5 - i3;           // +1
  auto t06 = fadd(i5, k1, i3);  // +1

  auto i2 = from.Load(2, 0);
  auto i6 = from.Load(6, 0);
  auto t02 = i6 + i2;  // 1
  const auto c2 = broadcast<1>(c1234);
  SIMD_FENCE;

  auto i0 = from.Load(0, 0);
  auto i4 = from.Load(4, 0);
  auto t03 = i6 - i2;    // !
  auto ct05 = c2 * t05;  // !
  SIMD_FENCE;

  auto t12 = t07 - t05;                 // 1
  const auto c1 = broadcast<0>(c1234);  // 1

  auto t00 = fadd(i0, k1, i4);          // +2
  const auto c3 = broadcast<2>(c1234);  // 2

  auto t09 = fsub(t04, k1, t06);
  auto t14 = mul_add(c1, t03, t02);  // +3

  auto t08 = fadd(t04, k1, t06);        // 1
  const auto c4 = broadcast<3>(c1234);  // 2

  auto t01 = i0 - i4;                 // +1
  auto t17 = mul_add(c3, t12, ct05);  // !
  SIMD_FENCE;

  //

  auto t10 = fadd(t00, k1, t02);
  auto ct07 = c4 * t07;  // !

  auto t15 = fsub(t01, k1, t14);  // 1
  auto ct09 = c1 * t09;

  auto t11 = fsub(t00, k1, t02);  // 6

  auto t19 = t08 + t17;  // !

  auto t16 = fadd(t01, k1, t14);

  auto d0 = fadd(t10, k1, t08);       // dep-3; 4
  auto t18 = mul_add(c3, t12, ct07);  // !

  auto t20 = ct09 + t19;         // !
  auto d7 = fsub(t10, k1, t08);  // 1

  auto d1 = fsub(t15, k1, t19);  // 5

  //

  auto d5 = t16 - t20;  // !
  auto d2 = fadd(t16, k1, t20);

  auto t21 = t18 - t20;  // !

  //

  // Begin transposing finished d#

  auto d6 = t15 + t19;  // 1
  const auto q5 = interleave_lo(d5, d7);

  auto d4 = t11 - t21;                    // !
  const auto q7 = interleave_hi(d5, d7);  // 8

  auto d3 = t11 + t21;  // !
  const auto q0 = interleave_lo(d0, d2);

  const auto q2 = interleave_hi(d0, d2);  // 8

  const auto q4 = interleave_lo(d4, d6);

  const auto q1 = interleave_lo(d1, d3);

  const auto r4 = interleave_lo(q4, q5);

  const auto r0 = interleave_lo(q0, q1);

  i0 = concat_lo_lo(r4, r0);

  i4 = concat_hi_hi(r4, r0);
  const auto _c1234 = load_dup128(d, c1234_lanes);

  const auto q3 = interleave_hi(d1, d3);

  // Begin second column-IDCT for transposed r#

  const auto q6 = interleave_hi(d4, d6);

  t00 = fadd(i0, k1, i4);
  const auto r2 = interleave_lo(q2, q3);

  t01 = fsub(i0, k1, i4);
  const auto r6 = interleave_lo(q6, q7);

  i2 = concat_lo_lo(r6, r2);

  i6 = concat_hi_hi(r6, r2);

  const auto r7 = interleave_hi(q6, q7);

  const auto r3 = interleave_hi(q2, q3);

  t03 = i6 - i2;
  i7 = concat_hi_hi(r7, r3);

  t02 = i6 + i2;
  const auto r5 = interleave_hi(q4, q5);

  const auto r1 = interleave_hi(q0, q1);
  const auto _c1 = broadcast<0>(_c1234);

  i1 = concat_lo_lo(r5, r1);
  auto ct03 = _c1 * t03;

  t10 = fadd(t00, k1, t02);  // 5
  i5 = concat_hi_hi(r5, r1);

  i3 = concat_lo_lo(r7, r3);

  t05 = i7 - i1;  // !
  const auto _c2 = broadcast<1>(_c1234);

  t04 = fadd(i7, k1, i1);  // 1

  t07 = i5 - i3;

  t06 = i5 + i3;
  ct05 = _c2 * t05;  // !

  t14 = ct03 + t02;  // 1

  t12 = t07 - t05;

  t08 = t04 + t06;

  t09 = t04 - t06;

  t15 = fsub(t01, k1, t14);      // 3
  t17 = mul_add(c3, t12, ct05);  // !

  d0 = t10 + t08;

  d7 = t10 - t08;

  ct09 = _c1 * t09;

  const auto _c4 = broadcast<3>(_c1234);
  to.Store(d0, 0, 0);
  SIMD_FENCE;

  t19 = t08 + t17;   // !
  ct07 = _c4 * t07;  // !
  to.Store(d7, 7, 0);
  SIMD_FENCE;

  t11 = t00 - t02;  // 8

  t16 = t01 + t14;  // 3

  d1 = t15 - t19;
  t20 = ct09 + t19;  // !

  d6 = t15 + t19;
  const auto _c3 = broadcast<2>(_c1234);

  t18 = mul_add(_c3, t12, ct07);  // !

  d2 = t16 + t20;
  to.Store(d1, 1, 0);
  SIMD_FENCE;

  d5 = t16 - t20;
  to.Store(d6, 6, 0);
  SIMD_FENCE;

  t21 = t18 - t20;  // !

  d4 = t11 - t21;
  to.Store(d2, 2, 0);

  d3 = t11 + t21;
  to.Store(d5, 5, 0);

  to.Store(d4, 4, 0);
  to.Store(d3, 3, 0);
}

}  // namespace pik

#endif  // SIMD_TARGET_VALUE

#endif  // THIRD_PARTY_DCT_SIMD_8_H_
