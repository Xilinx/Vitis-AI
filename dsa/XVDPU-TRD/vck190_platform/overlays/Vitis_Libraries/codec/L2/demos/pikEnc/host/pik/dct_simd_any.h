// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_DCT_SIMD_ANY_H_
#define PIK_DCT_SIMD_ANY_H_

#include "pik/block.h"
#include "pik/compiler_specific.h"
#include "pik/simd/simd.h"

#include <iostream>

namespace pik {

// DCT building blocks that does not require specific SIMD vector length.

template <class From, class To>
SIMD_ATTR PIK_INLINE void CopyBlock8(const From& from, const To& to) {
  const BlockDesc<8> d;
  for (size_t i = 0; i < 8; i += d.N) {
    const auto i0 = from.Load(0, i);
    const auto i1 = from.Load(1, i);
    const auto i2 = from.Load(2, i);
    const auto i3 = from.Load(3, i);
    const auto i4 = from.Load(4, i);
    const auto i5 = from.Load(5, i);
    const auto i6 = from.Load(6, i);
    const auto i7 = from.Load(7, i);
    to.Store(i0, 0, i);
    to.Store(i1, 1, i);
    to.Store(i2, 2, i);
    to.Store(i3, 3, i);
    to.Store(i4, 4, i);
    to.Store(i5, 5, i);
    to.Store(i6, 6, i);
    to.Store(i7, 7, i);
  }
}

template <class V>
SIMD_ATTR PIK_INLINE void ColumnDCT8(V& i0, V& i1, V& i2, V& i3, V& i4, V& i5,
                                     V& i6, V& i7) {
  const BlockDesc<8> d;

  const auto c1 = set1(d, 0.707106781186548f);  // 1 / sqrt(2)
  const auto c2 = set1(d, 0.382683432365090f);  // cos(3 * pi / 8)
  const auto c3 = set1(d, 1.30656296487638f);   // 1 / (2 * cos(3 * pi / 8))
  const auto c4 = set1(d, 0.541196100146197f);  // sqrt(2) * cos(3 * pi / 8)

  const auto t00 = i0 + i7;
  const auto t01 = i0 - i7;
  const auto t02 = i3 + i4;
  const auto t03 = i3 - i4;
  const auto t04 = i2 + i5;
  const auto t05 = i2 - i5;
  const auto t06 = i1 + i6;
  const auto t07 = i1 - i6;
  const auto t08 = t00 + t02;
  const auto t09 = t00 - t02;
  const auto t10 = t06 + t04;
  const auto t11 = t06 - t04;
  const auto t12 = t07 + t05;
  const auto t13 = t01 + t07;
  const auto t14 = t05 + t03;
  const auto t15 = t11 + t09;
  const auto t16 = t14 - t13;
  const auto t17 = c1 * t15;
  const auto t18 = c1 * t12;
  const auto t19 = c2 * t16;
  const auto t20 = t01 + t18;
  const auto t21 = t01 - t18;
  const auto t22 = mul_add(c3, t13, t19);
  const auto t23 = mul_add(c4, t14, t19);
  i0 = t08 + t10;
  i1 = t20 + t22;
  i2 = t09 + t17;
  i3 = t21 - t23;
  i4 = t08 - t10;
  i5 = t21 + t23;
  i6 = t09 - t17;
  i7 = t20 - t22;
}

// "A low multiplicative complexity fast recursive DCT-2 algorithm"
// Maxim Vashkevich, Alexander Pertrovsky, 27 Jul 2012
template <class V>
SIMD_ATTR PIK_INLINE void ColumnDCT16(V& i00, V& i01, V& i02, V& i03, V& i04,
                                      V& i05, V& i06, V& i07, V& i08, V& i09,
                                      V& i10, V& i11, V& i12, V& i13, V& i14,
                                      V& i15) {
  const BlockDesc<16> d;

  const auto c1_16 = set1(d, 1.9615705608064609f);   // 2 * cos(1 * pi / 16)
  const auto c2_16 = set1(d, 1.8477590650225735f);   // 2 * cos(2 * pi / 16)
  const auto c3_16 = set1(d, 1.6629392246050905f);   // 2 * cos(3 * pi / 16)
  const auto c4_16 = set1(d, 1.4142135623730951f);   // 2 * cos(4 * pi / 16)
  const auto c5_16 = set1(d, 1.1111404660392046f);   // 2 * cos(5 * pi / 16)
  const auto c6_16 = set1(d, 0.7653668647301797f);   // 2 * cos(6 * pi / 16)
  const auto c7_16 = set1(d, 0.39018064403225666f);  // 2 * cos(7 * pi / 16)

  const auto t00 = i00 + i15;
  const auto t01 = i01 + i14;
  const auto t02 = i02 + i13;
  const auto t03 = i03 + i12;
  const auto t04 = i04 + i11;
  const auto t05 = i05 + i10;
  const auto t06 = i06 + i09;
  const auto t07 = i07 + i08;
  const auto t08 = i00 - i15;
  const auto t09 = i01 - i14;
  const auto t10 = i02 - i13;
  const auto t11 = i03 - i12;
  const auto t12 = i04 - i11;
  const auto t13 = i05 - i10;
  const auto t14 = i06 - i09;
  const auto t15 = i07 - i08;

  const auto t16 = t00 + t07;
  const auto t17 = t01 + t06;
  const auto t18 = t02 + t05;
  const auto t19 = t03 + t04;
  const auto t20 = t00 - t07;
  const auto t21 = t01 - t06;
  const auto t22 = t02 - t05;
  const auto t23 = t03 - t04;
  const auto t24 = t16 + t19;
  const auto t25 = t17 + t18;
  const auto t26 = t16 - t19;
  const auto t27 = t17 - t18;
  i00 = t24 + t25;
  i08 = t24 - t25;
  const auto t30 = t26 - t27;
  const auto t31 = t27 * c4_16;
  i04 = t30 + t31;
  i12 = t30 - t31;
  const auto t34 = t20 - t23;
  const auto t35 = t21 - t22;
  const auto t36 = t22 * c4_16;
  const auto t37 = t23 * c4_16;
  const auto t38 = t34 + t36;
  const auto t39 = t35 + t37;
  const auto t40 = t34 - t36;
  const auto t41 = t35 - t37;
  const auto t42 = t38 - t39;
  const auto t43 = t39 * c2_16;
  i02 = t42 + t43;
  i14 = t42 - t43;
  const auto t46 = t40 - t41;
  const auto t47 = t41 * c6_16;
  i06 = t46 + t47;
  i10 = t46 - t47;
  const auto t50 = t08 - t15;
  const auto t51 = t09 - t14;
  const auto t52 = t10 - t13;
  const auto t53 = t11 - t12;
  const auto t54 = t12 * c4_16;
  const auto t55 = t13 * c4_16;
  const auto t56 = t14 * c4_16;
  const auto t57 = t15 * c4_16;
  const auto t58 = t50 + t54;
  const auto t59 = t51 + t55;
  const auto t60 = t52 + t56;
  const auto t61 = t53 + t57;
  const auto t62 = t50 - t54;
  const auto t63 = t51 - t55;
  const auto t64 = t52 - t56;
  const auto t65 = t53 - t57;
  const auto t66 = t58 - t61;
  const auto t67 = t59 - t60;
  const auto t68 = t60 * c2_16;
  const auto t69 = t61 * c2_16;
  const auto t70 = t66 + t68;
  const auto t71 = t67 + t69;
  const auto t72 = t66 - t68;
  const auto t73 = t67 - t69;
  const auto t74 = t70 - t71;
  const auto t75 = t71 * c1_16;
  i01 = t74 + t75;
  i15 = t74 - t75;
  const auto t78 = t72 - t73;
  const auto t79 = t73 * c7_16;
  i07 = t78 + t79;
  i09 = t78 - t79;
  const auto t82 = t62 - t65;
  const auto t83 = t63 - t64;
  const auto t84 = t64 * c6_16;
  const auto t85 = t65 * c6_16;
  const auto t86 = t82 + t84;
  const auto t87 = t83 + t85;
  const auto t88 = t82 - t84;
  const auto t89 = t83 - t85;
  const auto t90 = t86 - t87;
  const auto t91 = t87 * c3_16;
  i03 = t90 + t91;
  i13 = t90 - t91;
  const auto t94 = t88 - t89;
  const auto t95 = t89 * c5_16;
  i05 = t94 + t95;
  i11 = t94 - t95;
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ColumnDCT8(const From& from, const To& to) {
  const BlockDesc<8> d;

  for (size_t i = 0; i < 8; i += d.N) {
    auto i0 = from.template LoadPart<8>(0, i);
    auto i1 = from.template LoadPart<8>(1, i);
    auto i2 = from.template LoadPart<8>(2, i);
    auto i3 = from.template LoadPart<8>(3, i);
    auto i4 = from.template LoadPart<8>(4, i);
    auto i5 = from.template LoadPart<8>(5, i);
    auto i6 = from.template LoadPart<8>(6, i);
    auto i7 = from.template LoadPart<8>(7, i);
    ColumnDCT8(i0, i1, i2, i3, i4, i5, i6, i7);
    to.template StorePart<8>(i0, 0, i);
    to.template StorePart<8>(i1, 1, i);
    to.template StorePart<8>(i2, 2, i);
    to.template StorePart<8>(i3, 3, i);
    to.template StorePart<8>(i4, 4, i);
    to.template StorePart<8>(i5, 5, i);
    to.template StorePart<8>(i6, 6, i);
    to.template StorePart<8>(i7, 7, i);
  }
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ColumnDCT16(const From& from, const To& to) {
  const BlockDesc<16> d;

  for (size_t i = 0; i < 16; i += d.N) {
    auto i00 = from.Load(0, i);
    auto i01 = from.Load(1, i);
    auto i02 = from.Load(2, i);
    auto i03 = from.Load(3, i);
    auto i04 = from.Load(4, i);
    auto i05 = from.Load(5, i);
    auto i06 = from.Load(6, i);
    auto i07 = from.Load(7, i);
    auto i08 = from.Load(8, i);
    auto i09 = from.Load(9, i);
    auto i10 = from.Load(10, i);
    auto i11 = from.Load(11, i);
    auto i12 = from.Load(12, i);
    auto i13 = from.Load(13, i);
    auto i14 = from.Load(14, i);
    auto i15 = from.Load(15, i);
    ColumnDCT16(i00, i01, i02, i03, i04, i05, i06, i07, i08, i09, i10, i11, i12,
                i13, i14, i15);
    to.Store(i00, 0, i);
    to.Store(i01, 1, i);
    to.Store(i02, 2, i);
    to.Store(i03, 3, i);
    to.Store(i04, 4, i);
    to.Store(i05, 5, i);
    to.Store(i06, 6, i);
    to.Store(i07, 7, i);
    to.Store(i08, 8, i);
    to.Store(i09, 9, i);
    to.Store(i10, 10, i);
    to.Store(i11, 11, i);
    to.Store(i12, 12, i);
    to.Store(i13, 13, i);
    to.Store(i14, 14, i);
    to.Store(i15, 15, i);
  }
}

// NB: ColumnIDCT8(ColumnDCT8(I)) = 8.0 * I
template <class V>
SIMD_ATTR PIK_INLINE void ColumnIDCT8(V& i0, V& i1, V& i2, V& i3, V& i4, V& i5,
                                      V& i6, V& i7) {
  const BlockDesc<8> d;

  const auto c1 = set1(d, 1.41421356237310f);  // sqrt(2)
  const auto c2 = set1(d, 2.61312592975275f);  // 1 / cos(3 * pi / 8)
  const auto c3 = set1(d, 0.76536686473018f);  // 2 * cos(3 * pi / 8)
  const auto c4 = set1(d, 1.08239220029239f);  // 2 * sqrt(2) * cos(3 * pi / 8)

  const auto t00 = i0 + i4;
  const auto t01 = i0 - i4;
  const auto t02 = i6 + i2;
  const auto t03 = i6 - i2;
  const auto t04 = i7 + i1;
  const auto t05 = i7 - i1;
  const auto t06 = i5 + i3;
  const auto t07 = i5 - i3;
  const auto t08 = t04 + t06;
  const auto t09 = t04 - t06;
  const auto t10 = t00 + t02;
  const auto t11 = t00 - t02;
  const auto t12 = t07 - t05;
  const auto t13 = c3 * t12;
  const auto t14 = mul_add(c1, t03, t02);
  const auto t15 = t01 - t14;
  const auto t16 = t01 + t14;
  const auto t17 = mul_add(c2, t05, t13);
  const auto t18 = mul_add(c4, t07, t13);
  const auto t19 = t08 + t17;
  const auto t20 = mul_add(c1, t09, t19);
  const auto t21 = t18 - t20;
  i0 = t10 + t08;
  i1 = t15 - t19;
  i2 = t16 + t20;
  i3 = t11 + t21;
  i4 = t11 - t21;
  i5 = t16 - t20;
  i6 = t15 + t19;
  i7 = t10 - t08;
}

// "A low multiplicative complexity fast recursive DCT-2 algorithm"
// Maxim Vashkevich, Alexander Pertrovsky, 27 Jul 2012
template <class V>
SIMD_ATTR PIK_INLINE void ColumnIDCT16(V& i00, V& i01, V& i02, V& i03, V& i04,
                                       V& i05, V& i06, V& i07, V& i08, V& i09,
                                       V& i10, V& i11, V& i12, V& i13, V& i14,
                                       V& i15) {
  const BlockDesc<16> d;

  const auto c1_16 = set1(d, 0.5097955791041592f);  // 0.5 / cos(1 * pi / 16)
  const auto c2_16 = set1(d, 0.541196100146197f);   // 0.5 / cos(2 * pi / 16)
  const auto c3_16 = set1(d, 0.6013448869350453f);  // 0.5 / cos(3 * pi / 16)
  const auto c4_16 = set1(d, 0.7071067811865475f);  // 0.5 / cos(4 * pi / 16)
  const auto c5_16 = set1(d, 0.8999762231364156f);  // 0.5 / cos(5 * pi / 16)
  const auto c6_16 = set1(d, 1.3065629648763764f);  // 0.5 / cos(6 * pi / 16)
  const auto c7_16 = set1(d, 2.5629154477415055f);  // 0.5 / cos(7 * pi / 16)

  const auto t00 = i00 + i08;
  const auto t01 = i00 - i08;
  const auto t02 = i04 + i12;
  const auto t03 = i04 - i12;
  const auto t04 = t03 * c4_16;
  const auto t05 = t02 + t04;
  const auto t06 = t00 + t05;
  const auto t07 = t01 + t04;
  const auto t08 = t00 - t05;
  const auto t09 = t01 - t04;
  const auto t10 = i02 + i14;
  const auto t11 = i02 - i14;
  const auto t12 = t11 * c2_16;
  const auto t13 = t10 + t12;
  const auto t14 = i06 + i10;
  const auto t15 = i06 - i10;
  const auto t16 = t15 * c6_16;
  const auto t17 = t14 + t16;
  const auto t18 = t13 + t17;
  const auto t19 = t12 + t16;
  const auto t20 = t13 - t17;
  const auto t21 = t12 - t16;
  const auto t22 = t20 * c4_16;
  const auto t23 = t21 * c4_16;
  const auto t24 = t18 + t23;
  const auto t25 = t19 + t22;
  const auto t26 = t06 + t24;
  const auto t27 = t07 + t25;
  const auto t28 = t09 + t22;
  const auto t29 = t08 + t23;
  const auto t30 = t06 - t24;
  const auto t31 = t07 - t25;
  const auto t32 = t09 - t22;
  const auto t33 = t08 - t23;
  const auto t34 = i01 + i15;
  const auto t35 = i01 - i15;
  const auto t36 = t35 * c1_16;
  const auto t37 = t34 + t36;
  const auto t38 = i07 + i09;
  const auto t39 = i07 - i09;
  const auto t40 = t39 * c7_16;
  const auto t41 = t38 + t40;
  const auto t42 = t37 + t41;
  const auto t43 = t36 + t40;
  const auto t44 = t37 - t41;
  const auto t45 = t36 - t40;
  const auto t46 = t44 * c2_16;
  const auto t47 = t45 * c2_16;
  const auto t48 = t42 + t47;
  const auto t49 = t43 + t46;
  const auto t50 = i03 + i13;
  const auto t51 = i03 - i13;
  const auto t52 = t51 * c3_16;
  const auto t53 = t50 + t52;
  const auto t54 = i05 + i11;
  const auto t55 = i05 - i11;
  const auto t56 = t55 * c5_16;
  const auto t57 = t54 + t56;
  const auto t58 = t53 + t57;
  const auto t59 = t52 + t56;
  const auto t60 = t53 - t57;
  const auto t61 = t52 - t56;
  const auto t62 = t60 * c6_16;
  const auto t63 = t61 * c6_16;
  const auto t64 = t58 + t63;
  const auto t65 = t59 + t62;
  const auto t66 = t48 + t64;
  const auto t67 = t49 + t65;
  const auto t68 = t46 + t62;
  const auto t69 = t47 + t63;
  const auto t70 = t48 - t64;
  const auto t71 = t49 - t65;
  const auto t72 = t46 - t62;
  const auto t73 = t47 - t63;
  const auto t74 = t70 * c4_16;
  const auto t75 = t71 * c4_16;
  const auto t76 = t72 * c4_16;
  const auto t77 = t73 * c4_16;
  const auto t78 = t66 + t77;
  const auto t79 = t67 + t76;
  const auto t80 = t68 + t75;
  const auto t81 = t69 + t74;
  i00 = t26 + t78;
  i01 = t27 + t79;
  i02 = t28 + t80;
  i03 = t29 + t81;
  i04 = t33 + t74;
  i05 = t32 + t75;
  i06 = t31 + t76;
  i07 = t30 + t77;
  i15 = t26 - t78;
  i14 = t27 - t79;
  i13 = t28 - t80;
  i12 = t29 - t81;
  i11 = t33 - t74;
  i10 = t32 - t75;
  i09 = t31 - t76;
  i08 = t30 - t77;
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ColumnIDCT8(const From& from, const To& to) {
  const BlockDesc<8> d;

  for (size_t i = 0; i < 8; i += d.N) {
    auto i0 = from.Load(0, i);
    auto i1 = from.Load(1, i);
    auto i2 = from.Load(2, i);
    auto i3 = from.Load(3, i);
    auto i4 = from.Load(4, i);
    auto i5 = from.Load(5, i);
    auto i6 = from.Load(6, i);
    auto i7 = from.Load(7, i);
    ColumnIDCT8(i0, i1, i2, i3, i4, i5, i6, i7);
    to.Store(i0, 0, i);
    to.Store(i1, 1, i);
    to.Store(i2, 2, i);
    to.Store(i3, 3, i);
    to.Store(i4, 4, i);
    to.Store(i5, 5, i);
    to.Store(i6, 6, i);
    to.Store(i7, 7, i);
  }
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ColumnIDCT16(const From& from, const To& to) {
  const BlockDesc<16> d;

  for (size_t i = 0; i < 16; i += d.N) {
    auto i00 = from.Load(0, i);
    auto i01 = from.Load(1, i);
    auto i02 = from.Load(2, i);
    auto i03 = from.Load(3, i);
    auto i04 = from.Load(4, i);
    auto i05 = from.Load(5, i);
    auto i06 = from.Load(6, i);
    auto i07 = from.Load(7, i);
    auto i08 = from.Load(8, i);
    auto i09 = from.Load(9, i);
    auto i10 = from.Load(10, i);
    auto i11 = from.Load(11, i);
    auto i12 = from.Load(12, i);
    auto i13 = from.Load(13, i);
    auto i14 = from.Load(14, i);
    auto i15 = from.Load(15, i);
    ColumnIDCT16(i00, i01, i02, i03, i04, i05, i06, i07, i08, i09, i10, i11,
                 i12, i13, i14, i15);
    to.Store(i00, 0, i);
    to.Store(i01, 1, i);
    to.Store(i02, 2, i);
    to.Store(i03, 3, i);
    to.Store(i04, 4, i);
    to.Store(i05, 5, i);
    to.Store(i06, 6, i);
    to.Store(i07, 7, i);
    to.Store(i08, 8, i);
    to.Store(i09, 9, i);
    to.Store(i10, 10, i);
    to.Store(i11, 11, i);
    to.Store(i12, 12, i);
    to.Store(i13, 13, i);
    to.Store(i14, 14, i);
    to.Store(i15, 15, i);
  }
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ColumnDCT4(const From& from, const To& to) {
  const BlockDesc<4> d;
  const auto c2_8 = set1(d, 1.414213562373095048f);  // 2 * cos(2 * pi / 8)
  for (size_t i = 0; i < 4; i += d.N) {
    auto i0 = from.template LoadPart<4>(0, i);
    auto i1 = from.template LoadPart<4>(1, i);
    auto i2 = from.template LoadPart<4>(2, i);
    auto i3 = from.template LoadPart<4>(3, i);
    auto t0 = i0 + i3;
    auto t1 = i1 + i2;
    auto t2 = i0 - i3;
    auto t3 = i1 - i2;
    auto t4 = t0 + t1;
    auto t5 = t0 - t1;
    auto t6 = t2 - t3;
    auto t7 = t3 * c2_8;
    auto t8 = t6 + t7;
    auto t9 = t6 - t7;
    to.template StorePart<4>(t4, 0, i);
    to.template StorePart<4>(t8, 1, i);
    to.template StorePart<4>(t5, 2, i);
    to.template StorePart<4>(t9, 3, i);
  }
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ColumnIDCT4(const From& from, const To& to) {
  const BlockDesc<4> d;
  const auto c2_8 = set1(d, 0.7071067811865475244f);  // 0.5 / cos(2 * pi / 8)
  for (size_t i = 0; i < 4; i += d.N) {
    auto i0 = from.template LoadPart<4>(0, i);
    auto i1 = from.template LoadPart<4>(1, i);
    auto i2 = from.template LoadPart<4>(2, i);
    auto i3 = from.template LoadPart<4>(3, i);
    auto t0 = i0 + i2;
    auto t1 = i0 - i2;
    auto t2 = i1 + i3;
    auto t3 = i1 - i3;
    auto t4 = t3 * c2_8;
    auto t5 = t2 + t4;
    auto t6 = t0 + t5;
    auto t7 = t1 + t4;
    auto t8 = t0 - t5;
    auto t9 = t1 - t4;
    to.template StorePart<4>(t6, 0, i);
    to.template StorePart<4>(t7, 1, i);
    to.template StorePart<4>(t9, 2, i);
    to.template StorePart<4>(t8, 3, i);
  }
}

template <class V>
SIMD_ATTR PIK_INLINE void ColumnDCT32(V& i00, V& i01, V& i02, V& i03, V& i04,
                                      V& i05, V& i06, V& i07, V& i08, V& i09,
                                      V& i10, V& i11, V& i12, V& i13, V& i14,
                                      V& i15, V& i16, V& i17, V& i18, V& i19,
                                      V& i20, V& i21, V& i22, V& i23, V& i24,
                                      V& i25, V& i26, V& i27, V& i28, V& i29,
                                      V& i30, V& i31) {
  const BlockDesc<32> d;
  const auto c2_64 = set1(d, 1.990369453344393857f);   // 2 * cos(2 * pi / 64)
  const auto c4_64 = set1(d, 1.961570560806460861f);   // 2 * cos(4 * pi / 64)
  const auto c6_64 = set1(d, 1.913880671464417649f);   // 2 * cos(6 * pi / 64)
  const auto c8_64 = set1(d, 1.847759065022573477f);   // 2 * cos(8 * pi / 64)
  const auto c10_64 = set1(d, 1.763842528696710099f);  // 2 * cos(10 * pi / 64)
  const auto c12_64 = set1(d, 1.662939224605090471f);  // 2 * cos(12 * pi / 64)
  const auto c14_64 = set1(d, 1.546020906725473987f);  // 2 * cos(14 * pi / 64)
  const auto c16_64 = set1(d, 1.414213562373095145f);  // 2 * cos(16 * pi / 64)
  const auto c18_64 = set1(d, 1.268786568327290976f);  // 2 * cos(18 * pi / 64)
  const auto c20_64 = set1(d, 1.111140466039204577f);  // 2 * cos(20 * pi / 64)
  const auto c22_64 = set1(d, 0.942793473651995617f);  // 2 * cos(22 * pi / 64)
  const auto c24_64 = set1(d, 0.765366864730179675f);  // 2 * cos(24 * pi / 64)
  const auto c26_64 = set1(d, 0.580569354508924662f);  // 2 * cos(26 * pi / 64)
  const auto c28_64 = set1(d, 0.390180644032256663f);  // 2 * cos(28 * pi / 64)
  const auto c30_64 = set1(d, 0.196034280659121540f);  // 2 * cos(30 * pi / 64)

  const auto t00 = i00 + i31;
  const auto t01 = i01 + i30;
  const auto t02 = i02 + i29;
  const auto t03 = i03 + i28;
  const auto t04 = i04 + i27;
  const auto t05 = i05 + i26;
  const auto t06 = i06 + i25;
  const auto t07 = i07 + i24;
  const auto t08 = i08 + i23;
  const auto t09 = i09 + i22;
  const auto t10 = i10 + i21;
  const auto t11 = i11 + i20;
  const auto t12 = i12 + i19;
  const auto t13 = i13 + i18;
  const auto t14 = i14 + i17;
  const auto t15 = i15 + i16;
  const auto t16 = i00 - i31;
  const auto t17 = i01 - i30;
  const auto t18 = i02 - i29;
  const auto t19 = i03 - i28;
  const auto t20 = i04 - i27;
  const auto t21 = i05 - i26;
  const auto t22 = i06 - i25;
  const auto t23 = i07 - i24;
  const auto t24 = i08 - i23;
  const auto t25 = i09 - i22;
  const auto t26 = i10 - i21;
  const auto t27 = i11 - i20;
  const auto t28 = i12 - i19;
  const auto t29 = i13 - i18;
  const auto t30 = i14 - i17;
  const auto t31 = i15 - i16;
  const auto t32 = t00 + t15;
  const auto t33 = t01 + t14;
  const auto t34 = t02 + t13;
  const auto t35 = t03 + t12;
  const auto t36 = t04 + t11;
  const auto t37 = t05 + t10;
  const auto t38 = t06 + t09;
  const auto t39 = t07 + t08;
  const auto t40 = t00 - t15;
  const auto t41 = t01 - t14;
  const auto t42 = t02 - t13;
  const auto t43 = t03 - t12;
  const auto t44 = t04 - t11;
  const auto t45 = t05 - t10;
  const auto t46 = t06 - t09;
  const auto t47 = t07 - t08;
  const auto t48 = t32 + t39;
  const auto t49 = t33 + t38;
  const auto t50 = t34 + t37;
  const auto t51 = t35 + t36;
  const auto t52 = t32 - t39;
  const auto t53 = t33 - t38;
  const auto t54 = t34 - t37;
  const auto t55 = t35 - t36;
  const auto t56 = t48 + t51;
  const auto t57 = t49 + t50;
  const auto t58 = t48 - t51;
  const auto t59 = t49 - t50;
  const auto t60 = t56 + t57;
  const auto t61 = t56 - t57;
  const auto t62 = t58 - t59;
  const auto t63 = t59 * c16_64;
  const auto t64 = t62 + t63;
  const auto t65 = t62 - t63;
  const auto t66 = t52 - t55;
  const auto t67 = t53 - t54;
  const auto t68 = t54 * c16_64;
  const auto t69 = t55 * c16_64;
  const auto t70 = t66 + t68;
  const auto t71 = t67 + t69;
  const auto t72 = t66 - t68;
  const auto t73 = t67 - t69;
  const auto t74 = t70 - t71;
  const auto t75 = t71 * c8_64;
  const auto t76 = t74 + t75;
  const auto t77 = t74 - t75;
  const auto t78 = t72 - t73;
  const auto t79 = t73 * c24_64;
  const auto t80 = t78 + t79;
  const auto t81 = t78 - t79;
  const auto t82 = t40 - t47;
  const auto t83 = t41 - t46;
  const auto t84 = t42 - t45;
  const auto t85 = t43 - t44;
  const auto t86 = t44 * c16_64;
  const auto t87 = t45 * c16_64;
  const auto t88 = t46 * c16_64;
  const auto t89 = t47 * c16_64;
  const auto t90 = t82 + t86;
  const auto t91 = t83 + t87;
  const auto t92 = t84 + t88;
  const auto t93 = t85 + t89;
  const auto t94 = t82 - t86;
  const auto t95 = t83 - t87;
  const auto t96 = t84 - t88;
  const auto t97 = t85 - t89;
  const auto t98 = t90 - t93;
  const auto t99 = t91 - t92;
  const auto t100 = t92 * c8_64;
  const auto t101 = t93 * c8_64;
  const auto t102 = t98 + t100;
  const auto t103 = t99 + t101;
  const auto t104 = t98 - t100;
  const auto t105 = t99 - t101;
  const auto t106 = t102 - t103;
  const auto t107 = t103 * c4_64;
  const auto t108 = t106 + t107;
  const auto t109 = t106 - t107;
  const auto t110 = t104 - t105;
  const auto t111 = t105 * c28_64;
  const auto t112 = t110 + t111;
  const auto t113 = t110 - t111;
  const auto t114 = t94 - t97;
  const auto t115 = t95 - t96;
  const auto t116 = t96 * c24_64;
  const auto t117 = t97 * c24_64;
  const auto t118 = t114 + t116;
  const auto t119 = t115 + t117;
  const auto t120 = t114 - t116;
  const auto t121 = t115 - t117;
  const auto t122 = t118 - t119;
  const auto t123 = t119 * c12_64;
  const auto t124 = t122 + t123;
  const auto t125 = t122 - t123;
  const auto t126 = t120 - t121;
  const auto t127 = t121 * c20_64;
  const auto t128 = t126 + t127;
  const auto t129 = t126 - t127;
  const auto t130 = t16 - t31;
  const auto t131 = t17 - t30;
  const auto t132 = t18 - t29;
  const auto t133 = t19 - t28;
  const auto t134 = t20 - t27;
  const auto t135 = t21 - t26;
  const auto t136 = t22 - t25;
  const auto t137 = t23 - t24;
  const auto t138 = t24 * c16_64;
  const auto t139 = t25 * c16_64;
  const auto t140 = t26 * c16_64;
  const auto t141 = t27 * c16_64;
  const auto t142 = t28 * c16_64;
  const auto t143 = t29 * c16_64;
  const auto t144 = t30 * c16_64;
  const auto t145 = t31 * c16_64;
  const auto t146 = t130 + t138;
  const auto t147 = t131 + t139;
  const auto t148 = t132 + t140;
  const auto t149 = t133 + t141;
  const auto t150 = t134 + t142;
  const auto t151 = t135 + t143;
  const auto t152 = t136 + t144;
  const auto t153 = t137 + t145;
  const auto t154 = t130 - t138;
  const auto t155 = t131 - t139;
  const auto t156 = t132 - t140;
  const auto t157 = t133 - t141;
  const auto t158 = t134 - t142;
  const auto t159 = t135 - t143;
  const auto t160 = t136 - t144;
  const auto t161 = t137 - t145;
  const auto t162 = t146 - t153;
  const auto t163 = t147 - t152;
  const auto t164 = t148 - t151;
  const auto t165 = t149 - t150;
  const auto t166 = t150 * c8_64;
  const auto t167 = t151 * c8_64;
  const auto t168 = t152 * c8_64;
  const auto t169 = t153 * c8_64;
  const auto t170 = t162 + t166;
  const auto t171 = t163 + t167;
  const auto t172 = t164 + t168;
  const auto t173 = t165 + t169;
  const auto t174 = t162 - t166;
  const auto t175 = t163 - t167;
  const auto t176 = t164 - t168;
  const auto t177 = t165 - t169;
  const auto t178 = t170 - t173;
  const auto t179 = t171 - t172;
  const auto t180 = t172 * c4_64;
  const auto t181 = t173 * c4_64;
  const auto t182 = t178 + t180;
  const auto t183 = t179 + t181;
  const auto t184 = t178 - t180;
  const auto t185 = t179 - t181;
  const auto t186 = t182 - t183;
  const auto t187 = t183 * c2_64;
  const auto t188 = t186 + t187;
  const auto t189 = t186 - t187;
  const auto t190 = t184 - t185;
  const auto t191 = t185 * c30_64;
  const auto t192 = t190 + t191;
  const auto t193 = t190 - t191;
  const auto t194 = t174 - t177;
  const auto t195 = t175 - t176;
  const auto t196 = t176 * c28_64;
  const auto t197 = t177 * c28_64;
  const auto t198 = t194 + t196;
  const auto t199 = t195 + t197;
  const auto t200 = t194 - t196;
  const auto t201 = t195 - t197;
  const auto t202 = t198 - t199;
  const auto t203 = t199 * c14_64;
  const auto t204 = t202 + t203;
  const auto t205 = t202 - t203;
  const auto t206 = t200 - t201;
  const auto t207 = t201 * c18_64;
  const auto t208 = t206 + t207;
  const auto t209 = t206 - t207;
  const auto t210 = t154 - t161;
  const auto t211 = t155 - t160;
  const auto t212 = t156 - t159;
  const auto t213 = t157 - t158;
  const auto t214 = t158 * c24_64;
  const auto t215 = t159 * c24_64;
  const auto t216 = t160 * c24_64;
  const auto t217 = t161 * c24_64;
  const auto t218 = t210 + t214;
  const auto t219 = t211 + t215;
  const auto t220 = t212 + t216;
  const auto t221 = t213 + t217;
  const auto t222 = t210 - t214;
  const auto t223 = t211 - t215;
  const auto t224 = t212 - t216;
  const auto t225 = t213 - t217;
  const auto t226 = t218 - t221;
  const auto t227 = t219 - t220;
  const auto t228 = t220 * c12_64;
  const auto t229 = t221 * c12_64;
  const auto t230 = t226 + t228;
  const auto t231 = t227 + t229;
  const auto t232 = t226 - t228;
  const auto t233 = t227 - t229;
  const auto t234 = t230 - t231;
  const auto t235 = t231 * c6_64;
  const auto t236 = t234 + t235;
  const auto t237 = t234 - t235;
  const auto t238 = t232 - t233;
  const auto t239 = t233 * c26_64;
  const auto t240 = t238 + t239;
  const auto t241 = t238 - t239;
  const auto t242 = t222 - t225;
  const auto t243 = t223 - t224;
  const auto t244 = t224 * c20_64;
  const auto t245 = t225 * c20_64;
  const auto t246 = t242 + t244;
  const auto t247 = t243 + t245;
  const auto t248 = t242 - t244;
  const auto t249 = t243 - t245;
  const auto t250 = t246 - t247;
  const auto t251 = t247 * c10_64;
  const auto t252 = t250 + t251;
  const auto t253 = t250 - t251;
  const auto t254 = t248 - t249;
  const auto t255 = t249 * c22_64;
  const auto t256 = t254 + t255;
  const auto t257 = t254 - t255;

  i00 = t60;
  i01 = t188;
  i02 = t108;
  i03 = t236;
  i04 = t76;
  i05 = t252;
  i06 = t124;
  i07 = t204;
  i08 = t64;
  i09 = t208;
  i10 = t128;
  i11 = t256;
  i12 = t80;
  i13 = t240;
  i14 = t112;
  i15 = t192;
  i16 = t61;
  i17 = t193;
  i18 = t113;
  i19 = t241;
  i20 = t81;
  i21 = t257;
  i22 = t129;
  i23 = t209;
  i24 = t65;
  i25 = t205;
  i26 = t125;
  i27 = t253;
  i28 = t77;
  i29 = t237;
  i30 = t109;
  i31 = t189;
}

template <class V>
SIMD_ATTR PIK_INLINE void ColumnIDCT32(V& i00, V& i01, V& i02, V& i03, V& i04,
                                       V& i05, V& i06, V& i07, V& i08, V& i09,
                                       V& i10, V& i11, V& i12, V& i13, V& i14,
                                       V& i15, V& i16, V& i17, V& i18, V& i19,
                                       V& i20, V& i21, V& i22, V& i23, V& i24,
                                       V& i25, V& i26, V& i27, V& i28, V& i29,
                                       V& i30, V& i31) {
  const BlockDesc<32> d;
  const auto c2_64 = set1(d, 0.502419286188155678f);  // 0.5 / cos(2 * pi / 64)
  const auto c4_64 = set1(d, 0.509795579104159180f);  // 0.5 / cos(4 * pi / 64)
  const auto c6_64 = set1(d, 0.522498614939688855f);  // 0.5 / cos(6 * pi / 64)
  const auto c8_64 = set1(d, 0.541196100146197012f);  // 0.5 / cos(8 * pi / 64)
  const auto c10_64 =
      set1(d, 0.566944034816357689f);  // 0.5 / cos(10 * pi / 64)
  const auto c12_64 =
      set1(d, 0.601344886935045286f);  // 0.5 / cos(12 * pi / 64)
  const auto c14_64 =
      set1(d, 0.646821783359990077f);  // 0.5 / cos(14 * pi / 64)
  const auto c16_64 =
      set1(d, 0.707106781186547462f);  // 0.5 / cos(16 * pi / 64)
  const auto c18_64 =
      set1(d, 0.788154623451250202f);  // 0.5 / cos(18 * pi / 64)
  const auto c20_64 =
      set1(d, 0.899976223136415565f);  // 0.5 / cos(20 * pi / 64)
  const auto c22_64 =
      set1(d, 1.060677685990347063f);  // 0.5 / cos(22 * pi / 64)
  const auto c24_64 =
      set1(d, 1.306562964876376354f);  // 0.5 / cos(24 * pi / 64)
  const auto c26_64 =
      set1(d, 1.722447098238334195f);  // 0.5 / cos(26 * pi / 64)
  const auto c28_64 =
      set1(d, 2.562915447741505481f);  // 0.5 / cos(28 * pi / 64)
  const auto c30_64 =
      set1(d, 5.101148618689155256f);  // 0.5 / cos(30 * pi / 64)

  const auto t00 = i00 + i16;
  const auto t01 = i00 - i16;
  const auto t02 = i08 + i24;
  const auto t03 = i08 - i24;
  const auto t04 = t03 * c16_64;
  const auto t05 = t02 + t04;
  const auto t06 = t00 + t05;
  const auto t07 = t01 + t04;
  const auto t08 = t00 - t05;
  const auto t09 = t01 - t04;
  const auto t10 = i04 + i28;
  const auto t11 = i04 - i28;
  const auto t12 = t11 * c8_64;
  const auto t13 = t10 + t12;
  const auto t14 = i12 + i20;
  const auto t15 = i12 - i20;
  const auto t16 = t15 * c24_64;
  const auto t17 = t14 + t16;
  const auto t18 = t13 + t17;
  const auto t19 = t12 + t16;
  const auto t20 = t13 - t17;
  const auto t21 = t12 - t16;
  const auto t22 = t20 * c16_64;
  const auto t23 = t21 * c16_64;
  const auto t24 = t18 + t23;
  const auto t25 = t19 + t22;
  const auto t26 = t06 + t24;
  const auto t27 = t07 + t25;
  const auto t28 = t09 + t22;
  const auto t29 = t08 + t23;
  const auto t30 = t06 - t24;
  const auto t31 = t07 - t25;
  const auto t32 = t09 - t22;
  const auto t33 = t08 - t23;
  const auto t34 = i02 + i30;
  const auto t35 = i02 - i30;
  const auto t36 = t35 * c4_64;
  const auto t37 = t34 + t36;
  const auto t38 = i14 + i18;
  const auto t39 = i14 - i18;
  const auto t40 = t39 * c28_64;
  const auto t41 = t38 + t40;
  const auto t42 = t37 + t41;
  const auto t43 = t36 + t40;
  const auto t44 = t37 - t41;
  const auto t45 = t36 - t40;
  const auto t46 = t44 * c8_64;
  const auto t47 = t45 * c8_64;
  const auto t48 = t42 + t47;
  const auto t49 = t43 + t46;
  const auto t50 = i06 + i26;
  const auto t51 = i06 - i26;
  const auto t52 = t51 * c12_64;
  const auto t53 = t50 + t52;
  const auto t54 = i10 + i22;
  const auto t55 = i10 - i22;
  const auto t56 = t55 * c20_64;
  const auto t57 = t54 + t56;
  const auto t58 = t53 + t57;
  const auto t59 = t52 + t56;
  const auto t60 = t53 - t57;
  const auto t61 = t52 - t56;
  const auto t62 = t60 * c24_64;
  const auto t63 = t61 * c24_64;
  const auto t64 = t58 + t63;
  const auto t65 = t59 + t62;
  const auto t66 = t48 + t64;
  const auto t67 = t49 + t65;
  const auto t68 = t46 + t62;
  const auto t69 = t47 + t63;
  const auto t70 = t48 - t64;
  const auto t71 = t49 - t65;
  const auto t72 = t46 - t62;
  const auto t73 = t47 - t63;
  const auto t74 = t70 * c16_64;
  const auto t75 = t71 * c16_64;
  const auto t76 = t72 * c16_64;
  const auto t77 = t73 * c16_64;
  const auto t78 = t66 + t77;
  const auto t79 = t67 + t76;
  const auto t80 = t68 + t75;
  const auto t81 = t69 + t74;
  const auto t82 = t26 + t78;
  const auto t83 = t27 + t79;
  const auto t84 = t28 + t80;
  const auto t85 = t29 + t81;
  const auto t86 = t33 + t74;
  const auto t87 = t32 + t75;
  const auto t88 = t31 + t76;
  const auto t89 = t30 + t77;
  const auto t90 = t26 - t78;
  const auto t91 = t27 - t79;
  const auto t92 = t28 - t80;
  const auto t93 = t29 - t81;
  const auto t94 = t33 - t74;
  const auto t95 = t32 - t75;
  const auto t96 = t31 - t76;
  const auto t97 = t30 - t77;
  const auto t98 = i01 + i31;
  const auto t99 = i01 - i31;
  const auto t100 = t99 * c2_64;
  const auto t101 = t98 + t100;
  const auto t102 = i15 + i17;
  const auto t103 = i15 - i17;
  const auto t104 = t103 * c30_64;
  const auto t105 = t102 + t104;
  const auto t106 = t101 + t105;
  const auto t107 = t100 + t104;
  const auto t108 = t101 - t105;
  const auto t109 = t100 - t104;
  const auto t110 = t108 * c4_64;
  const auto t111 = t109 * c4_64;
  const auto t112 = t106 + t111;
  const auto t113 = t107 + t110;
  const auto t114 = i07 + i25;
  const auto t115 = i07 - i25;
  const auto t116 = t115 * c14_64;
  const auto t117 = t114 + t116;
  const auto t118 = i09 + i23;
  const auto t119 = i09 - i23;
  const auto t120 = t119 * c18_64;
  const auto t121 = t118 + t120;
  const auto t122 = t117 + t121;
  const auto t123 = t116 + t120;
  const auto t124 = t117 - t121;
  const auto t125 = t116 - t120;
  const auto t126 = t124 * c28_64;
  const auto t127 = t125 * c28_64;
  const auto t128 = t122 + t127;
  const auto t129 = t123 + t126;
  const auto t130 = t112 + t128;
  const auto t131 = t113 + t129;
  const auto t132 = t110 + t126;
  const auto t133 = t111 + t127;
  const auto t134 = t112 - t128;
  const auto t135 = t113 - t129;
  const auto t136 = t110 - t126;
  const auto t137 = t111 - t127;
  const auto t138 = t134 * c8_64;
  const auto t139 = t135 * c8_64;
  const auto t140 = t136 * c8_64;
  const auto t141 = t137 * c8_64;
  const auto t142 = t130 + t141;
  const auto t143 = t131 + t140;
  const auto t144 = t132 + t139;
  const auto t145 = t133 + t138;
  const auto t146 = i03 + i29;
  const auto t147 = i03 - i29;
  const auto t148 = t147 * c6_64;
  const auto t149 = t146 + t148;
  const auto t150 = i13 + i19;
  const auto t151 = i13 - i19;
  const auto t152 = t151 * c26_64;
  const auto t153 = t150 + t152;
  const auto t154 = t149 + t153;
  const auto t155 = t148 + t152;
  const auto t156 = t149 - t153;
  const auto t157 = t148 - t152;
  const auto t158 = t156 * c12_64;
  const auto t159 = t157 * c12_64;
  const auto t160 = t154 + t159;
  const auto t161 = t155 + t158;
  const auto t162 = i05 + i27;
  const auto t163 = i05 - i27;
  const auto t164 = t163 * c10_64;
  const auto t165 = t162 + t164;
  const auto t166 = i11 + i21;
  const auto t167 = i11 - i21;
  const auto t168 = t167 * c22_64;
  const auto t169 = t166 + t168;
  const auto t170 = t165 + t169;
  const auto t171 = t164 + t168;
  const auto t172 = t165 - t169;
  const auto t173 = t164 - t168;
  const auto t174 = t172 * c20_64;
  const auto t175 = t173 * c20_64;
  const auto t176 = t170 + t175;
  const auto t177 = t171 + t174;
  const auto t178 = t160 + t176;
  const auto t179 = t161 + t177;
  const auto t180 = t158 + t174;
  const auto t181 = t159 + t175;
  const auto t182 = t160 - t176;
  const auto t183 = t161 - t177;
  const auto t184 = t158 - t174;
  const auto t185 = t159 - t175;
  const auto t186 = t182 * c24_64;
  const auto t187 = t183 * c24_64;
  const auto t188 = t184 * c24_64;
  const auto t189 = t185 * c24_64;
  const auto t190 = t178 + t189;
  const auto t191 = t179 + t188;
  const auto t192 = t180 + t187;
  const auto t193 = t181 + t186;
  const auto t194 = t142 + t190;
  const auto t195 = t143 + t191;
  const auto t196 = t144 + t192;
  const auto t197 = t145 + t193;
  const auto t198 = t138 + t186;
  const auto t199 = t139 + t187;
  const auto t200 = t140 + t188;
  const auto t201 = t141 + t189;
  const auto t202 = t142 - t190;
  const auto t203 = t143 - t191;
  const auto t204 = t144 - t192;
  const auto t205 = t145 - t193;
  const auto t206 = t138 - t186;
  const auto t207 = t139 - t187;
  const auto t208 = t140 - t188;
  const auto t209 = t141 - t189;
  const auto t210 = t202 * c16_64;
  const auto t211 = t203 * c16_64;
  const auto t212 = t204 * c16_64;
  const auto t213 = t205 * c16_64;
  const auto t214 = t206 * c16_64;
  const auto t215 = t207 * c16_64;
  const auto t216 = t208 * c16_64;
  const auto t217 = t209 * c16_64;
  const auto t218 = t194 + t217;
  const auto t219 = t195 + t216;
  const auto t220 = t196 + t215;
  const auto t221 = t197 + t214;
  const auto t222 = t198 + t213;
  const auto t223 = t199 + t212;
  const auto t224 = t200 + t211;
  const auto t225 = t201 + t210;
  const auto t226 = t82 + t218;
  const auto t227 = t83 + t219;
  const auto t228 = t84 + t220;
  const auto t229 = t85 + t221;
  const auto t230 = t86 + t222;
  const auto t231 = t87 + t223;
  const auto t232 = t88 + t224;
  const auto t233 = t89 + t225;
  const auto t234 = t97 + t210;
  const auto t235 = t96 + t211;
  const auto t236 = t95 + t212;
  const auto t237 = t94 + t213;
  const auto t238 = t93 + t214;
  const auto t239 = t92 + t215;
  const auto t240 = t91 + t216;
  const auto t241 = t90 + t217;
  const auto t242 = t82 - t218;
  const auto t243 = t83 - t219;
  const auto t244 = t84 - t220;
  const auto t245 = t85 - t221;
  const auto t246 = t86 - t222;
  const auto t247 = t87 - t223;
  const auto t248 = t88 - t224;
  const auto t249 = t89 - t225;
  const auto t250 = t97 - t210;
  const auto t251 = t96 - t211;
  const auto t252 = t95 - t212;
  const auto t253 = t94 - t213;
  const auto t254 = t93 - t214;
  const auto t255 = t92 - t215;
  const auto t256 = t91 - t216;
  const auto t257 = t90 - t217;

  i00 = t226;
  i01 = t227;
  i02 = t228;
  i03 = t229;
  i04 = t230;
  i05 = t231;
  i06 = t232;
  i07 = t233;
  i08 = t234;
  i09 = t235;
  i10 = t236;
  i11 = t237;
  i12 = t238;
  i13 = t239;
  i14 = t240;
  i15 = t241;
  i16 = t257;
  i17 = t256;
  i18 = t255;
  i19 = t254;
  i20 = t253;
  i21 = t252;
  i22 = t251;
  i23 = t250;
  i24 = t249;
  i25 = t248;
  i26 = t247;
  i27 = t246;
  i28 = t245;
  i29 = t244;
  i30 = t243;
  i31 = t242;
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ColumnDCT32(const From& from, const To& to) {
  const BlockDesc<32> d;

  for (size_t i = 0; i < 32; i += d.N) {
    auto i00 = from.Load(0, i);
    auto i01 = from.Load(1, i);
    auto i02 = from.Load(2, i);
    auto i03 = from.Load(3, i);
    auto i04 = from.Load(4, i);
    auto i05 = from.Load(5, i);
    auto i06 = from.Load(6, i);
    auto i07 = from.Load(7, i);
    auto i08 = from.Load(8, i);
    auto i09 = from.Load(9, i);
    auto i10 = from.Load(10, i);
    auto i11 = from.Load(11, i);
    auto i12 = from.Load(12, i);
    auto i13 = from.Load(13, i);
    auto i14 = from.Load(14, i);
    auto i15 = from.Load(15, i);
    auto i16 = from.Load(16, i);
    auto i17 = from.Load(17, i);
    auto i18 = from.Load(18, i);
    auto i19 = from.Load(19, i);
    auto i20 = from.Load(20, i);
    auto i21 = from.Load(21, i);
    auto i22 = from.Load(22, i);
    auto i23 = from.Load(23, i);
    auto i24 = from.Load(24, i);
    auto i25 = from.Load(25, i);
    auto i26 = from.Load(26, i);
    auto i27 = from.Load(27, i);
    auto i28 = from.Load(28, i);
    auto i29 = from.Load(29, i);
    auto i30 = from.Load(30, i);
    auto i31 = from.Load(31, i);
    ColumnDCT32(i00, i01, i02, i03, i04, i05, i06, i07, i08, i09, i10, i11, i12,
                i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23, i24, i25,
                i26, i27, i28, i29, i30, i31);
    to.Store(i00, 0, i);
    to.Store(i01, 1, i);
    to.Store(i02, 2, i);
    to.Store(i03, 3, i);
    to.Store(i04, 4, i);
    to.Store(i05, 5, i);
    to.Store(i06, 6, i);
    to.Store(i07, 7, i);
    to.Store(i08, 8, i);
    to.Store(i09, 9, i);
    to.Store(i10, 10, i);
    to.Store(i11, 11, i);
    to.Store(i12, 12, i);
    to.Store(i13, 13, i);
    to.Store(i14, 14, i);
    to.Store(i15, 15, i);
    to.Store(i16, 16, i);
    to.Store(i17, 17, i);
    to.Store(i18, 18, i);
    to.Store(i19, 19, i);
    to.Store(i20, 20, i);
    to.Store(i21, 21, i);
    to.Store(i22, 22, i);
    to.Store(i23, 23, i);
    to.Store(i24, 24, i);
    to.Store(i25, 25, i);
    to.Store(i26, 26, i);
    to.Store(i27, 27, i);
    to.Store(i28, 28, i);
    to.Store(i29, 29, i);
    to.Store(i30, 30, i);
    to.Store(i31, 31, i);
  }
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void ColumnIDCT32(const From& from, const To& to) {
  const BlockDesc<32> d;

  for (size_t i = 0; i < 32; i += d.N) {
    auto i00 = from.Load(0, i);
    auto i01 = from.Load(1, i);
    auto i02 = from.Load(2, i);
    auto i03 = from.Load(3, i);
    auto i04 = from.Load(4, i);
    auto i05 = from.Load(5, i);
    auto i06 = from.Load(6, i);
    auto i07 = from.Load(7, i);
    auto i08 = from.Load(8, i);
    auto i09 = from.Load(9, i);
    auto i10 = from.Load(10, i);
    auto i11 = from.Load(11, i);
    auto i12 = from.Load(12, i);
    auto i13 = from.Load(13, i);
    auto i14 = from.Load(14, i);
    auto i15 = from.Load(15, i);
    auto i16 = from.Load(16, i);
    auto i17 = from.Load(17, i);
    auto i18 = from.Load(18, i);
    auto i19 = from.Load(19, i);
    auto i20 = from.Load(20, i);
    auto i21 = from.Load(21, i);
    auto i22 = from.Load(22, i);
    auto i23 = from.Load(23, i);
    auto i24 = from.Load(24, i);
    auto i25 = from.Load(25, i);
    auto i26 = from.Load(26, i);
    auto i27 = from.Load(27, i);
    auto i28 = from.Load(28, i);
    auto i29 = from.Load(29, i);
    auto i30 = from.Load(30, i);
    auto i31 = from.Load(31, i);
    ColumnIDCT32(i00, i01, i02, i03, i04, i05, i06, i07, i08, i09, i10, i11,
                 i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23,
                 i24, i25, i26, i27, i28, i29, i30, i31);
    to.Store(i00, 0, i);
    to.Store(i01, 1, i);
    to.Store(i02, 2, i);
    to.Store(i03, 3, i);
    to.Store(i04, 4, i);
    to.Store(i05, 5, i);
    to.Store(i06, 6, i);
    to.Store(i07, 7, i);
    to.Store(i08, 8, i);
    to.Store(i09, 9, i);
    to.Store(i10, 10, i);
    to.Store(i11, 11, i);
    to.Store(i12, 12, i);
    to.Store(i13, 13, i);
    to.Store(i14, 14, i);
    to.Store(i15, 15, i);
    to.Store(i16, 16, i);
    to.Store(i17, 17, i);
    to.Store(i18, 18, i);
    to.Store(i19, 19, i);
    to.Store(i20, 20, i);
    to.Store(i21, 21, i);
    to.Store(i22, 22, i);
    to.Store(i23, 23, i);
    to.Store(i24, 24, i);
    to.Store(i25, 25, i);
    to.Store(i26, 26, i);
    to.Store(i27, 27, i);
    to.Store(i28, 28, i);
    to.Store(i29, 29, i);
    to.Store(i30, 30, i);
    to.Store(i31, 31, i);
  }
}

}  // namespace pik

#endif  // THIRD_PARTY_DCT_SIMD_ANY_H_
