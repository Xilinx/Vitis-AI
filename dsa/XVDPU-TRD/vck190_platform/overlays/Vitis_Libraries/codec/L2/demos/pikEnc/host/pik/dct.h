// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_DCT_H_
#define PIK_DCT_H_

// Fast SIMD floating-point DCT8-32.

#include <cmath>
#include <cstring>
#include "pik/block.h"
#include "pik/compiler_specific.h"
#include "pik/dct.h"
#include "pik/dct_simd_4.h"
#include "pik/dct_simd_8.h"
#include "pik/dct_simd_any.h"
#include "pik/simd/simd.h"
#include "pik/status.h"

namespace pik {

// Final scaling factors of outputs/inputs in the Arai, Agui, and Nakajima
// algorithm computing the DCT/IDCT (described in the book JPEG: Still Image
// Data Compression Standard, section 4.3.5) and the "A low multiplicative
// complexity fast recursive DCT-2 algorithm" (Maxim Vashkevich, Alexander
// Pertrovsky) algorithm. Note that the DCT and the IDCT scales of these two
// algorithms are flipped. We use the first algorithm for DCT8, and the second
// one for all other DCTs.
/* Python snippet to produce these tables for the Arai, Agui, Nakajima
 * algorithm:
 *
from mpmath import *
N = 8
def iscale(u):
  eps = sqrt(mpf(0.5)) if u == 0 else mpf(1.0)
  return sqrt(mpf(2) / mpf(N)) * eps * cos(mpf(u) * pi / mpf(2 * N))
def scale(u):
  return mpf(1) / (mpf(N) * iscale(i))
mp.dps = 18
print(", ".join([str(scale(i)) + 'f' for i in range(N)]))
print(", ".join([str(iscale(i)) + 'f' for i in range(N)]))
 */
static constexpr const float kDCTScales2[2] = {0.707106781186547524f,
                                               0.707106781186547524f};
static constexpr const float kIDCTScales2[2] = {0.707106781186547524f,
                                                0.707106781186547524f};
static constexpr const float kDCTScales4[4] = {0.5f, 0.653281482438188264f,
                                               0.5f, 0.270598050073098492f};
static constexpr const float kIDCTScales4[4] = {0.5f, 0.382683432365089772f,
                                                0.5f, 0.923879532511286756f};
static constexpr const float kDCTScales8[8] = {
    0.353553390593273762f, 0.254897789552079584f, 0.270598050073098492f,
    0.30067244346752264f,  0.353553390593273762f, 0.449988111568207852f,
    0.653281482438188264f, 1.28145772387075309f};

static constexpr const float kIDCTScales8[8] = {
    0.353553390593273762f, 0.490392640201615225f, 0.461939766255643378f,
    0.415734806151272619f, 0.353553390593273762f, 0.277785116509801112f,
    0.191341716182544886f, 0.0975451610080641339f};

static constexpr const float kIDCTScales16[16] = {0.25f,
                                                  0.177632042131274808f,
                                                  0.180239955501736978f,
                                                  0.184731156892216368f,
                                                  0.191341716182544886f,
                                                  0.200444985785954314f,
                                                  0.212607523691814112f,
                                                  0.228686034616512494f,
                                                  0.25f,
                                                  0.278654739432954475f,
                                                  0.318189645143208485f,
                                                  0.375006192208515097f,
                                                  0.461939766255643378f,
                                                  0.608977011699708658f,
                                                  0.906127446352887843f,
                                                  1.80352839005774887f};

static constexpr const float kDCTScales16[16] = {0.25f,
                                                 0.351850934381595615f,
                                                 0.346759961330536865f,
                                                 0.33832950029358817f,
                                                 0.326640741219094132f,
                                                 0.311806253246667808f,
                                                 0.293968900604839679f,
                                                 0.273300466750439372f,
                                                 0.25f,
                                                 0.224291896585659071f,
                                                 0.196423739596775545f,
                                                 0.166663914619436624f,
                                                 0.135299025036549246f,
                                                 0.102631131880589345f,
                                                 0.0689748448207357531f,
                                                 0.0346542922997728657f};

static constexpr const float kIDCTScales32[32] = {
    0.176776695296636881f, 0.125150749558799075f, 0.125604821547038926f,
    0.126367739974385915f, 0.127448894776039792f, 0.128861827480656137f,
    0.13062465373492222f,  0.132760647772446044f, 0.135299025036549246f,
    0.138275974008611132f, 0.141736008704089426f, 0.145733742051533468f,
    0.15033622173376132f,  0.155626030758916204f, 0.161705445839997532f,
    0.168702085363751436f, 0.176776695296636881f, 0.186134067750574612f,
    0.197038655862812556f, 0.20983741135388176f,  0.224994055784103926f,
    0.243142059465490173f, 0.265169421497586868f, 0.292359983358221239f,
    0.326640741219094132f, 0.371041154078541569f, 0.430611774559583482f,
    0.514445252488352888f, 0.640728861935376545f, 0.851902104617179697f,
    1.27528715467229096f,  2.5475020308870142f};

static constexpr const float kDCTScales32[32] = {
    0.176776695296636881f,  0.249698864051293098f,  0.248796181668049222f,
    0.247294127491195243f,  0.245196320100807612f,  0.242507813298635998f,
    0.239235083933052216f,  0.235386016295755195f,  0.230969883127821689f,
    0.225997323280860833f,  0.220480316087088757f,  0.214432152500068017f,
    0.207867403075636309f,  0.200801882870161227f,  0.19325261334068424f,
    0.185237781338739773f,  0.176776695296636881f,  0.1678897387117546f,
    0.158598321040911375f,  0.148924826123108336f,  0.138892558254900556f,
    0.128525686048305432f,  0.117849184206499412f,  0.106888773357570524f,
    0.0956708580912724429f, 0.0842224633480550127f, 0.0725711693136155919f,
    0.0607450449758159725f, 0.048772580504032067f,  0.0366826186138404379f,
    0.0245042850823901505f, 0.0122669185818545036f};

template <size_t N>
constexpr const float* DCTScales() {
  return N == 2 ? kDCTScales2
                : (N == 4 ? kDCTScales4
                          : (N == 8 ? kDCTScales8
                                    : (N == 16 ? kDCTScales16 : kDCTScales32)));
}

template <size_t N>
constexpr const float* IDCTScales() {
  return N == 2
             ? kIDCTScales2
             : (N == 4 ? kIDCTScales4
                       : (N == 8 ? kIDCTScales8
                                 : (N == 16 ? kIDCTScales16 : kIDCTScales32)));
}

// Relative L1 norm of IDCT of a vector of 0s with a single 1 in position i
// (with respect to the L1 norm of a DC-only vector).
static constexpr const float kL1Norm2[2] = {
    1.0000000000000000000f,
    1.0000000000000000000f,
};
static constexpr const float kL1Norm4[4] = {
    1.0000000000000000000f,  //
    0.9238795325112867561f,  // cos(pi/8)
    1.0000000000000000000f,  //
    0.9238795325112867561f,  // cos(pi/8)
};
static constexpr const float kL1Norm8[8] = {
    1.0000000000000000000f,  //
    0.9061274463528878431f,  // cos(pi/8) * cos(pi/16)
    0.9238795325112867561f,  // cos(pi/8)
    0.9061274463528878431f,  // cos(pi/8) * cos(pi/16)
    1.0000000000000000000f,  //
    0.9061274463528878431f,  // cos(pi/8) * cos(pi/16)
    0.9238795325112867561f,  // cos(pi/8)
    0.9061274463528878431f,  // cos(pi/8) * cos(pi/16)

};
static constexpr const float kL1Norm16[16] = {
    1.0000000000000000000f,  //
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9061274463528878431f,  // cos(pi/8) * cos(pi/16)
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9238795325112867561f,  // cos(pi/8)
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9061274463528878431f,  // cos(pi/8) * cos(pi/16)
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    1.0000000000000000000f,  //
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9061274463528878431f,  // cos(pi/8) * cos(pi/16)
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9238795325112867561f,  // cos(pi/8)
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9061274463528878431f,  // cos(pi/8) * cos(pi/16)
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
};
static constexpr const float kL1Norm32[32] = {
    1.0000000000000000000f,  //
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9061274463528878431f,  // cos(pi/8) * cos(pi/16)
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9238795325112867561f,  // cos(pi/8)
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9061274463528878431f,  // cos(pi/8) * cos(pi/16)
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    1.0000000000000000000f,  //
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9061274463528878431f,  // cos(pi/8) * cos(pi/16)
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9238795325112867561f,  // cos(pi/8)
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9061274463528878431f,  // cos(pi/8) * cos(pi/16)
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
    0.9017641950288744354f,  // cos(pi/8) * cos(pi/16) * cos(pi/32)
    0.9006779805633546924f,  // cos(pi/8) * cos(pi/16) * cos(pi/32) * cos(pi/64)
};

static constexpr const float kL1NormInv2[2] = {
    1.000000000000000000f,
    1.000000000000000000f,
};
static constexpr const float kL1NormInv4[4] = {
    1.000000000000000000f,
    1.082392200292393968f,
    1.000000000000000000f,
    1.082392200292393968f,
};
static constexpr const float kL1NormInv8[8] = {
    1.000000000000000000f, 1.103597517131772049f, 1.082392200292393968f,
    1.103597517131772049f, 1.000000000000000000f, 1.103597517131772049f,
    1.082392200292393968f, 1.103597517131772049f,
};
static constexpr const float kL1NormInv16[16] = {
    1.000000000000000000f, 1.108937353592731700f, 1.103597517131772049f,
    1.108937353592731700f, 1.082392200292393968f, 1.108937353592731700f,
    1.103597517131772049f, 1.108937353592731700f, 1.000000000000000000f,
    1.108937353592731700f, 1.103597517131772049f, 1.108937353592731700f,
    1.082392200292393968f, 1.108937353592731700f, 1.103597517131772049f,
    1.108937353592731700f,
};
static constexpr const float kL1NormInv32[32] = {
    1.000000000000000000, 1.110274728127050414, 1.108937353592731379,
    1.110274728127050414, 1.103597517131772010, 1.110274728127050636,
    1.108937353592731379, 1.110274728127050414, 1.082392200292393580,
    1.110274728127050414, 1.108937353592730934, 1.110274728127050414,
    1.103597517131771788, 1.110274728127050414, 1.108937353592731156,
    1.110274728127050414, 0.999999999999999556, 1.110274728127049970,
    1.108937353592731601, 1.110274728127051080, 1.103597517131771788,
    1.110274728127050414, 1.108937353592732045, 1.110274728127050192,
    1.082392200292394691, 1.110274728127049526, 1.108937353592733155,
    1.110274728127050858, 1.103597517131772232, 1.110274728127051969,
    1.108937353592732933, 1.110274728127050414,
};

template <size_t N>
constexpr const float* L1Norm() {
  return N == 2
             ? kL1Norm2
             : (N == 4
                    ? kL1Norm4
                    : (N == 8 ? kL1Norm8 : (N == 16 ? kL1Norm16 : kL1Norm32)));
}

template <size_t N>
constexpr const float* L1NormInv() {
  return N == 2 ? kL1NormInv2
                : (N == 4 ? kL1NormInv4
                          : (N == 8 ? kL1NormInv8
                                    : (N == 16 ? kL1NormInv16 : kL1NormInv32)));
}

// https://en.wikipedia.org/wiki/In-place_matrix_transposition#Square_matrices
template <size_t N, class From, class To>
SIMD_ATTR PIK_INLINE void GenericTransposeBlockInplace(const From& from,
                                                       const To& to) {
  // This does not guarantee anything, just saves from the most stupid mistakes.
  PIK_ASSERT(from.Address(0, 0) == to.Address(0, 0));
  for (size_t n = 0; n < N - 1; ++n) {
    for (size_t m = n + 1; m < N; ++m) {
      // Swap
      const float tmp = from.Read(m, n);
      to.Write(from.Read(n, m), m, n);
      to.Write(tmp, n, m);
    }
  }
}

template <size_t N, class From, class To>
SIMD_ATTR PIK_INLINE void GenericTransposeBlock(const From& from,
                                                const To& to) {
  // This does not guarantee anything, just saves from the most stupid mistakes.
  PIK_ASSERT(from.Address(0, 0) != to.Address(0, 0));
  for (size_t n = 0; n < N; ++n) {
    for (size_t m = 0; m < N; ++m) {
      to.Write(from.Read(n, m), m, n);
    }
  }
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void TransposeBlock8(const From& from, const To& to) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
  TransposeBlock8_V8(from, to);
#elif SIMD_TARGET_VALUE == SIMD_NONE
  if (from.Address(0, 0) == to.Address(0, 0)) {
    GenericTransposeBlockInplace<8>(from, to);
  } else {
    GenericTransposeBlock<8>(from, to);
  }
#else  // generic 128-bit
  TransposeBlock8_V4(from, to);
#endif
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void TransposeBlock16(const From& from, const To& to) {
  SIMD_ALIGN float tmp[8 * 8];
  TransposeBlock8(from, to);
  TransposeBlock8(from.View(0, 8), ToBlock<8>(tmp));
  TransposeBlock8(from.View(8, 0), to.View(0, 8));
  CopyBlock8(FromBlock<8>(tmp), to.View(8, 0));
  TransposeBlock8(from.View(8, 8), to.View(8, 8));
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void TransposeBlock32(const From& from, const To& to) {
  SIMD_ALIGN float tmp[8 * 8];
  TransposeBlock8(from, to);
  TransposeBlock8(from.View(0, 8), ToBlock<8>(tmp));
  TransposeBlock8(from.View(8, 0), to.View(0, 8));
  CopyBlock8(FromBlock<8>(tmp), to.View(8, 0));
  TransposeBlock8(from.View(8, 8), to.View(8, 8));
  TransposeBlock8(from.View(0, 16), ToBlock<8>(tmp));
  TransposeBlock8(from.View(16, 0), to.View(0, 16));
  CopyBlock8(FromBlock<8>(tmp), to.View(16, 0));
  TransposeBlock8(from.View(8, 16), ToBlock<8>(tmp));
  TransposeBlock8(from.View(16, 8), to.View(8, 16));
  CopyBlock8(FromBlock<8>(tmp), to.View(16, 8));
  TransposeBlock8(from.View(16, 16), to.View(16, 16));
  TransposeBlock8(from.View(0, 24), ToBlock<8>(tmp));
  TransposeBlock8(from.View(24, 0), to.View(0, 24));
  CopyBlock8(FromBlock<8>(tmp), to.View(24, 0));
  TransposeBlock8(from.View(8, 24), ToBlock<8>(tmp));
  TransposeBlock8(from.View(24, 8), to.View(8, 24));
  CopyBlock8(FromBlock<8>(tmp), to.View(24, 8));
  TransposeBlock8(from.View(16, 24), ToBlock<8>(tmp));
  TransposeBlock8(from.View(24, 16), to.View(16, 24));
  CopyBlock8(FromBlock<8>(tmp), to.View(24, 16));
  TransposeBlock8(from.View(24, 24), to.View(24, 24));
}

// Computes the in-place NxN transposed-scaled-DCT (tsDCT) of block.
// Requires that block is SIMD_ALIGN'ed.
//
// Final DCT coefficients could be obtained the following way:
//   unscaled(f)[x, y] = f[x, y] * DCTScales<N>[x] * DCTScales<N>[y]
//   untransposed(f)[x, y] = f[y, x]
//   DCT(input) = unscaled(untransposed(tsDCT(input)))
//
// NB: DCT denotes scaled variant of DCT-II, which is orthonormal.
//
// See also DCTSlow, ComputeDCT
template <size_t N>
struct ComputeTransposedScaledDCT;

template <>
struct ComputeTransposedScaledDCT<32> {
  template <class From, class To>
  SIMD_ATTR PIK_INLINE void operator()(const From& from, const To& to) {
    SIMD_ALIGN float block[32 * 32];
    ColumnDCT32(from, ToBlock<32>(block));
    TransposeBlock32(FromBlock<32>(block), ToBlock<32>(block));
    ColumnDCT32(FromBlock<32>(block), to);
  }
};

template <>
struct ComputeTransposedScaledDCT<16> {
  template <class From, class To>
  SIMD_ATTR PIK_INLINE void operator()(const From& from, const To& to) {
    SIMD_ALIGN float block[16 * 16];
    ColumnDCT16(from, ToBlock<16>(block));
    TransposeBlock16(FromBlock<16>(block), ToBlock<16>(block));
    ColumnDCT16(FromBlock<16>(block), to);
  }
};

template <>
struct ComputeTransposedScaledDCT<8> {
  template <class From, class To>
  SIMD_ATTR PIK_INLINE void operator()(const From& from, const To& to) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
    ComputeTransposedScaledDCT8_V8(from, to);
#elif SIMD_TARGET_VALUE == SIMD_NONE
    SIMD_ALIGN float block[8 * 8];
    ColumnDCT8(from, ToBlock<8>(block));
    TransposeBlock8(FromBlock<8>(block), ToBlock<8>(block));
    ColumnDCT8(FromBlock<8>(block), to);
#else
    ComputeTransposedScaledDCT8_V4(from, to);
#endif
  }
};

template <>
struct ComputeTransposedScaledDCT<4> {
  template <class From, class To>
  SIMD_ATTR PIK_INLINE void operator()(const From& from, const To& to) {
    SIMD_ALIGN float block[4 * 4];
    ColumnDCT4(from, ToBlock<4>(block));
    GenericTransposeBlockInplace<4>(FromBlock<4>(block), ToBlock<4>(block));
    ColumnDCT4(FromBlock<4>(block), to);
  }
};

template <>
struct ComputeTransposedScaledDCT<2> {
  template <class From, class To>
  SIMD_ATTR PIK_INLINE void operator()(const From& from, const To& to) {
    const float a00 = from.Read(0, 0);
    const float a01 = from.Read(0, 1);
    const float a10 = from.Read(1, 0);
    const float a11 = from.Read(1, 1);
    to.Write(a00 + a01 + a10 + a11, 0, 0);
    to.Write(a00 + a01 - a10 - a11, 0, 1);
    to.Write(a00 - a01 + a10 - a11, 1, 0);
    to.Write(a00 - a01 - a10 + a11, 1, 1);
  }
};

// Computes the in-place NxN transposed-scaled-iDCT (tsIDCT)of block.
// Requires that block is SIMD_ALIGN'ed.
//
// Final DCT coefficients could be obtained the following way:
//   unscaled(f)[x, y] = f[x, y] * IDCTScales<N>[x] * IDCTScales<N>[y]
//   untransposed(f)[x, y] = f[y, x]
//   IDCT(input) = tsIDCT(untransposed(unscaled(input)))
//
// NB: IDCT denotes scaled variant of DCT-III, which is orthonormal.
//
// See also IDCTSlow, ComputeIDCT.
template <size_t N>
struct ComputeTransposedScaledIDCT;

template <>
struct ComputeTransposedScaledIDCT<32> {
  template <class From, class To>
  SIMD_ATTR PIK_INLINE void operator()(const From& from, const To& to) {
    SIMD_ALIGN float block[32 * 32];
    ColumnIDCT32(from, ToBlock<32>(block));
    TransposeBlock32(FromBlock<32>(block), ToBlock<32>(block));
    ColumnIDCT32(FromBlock<32>(block), to);
  }
};

template <>
struct ComputeTransposedScaledIDCT<16> {
  template <class From, class To>
  SIMD_ATTR PIK_INLINE void operator()(const From& from, const To& to) {
    SIMD_ALIGN float block[16 * 16];
    ColumnIDCT16(from, ToBlock<16>(block));
    TransposeBlock16(FromBlock<16>(block), ToBlock<16>(block));
    ColumnIDCT16(FromBlock<16>(block), to);
  }
};

template <>
struct ComputeTransposedScaledIDCT<8> {
  template <class From, class To>
  SIMD_ATTR PIK_INLINE void operator()(const From& from, const To& to) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
    ComputeTransposedScaledIDCT8_V8(from, to);
#elif SIMD_TARGET_VALUE == SIMD_NONE
    SIMD_ALIGN float block[8 * 8];
    ColumnIDCT8(from, ToBlock<8>(block));
    TransposeBlock8(FromBlock<8>(block), ToBlock<8>(block));
    ColumnIDCT8(FromBlock<8>(block), to);
#else
    ComputeTransposedScaledIDCT8_V4(from, to);
#endif
  }
};

template <>
struct ComputeTransposedScaledIDCT<4> {
  template <class From, class To>
  SIMD_ATTR PIK_INLINE void operator()(const From& from, const To& to) {
    SIMD_ALIGN float block[4 * 4];
    ColumnIDCT4(from, ToBlock<4>(block));
    GenericTransposeBlockInplace<4>(FromBlock<4>(block), ToBlock<4>(block));
    ColumnIDCT4(FromBlock<4>(block), to);
  }
};

template <>
struct ComputeTransposedScaledIDCT<2> {
  template <class From, class To>
  SIMD_ATTR PIK_INLINE void operator()(const From& from, const To& to) {
    const float a00 = from.Read(0, 0);
    const float a01 = from.Read(0, 1);
    const float a10 = from.Read(1, 0);
    const float a11 = from.Read(1, 1);

    //std::cout<<"std_IDCT: a00="<<a00<<" a01="<<a01<<" a10"<<a10<<" a11"<<a11<<std::endl;

    to.Write(a00 + a01 + a10 + a11, 0, 0);
    to.Write(a00 + a01 - a10 - a11, 0, 1);
    to.Write(a00 - a01 + a10 - a11, 1, 0);
    to.Write(a00 - a01 - a10 + a11, 1, 1);
  }
};

// Similar to ComputeTransposedScaledDCT, but only DC coefficient is calculated.
template <size_t N, class From>
static SIMD_ATTR PIK_INLINE float ComputeScaledDC(const From& from) {
  static_assert(N == 8, "Currently only 8x8 is supported");

#if SIMD_TARGET_VALUE == SIMD_AVX2
  return ComputeScaledDC8_V8(from);
#elif SIMD_TARGET_VALUE == SIMD_NONE
  const BlockDesc<N> d;
  auto sum = setzero(d);
  for (size_t iy = 0; iy < N; ++iy) {
    for (size_t ix = 0; ix < N; ix += d.N) {
      sum += from.Load(iy, ix);
    }
  }
  sum = ext::sum_of_lanes(sum);
  return get_part(SIMD_PART(float, 1)(), sum);
#else
  return ComputeScaledDC8_V4(from);
#endif
}

}  // namespace pik

#endif  // PIK_DCT_H_
