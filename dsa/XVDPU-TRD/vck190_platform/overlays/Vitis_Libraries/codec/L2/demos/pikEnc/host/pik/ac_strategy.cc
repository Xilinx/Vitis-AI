// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/ac_strategy.h"
#include "pik/block.h"
#include "pik/common.h"
#include "pik/dct.h"
#include "pik/entropy_coder.h"
#include "pik/image.h"
#include "pik/opsin_params.h"
#include "pik/profiler.h"
#include "pik/quantizer.h"
#include "pik/simd/simd.h"

#include <iostream>

#define ENABLE_DIAGONAL_LINES_EXPERIMENT 0

namespace pik {
namespace {

// 1-dimensional DCT's of different sizes (Except DCT2x2)
// TODO(lode): SIMDify and place in dct_simd_any.h
template <class V> void DCT2(V &i0, V &i1) {
  V r0 = (i0 + i1) * 0.5;
  V r1 = (i0 - i1) * 0.5;
  i0 = r0;
  i1 = r1;
}

template <class V> void IDCT2(V &i0, V &i1) {
  V r0 = (i0 + i1);
  V r1 = (i0 - i1);
  i0 = r0;
  i1 = r1;
}

template <class V> void DCT2x2(V &i0, V &i1, V &i2, V &i3) {
  V r0 = (i0 + i1 + i2 + i3) * 0.25f;
  V r1 = (i0 - i1 + i2 - i3) * 0.25f;
  V r2 = (i0 + i1 - i2 - i3) * 0.25f;
  V r3 = (i0 - i1 - i2 + i3) * 0.25f;
  i0 = r0;
  i1 = r1;
  i2 = r2;
  i3 = r3;
}

template <class V> void IDCT2x2(V &i0, V &i1, V &i2, V &i3) {
  V r0 = i0 + i1 + i2 + i3;
  V r1 = i0 - i1 + i2 - i3;
  V r2 = i0 + i1 - i2 - i3;
  V r3 = i0 - i1 - i2 + i3;
  i0 = r0;
  i1 = r1;
  i2 = r2;
  i3 = r3;
}

template <class V> void DCT3(V &i0, V &i1, V &i2) {
  V r0 = (i0 + i1 + i2) * (1 / 3.0f);
  V r1 = (i0 - i2) * 0.57735029;
  V r2 = (i0 + i2) * (1 / 3.0f) - i1 * (2 / 3.0f);
  i0 = r0;
  i1 = r1;
  i2 = r2;
}

template <class V> void IDCT3(V &i0, V &i1, V &i2) {
  V t0 = i0;
  V t1 = i1 * 0.86602540f;
  V t2a = i2 * 0.5f;
  V t2b = i2;
  i0 = t0 + t2a + t1;
  i1 = t0 - t2b;
  i2 = t0 + t2a - t1;
}

template <class V> void DCT4(V &i0, V &i1, V &i2, V &i3) {
  static const V c2_8 = 1.414213562373095048f; // 2 * cos(2 * pi / 8)
  V t0 = i0 + i3;
  V t1 = i1 + i2;
  V t2 = i0 - i3;
  V t3 = i1 - i2;
  V t4 = t0 + t1;
  V t5 = t0 - t1;
  V t6 = t2 - t3;
  V t7 = t3 * c2_8;
  V t8 = t6 + t7;
  V t9 = t6 - t7;
  i0 = t4 * (0.5f / 2);
  i1 = t8 * (0.653281482438188264f / 2);
  i2 = t5 * (0.5f / 2);
  i3 = t9 * (0.270598050073098492f / 2);
}

template <class V> void IDCT4(V &i0, V &i1, V &i2, V &i3) {
  static const V c2_8 = 0.7071067811865475244f; // 0.5 / cos(2 * pi / 8)
  i0 *= (0.5f * 2);
  i1 *= (0.382683432365089772f * 2);
  i2 *= (0.5f * 2);
  i3 *= (0.923879532511286756f * 2);
  V t0 = i0 + i2;
  V t1 = i0 - i2;
  V t2 = i1 + i3;
  V t3 = i1 - i3;
  V t4 = t3 * c2_8;
  V t5 = t2 + t4;
  V t6 = t0 + t5;
  V t7 = t1 + t4;
  V t8 = t0 - t5;
  V t9 = t1 - t4;
  i0 = t6;
  i1 = t7;
  i2 = t9;
  i3 = t8;
}

template <class V> void DCT6(V &i0, V &i1, V &i2, V &i3, V &i4, V &i5) {
  V t0 = (i1 - i4) * 0.23570227;
  V r0 = (i0 + i1 + i2 + i3 + i4 + i5) * (1 / 6.0f);
  V r1 = (i0 - i5) * 0.32197529 + t0 + (i2 - i3) * 0.08627302;
  V r2 = (i0 - i2 - i3 + i5) * 0.28867514;
  V r3 = (i0 - i1 - i2 + i3 + i4 - i5) * 0.23570227;
  V r4 = (i0 + i2 + i3 + i5) * (1 / 6.0f) - (i1 + i4) * (1 / 3.0f);
  V r5 = (i0 - i5) * 0.08627302 - t0 + (i2 - i3) * 0.32197529;
  i0 = r0;
  i1 = r1;
  i2 = r2;
  i3 = r3;
  i4 = r4;
  i5 = r5;
}

template <class V> void IDCT6(V &i0, V &i1, V &i2, V &i3, V &i4, V &i5) {
  // TODO(lode): maybe more multiplies can be removed by combining some terms.
  V i0a = i0;
  V i1a = i1 * 0.96592583;
  V i1b = i1 * 0.70710678;
  V i1c = i1 * 0.25881905;
  V i2a = i2 * 0.86602540;
  V i3a = i3 * 0.70710678;
  V i4a = i4 * 0.5;
  V i4b = i4;
  V i5a = i5 * 0.25881905;
  V i5b = i5 * 0.70710678;
  V i5c = i5 * 0.96592583;
  i0 = i0a + i1a + i3a + i4a + i5a + i2a;
  i1 = i0a + i1b - i3a - i4b - i5b;
  i2 = i0a + i1c - i3a + i4a + i5c - i2a;
  i3 = i0a - i1c + i3a + i4a - i5c - i2a;
  i4 = i0a - i1b + i3a - i4b + i5b;
  i5 = i0a - i1a - i3a + i4a - i5a + i2a;
}

template <class V>
static void DCT8(V &i0, V &i1, V &i2, V &i3, V &i4, V &i5, V &i6, V &i7) {
  static const V c1 = 0.707106781186548f; // 1 / sqrt(2)
  static const V c2 = 0.382683432365090f; // cos(3 * pi / 8)
  static const V c3 = 1.30656296487638f;  // 1 / (2 * cos(3 * pi / 8))
  static const V c4 = 0.541196100146197f; // sqrt(2) * cos(3 * pi / 8)
  const V t00 = i0 + i7;
  const V t01 = i0 - i7;
  const V t02 = i3 + i4;
  const V t03 = i3 - i4;
  const V t04 = i2 + i5;
  const V t05 = i2 - i5;
  const V t06 = i1 + i6;
  const V t07 = i1 - i6;
  const V t08 = t00 + t02;
  const V t09 = t00 - t02;
  const V t10 = t06 + t04;
  const V t11 = t06 - t04;
  const V t12 = t07 + t05;
  const V t13 = t01 + t07;
  const V t14 = t05 + t03;
  const V t15 = t11 + t09;
  const V t16 = t14 - t13;
  const V t17 = c1 * t15;
  const V t18 = c1 * t12;
  const V t19 = c2 * t16;
  const V t20 = t01 + t18;
  const V t21 = t01 - t18;
  const V t22 = c3 * t13 + t19;
  const V t23 = c4 * t14 + t19;
  i0 = (t08 + t10) * (0.353553390593273762f / 2.8284271247461903f);
  i1 = (t20 + t22) * (0.254897789552079584f / 2.8284271247461903f);
  i2 = (t09 + t17) * (0.270598050073098492f / 2.8284271247461903f);
  i3 = (t21 - t23) * (0.30067244346752264f / 2.8284271247461903f);
  i4 = (t08 - t10) * (0.353553390593273762f / 2.8284271247461903f);
  i5 = (t21 + t23) * (0.449988111568207852f / 2.8284271247461903f);
  i6 = (t09 - t17) * (0.653281482438188264f / 2.8284271247461903f);
  i7 = (t20 - t22) * (1.28145772387075309f / 2.8284271247461903f);
}

template <class V>
static void IDCT8(V &i0, V &i1, V &i2, V &i3, V &i4, V &i5, V &i6, V &i7) {
  static const V c1 = 1.41421356237310; // sqrt(2)
  static const V c2 = 2.61312592975275; // 1 / cos(3 * pi / 8)
  static const V c3 = 0.76536686473018; // 2 * cos(3 * pi / 8)
  static const V c4 = 1.08239220029239; // 2 * sqrt(2) * cos(3 * pi / 8)
  i0 *= (0.353553390593273762 * 2.8284271247461903f);
  i1 *= (0.490392640201615225 * 2.8284271247461903f);
  i2 *= (0.461939766255643378 * 2.8284271247461903f);
  i3 *= (0.415734806151272619 * 2.8284271247461903f);
  i4 *= (0.353553390593273762 * 2.8284271247461903f);
  i5 *= (0.277785116509801112 * 2.8284271247461903f);
  i6 *= (0.191341716182544886 * 2.8284271247461903f);
  i7 *= (0.0975451610080641339 * 2.8284271247461903f);
  const V t00 = i0 + i4;
  const V t01 = i0 - i4;
  const V t02 = i6 + i2;
  const V t03 = i6 - i2;
  const V t04 = i7 + i1;
  const V t05 = i7 - i1;
  const V t06 = i5 + i3;
  const V t07 = i5 - i3;
  const V t08 = t04 + t06;
  const V t09 = t04 - t06;
  const V t10 = t00 + t02;
  const V t11 = t00 - t02;
  const V t12 = t07 - t05;
  const V t13 = c3 * t12;
  const V t14 = c1 * t03 + t02;
  const V t15 = t01 - t14;
  const V t16 = t01 + t14;
  const V t17 = c2 * t05 + t13;
  const V t18 = c4 * t07 + t13;
  const V t19 = t08 + t17;
  const V t20 = c1 * t09 + t19;
  const V t21 = t18 - t20;
  i0 = t10 + t08;
  i1 = t15 - t19;
  i2 = t16 + t20;
  i3 = t11 + t21;
  i4 = t11 - t21;
  i5 = t16 - t20;
  i6 = t15 + t19;
  i7 = t10 - t08;
}

// True if we should try to find a non-trivial AC strategy.
const constexpr bool kChooseAcStrategy = true;

// Returns the value such that ComputeTransposedScaledDCT<N>() of a block with
// this value in position (x, y) and 0s everywhere else will have the average of
// absolute values of 1.
template <size_t N> constexpr float DCTTotalScale(size_t x, size_t y) {
  return N * DCTScales<N>()[x] * DCTScales<N>()[y] * L1NormInv<N>()[x] *
         L1NormInv<N>()[y];
}
template <size_t N> constexpr float DCTInvTotalScale(size_t x, size_t y) {
  return N * IDCTScales<N>()[x] * IDCTScales<N>()[y] * L1Norm<N>()[x] *
         L1Norm<N>()[y];
}

// Computes the lowest-frequency LFxLF-sized square in output, which is a
// DCTN-sized DCT block, by doing a NxN DCT on the input block.
template <size_t DCTN, size_t LF, size_t N>
SIMD_ATTR PIK_INLINE void
ReinterpretingDCT(const float *input, const size_t input_stride, float *output,
                  const size_t output_stride) {
  static_assert(LF == N,
                "ReinterpretingDCT should only be called with LF == N");
  SIMD_ALIGN float block[N * N] = {};
  for (size_t y = 0; y < N; y++) {
    for (size_t x = 0; x < N; x++) {
      block[y * N + x] = input[y * input_stride + x];
    }
  }
  ComputeTransposedScaledDCT<N>()(FromBlock<N>(block), ScaleToBlock<N>(block));
  for (size_t y = 0; y < LF; y++) {
    for (size_t x = 0; x < LF; x++) {
      output[y * output_stride + x] = block[y * N + x] *
                                      DCTTotalScale<N>(x, y) *
                                      DCTInvTotalScale<DCTN>(x, y);
      //std::cout<<"dct: y="<<y<<" x="<<x<<" dc="<<input[y*input_stride+x]<<" out="<<block[y*N+x]<<" dc_stride="<<input_stride
      //  	  <<" scale="<<DCTTotalScale<N>(x, y) * DCTInvTotalScale<DCTN>(x, y)<<std::endl;
    }
  }
}

// Inverse of ReinterpretingDCT.
template <size_t DCTN, size_t LF, size_t N>
SIMD_ATTR PIK_INLINE void
ReinterpretingIDCT(const float *input, const size_t input_stride, float *output,
                   const size_t output_stride) {
  SIMD_ALIGN float block[N * N] = {};
  for (size_t y = 0; y < LF; y++) {
    for (size_t x = 0; x < LF; x++) {
      block[y * N + x] = input[y * input_stride + x] *
                         DCTInvTotalScale<N>(x, y) * DCTTotalScale<DCTN>(x, y);
      //std::cout<<"std_IDCT: id="<<N*y+x<<" value="<<input[y * input_stride + x]<<" scaled="<<block[y * N + x]<<std::endl;
    }
  }
  ComputeTransposedScaledIDCT<N>()(FromBlock<N>(block), ToBlock<N>(block));

  for (size_t y = 0; y < N; y++) {
    for (size_t x = 0; x < N; x++) {
      output[y * output_stride + x] = block[y * N + x];
      //std::cout<<"std_IDCT: id="<<N*y+x<<" idct="<<block[y * N + x]<<std::endl;
    }
  }
}

template <size_t S>
void DCT2TopBlock(const float *block, size_t stride, float *out) {
  static_assert(kBlockDim % S == 0, "S should be a divisor of kBlockDim");
  static_assert(S % 2 == 0, "S should be even");
  float temp[kBlockDim * kBlockDim];
  constexpr size_t num_2x2 = S / 2;
  for (size_t y = 0; y < num_2x2; y++) {
    for (size_t x = 0; x < num_2x2; x++) {
      float c00 = block[y * 2 * stride + x * 2];
      float c01 = block[y * 2 * stride + x * 2 + 1];
      float c10 = block[(y * 2 + 1) * stride + x * 2];
      float c11 = block[(y * 2 + 1) * stride + x * 2 + 1];
      float r00 = c00 + c01 + c10 + c11;
      float r01 = c00 + c01 - c10 - c11;
      float r10 = c00 - c01 + c10 - c11;
      float r11 = c00 - c01 - c10 + c11;
      r00 *= 0.25f;
      r01 *= 0.25f;
      r10 *= 0.25f;
      r11 *= 0.25f;
      temp[y * kBlockDim + x] = r00;
      temp[y * kBlockDim + num_2x2 + x] = r01;
      temp[(y + num_2x2) * kBlockDim + x] = r10;
      temp[(y + num_2x2) * kBlockDim + num_2x2 + x] = r11;
    }
  }
  for (size_t y = 0; y < S; y++) {
    for (size_t x = 0; x < S; x++) {
      out[y * kBlockDim + x] = temp[y * kBlockDim + x];
    }
  }
}

template <size_t S>
void IDCT2TopBlock(const float *block, size_t stride_out, float *out) {
  static_assert(kBlockDim % S == 0, "S should be a divisor of kBlockDim");
  static_assert(S % 2 == 0, "S should be even");
  float temp[kBlockDim * kBlockDim];
  constexpr size_t num_2x2 = S / 2;
  for (size_t y = 0; y < num_2x2; y++) {
    for (size_t x = 0; x < num_2x2; x++) {
      float c00 = block[y * kBlockDim + x];
      float c01 = block[y * kBlockDim + num_2x2 + x];
      float c10 = block[(y + num_2x2) * kBlockDim + x];
      float c11 = block[(y + num_2x2) * kBlockDim + num_2x2 + x];
      float r00 = c00 + c01 + c10 + c11;
      float r01 = c00 + c01 - c10 - c11;
      float r10 = c00 - c01 + c10 - c11;
      float r11 = c00 - c01 - c10 + c11;
      temp[y * 2 * kBlockDim + x * 2] = r00;
      temp[y * 2 * kBlockDim + x * 2 + 1] = r01;
      temp[(y * 2 + 1) * kBlockDim + x * 2] = r10;
      temp[(y * 2 + 1) * kBlockDim + x * 2 + 1] = r11;
    }
  }
  for (size_t y = 0; y < S; y++) {
    for (size_t x = 0; x < S; x++) {
      out[y * stride_out + x] = temp[y * kBlockDim + x];
    }
  }
}

} // namespace

// Macros to index pixels, coefficients or temporary buffer with either x, y
// coordinates or with a single index. Short name on purpose but undefined
// below.
#define C(x, y) coefficients[(y)*kBlockDim + (x)]
#define P(x, y) pixels[(y)*pixels_stride + (x)]
#define T(x, y) temp[(y)*8 + (x)]
#define C1(i) C((i / 8), (i & 7))
#define P1(i) P((i / 8), (i & 7))
#define T1(i) temp[(i)]
#define ARRAYSIZE(a) sizeof(a) / sizeof(*a)

// These definitions are needed before C++17.
constexpr size_t AcStrategy::kMaxCoeffBlocks;
constexpr size_t AcStrategy::kMaxBlockDim;
constexpr size_t AcStrategy::kMaxCoeffArea;
constexpr size_t AcStrategy::kLLFMaskDim;

// Define hardcoded tables for the specific 45 degree case of the diagonal
// lines experiment.

// Indices of the 4 groups of pixels for the 4 DC's for the diagonal lines
// strategy.
static const size_t kLinesDc00indices[] = {0,  8,  9,  16, 17, 18, 25,
                                           26, 27, 34, 35, 36, 43, 44,
                                           45, 52, 53, 54, 61, 62, 63};
static const size_t kLinesDc01indices[] = {24, 32, 33, 40, 41, 42, 48, 49,
                                           50, 51, 56, 57, 58, 59, 60};
static const size_t kLinesDc10indices[] = {3,  4,  5,  6,  7,  12, 13, 14,
                                           15, 21, 22, 23, 30, 31, 39};
static const size_t kLinesDc11indices[] = {1,  2,  10, 11, 19, 20, 28,
                                           29, 37, 38, 46, 47, 55};
static const size_t kLinesNumDc00 = ARRAYSIZE(kLinesDc00indices);
static const size_t kLinesNumDc01 = ARRAYSIZE(kLinesDc01indices);
static const size_t kLinesNumDc10 = ARRAYSIZE(kLinesDc10indices);
static const size_t kLinesNumDc11 = ARRAYSIZE(kLinesDc11indices);

// Pixel indices of the different diagonal DCT's used in 8x8 block for the
// diagonal lines strategy.
static const size_t kLinesDct3indices[][3] = {
    {6, 7, 15}, {5, 14, 23}, {40, 49, 58}, {48, 56, 57}};
static const size_t kLinesDct4indices[][4] = {
    {4, 13, 22, 31}, {3, 12, 30, 39}, {24, 33, 51, 60}, {32, 41, 50, 59}};
static const size_t kLinesDct6indices[][6] = {{2, 11, 20, 29, 38, 47},
                                              {1, 10, 19, 37, 46, 55},
                                              {8, 17, 26, 44, 53, 62},
                                              {16, 25, 34, 43, 52, 61}};
static const size_t kLinesDct8indices[][8] = {{0, 9, 18, 27, 36, 45, 54, 63}};
static const size_t kLinesNumDct3 = ARRAYSIZE(kLinesDct3indices);
static const size_t kLinesNumDct4 = ARRAYSIZE(kLinesDct4indices);
static const size_t kLinesNumDct6 = ARRAYSIZE(kLinesDct6indices);
static const size_t kLinesNumDct8 = ARRAYSIZE(kLinesDct8indices);

// Coefficient indices of the different diagonal DCT's used in 8x8 block for the
// diagonal lines strategy.
static const size_t kLinesDctC3indices[][3] = {
    {6, 7, 15}, {5, 14, 23}, {40, 49, 58}, {48, 56, 57}};
static const size_t kLinesDctC4indices[][4] = {
    {4, 13, 22, 31}, {3, 12, 21, 30}, {24, 33, 42, 51}, {32, 41, 50, 59}};
static const size_t kLinesDctC6indices[][6] = {{2, 11, 20, 29, 38, 47},
                                               {10, 19, 28, 37, 46, 55},
                                               {17, 26, 35, 44, 53, 62},
                                               {16, 25, 34, 43, 52, 61}};
static const size_t kLinesDctC8indices[][8] = {
    {18, 27, 36, 45, 54, 39, 60, 63}};

// Computes and returns DC, and also subtracts it from the corresponding pixels.
static float ComputeDCPart(float *pixels, size_t pixels_stride,
                           const size_t *indices, size_t num) {
  float dc = 0;
  for (size_t i = 0; i < num; i++) {
    dc += P1(indices[i]);
  }
  dc /= num;
  for (size_t i = 0; i < num; i++) {
    P1(indices[i]) -= dc;
  }
  return dc;
}

static void RestoreDCPart(float *pixels, size_t pixels_stride,
                          const size_t *indices, size_t num, float dc) {
  for (size_t i = 0; i < num; i++) {
    P1(indices[i]) += dc;
  }
}

// Does the diagonal DCT's of size N as defined by the corresponding pixel and
// coefficient index arrays, for the diagonal lines strategy.
template <size_t N>
static void DoDCTs(const size_t indices_p[][N], const size_t indices_c[][N],
                   size_t num, const float *pixels, size_t pixels_stride,
                   float *coefficients) {
  for (size_t i = 0; i < num; i++) {
    for (size_t j = 0; j < N; j++) {
      C1(indices_c[i][j]) = P1(indices_p[i][j]);
    }
    // C++ has no static_if, so gives error when trying to index indices_c
    // directly, but turning it into a pointer fixes it.
    const size_t *indices = &indices_c[i][0];
    // Nothing to do for N == 1.
    if (N == 2) {
      DCT2(C1(indices[0]), C1(indices[1]));
    }
    if (N == 3) {
      DCT3(C1(indices[0]), C1(indices[1]), C1(indices[2]));
    }
    if (N == 4) {
      DCT4(C1(indices[0]), C1(indices[1]), C1(indices[2]), C1(indices[3]));
    }
    if (N == 6) {
      DCT6(C1(indices[0]), C1(indices[1]), C1(indices[2]), C1(indices[3]),
           C1(indices[4]), C1(indices[5]));
    }
    if (N == 8) {
      DCT8(C1(indices[0]), C1(indices[1]), C1(indices[2]), C1(indices[3]),
           C1(indices[4]), C1(indices[5]), C1(indices[6]), C1(indices[7]));
    }
  }
}

template <size_t N>
static void DoIDCTs(const size_t indices_p[][N], const size_t indices_c[][N],
                    size_t num, float *pixels, size_t pixels_stride,
                    const float *coefficients) {
  for (size_t i = 0; i < num; i++) {
    for (size_t j = 0; j < N; j++) {
      P1(indices_p[i][j]) = C1(indices_c[i][j]);
    }
    const size_t *indices = &indices_p[i][0];
    // Nothing to do for N == 1.
    if (N == 2) {
      IDCT2(P1(indices[0]), P1(indices[1]));
    }
    if (N == 3) {
      IDCT3(P1(indices[0]), P1(indices[1]), P1(indices[2]));
    }
    if (N == 4) {
      IDCT4(P1(indices[0]), P1(indices[1]), P1(indices[2]), P1(indices[3]));
    }
    if (N == 6) {
      IDCT6(P1(indices[0]), P1(indices[1]), P1(indices[2]), P1(indices[3]),
            P1(indices[4]), P1(indices[5]));
    }
    if (N == 8) {
      IDCT8(P1(indices[0]), P1(indices[1]), P1(indices[2]), P1(indices[3]),
            P1(indices[4]), P1(indices[5]), P1(indices[6]), P1(indices[7]));
    }
  }
}

SIMD_ATTR void AcStrategy::TransformFromPixels(
    const float *PIK_RESTRICT pixels, size_t pixels_stride,
    float *PIK_RESTRICT coefficients, size_t coefficients_stride) const {

  if (block_ != 0)
    return;
  switch (strategy_) {
  case Type::LINES: {
    SIMD_ALIGN float temp[kBlockDim * kBlockDim];

    for (size_t y = 0; y < 8; y++) {
      for (size_t x = 0; x < 8; x++) {
        C(x, y) = 0;
      }
    }

    for (size_t y = 0; y < 8; y++) {
      for (size_t x = 0; x < 8; x++) {
        T(x, y) = P(x, y);
      }
    }

    float dc00 = ComputeDCPart(temp, 8, kLinesDc00indices, kLinesNumDc00);
    float dc01 = ComputeDCPart(temp, 8, kLinesDc01indices, kLinesNumDc01);
    float dc10 = ComputeDCPart(temp, 8, kLinesDc10indices, kLinesNumDc10);
    float dc11 = ComputeDCPart(temp, 8, kLinesDc11indices, kLinesNumDc11);
    DCT2x2(dc00, dc01, dc10, dc11);

    C(0, 0) = dc00;
    C(0, 1) = dc01;
    C(1, 0) = dc10;
    C(1, 1) = dc11;

    DoDCTs<3>(kLinesDct3indices, kLinesDctC3indices, kLinesNumDct3, temp, 8,
              coefficients);
    DoDCTs<4>(kLinesDct4indices, kLinesDctC4indices, kLinesNumDct4, temp, 8,
              coefficients);
    DoDCTs<6>(kLinesDct6indices, kLinesDctC6indices, kLinesNumDct6, temp, 8,
              coefficients);
    DoDCTs<8>(kLinesDct8indices, kLinesDctC8indices, kLinesNumDct8, temp, 8,
              coefficients);
    break;
  }
  case Type::IDENTITY: {
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        float block_dc = 0;
        for (size_t iy = 0; iy < 4; iy++) {
          for (size_t ix = 0; ix < 4; ix++) {
            block_dc += pixels[(y * 4 + iy) * pixels_stride + x * 4 + ix];
          }
        }
        block_dc *= 1.0f / 16;
        for (size_t iy = 0; iy < 4; iy++) {
          for (size_t ix = 0; ix < 4; ix++) {
            if (ix == 1 && iy == 1)
              continue;
            coefficients[(y + iy * 2) * 8 + x + ix * 2] =
                pixels[(y * 4 + iy) * pixels_stride + x * 4 + ix] -
                pixels[(y * 4 + 1) * pixels_stride + x * 4 + 1];
          }
        }
        coefficients[(y + 2) * 8 + x + 2] = coefficients[y * 8 + x];
        coefficients[y * 8 + x] = block_dc;
      }
    }
    float block00 = coefficients[0];
    float block01 = coefficients[1];
    float block10 = coefficients[8];
    float block11 = coefficients[9];
    coefficients[0] = (block00 + block01 + block10 + block11) * 0.25f;
    coefficients[1] = (block00 + block01 - block10 - block11) * 0.25f;
    coefficients[8] = (block00 - block01 + block10 - block11) * 0.25f;
    coefficients[9] = (block00 - block01 - block10 + block11) * 0.25f;
    break;
  }
  case Type::DCT4X4_NOHF:
  case Type::DCT4X4: {
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        float block[4 * 4];
        ComputeTransposedScaledDCT<4>()(
            FromLines<4>(pixels + y * 4 * pixels_stride + x * 4, pixels_stride),
            ScaleToBlock<4>(block));

        //for(int k=0;k<16;k++)
        //std::cout<<"std_dct4_before: by="<<y<<" bx="<<x<<" "<<block[k]<<std::endl;

        for (size_t iy = 0; iy < 4; iy++) {
          for (size_t ix = 0; ix < 4; ix++) {
            coefficients[(y + iy * 2) * 8 + x + ix * 2] = block[iy * 4 + ix];
          }
        }
      }
    }
    float block00 = coefficients[0];
    float block01 = coefficients[1];
    float block10 = coefficients[8];
    float block11 = coefficients[9];

    coefficients[0] = (block00 + block01 + block10 + block11) * 0.25f;
    coefficients[1] = (block00 + block01 - block10 - block11) * 0.25f;
    coefficients[8] = (block00 - block01 + block10 - block11) * 0.25f;
    coefficients[9] = (block00 - block01 - block10 + block11) * 0.25f;

    //for(int k=0;k<64;k++)
      //std::cout<<"std_dct4_after: id="<<k<<" "<<coefficients[k]<<std::endl;

    break;
  }
  case Type::DCT2X2: {
    DCT2TopBlock<8>(pixels, pixels_stride, coefficients);
    DCT2TopBlock<4>(coefficients, kBlockDim, coefficients);
    DCT2TopBlock<2>(coefficients, kBlockDim, coefficients);
    break;
  }
  case Type::DCT16X16: {
    // TODO(veluca): Generalize ScaleToBlock and related classes to handle
    // non-contiguous blocks.
    SIMD_ALIGN float output[4 * kBlockDim * kBlockDim];
    ComputeTransposedScaledDCT<2 * kBlockDim>()(
        FromLines<2 * kBlockDim>(pixels, pixels_stride),
        ScaleToBlock<2 * kBlockDim>(output));

    //for(int k=0;k<256;k++)
      //std::cout<<"std_dct16: k="<<k<<" out="<<output[k]<<std::endl;
/*
    std::cout<<"dct16"<<std::endl;
    for(int by=0;by<2;by++){
    	for(int bx=0;bx<2;bx++){
    		for(int y=0;y<8;y++){
    			for(int x=0;x<8;x++){
    				std::cout<<output[by*128+y*16+bx*8+x]<<",";
    			}
    		}
    		std::cout<<std::endl;
    	}
    }

    std::cout<<"dct16_orig"<<std::endl;
    for(int by=0;by<16;by++){
    	for(int bx=0;bx<16;bx++){
    	    std::cout<<output[by*16+bx]<<",";
    	}
    	std::cout<<std::endl;
    }

    std::cout<<"coefficients_stride="<<coefficients_stride<<std::endl;
*/
    for (size_t i = 0; i < 2; i++) {
      memcpy(coefficients + coefficients_stride * i,
             output + 2 * kBlockDim * kBlockDim * i,
             sizeof(float) * 2 * kBlockDim * kBlockDim);
    }
    break;
  }
  case Type::DCT32X32: {
    // TODO(veluca): Generalize ScaleToBlock and related classes to handle
    // non-contiguous blocks.
    SIMD_ALIGN float output[16 * kBlockDim * kBlockDim];
    ComputeTransposedScaledDCT<4 * kBlockDim>()(
        FromLines<4 * kBlockDim>(pixels, pixels_stride),
        ScaleToBlock<4 * kBlockDim>(output));
    for (size_t i = 0; i < 4; i++) {
      memcpy(coefficients + coefficients_stride * i,
             output + 4 * kBlockDim * kBlockDim * i,
             sizeof(float) * 4 * kBlockDim * kBlockDim);
    }
    break;
  }
  case Type::DCT_NOHF:
  case Type::DCT: {
    ComputeTransposedScaledDCT<kBlockDim>()(
        FromLines<kBlockDim>(pixels, pixels_stride),
        ScaleToBlock<kBlockDim>(coefficients));
    break;
  }
  }
}

SIMD_ATTR void AcStrategy::TransformToPixels(const float *coefficients,
                                             size_t coefficients_stride,
                                             float *pixels,
                                             size_t pixels_stride) const {
  if (block_ != 0)
    return;
  switch (strategy_) {
  case Type::LINES: {
    SIMD_ALIGN float temp[kBlockDim * kBlockDim];

    for (size_t y = 0; y < 8; y++) {
      for (size_t x = 0; x < 8; x++) {
        T(x, y) = 0;
      }
    }

    DoIDCTs<3>(kLinesDct3indices, kLinesDctC3indices, kLinesNumDct3, temp, 8,
               coefficients);
    DoIDCTs<4>(kLinesDct4indices, kLinesDctC4indices, kLinesNumDct4, temp, 8,
               coefficients);
    DoIDCTs<6>(kLinesDct6indices, kLinesDctC6indices, kLinesNumDct6, temp, 8,
               coefficients);
    DoIDCTs<8>(kLinesDct8indices, kLinesDctC8indices, kLinesNumDct8, temp, 8,
               coefficients);

    float dc00 = C(0, 0);
    float dc01 = C(0, 1);
    float dc10 = C(1, 0);
    float dc11 = C(1, 1);

    IDCT2x2(dc00, dc01, dc10, dc11);

    RestoreDCPart(temp, 8, kLinesDc00indices, kLinesNumDc00, dc00);
    RestoreDCPart(temp, 8, kLinesDc01indices, kLinesNumDc01, dc01);
    RestoreDCPart(temp, 8, kLinesDc10indices, kLinesNumDc10, dc10);
    RestoreDCPart(temp, 8, kLinesDc11indices, kLinesNumDc11, dc11);

    // 4 pixels were not filled in, interpolate them here
    // TODO(lode): use bicubic interpolation, and support this in the general
    // case of working at any angle and fitting any size to any size.
    T1(21) = (T1(12) + T1(30)) * 0.5f;
    T1(28) = (T1(19) + T1(37)) * 0.5f;
    T1(35) = (T1(26) + T1(44)) * 0.5f;
    T1(42) = (T1(33) + T1(51)) * 0.5f;

    for (size_t y = 0; y < 8; y++) {
      for (size_t x = 0; x < 8; x++) {
        P(x, y) = T(x, y);
      }
    }
    break;
  }
  case Type::IDENTITY: {
    float dcs[4] = {};
    float block00 = coefficients[0];
    float block01 = coefficients[1];
    float block10 = coefficients[8];
    float block11 = coefficients[9];
    dcs[0] = block00 + block01 + block10 + block11;
    dcs[1] = block00 + block01 - block10 - block11;
    dcs[2] = block00 - block01 + block10 - block11;
    dcs[3] = block00 - block01 - block10 + block11;
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        float block_dc = dcs[y * 2 + x];
        float residual_sum = 0;
        for (size_t iy = 0; iy < 4; iy++) {
          for (size_t ix = 0; ix < 4; ix++) {
            if (ix == 0 && iy == 0)
              continue;
            residual_sum += coefficients[(y + iy * 2) * 8 + x + ix * 2];
          }
        }
        pixels[(4 * y + 1) * pixels_stride + 4 * x + 1] =
            block_dc - residual_sum * (1.0f / 16);
        for (size_t iy = 0; iy < 4; iy++) {
          for (size_t ix = 0; ix < 4; ix++) {
            if (ix == 1 && iy == 1)
              continue;
            pixels[(y * 4 + iy) * pixels_stride + x * 4 + ix] =
                coefficients[(y + iy * 2) * 8 + x + ix * 2] +
                pixels[(4 * y + 1) * pixels_stride + 4 * x + 1];
          }
        }
        pixels[y * 4 * pixels_stride + x * 4] =
            coefficients[(y + 2) * 8 + x + 2] +
            pixels[(4 * y + 1) * pixels_stride + 4 * x + 1];
      }
    }
    break;
  }
  case Type::DCT4X4_NOHF:
  case Type::DCT4X4: {
    float dcs[4] = {};
    float block00 = coefficients[0];
    float block01 = coefficients[1];
    float block10 = coefficients[8];
    float block11 = coefficients[9];
    dcs[0] = block00 + block01 + block10 + block11;
    dcs[1] = block00 + block01 - block10 - block11;
    dcs[2] = block00 - block01 + block10 - block11;
    dcs[3] = block00 - block01 - block10 + block11;
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        float block[4 * 4];
        block[0] = dcs[y * 2 + x];
        for (size_t iy = 0; iy < 4; iy++) {
          for (size_t ix = 0; ix < 4; ix++) {
            if (ix == 0 && iy == 0)
              continue;
            block[iy * 4 + ix] = coefficients[(y + iy * 2) * 8 + x + ix * 2];
          }
        }
        ComputeTransposedScaledIDCT<4>()(
            FromBlock<4>(block),
            ToLines<4>(pixels + y * 4 * pixels_stride + x * 4, pixels_stride));
      }
    }
    break;
  }
  case Type::DCT2X2: {
    SIMD_ALIGN float coeffs[kBlockDim * kBlockDim];
    memcpy(coeffs, coefficients, sizeof(float) * kBlockDim * kBlockDim);
    IDCT2TopBlock<2>(coeffs, kBlockDim, coeffs);
    IDCT2TopBlock<4>(coeffs, kBlockDim, coeffs);
    IDCT2TopBlock<8>(coeffs, kBlockDim, coeffs);
    for (size_t y = 0; y < kBlockDim; y++) {
      for (size_t x = 0; x < kBlockDim; x++) {
        pixels[y * pixels_stride + x] = coeffs[y * kBlockDim + x];
      }
    }
    break;
  }
  case Type::DCT16X16: {
    // TODO(veluca): Generalize ScaleToBlock and related classes to handle
    // non-contiguous blocks.
    SIMD_ALIGN float input[16 * kBlockDim * kBlockDim];
    for (size_t i = 0; i < 2; i++) {
      memcpy(input + 2 * kBlockDim * kBlockDim * i,
             coefficients + coefficients_stride * i,
             sizeof(float) * 2 * kBlockDim * kBlockDim);
    }
    ComputeTransposedScaledIDCT<2 * kBlockDim>()(
        FromBlock<2 * kBlockDim>(input),
        ToLines<2 * kBlockDim>(pixels, pixels_stride));
    break;
  }
  case Type::DCT32X32: {
    // TODO(veluca): Generalize ScaleToBlock and related classes to handle
    // non-contiguous blocks.
    SIMD_ALIGN float input[16 * kBlockDim * kBlockDim];
    for (size_t i = 0; i < 4; i++) {
      memcpy(input + 4 * kBlockDim * kBlockDim * i,
             coefficients + coefficients_stride * i,
             sizeof(float) * 4 * kBlockDim * kBlockDim);
    }
    ComputeTransposedScaledIDCT<4 * kBlockDim>()(
        FromBlock<4 * kBlockDim>(input),
        ToLines<4 * kBlockDim>(pixels, pixels_stride));
    break;
  }
  case Type::DCT_NOHF:
  case Type::DCT: {
    ComputeTransposedScaledIDCT<kBlockDim>()(
        FromBlock<kBlockDim>(coefficients),
        ToLines<kBlockDim>(pixels, pixels_stride));
    break;
  }
  }
}

#undef ARRAYSIZE
#undef C
#undef T
#undef P
#undef C1
#undef T1
#undef P1

SIMD_ATTR void AcStrategy::LowestFrequenciesFromDC(const float *PIK_RESTRICT dc,
                                                   size_t dc_stride, float *llf,
                                                   size_t llf_stride) const {
  if (block_)
    return;
  switch (strategy_) {
  case Type::DCT_NOHF:
  case Type::DCT:
  case Type::LINES: {
    llf[0] = dc[0];
    std::cout<<"std_dc_:"<<std::setprecision(8)<<dc[0]<<std::endl;
    break;
  }
  case Type::DCT16X16: {
    float tmp[4] = {};
    ReinterpretingDCT<2 * kBlockDim, 2, 2>(dc, dc_stride, tmp, 2);
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        llf[y * llf_stride + x] = tmp[y * 2 + x];
        //std::cout<<"dct16: y="<<y<<" x="<<x<<" dc="<<dc[y*dc_stride+x]<<" out="<<tmp[y*2+x]<<" dc_stride="<<dc_stride<<std::endl;
      }
    }
    break;
  }
  case Type::DCT32X32: {
    float tmp[16] = {};
    ReinterpretingDCT<4 * kBlockDim, 4, 4>(dc, dc_stride, tmp, 4);
    for (size_t y = 0; y < 4; y++) {
      for (size_t x = 0; x < 4; x++) {
        llf[y * llf_stride + x] = tmp[y * 4 + x];
        std::cout<<"dct32: y="<<y<<" x="<<x<<" dc="<<dc[y*dc_stride+x]<<" out="<<tmp[y*4+x]<<" dc_stride="<<dc_stride<<std::endl;
      }
    }
    break;
  }
  case Type::DCT2X2:
  case Type::DCT4X4_NOHF:
  case Type::DCT4X4:
  case Type::IDENTITY:
    llf[0] = dc[0];
    break;
  };
}

SIMD_ATTR void
AcStrategy::DCFromLowestFrequencies(const float *PIK_RESTRICT block,
                                    size_t block_stride, float *dc,
                                    size_t dc_stride) const {
  if (block_)
    return;
  switch (strategy_) {
  case Type::DCT_NOHF:
  case Type::DCT:
  case Type::LINES:
    dc[0] = block[0];
    break;
  case Type::DCT16X16: {
    float dest[4] = {};
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        dest[2 * y + x] = block[2 * kBlockDim * y + x];
        //std::cout<<"std_IDCT: id="<<2*y+x<<" value="<<dest[2*y+x]<<std::endl;
      }
    }
    ReinterpretingIDCT<2 * kBlockDim, 2, 2>(dest, 2, dc, dc_stride);
    break;
  }
  case Type::DCT32X32: {
    float dest[16] = {};
    for (size_t y = 0; y < 4; y++) {
      for (size_t x = 0; x < 4; x++) {
        dest[4 * y + x] = block[4 * kBlockDim * y + x];
      }
    }
    ReinterpretingIDCT<4 * kBlockDim, 4, 4>(dest, 4, dc, dc_stride);
    break;
  }
  case Type::DCT2X2:
  case Type::DCT4X4:
  case Type::DCT4X4_NOHF:
  case Type::IDENTITY:
    dc[0] = block[0];
    break;
  }
}

SIMD_ATTR void AcStrategy::DC2x2FromLowestFrequencies(
    const float *PIK_RESTRICT llf, size_t llf_stride, float *PIK_RESTRICT dc2x2,
    size_t dc2x2_stride) const {
  if (block_)
    return;
  constexpr size_t N = kBlockDim;
  switch (strategy_) {
  case Type::DCT_NOHF:
  case Type::DCT:
  case Type::LINES: {
    ReinterpretingIDCT<N, 1, 2>(llf, 0, dc2x2, dc2x2_stride);
    break;
  }
  case Type::DCT16X16: {
    float dest[16] = {};
    dest[0] = llf[0];
    dest[1] = llf[1];
    dest[4] = llf[llf_stride];
    dest[5] = llf[llf_stride + 1];
    ReinterpretingIDCT<2 * N, 2, 4>(dest, 4, dc2x2, dc2x2_stride);
    break;
  }
  case Type::DCT32X32: {
    float dest[64] = {};
    for (size_t iy = 0; iy < 4; iy++) {
      for (size_t ix = 0; ix < 4; ix++) {
        dest[iy * 8 + ix] = llf[iy * llf_stride + ix];
      }
    }
    ReinterpretingIDCT<4 * N, 4, 8>(dest, 8, dc2x2, dc2x2_stride);
    break;
  }
  case Type::DCT2X2:
  case Type::DCT4X4:
  case Type::DCT4X4_NOHF:
  case Type::IDENTITY:
    dc2x2[0] = llf[0];
    dc2x2[1] = llf[0];
    dc2x2[dc2x2_stride] = llf[0];
    dc2x2[dc2x2_stride + 1] = llf[0];
    break;
  }
}

SIMD_ATTR void AcStrategy::DC2x2FromLowFrequencies(const float *block,
                                                   size_t block_stride,
                                                   float *dc2x2,
                                                   size_t dc2x2_stride) const {
  if (block_)
    return;
  switch (strategy_) {
  case Type::DCT_NOHF:
  case Type::DCT:
  case Type::LINES:
    ReinterpretingIDCT<kBlockDim, 2, 2>(block, kBlockDim, dc2x2, dc2x2_stride);
    break;
  case Type::DCT16X16: {
    float dest[16] = {};
    for (size_t y = 0; y < 4; y++) {
      for (size_t x = 0; x < 4; x++) {
        dest[4 * y + x] = block[2 * kBlockDim * y + x];
      }
    }
    ReinterpretingIDCT<2 * kBlockDim, 4, 4>(dest, 4, dc2x2, dc2x2_stride);
    break;
  }
  case Type::DCT32X32: {
    float dest[64] = {};
    for (size_t y = 0; y < 8; y++) {
      for (size_t x = 0; x < 8; x++) {
        dest[8 * y + x] = block[4 * kBlockDim * y + x];
      }
    }
    ReinterpretingIDCT<4 * kBlockDim, 8, 8>(dest, 8, dc2x2, dc2x2_stride);
    break;
  }
  case Type::DCT2X2:
  case Type::DCT4X4:
  case Type::DCT4X4_NOHF:
  case Type::IDENTITY:
    float block00 = block[0];
    float block01 = block[1];
    float block10 = block[kBlockDim];
    float block11 = block[kBlockDim + 1];
    dc2x2[0] = block00 + block01 + block10 + block11;
    dc2x2[1] = block00 + block01 - block10 - block11;
    dc2x2[dc2x2_stride] = block00 - block01 + block10 - block11;
    dc2x2[dc2x2_stride + 1] = block00 - block01 - block10 + block11;
    break;
  }
}

SIMD_ATTR void AcStrategy::LowFrequenciesFromDC2x2(const float *dc2x2,
                                                   size_t dc2x2_stride,
                                                   float *block,
                                                   size_t block_stride) const {
  if (block_)
    return;
  switch (strategy_) {
  case Type::DCT_NOHF:
  case Type::DCT:
  case Type::LINES:
    ReinterpretingDCT<kBlockDim, 2, 2>(dc2x2, dc2x2_stride, block,
                                       block_stride);
    break;
  case Type::DCT16X16: {
    float dest[16] = {};
    ReinterpretingDCT<2 * kBlockDim, 4, 4>(dc2x2, dc2x2_stride, dest, 4);
    for (size_t y = 0; y < 4; y++) {
      for (size_t x = 0; x < 4; x++) {
        block[block_stride * y + x] = dest[y * 4 + x];
      }
    }
    break;
  }
  case Type::DCT32X32: {
    float dest[64] = {};
    ReinterpretingDCT<4 * kBlockDim, 8, 8>(dc2x2, dc2x2_stride, dest, 8);
    for (size_t y = 0; y < 8; y++) {
      for (size_t x = 0; x < 8; x++) {
        block[block_stride * y + x] = dest[y * 8 + x];
      }
    }
    break;
  }
  case Type::DCT2X2:
  case Type::DCT4X4:
  case Type::DCT4X4_NOHF:
  case Type::IDENTITY:
    float block00 = dc2x2[0];
    float block01 = dc2x2[1];
    float block10 = dc2x2[dc2x2_stride];
    float block11 = dc2x2[dc2x2_stride + 1];
    block[0] = (block00 + block01 + block10 + block11) * 0.25f;
    block[1] = (block00 + block01 - block10 - block11) * 0.25f;
    block[block_stride] = (block00 - block01 + block10 - block11) * 0.25f;
    block[block_stride + 1] = (block00 - block01 - block10 + block11) * 0.25f;
  }
}

void AcStrategyImage::SetFromRaw(const Rect &rect, const ImageB &raw_layers) {
  PIK_ASSERT(rect.IsInside(layers_));
  PIK_ASSERT(rect.xsize() <= raw_layers.xsize());
  PIK_ASSERT(rect.ysize() <= raw_layers.ysize());
  size_t stride = layers_.PixelsPerRow();
  for (size_t y = 0; y < rect.ysize(); ++y) {
    uint8_t *PIK_RESTRICT row = rect.Row(&layers_, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      row[x] = INVALID;
    }
  }
  for (size_t y = 0; y < rect.ysize(); ++y) {
    const uint8_t *PIK_RESTRICT row_in = raw_layers.Row(y);
    uint8_t *PIK_RESTRICT row = rect.Row(&layers_, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      if (row[x] != INVALID)
        continue;
      uint8_t raw_strategy = row_in[x];
#ifdef ADDRESS_SANITIZER
      PIK_ASSERT(AcStrategy::IsRawStrategyValid(raw_strategy));
#endif
      AcStrategy acs = AcStrategy::FromRawStrategy(raw_strategy);
#ifdef ADDRESS_SANITIZER
      PIK_ASSERT(y + acs.covered_blocks_y() <= rect.ysize());
      PIK_ASSERT(x + acs.covered_blocks_x() <= rect.xsize());
#endif
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          row[x + ix + iy * stride] =
              (raw_strategy << 4) | (iy * acs.covered_blocks_x() + ix);
        }
      }
    }
  }
}

void AcStrategyImage::SetFromArray(const Rect &rect, uint32_t data[]) {
  size_t stride = layers_.PixelsPerRow();
  for (size_t y = 0; y < rect.ysize(); ++y) {
    uint8_t *PIK_RESTRICT row = rect.Row(&layers_, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      uint8_t raw_strategy = data[y*rect.ysize()+x];
      row[x + y * stride] = raw_strategy;
    }
  }
}

size_t AcStrategyImage::CountBlocks(AcStrategy::Type type) const {
  size_t ret = 0;
  for (size_t y = 0; y < layers_.ysize(); y++) {
    const uint8_t *PIK_RESTRICT row = layers_.ConstRow(y);
    for (size_t x = 0; x < layers_.xsize(); x++) {
      if (row[x] == (static_cast<uint8_t>(type) << 4))
        ret++;
    }
  }
  return ret;
}

SIMD_ATTR void FindBestAcStrategy(float butteraugli_target,
                                  const ImageF *quant_field,
                                  const DequantMatrices &dequant,
                                  const Image3F &src, ThreadPool *pool,
                                  AcStrategyImage *ac_strategy,
                                  PikInfo *aux_out) {
  // TODO(veluca): this function does *NOT* know the actual quantization field
  // values, and thus is not able to make choices taking into account the actual
  // quantization matrix.
  PROFILER_FUNC;
  size_t xsize_blocks = src.xsize() / kBlockDim;
  size_t ysize_blocks = src.ysize() / kBlockDim;
  Image3F coeffs = Image3F(xsize_blocks * kBlockDim * kBlockDim, ysize_blocks);
  TransposedScaledDCT(src, &coeffs);
  *ac_strategy = AcStrategyImage(xsize_blocks, ysize_blocks);
  if (!kChooseAcStrategy) {
    return;
  }
  std::vector<bool> disable_dct16(xsize_blocks * ysize_blocks);
  std::vector<bool> disable_dct32(xsize_blocks * ysize_blocks);
  const auto disable_large_transforms = [&](int bx, int by) SIMD_ATTR {
    // If we find a well-fitting DCT4x4 within the larger block,
    // we disable the larger block.
    {
      std::vector<float> blockval(4);
      // 4x4 DCT needs less focus on B channel, since at that resolution
      // blue needs to be correct only by average.
      static const float kColorWeights4x4[3] = {
          0.60349588292079182, 1.5435289569786645, 0.33080849938060852,
      };
      for (int ix = 0; ix < 4; ++ix) {
        float total_sum = 0;
        int offx = (ix & 1) * 4;
        int offy = (ix & 2) * 2;
        for (size_t c = 0; c < src.kNumPlanes; c++) {
          float sum = 0;
          for (size_t iy = 0; iy < 3; iy++) {
            const float *row0 =
                src.ConstPlaneRow(c, by * kBlockDim + offy + iy);
            const float *row1 =
                src.ConstPlaneRow(c, by * kBlockDim + offy + iy + 1);
            for (size_t dx = 0; dx < 3; dx++) {
              int x = bx * kBlockDim + offx + dx;
              sum += fabs(row0[x] - row0[x + 1]) + fabs(row0[x] - row1[x]);
            }
            {
              int x = bx * kBlockDim + offx + 3;
              sum += fabs(row0[x] - row1[x]);
            }
          }
          int iy = 3;
          const float *row0 = src.ConstPlaneRow(c, by * kBlockDim + offy + iy);
          for (size_t dx = 0; dx < 3; dx++) {
            int x = bx * kBlockDim + offx + dx;
            sum += fabs(row0[x] - row0[x + 1]);
          }
          total_sum += kColorWeights4x4[c] * sum;
        }
        blockval[ix] = total_sum;
      }
      float norm2 = 0.0;
      float norm4 = 0.0;
      float norm8 = 0.0;
      for (int ix = 0; ix < 4; ++ix) {
        float v = blockval[ix];
        v *= v;
        norm2 += v;
        v *= v;
        norm4 += v;
        v *= v;
        norm8 += v;
      }
      norm2 = std::pow(norm2 * (1.0 / 4), 0.5);
      norm4 = std::pow(norm4 * (1.0 / 4), 0.25);
      norm8 = std::pow(norm8 * (1.0 / 4), 0.125);
      norm2 += 0.03;

      float kMul1 = 0.86101693093148191;
      float loss_4x4 = kMul1 * norm8 / norm2;
      float kMul2 = -0.18168363725368566;
      loss_4x4 += kMul2 * norm4 / norm2;
      static const float loss_4x4_limit0 = 1.0861540086721586;
      if (loss_4x4 >= loss_4x4_limit0) {
    	  std::cout<<"std_disable16: by="<<by<<" bx="<<bx<<std::endl;

        // Probably not multi-threading safe.
        disable_dct32[(by & ~3) * xsize_blocks + (bx & ~3)] = true;
        disable_dct16[(by & ~1) * xsize_blocks + (bx & ~1)] = true;
      }
    }
  };
  const auto find_block_strategy = [&](int bx, int by) SIMD_ATTR {
#if ENABLE_DIAGONAL_LINES_EXPERIMENT
    return AcStrategy::Type::LINES;
#endif // ENABLE_DIAGONAL_LINES_EXPERIMENT
    // The quantized symbol distribution contracts with the increasing
    // butteraugli_target.
    const float discretization_factor =
        100 * (6.9654004856811754) / butteraugli_target;
    // A value below 1.0 to favor 8x8s when all things are equal.
    // 16x16 has wider reach of oscillations and this part of the
    // computation is not aware of visual masking. Inhomogeneous
    // visual masking will propagate accuracy further with 16x16 than
    // with 8x8 dcts.
    const float kFavor8x8Dct = 0.978192691479985;
    float kFavor8x8DctOver32x32 = 0.74742417168628905;
    if (butteraugli_target >= 6.0) {
      kFavor8x8DctOver32x32 = 0.737101360945845;
    }
    static const float kColorWeights[3] = {
        0.65285453568125873, 2.4740163893371157, 2.0140216656143393,
    };
    // DCT4X4
    {
      float blockval[4];
      // 4x4 DCT needs less focus on B channel, since at that resolution
      // blue needs to be correct only by average.
      static const float kColorWeights4x4[3] = {
          0.76084140985773008, 0.9344031093258709, 0.31536647913297183,
      };
      // DCT4X4 collection
      for (int ix = 0; ix < 4; ++ix) {
        float total_sum = 0;
        int offx = (ix & 1) * 4;
        int offy = (ix & 2) * 2;
        for (size_t c = 0; c < src.kNumPlanes; c++) {
          float sum = 0;
          for (size_t iy = 0; iy < 3; iy++) {
            const float *row0 =
                src.ConstPlaneRow(c, by * kBlockDim + offy + iy);
            const float *row1 =
                src.ConstPlaneRow(c, by * kBlockDim + offy + iy + 1);
            for (size_t dx = 0; dx < 3; dx++) {
              int x = bx * kBlockDim + offx + dx;
              sum += fabs(row0[x] - row0[x + 1]) + fabs(row0[x] - row1[x]);
            }
            {
              int x = bx * kBlockDim + offx + 3;
              sum += fabs(row0[x] - row1[x]);
            }
          }
          int iy = 3;
          const float *row0 = src.ConstPlaneRow(c, by * kBlockDim + offy + iy);
          for (size_t dx = 0; dx < 3; dx++) {
            int x = bx * kBlockDim + offx + dx;
            sum += fabs(row0[x] - row0[x + 1]);
          }
          total_sum += kColorWeights4x4[c] * sum;
        }
        blockval[ix] = total_sum;
      }
      float norm2 = 0.0;
      float norm4 = 0.0;
      float norm8 = 0.0;
      for (int ix = 0; ix < 4; ++ix) {
        float v = blockval[ix];
        v *= v;
        norm2 += v;
        v *= v;
        norm4 += v;
        v *= v;
        norm8 += v;
      }
      norm2 = std::pow(norm2 * (1.0 / 4), 0.5);
      norm4 = std::pow(norm4 * (1.0 / 4), 0.25);
      norm8 = std::pow(norm8 * (1.0 / 4), 0.125);
      norm2 += 0.03;

      float kMul1 = 0.84695221371792806;
      float loss_4x4 = kMul1 * norm8 / norm2;
      float kMulCross = 0.24239613587680031;
      float kMul2 = -0.012220022434342694;
      loss_4x4 += kMul2 * norm4 / norm2;
      static const float loss_4x4_limit0 = 1.079485914917413;
      if (loss_4x4 >= loss_4x4_limit0) {
        return AcStrategy::Type::DCT4X4;
      }
    }

    const float kPow = 0.99263297216052859;
    const float kPow2 = 0.018823021573462634;
    const float kExtremityWeight16x16 = 7.77;
    const float kExtremityWeight32x32 = 7.77;
    // DCT32
    if (!disable_dct32[by * xsize_blocks + bx] && bx + 3 < xsize_blocks &&
        by + 3 < ysize_blocks && (bx & 3) == 0 && (by & 3) == 0) {
      static const float kDiff = 0.9539527585329598;
      float dct8x8_entropy = 0;
      for (size_t c = 0; c < coeffs.kNumPlanes; c++) {
        float entropy = 0;
        float min_ext = 1e30;
        float max_ext = -1e30;
        for (size_t iy = 0; iy < 4 && by + iy < ysize_blocks; iy++) {
          const float *row = coeffs.ConstPlaneRow(c, by + iy);
          for (size_t ix = 0; ix < 4 && bx + ix < xsize_blocks; ix++) {
            float min8x8 = 1e30;
            float max8x8 = -1e30;
            for (int dy = 0; dy < 8; ++dy) {
              const float *row =
                  src.ConstPlaneRow(c, (by + iy) * kBlockDim + dy);
              for (int dx = 0; dx < 8; ++dx) {
                float v = row[(bx + ix) * kBlockDim + dx];
                if (v < min8x8)
                  min8x8 = v;
                if (v > max8x8)
                  max8x8 = v;
              }
            }
            float ext = max8x8 - min8x8;
            if (ext < min_ext)
              min_ext = ext;
            if (ext > max_ext)
              max_ext = ext;
          }
          int bx_actual = bx;
          for (size_t ix = 1; ix < kBlockDim * kBlockDim * 4; ix++) {
            // Skip the dc values at 0 and 64.
            if ((ix & 63) == 0) {
              bx_actual++;
              continue;
            }
            float mul = 1.0f / dequant.Matrix(0, kQuantKindDCT8, c)[ix & 63];
            float val = mul * row[bx * kBlockDim * kBlockDim + ix];
            val *= quant_field->ConstRow(by + iy)[bx_actual];
            float v = fabsf(val) * discretization_factor;
            entropy += 1 + kDiff - pow(kPow, v) - kDiff * pow(kPow2, v);
          }
        }
        entropy -= kExtremityWeight32x32 * (max_ext - min_ext);
        dct8x8_entropy += kColorWeights[c] * entropy;
      }

      float quant_inhomogeneity = 0;
      float max_quant = -1e30;
      for (int dy = 0; dy < 4; ++dy) {
        for (int dx = 0; dx < 4; ++dx) {
          float quant = quant_field->ConstRow(by + dy)[bx + dx];
          max_quant = std::max(max_quant, quant);
          quant_inhomogeneity -= quant;
        }
      }
      quant_inhomogeneity += 16 * max_quant;
      float kMulInho = (-47.780 * (-4.270639713545533)) / butteraugli_target;
      dct8x8_entropy += kMulInho * quant_inhomogeneity;
      float dct32x32_entropy = 0;
      for (size_t c = 0; c < src.kNumPlanes; c++) {
        float entropy = 0;
        SIMD_ALIGN float dct32x32[16 * kBlockDim * kBlockDim] = {};
        AcStrategy acs(AcStrategy::Type::DCT32X32, 0);
        acs.TransformFromPixels(
            src.PlaneRow(c, kBlockDim * by) + kBlockDim * bx,
            src.PixelsPerRow(), dct32x32, 4 * kBlockDim * kBlockDim);
        for (size_t k = 0; k < 16 * kBlockDim * kBlockDim; k++) {
          if (k < 4 || (k < 36 && k > 31) || (k < 68 && k > 63) ||
              (k < 100 && k > 95)) {
            // Leave out the lowest frequencies.
            continue;
          }
          float mul = 1.0f / dequant.Matrix(0, kQuantKindDCT32, c)[k];
          float val = mul * dct32x32[k];
          val *= max_quant;
          float v = fabsf(val) * discretization_factor;
          entropy += 1 + kDiff - pow(kPow, v) - kDiff * pow(kPow2, v);
        }
        dct32x32_entropy += kColorWeights[c] * entropy;
      }
      if (dct32x32_entropy < kFavor8x8DctOver32x32 * dct8x8_entropy) {
        return AcStrategy::Type::DCT32X32;
      }
    }

    // DCT16
    if (!disable_dct16[by * xsize_blocks + bx] && bx + 1 < xsize_blocks &&
        by + 1 < ysize_blocks && (bx & 1) == 0 && (by & 1) == 0) {
      static const float kDiff = 0.2494383590606063;
      float dct8x8_entropy = 0;
      for (size_t c = 0; c < coeffs.kNumPlanes; c++) {
        float entropy = 0;
        float min_ext = 1e30;
        float max_ext = -1e30;
        for (size_t iy = 0; iy < 2 && by + iy < ysize_blocks; iy++) {
          for (size_t ix = 0; ix < 2 && bx + ix < xsize_blocks; ix++) {
            float min8x8 = 1e30;
            float max8x8 = -1e30;
            for (int dy = 0; dy < 8; ++dy) {
              const float *row =
                  src.ConstPlaneRow(c, (by + iy) * kBlockDim + dy);
              for (int dx = 0; dx < 8; ++dx) {
                float v = row[(bx + ix) * kBlockDim + dx];
                if (v < min8x8)
                  min8x8 = v;
                if (v > max8x8)
                  max8x8 = v;
              }
            }
            float ext = max8x8 - min8x8;
            if (ext < min_ext)
              min_ext = ext;
            if (ext > max_ext)
              max_ext = ext;
          }
          const float *row = coeffs.ConstPlaneRow(c, by + iy);
          int bx_actual = bx;
          for (size_t ix = 1; ix < kBlockDim * kBlockDim * 2; ix++) {
            // Skip the dc values at 0 and 64.
            if (ix == 64) {
              bx_actual++;
              continue;
            }
            float mul = 1.0f / dequant.Matrix(0, kQuantKindDCT8, c)[ix & 63];
            float val = mul * row[bx * kBlockDim * kBlockDim + ix];
            val *= quant_field->ConstRow(by + iy)[bx_actual];
            float v = fabsf(val) * discretization_factor;
            entropy += 1 + kDiff - pow(kPow, v) - kDiff * pow(kPow2, v);
          }
        }
        entropy -= kExtremityWeight16x16 * (max_ext - min_ext);
        dct8x8_entropy += kColorWeights[c] * entropy;
      }
      float max_quant = std::max<float>(
          std::max<float>(quant_field->ConstRow(by)[bx],
                           quant_field->ConstRow(by)[bx + 1]),
          std::max<float>(quant_field->ConstRow(by + 1)[bx],
                           quant_field->ConstRow(by + 1)[bx + 1]));
      float quant_inhomogeneity =
          4 * max_quant -
          (quant_field->ConstRow(by)[bx] + quant_field->ConstRow(by)[bx + 1] +
           quant_field->ConstRow(by + 1)[bx] +
           quant_field->ConstRow(by + 1)[bx + 1]);
      float kMulInho = (-47.780 * (3.9429727851421288)) / butteraugli_target;
      dct8x8_entropy += kMulInho * quant_inhomogeneity;
      float dct16x16_entropy = 0;
      for (size_t c = 0; c < src.kNumPlanes; c++) {
        float entropy = 0;
        SIMD_ALIGN float dct16x16[4 * kBlockDim * kBlockDim] = {};
        AcStrategy acs = AcStrategy(AcStrategy::Type::DCT16X16, 0);
        acs.TransformFromPixels(
            src.PlaneRow(c, kBlockDim * by) + kBlockDim * bx,
            src.PixelsPerRow(), dct16x16, 2 * kBlockDim * kBlockDim);
        for (size_t k = 0; k < 4 * kBlockDim * kBlockDim; k++) {
          if (k < 2 || (k < 18 && k > 15)) {
            // Leave out the lowest frequencies.
            continue;
          }
          float mul = 1.0f / dequant.Matrix(0, kQuantKindDCT16, c)[k];
          float val = mul * dct16x16[k];
          val *= max_quant;
          float v = fabsf(val) * discretization_factor;
          entropy += 1 + kDiff - pow(kPow, v) - kDiff * pow(kPow2, v);
          //std::cout << "std v=" <<v<<" entropy="<< entropy << std::endl;
        }
        dct16x16_entropy += kColorWeights[c] * entropy;
        //std::cout << "std entropy:" << entropy << std::endl;
      }

      //std::cout<<"std by="<<by<<" bx="<<bx<<" dct16x16_entropy="<<dct16x16_entropy<<" dct8x8_entropy="<<dct8x8_entropy<<std::endl;
      if (dct16x16_entropy < kFavor8x8Dct * dct8x8_entropy) {
        return AcStrategy::Type::DCT16X16;
      }
    }
    return AcStrategy::Type::DCT;
  };
  ImageB raw_ac_strategy(xsize_blocks, ysize_blocks);
  RunOnPool(pool, 0, ysize_blocks, [&](int y, int _) {
    for (size_t x = 0; x < xsize_blocks; x++) {
      disable_large_transforms(x, y);
    }
  });
  RunOnPool(pool, 0, ysize_blocks, [&](int y, int _) {
    uint8_t *PIK_RESTRICT row = raw_ac_strategy.Row(y);
    for (size_t x = 0; x < xsize_blocks; x++) {
      row[x] = static_cast<uint8_t>(find_block_strategy(x, y));
    }
  });

  ac_strategy->SetFromRaw(Rect(raw_ac_strategy), raw_ac_strategy);
  if (aux_out != nullptr) {
    aux_out->num_dct2_blocks =
        ac_strategy->CountBlocks(AcStrategy::Type::DCT2X2);
    aux_out->num_dct4_blocks =
        ac_strategy->CountBlocks(AcStrategy::Type::DCT4X4);
    aux_out->num_dct16_blocks =
        ac_strategy->CountBlocks(AcStrategy::Type::DCT16X16);
    aux_out->num_dct32_blocks =
        ac_strategy->CountBlocks(AcStrategy::Type::DCT32X32);
  }
  if (ac_strategy->CountBlocks(AcStrategy::Type::DCT) ==
      xsize_blocks * ysize_blocks) {
    *ac_strategy = AcStrategyImage(xsize_blocks, ysize_blocks);
  }
  if (WantDebugOutput(aux_out)) {
    aux_out->DumpImage("ac_strategy_type", ac_strategy->ConstRaw());
  }
}

} // namespace pik
