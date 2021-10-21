// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// Empty if not included by foreach_target.
#ifdef SIMD_ATTR_IMPL

#ifndef EPF_NEW_SIGMA
#define EPF_NEW_SIGMA 0
#endif
#ifndef EPF_INDEP_RANGE
#define EPF_INDEP_RANGE 0
#endif

namespace pik {
namespace SIMD_NAMESPACE {
namespace {

using D16 = SIMD_FULL(int16_t);
using DF = SIMD_FULL(float);
const D16 d16;
const DF df;
using V16 = D16::V;
using VF = DF::V;

// Number of extra pixels on the top/bottom/left/right edges of the "guide" and
// "in" images relative to "out".
static constexpr int kBorder = 6;  // = Quad radius(2) + reference radius(4)

static constexpr int kSigmaShift = EdgePreservingFilter::kSigmaShift;
static constexpr int kMinSigma = EdgePreservingFilter::kMinSigma;
static constexpr int kMaxSigma = EdgePreservingFilter::kMaxSigma;

static constexpr float kFlushWeightToZeroIfBelow = 0.05f;

//------------------------------------------------------------------------------
// Distance: sum of absolute differences on patches

class Distance {
 public:
  // "Patches" are 3x4 areas with top-left pixel northwest of the reference
  // pixel or its 7x8 neighbors. The 4-pixel width ("quad") is dictated by
  // MPSADBW.
  static constexpr int kPatchArea = 4 * 3;

  static constexpr size_t kNeighbors = 7 * 8;

  // Maximum possible sum of 8-bit differences, used in tests.
  static constexpr int kMaxSAD = kPatchArea * 255;  // = 3060

  static SIMD_ATTR SIMD_INLINE void SumsOfAbsoluteDifferences(
      const uint8_t* SIMD_RESTRICT guide_m4, const size_t guide_stride,
      int16_t* SIMD_RESTRICT sad) {
#if SIMD_TARGET_VALUE == SIMD_NONE
    // 7x8 reference pixels (total search window: 9x11)
    // 56 * 12 * 3 = 2016 ops per pixel, counting abs as one op.
    for (int cy = -3; cy <= 3; ++cy) {
      for (int cx = -3; cx <= 4; ++cx) {
        int sad_sum = 0;
        // 3x4 patch
        for (int iy = -1; iy <= 1; ++iy) {
          const uint8_t* row_ref = guide_m4 + (iy + 4) * guide_stride;
          const uint8_t* row_wnd = guide_m4 + (cy + iy + 4) * guide_stride;
          for (int ix = -1; ix <= 2; ++ix) {
            sad_sum += std::abs(row_ref[ix] - row_wnd[cx + ix]);
          }
        }

        sad[(cy + 3) * 8 + cx + 3] = static_cast<int16_t>(sad_sum);
      }
    }
#elif SIMD_TARGET_VALUE != SIMD_AVX2
    const SIMD_PART(uint8_t, 16) d8;
    const SIMD_PART(int16_t, 8) d16;
    const SIMD_PART(uint32_t, 4) d32;
    const SIMD_PART(uint64_t, 2) d64;

    // Offset to the leftmost pixel of the search window.
    const int kWindow = -4;  // Starts at row0

    const uint8_t* SIMD_RESTRICT row0 = guide_m4;
    const uint8_t* SIMD_RESTRICT row1 = guide_m4 + 1 * guide_stride;
    const uint8_t* SIMD_RESTRICT row2 = guide_m4 + 2 * guide_stride;
    const uint8_t* SIMD_RESTRICT row3 = guide_m4 + 3 * guide_stride;
    const uint8_t* SIMD_RESTRICT row4 = guide_m4 + 4 * guide_stride;
    const uint8_t* SIMD_RESTRICT row5 = guide_m4 + 5 * guide_stride;
    const uint8_t* SIMD_RESTRICT row6 = guide_m4 + 6 * guide_stride;
    const uint8_t* SIMD_RESTRICT row7 = guide_m4 + 7 * guide_stride;
    const uint8_t* SIMD_RESTRICT row8 = guide_m4 + 8 * guide_stride;

    const uint8_t* ref_pos_t = row3 - 1;

    // "ref" := one four-byte quad from three rows (t/m/b = top/middle/bottom),
    // assembled into 128 bits.
    // Gather would be faster on SKX, but on HSW we reduce port 5 pressure by
    // loading m and b MINUS 4 and 8 bytes to shift those quads upwards.
    // This is safe because we're only shifting m and b => there are valid
    // pixels to load from the previous row. x = don't care/ignored.
    const auto ref_xxT = load_dup128(d8, ref_pos_t);
    const auto ref_xMx = load_dup128(d8, ref_pos_t + guide_stride - 4);
    const auto ref_Bxx = load_dup128(d8, ref_pos_t + 2 * guide_stride - 8);

    // 3 patch rows x 7 window rows (m3 to p3) = 21x 128-bit SAD.
    const auto wnd_p2 = load_unaligned(d8, row6 + kWindow);
    const auto wnd_p3 = load_unaligned(d8, row7 + kWindow);
    const auto wnd_p4 = load_unaligned(d8, row8 + kWindow);

    const auto ref_xMT =
        cast_to(d8, odd_even(cast_to(d32, ref_xMx), cast_to(d32, ref_xxT)));
    const auto ref =
        cast_to(d8, odd_even(cast_to(d64, ref_Bxx), cast_to(d64, ref_xMT)));

    // MPSADBW is 3 uops (p0 + 2p5) and 6 bytes.
    auto sad_6t = ext::mpsadbw<0>(wnd_p2, ref);
    const auto wnd_p0 = load_unaligned(d8, row4 + kWindow);
    const auto wnd_p1 = load_unaligned(d8, row5 + kWindow);

    const auto sad_6m = ext::mpsadbw<1>(wnd_p3, ref);
    const auto wnd_m2 = load_unaligned(d8, row2 + kWindow);

    const auto sad_6b = ext::mpsadbw<2>(wnd_p4, ref);
    // Begin adding together the SAD results from each of the t/m/b rows.
    sad_6t += sad_6m;
    const auto wnd_m1 = load_unaligned(d8, row3 + kWindow);

    auto sad_5t = ext::mpsadbw<0>(wnd_p1, ref);
    auto sad_4m = ext::mpsadbw<1>(wnd_p1, ref);
    sad_6t += sad_6b;
    const auto wnd_m4 = load_unaligned(d8, row0 + kWindow);

    const auto sad_5m = ext::mpsadbw<1>(wnd_p2, ref);
    const auto sad_4b = ext::mpsadbw<2>(wnd_p2, ref);

    const auto sad_5b = ext::mpsadbw<2>(wnd_p3, ref);
    const auto sad_4t = ext::mpsadbw<0>(wnd_p0, ref);

    auto sad_3t = ext::mpsadbw<0>(wnd_m1, ref);
    auto sad_2m = ext::mpsadbw<1>(wnd_m1, ref);
    sad_5t += sad_5m;
    sad_4m += sad_4b;
    const auto wnd_m3 = load_unaligned(d8, row1 + kWindow);

    const auto sad_3m = ext::mpsadbw<1>(wnd_p0, ref);
    const auto sad_2b = ext::mpsadbw<2>(wnd_p0, ref);
    sad_5t += sad_5b;
    sad_4m += sad_4t;

    const auto sad_3b = ext::mpsadbw<2>(wnd_p1, ref);
    const auto sad_2t = ext::mpsadbw<0>(wnd_m2, ref);

    auto sad_1b = ext::mpsadbw<2>(wnd_m1, ref);
    auto sad_0t = ext::mpsadbw<0>(wnd_m4, ref);
    sad_3t += sad_3m;
    sad_2m += sad_2b;

    const auto sad_1t = ext::mpsadbw<0>(wnd_m3, ref);
    const auto sad_0m = ext::mpsadbw<1>(wnd_m3, ref);
    sad_3t += sad_3b;
    sad_2m += sad_2t;

    const auto sad_1m = ext::mpsadbw<1>(wnd_m2, ref);
    const auto sad_0b = ext::mpsadbw<2>(wnd_m2, ref);

    sad_1b += sad_1t;
    sad_0t += sad_0m;
    sad_1b += sad_1m;
    sad_0t += sad_0b;

    store(sad_0t, d16, sad + 0 * d16.N);
    store(sad_1b, d16, sad + 1 * d16.N);
    store(sad_2m, d16, sad + 2 * d16.N);
    store(sad_3t, d16, sad + 3 * d16.N);
    store(sad_4m, d16, sad + 4 * d16.N);
    store(sad_5t, d16, sad + 5 * d16.N);
    store(sad_6t, d16, sad + 6 * d16.N);
#else   // AVX2
    const SIMD_FULL(uint8_t) d8;
    const SIMD_FULL(uint32_t) d32;
    const SIMD_FULL(uint64_t) d64;

    // Leftmost pixel of the search window and reference patch.
    const uint8_t* SIMD_RESTRICT wnd_pos_m4 = guide_m4 - 4;

    const uint8_t* SIMD_RESTRICT ref_pos_m1 = guide_m4 + 3 * guide_stride - 1;
    const size_t gbpr2 = 2 * guide_stride;
    const size_t gbpr4 = 4 * guide_stride;

    // "ref" := one four-byte quad from three rows (t/m/b = top/middle/bottom),
    // assembled into 128 bits, which are duplicated for use by SAD (its
    // arguments select which two quads/rows to use).
    // Gather would be faster on SKX, but on HSW we reduce port 5 pressure by
    // loading m and b MINUS 4 and 8 bytes to shift those quads upwards.
    // This is safe because we're only shifting m and b => there are valid
    // pixels to load from the previous row. x = don't care/ignored.
    const auto ref_xxT = load_dup128(d8, ref_pos_m1);
    const auto ref_xMx = load_dup128(d8, ref_pos_m1 + guide_stride - 4);
    const auto ref_Bxx = load_dup128(d8, ref_pos_m1 + gbpr2 - 8);

    // 3 patch rows x 7 window rows (m3 to p3) = 21x 128-bit SAD = 9 + 3 SAD(),
    // which requires windows to be duplicated into both 128-bit lanes.

    // SAD 10
    const auto ref_xMT =
        cast_to(d8, odd_even(cast_to(d32, ref_xMx), cast_to(d32, ref_xxT)));
    const auto wnd_m3 = load_dup128(d8, wnd_pos_m4 + 1 * guide_stride);
    const auto ref =
        cast_to(d8, odd_even(cast_to(d64, ref_Bxx), cast_to(d64, ref_xMT)));
    const auto wnd_m2 = load_dup128(d8, wnd_pos_m4 + gbpr2);
    auto sad_1t0m = ext::mpsadbw2<0, 1>(wnd_m3, ref);
    const auto wnd_m1 = load_dup128(d8, wnd_pos_m4 + 3 * guide_stride);
    const auto sad_1m0b = ext::mpsadbw2<1, 2>(wnd_m2, ref);
    const auto wnd_m4 = load_dup128(d8, wnd_pos_m4);
    const auto wnd_m1m4 = concat_hi_lo(wnd_m1, wnd_m4);
    const auto sad_1b0t = ext::mpsadbw2<2, 0>(wnd_m1m4, ref);
    sad_1t0m += sad_1m0b;
    sad_1t0m += sad_1b0t;
    store(sad_1t0m, d16, sad + 0 * d16.N);

    // SAD 32
    const auto wnd_p0 = load_dup128(d8, wnd_pos_m4 + gbpr4);
    const auto wnd_p1 = load_dup128(d8, wnd_pos_m4 + 5 * guide_stride);
    auto sad_3t2m = ext::mpsadbw2<0, 1>(wnd_m1, ref);
    const auto wnd_p1m2 = concat_hi_lo(wnd_p1, wnd_m2);
    const auto sad_3m2b = ext::mpsadbw2<1, 2>(wnd_p0, ref);
    const auto sad_3b2t = ext::mpsadbw2<2, 0>(wnd_p1m2, ref);
    sad_3t2m += sad_3m2b;
    sad_3t2m += sad_3b2t;
    store(sad_3t2m, d16, sad + 1 * d16.N);

    // SAD 54
    const auto wnd_p2 = load_dup128(d8, wnd_pos_m4 + 6 * guide_stride);
    const auto wnd_p3 = load_dup128(d8, wnd_pos_m4 + 7 * guide_stride);
    const auto wnd_p3p0 = concat_hi_lo(wnd_p3, wnd_p0);
    auto sad_5t4m = ext::mpsadbw2<0, 1>(wnd_p1, ref);
    const auto sad_5m4b = ext::mpsadbw2<1, 2>(wnd_p2, ref);
    const auto sad_5b4t = ext::mpsadbw2<2, 0>(wnd_p3p0, ref);
    sad_5t4m += sad_5m4b;
    sad_5t4m += sad_5b4t;
    store(sad_5t4m, d16, sad + 2 * d16.N);

    const auto wnd_p4 = load_dup128(d8, wnd_pos_m4 + 8 * guide_stride);
    auto sad_6 = ext::mpsadbw2<0, 0>(wnd_p2, ref);  // t
    const auto sad_6m = ext::mpsadbw2<1, 1>(wnd_p3, ref);
    const auto sad_6b = ext::mpsadbw2<2, 2>(wnd_p4, ref);
    sad_6 += sad_6m;
    sad_6 += sad_6b;
    // Both 128-bit blocks are identical - required by SameBlocks().
    store(sad_6, d16, sad + 3 * d16.N);
#endif  // AVX2
  }
};

//------------------------------------------------------------------------------
// Exponentially decreasing weight functions

// Max such that mul_high(kClampedSAD << kShiftSAD, -32768) + bias=127*128 > 0.
// Also used by WeightExp to match WeightFast behavior at large distances.
// Doubling this maximum requires doubling kMinSigma.
constexpr int16_t kClampedSAD = 507;

// Straightforward but slow: computes e^{-s*x}.
class WeightExp {
 public:
  // W(sigma) = 0.5 = exp(mul_ * sigma) => mul_ = ln(0.5) / sigma.
  void SetSigma(const int sigma) {
    mul_ = (1 << kSigmaShift) * -0.69314717f / sigma;
  }

  SIMD_ATTR void operator()(const V16 sad, VF* SIMD_RESTRICT lo,
                            VF* SIMD_RESTRICT hi) const {
    const auto clamped = min(sad, set1(d16, kClampedSAD));
    SIMD_ALIGN int16_t sad_lanes[d16.N];
    store(clamped, d16, sad_lanes);
    SIMD_ALIGN float weight_lanes[d16.N];
    for (size_t i = 0; i < d16.N; ++i) {
      weight_lanes[i] = expf(sad_lanes[i] * mul_);
    }
    *lo = load(df, weight_lanes);
    *hi = load(df, weight_lanes + df.N);
  }

  // All blocks of "sad" are identical, but this function does not make use
  // of that.
  SIMD_ATTR VF SameBlocks(const V16 sad) const {
    const auto clamped = min(sad, set1(d16, kClampedSAD));
    SIMD_ALIGN int16_t sad_lanes[d16.N];
    store(clamped, d16, sad_lanes);
    // 1 for scalar, otherwise a full f32 vector.
    const size_t N = (d16.N + 1) / 2;
    float weight_lanes[N];
    for (size_t i = 0; i < N; ++i) {
      weight_lanes[i] = expf(sad_lanes[i] * mul_);
    }
    return load(df, weight_lanes);
  }

 private:
  float mul_;
};

// Fast approximation using the 2^x in the IEEE-754 representation.
class WeightFast {
 public:
  using D32 = SIMD_FULL(int32_t);

  SIMD_ATTR WeightFast() : bias_(set1(d16, 127 << (23 - 16))) {}

  SIMD_ATTR SIMD_INLINE void SetMul(const int mul) {
    EPF_ASSERT(-32768 <= mul && mul <= -1);
    mul_ = set1(d16, mul);
  }

  // Uses MulTable => must define after that class.
  void SetSigma(const int sigma);

  // Fills two f32 vectors from one i16 vector. On AVX2, "lo" are the lower
  // halves of two vectors (avoids crossing blocks).
  SIMD_ATTR SIMD_INLINE void operator()(const V16 sad, VF* SIMD_RESTRICT lo,
                                        VF* SIMD_RESTRICT hi) const {
    const auto zero = setzero(d16);

    // Avoid 16-bit overflow; ensures biased_exp >= 0.
    const auto clamped = min(sad, set1(d16, kClampedSAD));

    // Pre-shift to increase the multiplier range.
    const auto prescaled = shift_left<kShiftSAD>(clamped);

    // _Decrease_ to an unbiased exponent and fill in some mantissa bits.
    const auto unbiased_exp = ext::mul_high(prescaled, mul_);

    // Add exponent bias.
    auto biased_exp = unbiased_exp + bias_;

    // Assemble into an IEEE-754 representation with mantissa = zero.
    const auto bits_lo = zip_lo(zero, biased_exp);
    const auto bits_hi = zip_hi(zero, biased_exp);

    // Approximates exp(-s * sad).
    *lo = cast_to(df, bits_lo);
    *hi = cast_to(df, bits_hi);
  }

  // Same as above, but with faster i16x8->i32x8 conversion on AVX2 because all
  // blocks of "sad" are equal.
  SIMD_ATTR SIMD_INLINE VF SameBlocks(const V16 sad) const {
    const auto clamped = min(sad, set1(d16, kClampedSAD));
    const auto prescaled = shift_left<kShiftSAD>(clamped);
    const auto unbiased_exp = ext::mul_high(prescaled, mul_);
    const auto biased_exp = unbiased_exp + bias_;

#if SIMD_TARGET_VALUE == SIMD_AVX2
    // Both blocks of biased_exp are identical, so we can MOVZX + shift into
    // the upper 16 bits using a single-cycle shuffle.
    SIMD_ALIGN constexpr int32_t kHi32From16[8] = {
        0x0100FFFF, 0x0302FFFF, 0x0504FFFF, 0x0706FFFF,
        0x0908FFFF, 0x0B0AFFFF, 0x0D0CFFFF, 0x0F0EFFFF,
    };
    const auto bits = table_lookup_bytes(cast_to(D32(), biased_exp),
                                         load(D32(), kHi32From16));
#else
    const auto bits = zip_lo(setzero(d16), biased_exp);
#endif

    return cast_to(df, bits);
  }

 private:
  // Larger shift = higher precision but narrower range of permissible SAD
  // (limited by 16-bit overflow, see kClampedSAD).
  static constexpr int kShiftSAD = 6;

  const V16 bias_;  // Upper 16 bits of the IEEE-754 exponent bias.
  V16 mul_;         // Set by SetMul.
};

// Used by WeightFast. Monostate.
class MulTable {
 public:
  // Single-threaded.
  static SIMD_ATTR void Init() {
    if (mul_table_[0] != 0) return;  // Already initialized

    WeightFast weight_func;
    const int gap = 1 << kSigmaShift;
    int mul = -32768;
    for (int sigma = kMinSigma; sigma <= kMaxSigma; sigma += gap) {
      float w = 0.0f;
      for (; mul < 0; ++mul) {
        weight_func.SetMul(mul);
        const auto weight =
            weight_func.SameBlocks(set1(d16, sigma >> kSigmaShift));
        w = get_part(SIMD_PART(float, 1)(), weight);
        if (w > 0.5f) {
          break;
        }
      }
      mul_table_[sigma] = mul;
    }

    // Fill in (sigma, sigma + gap) via linear interpolation
    for (int sigma = kMinSigma; sigma < kMaxSigma; sigma += gap) {
      const float mul_step =
          (mul_table_[sigma + gap] - mul_table_[sigma]) / float(gap);
      for (int i = 1; i < gap; ++i) {
        mul_table_[sigma + i] = mul_table_[sigma] + i * mul_step;
      }
    }
  }

  static int Get(size_t sigma) {
    EPF_ASSERT(kMinSigma <= sigma && sigma <= kMaxSigma);
    EPF_ASSERT(mul_table_[sigma] != 0);
    return mul_table_[sigma];
  }

 private:
  static int mul_table_[kMaxSigma + 1];
};
int MulTable::mul_table_[kMaxSigma + 1];

SIMD_ATTR void WeightFast::SetSigma(const int sigma) {
  const int mul = MulTable::Get(sigma);
  EPF_ASSERT(mul != 0);  // Must have called MulTable::Init first.
  SetMul(mul);
}

// Slow, only use for tests.
SIMD_ATTR float GetWeightForTest(const WeightFast& weight_func, int sad) {
  PIK_ASSERT(0 <= sad && sad <= Distance::kMaxSAD);
  VF lo, hi;
  weight_func(set1(d16, sad), &lo, &hi);

  const SIMD_PART(float, 1) df1;
  const float w0 = get_part(df1, lo);
  const float w1 = get_part(df1, hi);
  PIK_CHECK(w0 == w1);
  return w0;
}

// (Must be in same file to use WeightFast etc.)
class InternalWeightTests {
 public:
  static void Run() {
    MulTable::Init();
    TestEndpoints();
    TestWeaklyMonotonicallyDecreasing();
    TestFastMatchesExp();
  }

 private:
  // Returns weight, or aborts.
  static SIMD_ATTR float EnsureWeightEquals(const float expected,
                                            const int16_t sad, const int sigma,
                                            const WeightFast& weight_func,
                                            const float tolerance) {
    const float w = GetWeightForTest(weight_func, sad);
    if (std::abs(w - expected) > tolerance) {
      fprintf(stderr, "Weight %f too far from %f for sigma %d, sad %d\n", w,
              expected, sigma, sad);
      abort();
    }
    return w;
  }

  static void TestEndpoints() {
    WeightFast weight_func;
    // Only test at integral sigma because we can't represent fractional SAD,
    // and weight_{sigma+3}(sad) is too far from 0.5.
    for (int sigma = kMinSigma; sigma <= kMaxSigma; sigma += 1 << kSigmaShift) {
      weight_func.SetSigma(sigma);
      // Zero SAD => max weight 1.0
      EnsureWeightEquals(1.0f, 0, sigma, weight_func, 0.02f);
      // Half-width at half max => 0.5
      EnsureWeightEquals(0.5f, sigma >> kSigmaShift, sigma, weight_func, 0.02f);
    }
  }

  // WeightFast and WeightExp should return similar values.
  static SIMD_ATTR void TestFastMatchesExp() {
    WeightExp func_slow;
    WeightFast func;

    for (int sigma = kMinSigma; sigma <= kMaxSigma; ++sigma) {
      func_slow.SetSigma(sigma);
      func.SetSigma(sigma);

      for (int sad = 0; sad <= Distance::kMaxSAD; ++sad) {
        VF lo_slow, unused;
        func_slow(set1(d16, sad), &lo_slow, &unused);
        const float weight_slow = get_part(SIMD_PART(float, 1)(), lo_slow);
        // Max tolerance is required for very low sigma (0.75 vs 0.707).
        EnsureWeightEquals(weight_slow, sad, sigma, func, 0.05f);
      }
    }
  }

  // Weight(sad + 1) <= Weight(sad).
  static SIMD_ATTR void TestWeaklyMonotonicallyDecreasing() {
    WeightFast weight_func;
    // half width at half max
    weight_func.SetSigma(30 << kSigmaShift);

    const SIMD_PART(float, 1) df1;

    float last_w = 1.1f;
    for (int sad = 1; sad <= kMaxSigma >> kSigmaShift; ++sad) {
      VF lo, hi;
      weight_func(set1(d16, sad), &lo, &hi);
      const float w = get_part(df1, lo);
      PIK_CHECK(w <= last_w);
      last_w = w;
    }
  }
};

//------------------------------------------------------------------------------

class WeightedSum {
 public:
  static constexpr size_t kNeighbors = Distance::kNeighbors;

  static void Test() { TestHorzSums(); }

  template <class WeightFunc>
  static SIMD_ATTR SIMD_INLINE void Compute(
      const uint8_t* SIMD_RESTRICT guide_m4_r,
      const uint8_t* SIMD_RESTRICT guide_m4_g,
      const uint8_t* SIMD_RESTRICT guide_m4_b, const size_t guide_stride,
      const float* SIMD_RESTRICT in_m3_r, const float* SIMD_RESTRICT in_m3_g,
      const float* SIMD_RESTRICT in_m3_b, const size_t in_stride,
      const WeightFunc& weight_func, float* SIMD_RESTRICT out_r,
      float* SIMD_RESTRICT out_g, float* SIMD_RESTRICT out_b) {
    SIMD_ALIGN float weights[kNeighbors];
    ComputeWeights(guide_m4_r, guide_m4_g, guide_m4_b, guide_stride,
                   weight_func, weights);

    const auto kMinWeight = set1(df, kFlushWeightToZeroIfBelow);
    for (size_t i = 0; i < kNeighbors; i += df.N) {
      auto v = load(df, weights + i);
      v &= v >= kMinWeight;
      store(v, df, weights + i);
    }

    // Joint weights are better than per-channel!
    FromWeights(in_m3_r, in_stride, weights, out_r);
    FromWeights(in_m3_g, in_stride, weights, out_g);
    FromWeights(in_m3_b, in_stride, weights, out_b);
  }

  static SIMD_INLINE void CopyOriginalBlock(const Image3F& in, const size_t x,
                                            const size_t y,
                                            Image3F* SIMD_RESTRICT out) {
    for (size_t iy = 0; iy < kBlockDim; ++iy) {
      CopyBlockRow(in, 0, x, y + iy, out);
      CopyBlockRow(in, 1, x, y + iy, out);
      CopyBlockRow(in, 2, x, y + iy, out);
    }
  }

 private:
  static SIMD_INLINE void CopyBlockRow(const Image3F& in, const size_t c,
                                       const size_t x, const size_t y,
                                       Image3F* SIMD_RESTRICT out) {
    const float* SIMD_RESTRICT in_row =
        in.ConstPlaneRow(c, y + kBorder) + kBorder;
    float* SIMD_RESTRICT out_row = out->PlaneRow(c, y);
    memcpy(out_row + x, in_row + x, kBlockDim * sizeof(*in_row));
  }

  // 2465 ops per pixel (2016 + 56 * (5 + 3) + 1)
  // NOTE: weights may be stored interleaved.
  template <class WeightFunc>
  static SIMD_ATTR SIMD_INLINE void WeightsFromSAD(
      const int16_t* SIMD_RESTRICT sad, const WeightFunc& weight_func,
      float* SIMD_RESTRICT weights) {
#if SIMD_TARGET_VALUE == SIMD_NONE
    for (size_t i = 0; i < kNeighbors; ++i) {
      const auto sad_v = set1(d16, sad[i]);
      VF lo, unused;
      weight_func(sad_v, &lo, &unused);
      store(lo, df, weights + i);
    }
#elif SIMD_TARGET_VALUE != SIMD_AVX2
    f32x4 w0L, w0H, w1L, w1H, w2L, w2H, w3L, w3H, w4L, w4H, w5L, w5H, w6L, w6H;
    weight_func(load(d16, sad + 0 * d16.N), &w0L, &w0H);
    weight_func(load(d16, sad + 1 * d16.N), &w1L, &w1H);
    weight_func(load(d16, sad + 2 * d16.N), &w2L, &w2H);
    weight_func(load(d16, sad + 3 * d16.N), &w3L, &w3H);
    weight_func(load(d16, sad + 4 * d16.N), &w4L, &w4H);
    weight_func(load(d16, sad + 5 * d16.N), &w5L, &w5H);
    weight_func(load(d16, sad + 6 * d16.N), &w6L, &w6H);
    store(w0L, df, weights + 0 * df.N);
    store(w0H, df, weights + 1 * df.N);
    store(w1L, df, weights + 2 * df.N);
    store(w1H, df, weights + 3 * df.N);
    store(w2L, df, weights + 4 * df.N);
    store(w2H, df, weights + 5 * df.N);
    store(w3L, df, weights + 6 * df.N);
    store(w3H, df, weights + 7 * df.N);
    store(w4L, df, weights + 8 * df.N);
    store(w4H, df, weights + 9 * df.N);
    store(w5L, df, weights + 10 * df.N);
    store(w5H, df, weights + 11 * df.N);
    store(w6L, df, weights + 12 * df.N);
    store(w6H, df, weights + 13 * df.N);
#else  // AVX2
    decltype(setzero(df)) w10L, w10H, w32L, w32H, w54L, w54H, w6;
    weight_func(load(d16, sad + 0 * d16.N), &w10L, &w10H);
    weight_func(load(d16, sad + 1 * d16.N), &w32L, &w32H);
    weight_func(load(d16, sad + 2 * d16.N), &w54L, &w54H);
    w6 = weight_func.SameBlocks(load(d16, sad + 3 * d16.N));
    store(w10L, df, weights + 0 * df.N);
    store(w10H, df, weights + 1 * df.N);
    store(w32L, df, weights + 2 * df.N);
    store(w32H, df, weights + 3 * df.N);
    store(w54L, df, weights + 4 * df.N);
    store(w54H, df, weights + 5 * df.N);
    store(w6, df, weights + 6 * df.N);
#endif
  }

  // Returns weights for 7x8 neighbor pixels
  template <class WeightFunc>
  static SIMD_ATTR SIMD_INLINE void ComputeWeights(
      const uint8_t* SIMD_RESTRICT guide_m4_r,
      const uint8_t* SIMD_RESTRICT guide_m4_g,
      const uint8_t* SIMD_RESTRICT guide_m4_b, const size_t guide_stride,
      const WeightFunc& weight_func, float* SIMD_RESTRICT weights) {
    // It's important to include all channels, only computing for X and Y
    // channels misses/weakens some edges.
    SIMD_ALIGN int16_t sad_r[64];
    SIMD_ALIGN int16_t sad_g[64];
    SIMD_ALIGN int16_t sad_b[64];
    Distance::SumsOfAbsoluteDifferences(guide_m4_r, guide_stride, &sad_r[0]);
    Distance::SumsOfAbsoluteDifferences(guide_m4_g, guide_stride, &sad_g[0]);
    Distance::SumsOfAbsoluteDifferences(guide_m4_b, guide_stride, &sad_b[0]);

    SIMD_FULL(int16_t) d;
    for (size_t i = 0; i < 64; i += d.N) {
      const auto d0 = load(d, &sad_r[i]);
      const auto d1 = load(d, &sad_g[i]);
      const auto d2 = load(d, &sad_b[i]);
      // Better than sum and sum/4.
      const auto combined = max(max(d0, d1), d2);
      store(combined, d, &sad_r[i]);
    }

    // We actually see better results from central distance 0 as opposed to
    // the minimum non-center (i.e. max weight).

    WeightsFromSAD(&sad_r[0], weight_func, weights);
  }

  // Returns sum(num) / sum(den).
  template <class V>
  static SIMD_ATTR SIMD_INLINE SIMD_PART(float, 1)::V
      RatioOfHorizontalSums(const V num, const V den) {
    const SIMD_PART(float, 1) d;
    // Faster than concat_lo_lo/hi_hi plus single sum_of_lanes.
    const auto sum_den = any_part(d, ext::sum_of_lanes(den));
    const auto sum_num = any_part(d, ext::sum_of_lanes(num));
    const auto rcp_den = approximate_reciprocal(sum_den);
    return rcp_den * sum_num;
  }

  static SIMD_ATTR SIMD_INLINE void FromWeights(
      const float* SIMD_RESTRICT in_m3, const size_t in_stride,
      const float* SIMD_RESTRICT weights, float* SIMD_RESTRICT out) {
#if SIMD_TARGET_VALUE == SIMD_NONE
    float weighted_sum = 0.0f;
    float sum_weights = 0.0f;
    int i = 0;
    for (int cy = -3; cy <= 3; ++cy) {
      const float* SIMD_RESTRICT in_row =
          ByteOffset(in_m3, (cy + 3) * in_stride);
      for (int cx = -3; cx <= 4; ++cx) {
        const float neighbor = in_row[cx];
        const float weight = weights[i++];
        weighted_sum += neighbor * weight;
        sum_weights += weight;
      }
    }

    // Safe because weights[27] == 1.
    *out = weighted_sum / sum_weights;
#elif SIMD_TARGET_VALUE != SIMD_AVX2
    in_m3 -= 3;

    const auto w0L = load(df, weights + 0 * df.N);
    const auto w0H = load(df, weights + 1 * df.N);
    const auto w1L = load(df, weights + 2 * df.N);
    const auto w1H = load(df, weights + 3 * df.N);
    const auto w2L = load(df, weights + 4 * df.N);
    const auto w2H = load(df, weights + 5 * df.N);
    const auto w3L = load(df, weights + 6 * df.N);
    const auto w3H = load(df, weights + 7 * df.N);
    const auto w4L = load(df, weights + 8 * df.N);
    const auto w4H = load(df, weights + 9 * df.N);
    const auto w5L = load(df, weights + 10 * df.N);
    const auto w5H = load(df, weights + 11 * df.N);
    const auto w6L = load(df, weights + 12 * df.N);
    const auto w6H = load(df, weights + 13 * df.N);

    const auto n0L = load_unaligned(df, ByteOffset(in_m3, 0 * in_stride));
    const auto n1L = load_unaligned(df, ByteOffset(in_m3, 1 * in_stride));
    const auto n2L = load_unaligned(df, ByteOffset(in_m3, 2 * in_stride));
    const auto n3L = load_unaligned(df, ByteOffset(in_m3, 3 * in_stride));
    const auto n4L = load_unaligned(df, ByteOffset(in_m3, 4 * in_stride));
    const auto n5L = load_unaligned(df, ByteOffset(in_m3, 5 * in_stride));
    const auto n6L = load_unaligned(df, ByteOffset(in_m3, 6 * in_stride));
    const auto n0H =
        load_unaligned(df, ByteOffset(in_m3, 0 * in_stride) + df.N);
    const auto n1H =
        load_unaligned(df, ByteOffset(in_m3, 1 * in_stride) + df.N);
    const auto n2H =
        load_unaligned(df, ByteOffset(in_m3, 2 * in_stride) + df.N);
    const auto n3H =
        load_unaligned(df, ByteOffset(in_m3, 3 * in_stride) + df.N);
    const auto n4H =
        load_unaligned(df, ByteOffset(in_m3, 4 * in_stride) + df.N);
    const auto n5H =
        load_unaligned(df, ByteOffset(in_m3, 5 * in_stride) + df.N);
    const auto n6H =
        load_unaligned(df, ByteOffset(in_m3, 6 * in_stride) + df.N);

    const auto sum_weights = w0L + w0H + w1L + w1H + w2L + w2H + w3L + w3H +
                             w4L + w4H + w5L + w5H + w6L + w6H;

    auto weighted_sum = n0L * w0L;
    weighted_sum = mul_add(n0H, w0H, weighted_sum);
    weighted_sum = mul_add(n1L, w1L, weighted_sum);
    weighted_sum = mul_add(n1H, w1H, weighted_sum);
    weighted_sum = mul_add(n2L, w2L, weighted_sum);
    weighted_sum = mul_add(n2H, w2H, weighted_sum);
    weighted_sum = mul_add(n3L, w3L, weighted_sum);
    weighted_sum = mul_add(n3H, w3H, weighted_sum);
    weighted_sum = mul_add(n4L, w4L, weighted_sum);
    weighted_sum = mul_add(n4H, w4H, weighted_sum);
    weighted_sum = mul_add(n5L, w5L, weighted_sum);
    weighted_sum = mul_add(n5H, w5H, weighted_sum);
    weighted_sum = mul_add(n6L, w6L, weighted_sum);
    weighted_sum = mul_add(n6H, w6H, weighted_sum);

    store(RatioOfHorizontalSums(weighted_sum, sum_weights),
          SIMD_PART(float, 1)(), out);
#else  // AVX2
    in_m3 -= 3;
    const size_t kN2 = df.N / 2;

    // Weighted sum 10
    const auto n0 = load_unaligned(df, ByteOffset(in_m3, 0 * in_stride));
    const auto n1L = load_dup128(df, ByteOffset(in_m3, 1 * in_stride));
    const auto n1H = load_dup128(df, ByteOffset(in_m3, 1 * in_stride) + kN2);
    const auto w10L = load(df, weights + 0 * df.N);
    const auto w10H = load(df, weights + 1 * df.N);
    const auto n10L = concat_hi_lo(n1L, n0);
    const auto n10H = concat_hi_hi(n1H, n0);
    const auto sum01 = w10L + w10H;
    const auto mul0 = n10L * w10L;
    const auto mul1 = n10H * w10H;

    // Weighted sum 32
    const auto n2 = load_unaligned(df, ByteOffset(in_m3, 2 * in_stride));
    const auto n3L = load_dup128(df, ByteOffset(in_m3, 3 * in_stride));
    const auto n3H = load_dup128(df, ByteOffset(in_m3, 3 * in_stride) + kN2);
    const auto w32L = load(df, weights + 2 * df.N);
    const auto w32H = load(df, weights + 3 * df.N);
    const auto n32L = concat_hi_lo(n3L, n2);
    const auto n32H = concat_hi_hi(n3H, n2);
    const auto sum23 = w32L + w32H;
    const auto mul02 = mul_add(n32L, w32L, mul0);
    const auto mul13 = mul_add(n32H, w32H, mul1);

    // Weighted sum 54
    const auto n4 = load_unaligned(df, ByteOffset(in_m3, 4 * in_stride));
    const auto n5L = load_dup128(df, ByteOffset(in_m3, 5 * in_stride));
    const auto n5H = load_dup128(df, ByteOffset(in_m3, 5 * in_stride) + kN2);
    const auto w54L = load(df, weights + 4 * df.N);
    const auto w54H = load(df, weights + 5 * df.N);
    const auto n54L = concat_hi_lo(n5L, n4);
    const auto n54H = concat_hi_hi(n5H, n4);
    const auto sum0123 = sum01 + sum23;
    const auto mul024 = mul_add(n54L, w54L, mul02);
    const auto sum45 = w54L + w54H;
    const auto mul135 = mul_add(n54H, w54H, mul13);

    const auto mul012345 = mul024 + mul135;
    const auto sum012345 = sum0123 + sum45;

    // Weighted sum 6
    const auto n6 = load_unaligned(df, ByteOffset(in_m3, 6 * in_stride));
    const auto w6 = load(df, weights + 6 * df.N);
    const auto weighted_sum = mul_add(n6, w6, mul012345);
    const auto sum_weights = sum012345 + w6;

    store(RatioOfHorizontalSums(weighted_sum, sum_weights),
          SIMD_PART(float, 1)(), out);
#endif
  }

  static SIMD_ATTR SIMD_INLINE float Reciprocal12(const float x) {
    const SIMD_PART(float, 1) d;
    return get_part(d, approximate_reciprocal(set_part(d, x)));
  }

  static SIMD_ATTR void TestHorzSums() {
    const SIMD_PART(float, 1) df1;

    SIMD_ALIGN const float in0_lanes[8] = {256.8f, 128.7f, 64.6f, 32.5f,
                                           16.4f,  8.3f,   4.2f,  2.1f};
    SIMD_ALIGN const float in1_lanes[8] = {-0.1f, -1.2f, -2.3f, -3.4f,
                                           -4.5f, -5.6f, -6.7f, -7.8f};
    for (size_t i = 0; i < 8; i += df.N) {
      const auto in0 = load(df, in0_lanes + i);
      const auto in1 = load(df, in1_lanes + i);

      const float expected0 =
          std::accumulate(in0_lanes + i, in0_lanes + i + df.N, 0.0f);
      const float expected1 =
          std::accumulate(in1_lanes + i, in1_lanes + i + df.N, 0.0f);
      const float expected = Reciprocal12(expected1) * expected0;

      const float actual = get_part(df1, RatioOfHorizontalSums(in0, in1));
      PIK_CHECK(std::abs(expected - actual) < 2E-2f);
    }
  }
};

// POD
class SIMD_ALIGN MinMaxWorker {
 public:
  SIMD_ATTR void Init(const Image3F* SIMD_RESTRICT in,
                      Image3F* SIMD_RESTRICT padded) {
    in_ = in;
    padded_ = padded;
    xsize_ = in->xsize();
    ysize_ = in->ysize();
    aligned_x_end_ = xsize_ - (xsize_ % df.N);

    for (int c = 0; c < 3; ++c) {
      store(set1(df, FLT_MAX), df, min_[c]);
      store(set1(df, -FLT_MAX), df, max_[c]);
      scalar_min_[c] = FLT_MAX;
      scalar_max_[c] = -FLT_MAX;
    }
  }

  // iy may be out of bounds (for padding).
  SIMD_ATTR void Run(int64_t iy) {
    const size_t y = (static_cast<size_t>(iy));  // assumes 2's complement
    if (PIK_LIKELY(y < ysize_)) {
      for (size_t c = 0; c < 3; ++c) {
        PadAndUpdate(c, y);
      }
    } else {
      for (size_t c = 0; c < 3; ++c) {
        PadTopBottomRow(c, iy);
      }
    }
  }

  SIMD_ATTR void Assimilate(const MinMaxWorker& other) {
    for (int c = 0; c < 3; ++c) {
      const auto min1 = load(df, min_[c]);
      const auto min2 = load(df, other.min_[c]);
      store(min(min1, min2), df, min_[c]);
      const auto max1 = load(df, max_[c]);
      const auto max2 = load(df, other.max_[c]);
      store(max(max1, max2), df, max_[c]);
      scalar_min_[c] = std::min(scalar_min_[c], other.scalar_min_[c]);
      scalar_max_[c] = std::max(scalar_max_[c], other.scalar_max_[c]);
    }
  }

  SIMD_ATTR void Finalize(std::array<float, 3>* PIK_RESTRICT min,
                          std::array<float, 3>* PIK_RESTRICT max) const {
    for (int c = 0; c < 3; ++c) {
      (*min)[c] =
          std::min(scalar_min_[c], *std::min_element(min_[c], min_[c] + df.N));
      (*max)[c] =
          std::max(scalar_max_[c], *std::max_element(max_[c], max_[c] + df.N));
    }
  }

 private:
  // Interior, y is valid.
  SIMD_ATTR void PadAndUpdate(const size_t c, const size_t y) {
    const float* SIMD_RESTRICT row_in = in_->ConstPlaneRow(c, y);
    float* SIMD_RESTRICT row_out = padded_->PlaneRow(c, y + kBorder) + kBorder;

    // Ensure store alignment (faster than loading aligned)
    constexpr int64_t aligned_begin = (kBorder + df.N - 1) & ~(df.N - 1);

    // Local copies avoid stores in each iteration. Part+min also leads to
    // better code than std::min (VUCOMISS + CMOV).
    const SIMD_PART(float, 1) d1;
    auto my_min1 = load(d1, &scalar_min_[c]);
    auto my_max1 = load(d1, &scalar_max_[c]);

    // Left: mirror and vector alignment
    int64_t ix = -kBorder;
    for (; ix < aligned_begin - kBorder; ++ix) {
      const int64_t clamped_x = Mirror(ix, xsize_);
      const auto in = load(d1, row_in + clamped_x);
      my_min1 = min(my_min1, in);
      my_max1 = max(my_max1, in);
      store(in, d1, row_out + ix);
    }

    // Interior: whole vectors
    auto my_min = load(df, min_[c]);
    auto my_max = load(df, max_[c]);
    for (; ix + df.N <= xsize_; ix += df.N) {
      const auto in = load_unaligned(df, row_in + ix);
      my_min = min(my_min, in);
      my_max = max(my_max, in);
      store(in, df, row_out + ix);
    }
    store(my_min, df, min_[c]);
    store(my_max, df, max_[c]);

    // Right: vector remainder and mirror
    for (; ix < xsize_ + kBorder; ++ix) {
      const int64_t clamped_x = Mirror(ix, xsize_);
      const auto in = load(d1, row_in + clamped_x);
      my_min1 = min(my_min1, in);
      my_max1 = max(my_max1, in);
      store(in, d1, row_out + ix);
    }

    store(my_min1, d1, &scalar_min_[c]);
    store(my_max1, d1, &scalar_max_[c]);
  }

  // Border, no need to update min/max from mirrored values.
  SIMD_ATTR void PadTopBottomRow(const size_t c, const int64_t iy) {
    const int64_t clamped_y = WrapMirror()(iy, ysize_);
    const float* SIMD_RESTRICT row_in = in_->ConstPlaneRow(c, clamped_y);
    float* SIMD_RESTRICT row_out = padded_->PlaneRow(c, iy + kBorder) + kBorder;

    // Ensure store alignment (faster than loading aligned)
    constexpr int64_t aligned_begin = (kBorder + df.N - 1) & ~(df.N - 1);

    // Left: mirror and vector alignment
    int64_t ix = -kBorder;
    for (; ix < aligned_begin - kBorder; ++ix) {
      const int64_t clamped_x = Mirror(ix, xsize_);
      row_out[ix] = row_in[clamped_x];
    }

    // Interior: whole vectors
    for (; ix + df.N <= xsize_; ix += df.N) {
      const auto src = load_unaligned(df, row_in + ix);
      store(src, df, row_out + ix);
    }

    // Right: vector remainder and mirror
    for (; ix < xsize_ + kBorder; ++ix) {
      const int64_t clamped_x = Mirror(ix, xsize_);
      row_out[ix] = row_in[clamped_x];
    }
  }

  SIMD_ALIGN float min_[3][df.N];
  SIMD_ALIGN float max_[3][df.N];
  const Image3F* SIMD_RESTRICT in_;  // not owned
  Image3F* SIMD_RESTRICT padded_;    // not owned
  size_t xsize_;
  size_t ysize_;
  size_t aligned_x_end_;
  float scalar_min_[3];
  float scalar_max_[3];
};
static_assert(sizeof(MinMaxWorker) % sizeof(DF::V) == 0, "Align");

// Returns a new image with kBorder additional pixels on each side initialized
// by mirroring.
SIMD_ATTR void MinMax(const Image3F& in, ThreadPool* pool,
                      std::array<float, 3>* SIMD_RESTRICT min,
                      std::array<float, 3>* SIMD_RESTRICT max,
                      Image3F* SIMD_RESTRICT padded) {
  PROFILER_FUNC;
  // A bit too large for the stack. Must be aligned for min_/max_ members.
  const size_t num_workers = NumThreads(pool);
  auto workers_mem = AllocateArray(num_workers * sizeof(MinMaxWorker));
  MinMaxWorker* workers = reinterpret_cast<MinMaxWorker*>(workers_mem.get());
  for (size_t i = 0; i < num_workers; ++i) {
    workers[i].Init(&in, padded);
  }

  // Includes padding. ThreadPool requires task >= 0.
  RunOnPool(pool, 0, in.ysize() + 2 * kBorder,
            [workers](const int task, const int thread) {
              workers[thread].Run(task - kBorder);
            });

  // Reduction
  for (size_t i = 1; i < num_workers; ++i) {
    workers[0].Assimilate(workers[i]);
  }
  workers[0].Finalize(min, max);
}

// Returns a guide image for "in" (padded). u8 is required for the SAD
// hardware acceleration; precomputing is faster than converting a window for
// each pixel.
SIMD_ATTR Image3B MakeGuide(const Image3F& padded,
                            const std::array<float, 3>& min,
                            const std::array<float, 3>& max, ThreadPool* pool) {
  const size_t xsize = padded.xsize();
  const size_t ysize = padded.ysize();
  Image3B guide(xsize, ysize);

  const SIMD_FULL(int32_t) di;
  const SIMD_FULL(uint32_t) du;
  const SIMD_PART(uint8_t, df.N) d8;

  float c_min[3];
  float c_mul[3];

#if EPF_INDEP_RANGE
  const float channel_scale[3] = {1.0f / 16, 1.0f / 4, 1.0f};
  for (size_t c = 0; c < 3; ++c) {
    PIK_CHECK(max[c] >= min[c]);
    float range = max[c] - min[c];
    if (range == 0.0f) {
      // Prevent division by zero. Guide is zero because we subtract min.
      range = 1.0f;
    }
    c_mul[c] = 255.0f * channel_scale[c] / range;
    c_min[c] = min[c];
  }
#else
  const float all_max = *std::max_element(max.begin(), max.end());
  const float all_min = *std::min_element(min.begin(), min.end());
  const float range = all_max - all_min;
  c_mul[0] = c_mul[1] = c_mul[2] = range == 0.0f ? 1.0f : 255.0f / range;
  c_min[0] = c_min[1] = c_min[2] = all_min;
#endif

  RunOnPool(pool, 0, ysize, [&](const int task, const int thread) SIMD_ATTR {
    const size_t y = task;
    for (size_t c = 0; c < 3; ++c) {
      const float* SIMD_RESTRICT padded_row = padded.ConstPlaneRow(c, y);
      uint8_t* SIMD_RESTRICT guide_row = guide.PlaneRow(c, y);

      const auto vmul = set1(df, c_mul[c]);
      const auto vmin = set1(df, c_min[c]);

      size_t x = 0;
      for (; x < xsize; x += df.N) {
        const auto scaled = (load(df, padded_row + x) - vmin) * vmul;
        const auto i32 = convert_to(di, scaled);
        const auto bytes = u8_from_u32(cast_to(du, i32));
        store(bytes, d8, guide_row + x);
      }

      // MPSADBW will read 16 bytes but only 11 need be valid;
      // zero-initialize the rest.
      for (; x < xsize + 16 - 11; x += df.N) {
        store(setzero(d8), d8, guide_row + x);
      }

    }  // c
  });  // y

  return guide;
}

static PIK_INLINE int SigmaFromQuant(float signal, float stretch, int lut_id,
                                     const float* luts, EpfStats* stats) {
  constexpr size_t kTableSize = 16;
  // Larger signal => less quantization, less smoothing.

  const float* lut = luts + kTableSize * lut_id;

#if EPF_NEW_SIGMA
#error "Add new LUT"

#else
  // baseline
  const float min_signal = 0.022156f;
  const float max_signal = 0.531738;
  const float mul_signal = 29.435892;
#endif
  float unscaled_sigma;
  if (signal <= min_signal) {
#if EPF_ENABLE_STATS
    stats->less += 1;
#endif
    unscaled_sigma = lut[0];
  } else if (signal >= max_signal) {
#if EPF_ENABLE_STATS
    stats->greater += 1;
#endif
    unscaled_sigma = lut[kTableSize - 1];
  } else {
    const float pos = (signal - min_signal) * mul_signal;
    const int64_t trunc = static_cast<int64_t>(pos);
    PIK_ASSERT(0 <= trunc && trunc < kTableSize);
    const float frac = pos - trunc;
    PIK_ASSERT(0.0f <= frac && frac <= 1.0f);
    unscaled_sigma = frac * lut[trunc + 1] + (1.0f - frac) * lut[trunc];
  }
  static const float kBias = 0.5182760822018414;
  const int sigma = unscaled_sigma * stretch + kBias;
  // No need to clamp to kMinSigma, we skip blocks with very low sigma.
  return std::min(sigma, kMaxSigma);
}

SIMD_ATTR void AdaptiveFilter(const Image3F& in_guide, const Image3F& in,
                              const ImageI* ac_quant, float quant_scale,
                              const ImageB& lut_ids,
                              const AcStrategyImage& ac_strategy,
                              const EpfParams& epf_params,
                              Image3F* smoothed, EpfStats* epf_stats) {
  PIK_ASSERT(SameSize(in, *smoothed));
  const size_t xsize = smoothed->xsize();
  const size_t ysize = smoothed->ysize();
  PIK_CHECK(xsize != 0 && ysize != 0);
  PIK_CHECK((xsize | ysize) % kBlockDim == 0);
  const size_t ysize_blocks = DivCeil(ysize, kBlockDim);
  PROFILER_FUNC;

  PIK_ASSERT(epf_params.enable_adaptive);

  std::array<float, 3> min, max;

  Image3F padded_in(xsize + 2 * kBorder, ysize + 2 * kBorder);
  MinMax(in, nullptr, &min, &max, &padded_in);

  const size_t padded_in_stride = padded_in.bytes_per_row();

  Image3F padded_guide(xsize + 2 * kBorder, ysize + 2 * kBorder);
  MinMax(epf_params.use_sharpened ? in : in_guide, nullptr, &min, &max,
         &padded_guide);

  if (epf_stats != nullptr) {
    for (int c = 0; c < 3; ++c) {
      epf_stats->s_ranges[c].Notify(max[c] - min[c]);
    }
  }
  const float all_max = *std::max_element(max.begin(), max.end());
  const float all_min = *std::min_element(min.begin(), min.end());
  const float stretch = all_min == all_max ? 1.f : 255.0f / (all_max - all_min);

  Image3B guide = MakeGuide(padded_guide, min, max, nullptr);
  const size_t guide_stride = guide.bytes_per_row();

#if EPF_DUMP_SIGMA
  ImageB dump(DivCeil(xsize, kBlockDim), ysize_blocks);
#endif

#if !EPF_NEW_SIGMA
  quant_scale = 0.039324273f;
#endif

  std::vector<EpfStats> all_stats(NumThreads(nullptr));

  const float lut[] = {
      1.9815775622811198,
      1.9715084740908622,
      1.6819963065873933,
      1.2146133632942862,
      1.0395364091521881,
      0.93552327583169714,
      0.68568655651684773,
      0.51174440217871964,
      0.36397262821018583,
      0.31621830414136975,
      0.30262954326557712,
      0.246314237855494,
      0.21617524864418683,
      0.10,
      0.05,
      0.0,
      // TODO(robryk): This is a temporary test alternative LUT. Provide actual
      // alternatives.
      1.9815775622811198 / 2.0,
      1.9715084740908622 / 2.0,
      1.6819963065873933 / 2.0,
      1.2146133632942862 / 2.0,
      1.0395364091521881 / 2.0,
      0.93552327583169714 / 2.0,
      0.68568655651684773 / 2.0,
      0.51174440217871964 / 2.0,
      0.36397262821018583 / 2.0,
      0.31621830414136975 / 2.0,
      0.30262954326557712 / 2.0,
      0.246314237855494 / 2.0,
      0.21617524864418683 / 2.0,
      0.10 / 2.0,
      0.05 / 2.0,
      0.0 / 2.0,
  };

	  for(int task = 0; task < ysize_blocks; ++task) {
        const size_t by = task;
        EpfStats& stats = all_stats[0];
        const int* SIMD_RESTRICT ac_quant_row = ac_quant->Row(by);
        const uint8_t* SIMD_RESTRICT lut_id_row = lut_ids.Row(by);
        AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(by);
#if EPF_DUMP_SIGMA
        uint8_t* dump_row = dump.Row(by);
#endif

        WeightFast weight_func;

        for (size_t bx = 0; bx < xsize; bx += kBlockDim) {
          const float ac_q = ac_quant_row[bx / kBlockDim];
          const int lut_id = lut_id_row[bx / kBlockDim];
          const AcStrategy ac_strategy = ac_strategy_row[bx / kBlockDim];
          const float scale = ac_strategy.ARQuantScale();
          // fprintf(stderr, "%d %d %g (%g %g)\n", by, bx, ac_q, scale,
          // quant_scale);
          const float quant = ac_q * scale * quant_scale;
          const int sigma = SigmaFromQuant(quant, stretch, lut_id, lut, &stats);
#if EPF_ENABLE_STATS
          stats.s_sigma.Notify(sigma);
          stats.s_quant.Notify(quant);
          stats.total += 1;
#endif
#if EPF_DUMP_SIGMA
          dump_row[bx / kBlockDim] = std::min(std::max(0, sigma), 255);
#endif
          if (sigma < kMinSigma) {
            WeightedSum::CopyOriginalBlock(padded_in, bx, by * kBlockDim,
                                           smoothed);
#if EPF_ENABLE_STATS
            stats.skipped += 1;
#endif
            continue;
          }
          weight_func.SetSigma(sigma);

          for (size_t iy = 0; iy < kBlockDim; ++iy) {
            const size_t y = by * kBlockDim + iy;
            // "guide_m4" and "in_m3" are 4 and 3 rows above the current pixel.
            const uint8_t* SIMD_RESTRICT guide_m4_r =
                guide.ConstPlaneRow(0, y + kBorder - 4) + kBorder;
            const uint8_t* SIMD_RESTRICT guide_m4_g =
                guide.ConstPlaneRow(1, y + kBorder - 4) + kBorder;
            const uint8_t* SIMD_RESTRICT guide_m4_b =
                guide.ConstPlaneRow(2, y + kBorder - 4) + kBorder;
            const float* SIMD_RESTRICT in_m3_r =
                padded_in.ConstPlaneRow(0, y - 3 + kBorder) + kBorder;
            const float* SIMD_RESTRICT in_m3_g =
                padded_in.ConstPlaneRow(1, y - 3 + kBorder) + kBorder;
            const float* SIMD_RESTRICT in_m3_b =
                padded_in.ConstPlaneRow(2, y - 3 + kBorder) + kBorder;
            float* SIMD_RESTRICT out_r = smoothed->PlaneRow(0, y);
            float* SIMD_RESTRICT out_g = smoothed->PlaneRow(1, y);
            float* SIMD_RESTRICT out_b = smoothed->PlaneRow(2, y);

            for (size_t ix = 0; ix < kBlockDim; ++ix) {
              const size_t x = bx + ix;
              WeightedSum::Compute(
                  guide_m4_r + x, guide_m4_g + x, guide_m4_b + x, guide_stride,
                  in_m3_r + x, in_m3_g + x, in_m3_b + x, padded_in_stride,
                  weight_func, out_r + x, out_g + x, out_b + x);
            }  // ix
          }    // iy
        }      // bx
      }        // by

  if (epf_stats != nullptr) {
    for (EpfStats& stats : all_stats) {
      epf_stats->Assimilate(stats);
    }
  }

#if EPF_DUMP_SIGMA
  WriteImage(ImageFormatPNG(), dump, "/tmp/out/sigma.png");
#endif
}

// Closure for ThreadPool, with mutable per-thread state.
class FilterWorkers {
 public:
  explicit FilterWorkers(size_t num_workers, const Image3B& guide,
                         const Image3F& in, const int sigma,
                         Image3F* SIMD_RESTRICT out)
      : guide_(guide),
        in_(in),
        out_(out),
        // Must use out because in is padded.
        xsize_(out->xsize()),
        ysize_(out->ysize()) {
    guide_stride_ = guide.bytes_per_row();

    in_stride_ = in.bytes_per_row();

    PIK_ASSERT(kMinSigma <= sigma && sigma <= kMaxSigma);
    weight_func_.SetSigma(sigma);
  }

  // BLOCK y index ("1" is the second block)
  SIMD_ATTR void Run(const size_t by, const int thread) {
    for (size_t bx = 0; bx < xsize_; bx += kBlockDim) {
      for (size_t iy = 0; iy < kBlockDim; ++iy) {
        const size_t y = by * kBlockDim + iy;
        // "guide_m4" and "in_m3" are 4 and 3 rows above the current pixel.
        const uint8_t* SIMD_RESTRICT guide_m4_r =
            guide_.ConstPlaneRow(0, y + kBorder - 4) + kBorder;
        const uint8_t* SIMD_RESTRICT guide_m4_g =
            guide_.ConstPlaneRow(1, y + kBorder - 4) + kBorder;
        const uint8_t* SIMD_RESTRICT guide_m4_b =
            guide_.ConstPlaneRow(2, y + kBorder - 4) + kBorder;
        const float* SIMD_RESTRICT in_m3_r =
            in_.ConstPlaneRow(0, y - 3 + kBorder) + kBorder;
        const float* SIMD_RESTRICT in_m3_g =
            in_.ConstPlaneRow(1, y - 3 + kBorder) + kBorder;
        const float* SIMD_RESTRICT in_m3_b =
            in_.ConstPlaneRow(2, y - 3 + kBorder) + kBorder;
        float* SIMD_RESTRICT out_r = out_->PlaneRow(0, y);
        float* SIMD_RESTRICT out_g = out_->PlaneRow(1, y);
        float* SIMD_RESTRICT out_b = out_->PlaneRow(2, y);

        for (size_t ix = 0; ix < kBlockDim; ++ix) {
          const size_t x = bx + ix;
          WeightedSum::Compute(guide_m4_r + x, guide_m4_g + x, guide_m4_b + x,
                               guide_stride_, in_m3_r + x, in_m3_g + x,
                               in_m3_b + x, in_stride_, weight_func_, out_r + x,
                               out_g + x, out_b + x);
        }
      }
    }
  }

 private:
  const Image3B& guide_;
  const Image3F& in_;
  size_t guide_stride_;
  size_t in_stride_;
  Image3F* SIMD_RESTRICT out_;

  size_t xsize_;
  size_t ysize_;
  WeightFast weight_func_;
};

void Filter(const Image3F& in_guide, const Image3F& in,
            const EpfParams& epf_params, float* PIK_RESTRICT stretch,
            Image3F* smoothed) {
  PIK_ASSERT(SameSize(in, *smoothed));
  const size_t xsize = smoothed->xsize();
  const size_t ysize = smoothed->ysize();
  PIK_CHECK(xsize != 0 && ysize != 0);
  PIK_CHECK((xsize | ysize) % kBlockDim == 0);
  const size_t ysize_blocks = DivCeil(ysize, kBlockDim);
  PROFILER_FUNC;

  PIK_ASSERT(!epf_params.enable_adaptive);
  if (epf_params.sigma == 0) {
    CopyImageTo(in, smoothed);
    *stretch = 1.0f;
    return;
  }

  std::array<float, 3> min, max;
  Image3F padded_in(xsize + 2 * kBorder, ysize + 2 * kBorder);
  MinMax(in, /*pool=*/nullptr, &min, &max, &padded_in);

  Image3F padded_guide(xsize + 2 * kBorder, ysize + 2 * kBorder);
  MinMax(epf_params.use_sharpened ? in : in_guide, /*pool=*/nullptr, &min, &max,
         &padded_guide);

  const float all_max = *std::max_element(max.begin(), max.end());
  const float all_min = *std::min_element(min.begin(), min.end());
  *stretch = all_min == all_max ? 1.0f : 255.0f / (all_max - all_min);

  Image3B guide = MakeGuide(padded_guide, min, max, /*pool=*/nullptr);

  FilterWorkers workers(1, guide, padded_in, epf_params.sigma, smoothed);
  for (size_t y = 0; y < ysize_blocks; ++y) {
    workers.Run(y, /*thread=*/0);
  }
}

}  // namespace
}  // namespace SIMD_NAMESPACE

template <>
void InitEdgePreservingFilter::operator()<SIMD_TARGET>() const {
  SIMD_NAMESPACE::MulTable::Init();
}

template <>
void EdgePreservingFilter::operator()<SIMD_TARGET>(
    const Image3F& in_guide, const Image3F& in, const ImageI* ac_quant,
    float sigma_mul, const ImageB& lut_ids, const AcStrategyImage& ac_strategy,
    const EpfParams& epf_params, Image3F* smoothed,
    EpfStats* epf_stats) const {
  SIMD_NAMESPACE::AdaptiveFilter(in_guide, in, ac_quant, sigma_mul, lut_ids,
                                 ac_strategy, epf_params, smoothed,
                                 epf_stats);
}

template <>
void EdgePreservingFilter::operator()<SIMD_TARGET>(const Image3F& in_guide,
                                                   const Image3F& in,
                                                   const EpfParams& epf_params,
                                                   float* PIK_RESTRICT stretch,
                                                   Image3F* smoothed) const {
  SIMD_NAMESPACE::Filter(in_guide, in, epf_params, stretch, smoothed);
}

template <>
void EdgePreservingFilterTest::operator()<SIMD_TARGET>() const {
  SIMD_NAMESPACE::InternalWeightTests::Run();
  SIMD_NAMESPACE::WeightedSum::Test();
  fprintf(stderr, "Tests OK: %s\n", vec_name<SIMD_NAMESPACE::DF>());
}

template <>
float EdgePreservingFilterTest::operator()<SIMD_TARGET>(int sigma,
                                                        int sad) const {
  SIMD_NAMESPACE::MulTable::Init();
  SIMD_NAMESPACE::WeightFast weight_func;
  weight_func.SetSigma(sigma);
  return SIMD_NAMESPACE::GetWeightForTest(weight_func, sad);
}

}  // namespace pik

#endif  // SIMD_ATTR_IMPL
