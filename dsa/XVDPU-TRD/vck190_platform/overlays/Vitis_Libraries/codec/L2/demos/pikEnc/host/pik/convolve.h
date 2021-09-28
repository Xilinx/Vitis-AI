// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_CONVOLVE_H_
#define PIK_CONVOLVE_H_

// Fast SIMD 2D convolution.

#include <stddef.h>
#include <stdint.h>
#include <cassert>

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1

#include "pik/compiler_specific.h"
#include "pik/data_parallel.h"
#include "pik/image.h"
#include "pik/image_ops.h"
#include "pik/profiler.h"
#include "pik/simd/simd.h"
#include "pik/status.h"

namespace pik {

// Usable by any 3x3 kernel; applied as-is without flipping.
struct Weights3x3 {
  // top/middle/bottom left/center/right, replicated 4x via SIMD_REP4.
  float tl[4];
  float tc[4];
  float tr[4];
  float ml[4];
  float mc[4];
  float mr[4];
  float bl[4];
  float bc[4];
  float br[4];
};

struct WeightsSeparable5 {
  // Horizontal 1D, distances 0..2, each replicated 4x.
  float horz[3 * 4];
  float vert[3 * 4];
};

// For code-folding.
namespace kernel {

// Holds weights computed at runtime (e.g. inverse of another kernel).
class Variable3 {
 public:
  explicit Variable3(const float tl, const float tc, const float tr,
                     const float ml, const float mc, const float mr,
                     const float bl, const float bc, const float br) {
    for (size_t i = 0; i < 4; ++i) {
      weights_.tl[i] = tl;
      weights_.tc[i] = tc;
      weights_.tr[i] = tr;
      weights_.ml[i] = ml;
      weights_.mc[i] = mc;
      weights_.mr[i] = mr;
      weights_.bl[i] = bl;
      weights_.bc[i] = bc;
      weights_.br[i] = br;
    }
  }

  PIK_INLINE const Weights3x3& Weights() const { return weights_; }

 private:
  Weights3x3 weights_;
};

// Approximation of the Laplacian.
struct Laplacian3 {
  PIK_INLINE const Weights3x3& Weights() const {
    constexpr float w0 = -4.0f;
    constexpr float w1 = 1.0f;
    constexpr float w2 = 0.0f;
    static constexpr Weights3x3 weights = {
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)},
        {SIMD_REP4(w1)}, {SIMD_REP4(w0)}, {SIMD_REP4(w1)},
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)}};
    return weights;
  }
};

// Concentrates energy in low-frequency components (e.g. for antialiasing).
struct Lowpass3 {
  PIK_INLINE const Weights3x3& Weights() const {
    // Computed by research/convolve_weights.py's cubic spline approximations of
    // prolate spheroidal wave functions.
    constexpr float w0 = 0.36208932f;
    constexpr float w1 = 0.12820096f;
    constexpr float w2 = 0.03127668f;
    static constexpr Weights3x3 weights = {
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)},
        {SIMD_REP4(w1)}, {SIMD_REP4(w0)}, {SIMD_REP4(w1)},
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)}};
    return weights;
  }
};

struct Lowpass5 {
  PIK_INLINE const WeightsSeparable5& Weights() const {
    constexpr float w0 = 0.41714928f;
    constexpr float w1 = 0.25539268f;
    constexpr float w2 = 0.03603267f;
    static constexpr WeightsSeparable5 weights = {
        {SIMD_REP4(w0), SIMD_REP4(w1), SIMD_REP4(w2)},
        {SIMD_REP4(w0), SIMD_REP4(w1), SIMD_REP4(w2)}};
    return weights;
  }
};

struct Gaussian5Sigma1 {
  PIK_INLINE const WeightsSeparable5& Weights() const {
    constexpr float w0 = 0.38774f;
    constexpr float w1 = 0.24477f;
    constexpr float w2 = 0.06136f;
    static constexpr WeightsSeparable5 weights = {
        {SIMD_REP4(w0), SIMD_REP4(w1), SIMD_REP4(w2)},
        {SIMD_REP4(w0), SIMD_REP4(w1), SIMD_REP4(w2)}};
    return weights;
  }
};

struct Gaussian5Sigma2 {
  PIK_INLINE const WeightsSeparable5& Weights() const {
    constexpr float w0 = 0.250301f;
    constexpr float w1 = 0.221461f;
    constexpr float w2 = 0.153388f;
    static constexpr WeightsSeparable5 weights = {
        {SIMD_REP4(w0), SIMD_REP4(w1), SIMD_REP4(w2)},
        {SIMD_REP4(w0), SIMD_REP4(w1), SIMD_REP4(w2)}};
    return weights;
  }
};

}  // namespace kernel

// Non-vectorized implementations for validation.
namespace slow {

// Separable kernels, any radius.
template <int64_t kRadius, class Wrap>
class SeparableConvolution {
 public:
  template <class Kernel>
  static void Run(const ImageF& in, const size_t xsize, const size_t ysize,
                  const Kernel& kernel, ImageF* out) {
    const float* horz_weights = &kernel.Weights().horz[0];
    const float* vert_weights = &kernel.Weights().vert[0];
    for (size_t y = 0; y < ysize; ++y) {
      float* const PIK_RESTRICT row_out = out->Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] =
            ConvolvePixel(in, xsize, ysize, x, y, horz_weights, vert_weights);
      }
    }
  }

 private:
  static float ConvolvePixel(const ImageF& in, const size_t xsize,
                             const size_t ysize, const size_t x, const size_t y,
                             const float* PIK_RESTRICT horz_weights,
                             const float* PIK_RESTRICT vert_weights) {
    float mul = 0.0f;
    for (int dy = -kRadius; dy <= kRadius; ++dy) {
      const float wy = vert_weights[std::abs(dy) * 4];
      const size_t sy = Wrap()(y + dy, ysize);
      PIK_CHECK(sy < ysize);
      const float* const PIK_RESTRICT row = in.ConstRow(sy);
      for (int dx = -kRadius; dx <= kRadius; ++dx) {
        const float wx = horz_weights[std::abs(dx) * 4];
        const size_t sx = Wrap()(x + dx, xsize);
        PIK_CHECK(sx < xsize);
        mul += row[sx] * wx * wy;
      }
    }
    return mul;
  }
};

// Weights i=0..2 are for Manhattan distance i from center.
template <int64_t kRadius, class Wrap>
struct Symmetric3x3Convolution {
  static_assert(kRadius == 1, "Wrong kRadius");

  template <class Kernel>
  static void Run(const ImageF& in, const size_t xsize, const size_t ysize,
                  const Kernel& kernel, ImageF* out) {
    PIK_CHECK(xsize == out->xsize() && ysize == out->ysize());
    const Weights3x3& weights = kernel.Weights();

    for (size_t y = 0; y < ysize; ++y) {
      const float* const PIK_RESTRICT row_t = in.ConstRow(Wrap()(y - 1, ysize));
      const float* const PIK_RESTRICT row_m = in.ConstRow(y);
      const float* const PIK_RESTRICT row_b = in.ConstRow(Wrap()(y + 1, ysize));
      float* const PIK_RESTRICT row_out = out->Row(y);

      for (size_t x = 0; x < xsize; ++x) {
        float mul = row_m[x] * weights.mc[0];
        const int64_t xm1 = Wrap()(x - 1, xsize);
        const int64_t xp1 = Wrap()(x + 1, xsize);
        const float tl = row_t[xm1];
        const float ml = row_m[xm1];
        const float bl = row_b[xm1];
        const float tr = row_t[xp1];
        const float mr = row_m[xp1];
        const float br = row_b[xp1];
        mul += (row_t[x] + row_b[x] + ml + mr) * weights.tc[0];
        mul += (tl + tr + bl + br) * weights.tl[0];
        row_out[x] = mul;
      }
    }
  }
};

template <int64_t kRadius, class Wrap>
struct General3x3Convolution {
  static_assert(kRadius == 1, "Wrong kRadius");

  template <class Kernel>
  static void Run(const ImageF& in, const size_t xsize, const size_t ysize,
                  const Kernel& kernel, ImageF* out) {
    PIK_CHECK(xsize == out->xsize() && ysize == out->ysize());
    const Weights3x3& weights = kernel.Weights();

    for (size_t y = 0; y < ysize; ++y) {
      const float* const PIK_RESTRICT row_t = in.ConstRow(Wrap()(y - 1, ysize));
      const float* const PIK_RESTRICT row_m = in.ConstRow(y);
      const float* const PIK_RESTRICT row_b = in.ConstRow(Wrap()(y + 1, ysize));
      float* const PIK_RESTRICT row_out = out->Row(y);

      for (size_t x = 0; x < xsize; ++x) {
        const int64_t xm1 = Wrap()(x - 1, xsize);
        const int64_t xp1 = Wrap()(x + 1, xsize);
        const float tl = row_t[xm1];
        const float ml = row_m[xm1];
        const float bl = row_b[xm1];
        const float tr = row_t[xp1];
        const float mr = row_m[xp1];
        const float br = row_b[xp1];
        float r = 0.0f;
        r += tl * weights.tl[0] + row_t[x] * weights.tc[0] + tr * weights.tr[0];
        r += ml * weights.ml[0] + row_m[x] * weights.mc[0] + mr * weights.mr[0];
        r += bl * weights.bl[0] + row_b[x] * weights.bc[0] + br * weights.br[0];
        row_out[x] = r;
      }
    }
  }

  template <class Kernel>
  static void Run(const Image3F& in, const size_t xsize, const size_t ysize,
                  const Kernel& kernel, Image3F* out) {
    for (int c = 0; c < 3; ++c) {
      Run(in.Plane(c), xsize, ysize, kernel,
          const_cast<ImageF*>(&out->Plane(c)));
    }
  }
};

// Slow N*R^2 algorithm in case weights are not separable, but avoids
// bounds-checking overhead for interior pixels. Weights are the lower-right
// quadrant of the kernel and need not be pre-normalized.
template <int64_t kRadius, class Wrap>
class SymmetricConvolution {
 public:
  static void Run(const ImageF& in, const size_t xsize, const size_t ysize,
                  const float (&weights)[(kRadius + 1) * (kRadius + 1)],
                  ImageF* out) {
    // Normalize all weights (expand quadrant into entire kernel)
    double sum = 0.0f;
    for (int64_t ky = -kRadius; ky <= kRadius; ky++) {
      const int64_t wy = std::abs(ky);
      for (int64_t kx = -kRadius; kx <= kRadius; kx++) {
        const int64_t wx = std::abs(kx);
        sum += weights[wy * (kRadius + 1) + wx];
      }
    }
    const float mul = sum == 0.0f ? 1.0f : 1.0 / sum;
    float normalized[(kRadius + 1) * (kRadius + 1)];
    for (size_t i = 0; i < (kRadius + 1) * (kRadius + 1); ++i) {
      normalized[i] = weights[i] * mul;
    }

    int64_t iy = 0;
    for (; iy < kRadius; iy++) {
      ConvolveRow<Wrap>(in, xsize, ysize, iy, normalized, out);
    }
    for (; iy < ysize - kRadius; iy++) {
      ConvolveRow<WrapUnchanged>(in, xsize, ysize, iy, normalized, out);
    }
    for (; iy < ysize; iy++) {
      ConvolveRow<Wrap>(in, xsize, ysize, iy, normalized, out);
    }
  }

  static void Run(const Image3F& in, const size_t xsize, const size_t ysize,
                  const float (&weights)[(kRadius + 1) * (kRadius + 1)],
                  Image3F* out) {
    for (int c = 0; c < 3; ++c) {
      Run(in.Plane(c), xsize, ysize, weights,
          const_cast<ImageF*>(&out->Plane(c)));
    }
  }

 private:
  template <class WrapX, class WrapY>
  static float ConvolvePixel(
      const ImageF& in, const size_t xsize, const size_t ysize,
      const int64_t ix, const int64_t iy,
      const float (&weights)[(kRadius + 1) * (kRadius + 1)]) {
    float sum = 0.0;

    // ix: image; kx: kernel; wx: weight
    for (int64_t ky = -kRadius; ky <= kRadius; ky++) {
      const int64_t wy = std::abs(ky);
      const int64_t y = WrapY()(iy + ky, ysize);
      const float* PIK_RESTRICT row_in = in.ConstRow(y);

      for (int64_t kx = -kRadius; kx <= kRadius; kx++) {
        const int64_t wx = std::abs(kx);
        const int64_t x = WrapX()(ix + kx, xsize);

        sum += row_in[x] * weights[wy * (kRadius + 1) + wx];
      }
    }
    return sum;
  }

  template <class WrapY>
  static inline void ConvolveRow(
      const ImageF& in, const size_t xsize, const size_t ysize,
      const int64_t iy, const float (&weights)[(kRadius + 1) * (kRadius + 1)],
      ImageF* PIK_RESTRICT out) {
    float* PIK_RESTRICT row_out = out->Row(iy);
    int64_t ix = 0;
    for (; ix < kRadius; ix++) {
      row_out[ix] =
          ConvolvePixel<WrapMirror, WrapY>(in, xsize, ysize, ix, iy, weights);
    }
    for (; ix < xsize - kRadius; ix++) {
      row_out[ix] = ConvolvePixel<WrapUnchanged, WrapY>(in, xsize, ysize, ix,
                                                        iy, weights);
    }
    for (; ix < xsize; ix++) {
      row_out[ix] =
          ConvolvePixel<WrapMirror, WrapY>(in, xsize, ysize, ix, iy, weights);
    }
  }
};

}  // namespace slow

// Synthesizes left/right neighbors from a vector of center pixels.
class Neighbors {
  using D = SIMD_FULL(float);
  using V = D::V;
  static const D d;

 public:
  // Returns l[i] == c[i - 1].
  static SIMD_ATTR PIK_INLINE V L1(const V c, const V p) {
    // For AVX-512: try permutex2var_ps.
#if SIMD_TARGET_VALUE == SIMD_AVX2
    // c = PONM'LKJI, p = Hxxx'xxxx
    const V L_H = concat_lo_hi(c, p);
    return combine_shift_right_bytes<12>(c, L_H);  // ONML'KJIH
#elif SIMD_TARGET_VALUE == SIMD_NONE
    return p;
#else
    // c = LKJI, p = Hxxx
    return combine_shift_right_bytes<12>(c, p);  // KJIH
#endif
  }

  // Returns l[i] == c[Mirror(i - 1)].
  static SIMD_ATTR PIK_INLINE V FirstL1(const V c) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
    SIMD_ALIGN constexpr int lanes[8] = {0, 0, 1, 2, 3, 4, 5, 6};
    const auto indices = set_table_indices(d, lanes);
    // c = PONM'LKJI
    return table_lookup_lanes(c, indices);  // ONML'KJII
#elif SIMD_TARGET_VALUE == SIMD_NONE
    return c;
#else
    // c = LKJI
    return V(_mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(2, 1, 0, 0)));  // KJII
#endif
  }

  // Returns l[i] == c[Mirror(i - 2)].
  static SIMD_ATTR PIK_INLINE V FirstL2(const V c) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
    SIMD_ALIGN constexpr int lanes[8] = {1, 0, 0, 1, 2, 3, 4, 5};
    const auto indices = set_table_indices(d, lanes);
    // c = PONM'LKJI
    return table_lookup_lanes(c, indices);  // NMLK'JIIJ
#elif SIMD_TARGET_VALUE == SIMD_NONE
    return setzero(d);  // unsupported, avoid calling this.
#else
    // c = LKJI
    return V(_mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(1, 0, 0, 1)));  // JIIJ
#endif
  }

  // Returns r[i] == c[i + 1].
  static SIMD_ATTR PIK_INLINE V R1(const V c, const V n) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
    // c = PONM'LKJI, n = xxxx'xxxQ
    const V Q_M = concat_lo_hi(n, c);             // Right-aligned (lower lane)
    return combine_shift_right_bytes<4>(Q_M, c);  // QPON'MLKJ
#elif SIMD_TARGET_VALUE == SIMD_NONE
    return n;
#else
    // c = LKJI, n = xxxM
    return combine_shift_right_bytes<4>(n, c);  // MLKJ
#endif
  }

  // Returns r[i] == c[i + 1].
  static SIMD_ATTR PIK_INLINE V LastR1(const V c) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
    SIMD_ALIGN constexpr uint32_t lanes[8] = {1, 2, 3, 4, 5, 6, 7, 7};
    const auto indices = load(SIMD_FULL(uint32_t)(), lanes);
    // c = PONM'LKJI
    return V(_mm256_permutevar8x32_ps(c.raw, indices.raw));  // PPON'MLKJ
#elif SIMD_TARGET_VALUE == SIMD_NONE
    return c;
#else
    // c = LKJI
    const auto L = broadcast<3>(c);
    return combine_shift_right_bytes<4>(L, c);  // LLKJ
#endif
  }
};

// Requires kRadius valid (mirrored or neighbor) columns on either side of
// [0, xsize). It is also safe to load entire vectors. This behavior is
// required by TFNode with Borders(>0). In other cases, this assumption requires
// ConvolveT to PadImage, so LeftRightInvalid would be more efficient.
struct LeftRightValid {};

// No valid values outside [0, xsize), but the strategy may still safely load
// the preceding vector, and/or round xsize up to the vector lane count. This
// avoids needing PadImage.
struct LeftRightInvalid {};

// LeftRightInvalid requires xsize >= SIMD_FULL(float)::N + kConvolveMaxRadius.
static constexpr size_t kConvolveMaxRadius = 3;

// For use by set_table_indices.
static inline const int32_t* MirrorLanes(const size_t mod) {
  SIMD_FULL(float) d;
#if SIMD_TARGET_VALUE == SIMD_AVX2
  // last  part  mirrored
  // 01234567| 76543210   loadedReg 76543210 mirroredReg 01234567
  // 01234567|8 8765432   loadedReg 87654321 mirroredReg 23456788
  // 01234567|89 987654   loadedReg 98765432 mirroredReg 45678998
  // 01234567|89A A9876   loadedReg A9876543 mirroredReg 6789AA98
  // 01234567|89AB BA98
  // 01234567|89ABC CBA
  // 01234567|89ABCD DC
  // 01234567|89ABCDE E   loadedReg EDCBA987 mirroredReg EEDCBA98
  SIMD_ALIGN static constexpr int32_t idx_lanes[d.N * d.N] = {
      7, 6, 5, 4, 3, 2, 1, 0,  // 0
      7, 7, 6, 5, 4, 3, 2, 1,  // 1
      6, 7, 7, 6, 5, 4, 3, 2,  // 2
      5, 6, 7, 7, 6, 5, 4, 3,  // 3
      4, 5, 6, 7, 7, 6, 5, 4,  // 4
      3, 4, 5, 6, 7, 7, 6, 5,  // 5
      2, 3, 4, 5, 6, 7, 7, 6,  // 6
      1, 2, 3, 4, 5, 6, 7, 7,  // 7
  };
  return idx_lanes + mod * d.N;
#elif SIMD_TARGET_VALUE == SIMD_NONE
  (void)d;         // silence warning about unused d
  return nullptr;  // do not call
#else
  // 0123| 3210   loadedReg 3210 mirroredReg 0123
  // 0123|4 432   loadedReg 4321 mirroredReg 2344
  // 0123|45 54   loadedReg 5432 mirroredReg 4554
  // 0123|456 6   loadedReg 6543 mirroredReg 6654
  SIMD_ALIGN static constexpr int32_t idx_lanes[d.N * d.N] = {
      3, 2, 1, 0,  // 0
      3, 3, 2, 1,  // 1
      2, 3, 3, 2,  // 2
      1, 2, 3, 3,  // 3
  };
  return idx_lanes + mod * d.N;
#endif
}

namespace strategy {

// 3x3 convolution by symmetric kernel with a single scan through the input.
class Symmetric3 {
  using D = SIMD_FULL(float);
  using V = D::V;

 public:
  static constexpr int64_t kRadius = 1;

  // Only accesses pixels in [0, xsize).
  template <size_t kSizeModN, class WrapRow>
  static SIMD_ATTR PIK_INLINE void ConvolveRow(
      LeftRightInvalid, const float* const PIK_RESTRICT row_m,
      const size_t xsize, const int64_t stride, const WrapRow& wrap_row,
      const Weights3x3& weights, float* const PIK_RESTRICT row_out) {
    const D d;
    // t, m, b = top, middle, bottom row;
    const float* const PIK_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const PIK_RESTRICT row_b = wrap_row(row_m + stride, stride);

    // Must load in advance - compiler doesn't understand load_dup128 and
    // schedules them too late.
    const V w0 = load_dup128(d, weights.mc);
    const V w1 = load_dup128(d, weights.tc);
    const V w2 = load_dup128(d, weights.tl);

    // l, c, r = left, center, right. Leftmost vector: need FirstL1.
    {
      const V tc = load_unaligned(d, row_t + 0);
      const V mc = load_unaligned(d, row_m + 0);
      const V bc = load_unaligned(d, row_b + 0);
      const V tl = Neighbors::FirstL1(tc);
      const V tr = load_unaligned(d, row_t + 0 + 1);
      const V ml = Neighbors::FirstL1(mc);
      const V mr = load_unaligned(d, row_m + 0 + 1);
      const V bl = Neighbors::FirstL1(bc);
      const V br = load_unaligned(d, row_b + 0 + 1);
      const V conv =
          WeightedSum(tl, tc, tr, ml, mc, mr, bl, bc, br, w0, w1, w2);
      store(conv, d, row_out + 0);
    }

    // Loop as long as we can load enough new values:
    size_t x = d.N;
    for (; x + d.N + kRadius <= xsize; x += d.N) {
      const auto conv = ConvolveValid(row_t, row_m, row_b, x, w0, w1, w2);
      store(conv, d, row_out + x);
    }

    // For final (partial) vector:
    const V tc = load_unaligned(d, row_t + x);
    const V mc = load_unaligned(d, row_m + x);
    const V bc = load_unaligned(d, row_b + x);

    V tr, mr, br;
#if SIMD_TARGET_VALUE == SIMD_NONE
    tr = tc;  // Single-lane => mirrored right neighbor = center value.
    mr = mc;
    br = bc;
#else
    if (kSizeModN == 0) {
      // The above loop didn't handle the last vector because it needs an
      // additional right neighbor (generated via mirroring).
      auto mirror = set_table_indices(d, MirrorLanes(d.N - 1));
      tr = table_lookup_lanes(tc, mirror);
      mr = table_lookup_lanes(mc, mirror);
      br = table_lookup_lanes(bc, mirror);
    } else {
      auto mirror = set_table_indices(d, MirrorLanes((xsize % d.N) - 1));
      // Loads last valid value into uppermost lane and mirrors.
      tr = table_lookup_lanes(load_unaligned(d, row_t + xsize - d.N), mirror);
      mr = table_lookup_lanes(load_unaligned(d, row_m + xsize - d.N), mirror);
      br = table_lookup_lanes(load_unaligned(d, row_b + xsize - d.N), mirror);
    }
#endif

    const V tl = load_unaligned(d, row_t + x - 1);
    const V ml = load_unaligned(d, row_m + x - 1);
    const V bl = load_unaligned(d, row_b + x - 1);
    const V conv = WeightedSum(tl, tc, tr, ml, mc, mr, bl, bc, br, w0, w1, w2);
    store(conv, d, row_out + x);
  }

  template <size_t kSizeModN, class WrapRow>
  static SIMD_ATTR PIK_INLINE void ConvolveRow(
      LeftRightValid, const float* const PIK_RESTRICT row_m, const size_t xsize,
      const int64_t stride, const WrapRow& wrap_row,
      const Weights3x3& PIK_RESTRICT weights,
      float* const PIK_RESTRICT row_out) {
    const D d;
    // t, m, b = top, middle, bottom row;
    const float* const PIK_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const PIK_RESTRICT row_b = wrap_row(row_m + stride, stride);

    const V w0 = load_dup128(d, weights.mc);
    const V w1 = load_dup128(d, weights.tc);
    const V w2 = load_dup128(d, weights.tl);

    // l, c, r = left, center, right.
    for (size_t x = 0; x < xsize; x += d.N) {
      const V conv = ConvolveValid(row_t, row_m, row_b, x, w0, w1, w2);
      store(conv, d, row_out + x);
    }
  }

 private:
  // Returns sum{x_i * w_i}.
  template <class V>
  static SIMD_ATTR PIK_INLINE V WeightedSum(const V tl, const V tc, const V tr,
                                            const V ml, const V mc, const V mr,
                                            const V bl, const V bc, const V br,
                                            const V w0, const V w1,
                                            const V w2) {
    const V sum_tb = tc + bc;

    // Faster than 5 mul + 4 FMA.
    const V mul0 = mc * w0;
    const V sum_lr = ml + mr;

    const V x1 = sum_tb + sum_lr;
    const V mul1 = mul_add(x1, w1, mul0);

    const V sum_t2 = tl + tr;
    const V sum_b2 = bl + br;
    const V x2 = sum_t2 + sum_b2;
    const V mul2 = mul_add(x2, w2, mul1);
    return mul2;
  }

  static SIMD_ATTR PIK_INLINE V ConvolveValid(const float* PIK_RESTRICT row_t,
                                              const float* PIK_RESTRICT row_m,
                                              const float* PIK_RESTRICT row_b,
                                              const int64_t x, const V w0,
                                              const V w1, const V w2) {
    const D d;
    const V tc = load_unaligned(d, row_t + x);
    const V mc = load_unaligned(d, row_m + x);
    const V bc = load_unaligned(d, row_b + x);
    const V tl = load_unaligned(d, row_t + x - 1);
    const V tr = load_unaligned(d, row_t + x + 1);
    const V ml = load_unaligned(d, row_m + x - 1);
    const V mr = load_unaligned(d, row_m + x + 1);
    const V bl = load_unaligned(d, row_b + x - 1);
    const V br = load_unaligned(d, row_b + x + 1);
    return WeightedSum(tl, tc, tr, ml, mc, mr, bl, bc, br, w0, w1, w2);
  }
};

// 3x3, center column zero, right column = negated left column.
class GradX3 {
  using D = SIMD_FULL(float);
  using V = D::V;

 public:
  static constexpr int64_t kRadius = 1;

  // Only accesses pixels in [0, xsize).
  template <size_t kSizeModN, class WrapRow>
  static SIMD_ATTR PIK_INLINE void ConvolveRow(
      LeftRightInvalid, const float* const PIK_RESTRICT row_m,
      const size_t xsize, const int64_t stride, const WrapRow& wrap_row,
      const Weights3x3& weights, float* const PIK_RESTRICT row_out) {
    const D d;
    // t, m, b = top, middle, bottom row;
    const float* const PIK_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const PIK_RESTRICT row_b = wrap_row(row_m + stride, stride);

    // Must load in advance - compiler doesn't understand load_dup128 and
    // schedules them too late.
    const V wtb = load_dup128(d, weights.tl);
    const V wm = load_dup128(d, weights.ml);

    // l, c, r = left, center, right. Leftmost vector: need FirstL1.
    {
      const V tc = load_unaligned(d, row_t + 0);
      const V mc = load_unaligned(d, row_m + 0);
      const V bc = load_unaligned(d, row_b + 0);
      const V tl = Neighbors::FirstL1(tc);
      const V tr = load_unaligned(d, row_t + 0 + 1);
      const V ml = Neighbors::FirstL1(mc);
      const V mr = load_unaligned(d, row_m + 0 + 1);
      const V bl = Neighbors::FirstL1(bc);
      const V br = load_unaligned(d, row_b + 0 + 1);
      const V conv = WeightedSum(tl, tr, ml, mr, bl, br, wtb, wm);
      store(conv, d, row_out + 0);
    }

    // Loop as long as we can load enough new values:
    size_t x = d.N;
    for (; x + d.N + kRadius <= xsize; x += d.N) {
      const auto conv = ConvolveValid(row_t, row_m, row_b, x, wtb, wm);
      store(conv, d, row_out + x);
    }

    // For final (partial) vector:
    const V tc = load_unaligned(d, row_t + x);
    const V mc = load_unaligned(d, row_m + x);
    const V bc = load_unaligned(d, row_b + x);

    V tr, mr, br;
#if SIMD_TARGET_VALUE == SIMD_NONE
    tr = tc;  // Single-lane => mirrored right neighbor = center value.
    mr = mc;
    br = bc;
#else
    if (kSizeModN == 0) {
      // The above loop didn't handle the last vector because it needs an
      // additional right neighbor (generated via mirroring).
      auto mirror = set_table_indices(d, MirrorLanes(d.N - 1));
      tr = table_lookup_lanes(tc, mirror);
      mr = table_lookup_lanes(mc, mirror);
      br = table_lookup_lanes(bc, mirror);
    } else {
      auto mirror = set_table_indices(d, MirrorLanes((xsize % d.N) - 1));
      // Loads last valid value into uppermost lane and mirrors.
      tr = table_lookup_lanes(load_unaligned(d, row_t + xsize - d.N), mirror);
      mr = table_lookup_lanes(load_unaligned(d, row_m + xsize - d.N), mirror);
      br = table_lookup_lanes(load_unaligned(d, row_b + xsize - d.N), mirror);
    }
#endif

    const V tl = load_unaligned(d, row_t + x - 1);
    const V ml = load_unaligned(d, row_m + x - 1);
    const V bl = load_unaligned(d, row_b + x - 1);
    const V conv = WeightedSum(tl, tr, ml, mr, bl, br, wtb, wm);
    store(conv, d, row_out + x);
  }

  template <size_t kSizeModN, class WrapRow>
  static SIMD_ATTR PIK_INLINE void ConvolveRow(
      LeftRightValid, const float* const PIK_RESTRICT row_m, const size_t xsize,
      const int64_t stride, const WrapRow& wrap_row,
      const Weights3x3& PIK_RESTRICT weights,
      float* const PIK_RESTRICT row_out) {
    const D d;
    // t, m, b = top, middle, bottom row;
    const float* const PIK_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const PIK_RESTRICT row_b = wrap_row(row_m + stride, stride);

    const V wtb = load_dup128(d, weights.tl);
    const V wm = load_dup128(d, weights.ml);

    // l, c, r = left, center, right.
    for (size_t x = 0; x < xsize; x += d.N) {
      const V conv = ConvolveValid(row_t, row_m, row_b, x, wtb, wm);
      store(conv, d, row_out + x);
    }
  }

 private:
  // Returns sum{x_i * w_i}.
  template <class V>
  static SIMD_ATTR PIK_INLINE V WeightedSum(const V tl, const V tr, const V ml,
                                            const V mr, const V bl, const V br,
                                            const V wtb, const V wm) {
    const V sub_m = ml - mr;
    const V mul_m = sub_m * wm;
    const V sub_t = tl - tr;
    const V sub_b = bl - br;
    const V sum_tb = sub_t + sub_b;
    return mul_add(sum_tb, wtb, mul_m);
  }

  static SIMD_ATTR PIK_INLINE V ConvolveValid(const float* PIK_RESTRICT row_t,
                                              const float* PIK_RESTRICT row_m,
                                              const float* PIK_RESTRICT row_b,
                                              const int64_t x, const V wtb,
                                              const V wm) {
    const D d;
    const V tl = load_unaligned(d, row_t + x - 1);
    const V tr = load_unaligned(d, row_t + x + 1);
    const V ml = load_unaligned(d, row_m + x - 1);
    const V mr = load_unaligned(d, row_m + x + 1);
    const V bl = load_unaligned(d, row_b + x - 1);
    const V br = load_unaligned(d, row_b + x + 1);
    return WeightedSum(tl, tr, ml, mr, bl, br, wtb, wm);
  }
};

// 3x3, center row zero, bottom row = negated top row.
class GradY3 {
  using D = SIMD_FULL(float);
  using V = D::V;

 public:
  static constexpr int64_t kRadius = 1;

  // Only accesses pixels in [0, xsize).
  template <size_t kSizeModN, class WrapRow>
  static SIMD_ATTR PIK_INLINE void ConvolveRow(
      LeftRightInvalid, const float* const PIK_RESTRICT row_m,
      const size_t xsize, const int64_t stride, const WrapRow& wrap_row,
      const Weights3x3& weights, float* const PIK_RESTRICT row_out) {
    const D d;
    // t, m, b = top, middle, bottom row;
    const float* const PIK_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const PIK_RESTRICT row_b = wrap_row(row_m + stride, stride);

    // Must load in advance - compiler doesn't understand load_dup128 and
    // schedules them too late.
    const V wlr = load_dup128(d, weights.tl);
    const V wc = load_dup128(d, weights.tc);

    // l, c, r = left, center, right. Leftmost vector: need FirstL1.
    {
      const V tc = load_unaligned(d, row_t + 0);
      const V bc = load_unaligned(d, row_b + 0);
      const V tl = Neighbors::FirstL1(tc);
      const V tr = load_unaligned(d, row_t + 0 + 1);
      const V bl = Neighbors::FirstL1(bc);
      const V br = load_unaligned(d, row_b + 0 + 1);
      const V conv = WeightedSum(tl, tc, tr, bl, bc, br, wlr, wc);
      store(conv, d, row_out + 0);
    }

    // Loop as long as we can load enough new values:
    size_t x = d.N;
    for (; x + d.N + kRadius <= xsize; x += d.N) {
      const auto conv = ConvolveValid(row_t, row_b, x, wlr, wc);
      store(conv, d, row_out + x);
    }

    // For final (partial) vector:
    const V tc = load_unaligned(d, row_t + x);
    const V bc = load_unaligned(d, row_b + x);

    V tr, br;
#if SIMD_TARGET_VALUE == SIMD_NONE
    tr = tc;  // Single-lane => mirrored right neighbor = center value.
    br = bc;
#else
    if (kSizeModN == 0) {
      // The above loop didn't handle the last vector because it needs an
      // additional right neighbor (generated via mirroring).
      auto mirror = set_table_indices(d, MirrorLanes(d.N - 1));
      tr = table_lookup_lanes(tc, mirror);
      br = table_lookup_lanes(bc, mirror);
    } else {
      auto mirror = set_table_indices(d, MirrorLanes((xsize % d.N) - 1));
      // Loads last valid value into uppermost lane and mirrors.
      tr = table_lookup_lanes(load_unaligned(d, row_t + xsize - d.N), mirror);
      br = table_lookup_lanes(load_unaligned(d, row_b + xsize - d.N), mirror);
    }
#endif

    const V tl = load_unaligned(d, row_t + x - 1);
    const V bl = load_unaligned(d, row_b + x - 1);
    const V conv = WeightedSum(tl, tc, tr, bl, bc, br, wlr, wc);
    store(conv, d, row_out + x);
  }

  template <size_t kSizeModN, class WrapRow>
  static SIMD_ATTR PIK_INLINE void ConvolveRow(
      LeftRightValid, const float* const PIK_RESTRICT row_m, const size_t xsize,
      const int64_t stride, const WrapRow& wrap_row,
      const Weights3x3& PIK_RESTRICT weights,
      float* const PIK_RESTRICT row_out) {
    const D d;
    // t, m, b = top, middle, bottom row;
    const float* const PIK_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const PIK_RESTRICT row_b = wrap_row(row_m + stride, stride);

    const V wlr = load_dup128(d, weights.tl);
    const V wc = load_dup128(d, weights.tc);

    // l, c, r = left, center, right.
    for (size_t x = 0; x < xsize; x += d.N) {
      const V conv = ConvolveValid(row_t, row_b, x, wlr, wc);
      store(conv, d, row_out + x);
    }
  }

 private:
  // Returns sum{x_i * w_i}.
  template <class V>
  static SIMD_ATTR PIK_INLINE V WeightedSum(const V tl, const V tc, const V tr,
                                            const V bl, const V bc, const V br,
                                            const V wlr, const V wc) {
    const V sub_c = tc - bc;
    const V mul_c = sub_c * wc;
    const V sub_l = tl - bl;
    const V sub_r = tr - br;
    const V sum_lr = sub_l + sub_r;
    return mul_add(sum_lr, wlr, mul_c);
  }

  static SIMD_ATTR PIK_INLINE V ConvolveValid(const float* PIK_RESTRICT row_t,
                                              const float* PIK_RESTRICT row_b,
                                              const int64_t x, const V wlr,
                                              const V wc) {
    const D d;
    const V tc = load_unaligned(d, row_t + x);
    const V bc = load_unaligned(d, row_b + x);
    const V tl = load_unaligned(d, row_t + x - 1);
    const V tr = load_unaligned(d, row_t + x + 1);
    const V bl = load_unaligned(d, row_b + x - 1);
    const V br = load_unaligned(d, row_b + x + 1);
    return WeightedSum(tl, tc, tr, bl, bc, br, wlr, wc);
  }
};

// 3x3, all but corners zero, br = -tl.
class Corner3 {
  using D = SIMD_FULL(float);
  using V = D::V;

 public:
  static constexpr int64_t kRadius = 1;

  // Only accesses pixels in [0, xsize).
  template <size_t kSizeModN, class WrapRow>
  static SIMD_ATTR PIK_INLINE void ConvolveRow(
      LeftRightInvalid, const float* const PIK_RESTRICT row_m,
      const size_t xsize, const int64_t stride, const WrapRow& wrap_row,
      const Weights3x3& weights, float* const PIK_RESTRICT row_out) {
    const D d;
    // t, m, b = top, middle, bottom row;
    const float* const PIK_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const PIK_RESTRICT row_b = wrap_row(row_m + stride, stride);

    // Must load in advance - compiler doesn't understand load_dup128 and
    // schedules them too late.
    const V w = load_dup128(d, weights.tl);

    // l, c, r = left, center, right. Leftmost vector: need FirstL1.
    {
      const V tc = load_unaligned(d, row_t + 0);
      const V bc = load_unaligned(d, row_b + 0);
      const V tl = Neighbors::FirstL1(tc);
      const V tr = load_unaligned(d, row_t + 0 + 1);
      const V bl = Neighbors::FirstL1(bc);
      const V br = load_unaligned(d, row_b + 0 + 1);
      const V conv = WeightedSum(tl, tr, bl, br, w);
      store(conv, d, row_out + 0);
    }

    // Loop as long as we can load enough new values:
    size_t x = d.N;
    for (; x + d.N + kRadius <= xsize; x += d.N) {
      const auto conv = ConvolveValid(row_t, row_b, x, w);
      store(conv, d, row_out + x);
    }

    // For final (partial) vector:
    const V tc = load_unaligned(d, row_t + x);
    const V bc = load_unaligned(d, row_b + x);

    V tr, br;
#if SIMD_TARGET_VALUE == SIMD_NONE
    tr = tc;  // Single-lane => mirrored right neighbor = center value.
    br = bc;
#else
    if (kSizeModN == 0) {
      // The above loop didn't handle the last vector because it needs an
      // additional right neighbor (generated via mirroring).
      auto mirror = set_table_indices(d, MirrorLanes(d.N - 1));
      tr = table_lookup_lanes(tc, mirror);
      br = table_lookup_lanes(bc, mirror);
    } else {
      auto mirror = set_table_indices(d, MirrorLanes((xsize % d.N) - 1));
      // Loads last valid value into uppermost lane and mirrors.
      tr = table_lookup_lanes(load_unaligned(d, row_t + xsize - d.N), mirror);
      br = table_lookup_lanes(load_unaligned(d, row_b + xsize - d.N), mirror);
    }
#endif

    const V tl = load_unaligned(d, row_t + x - 1);
    const V bl = load_unaligned(d, row_b + x - 1);
    const V conv = WeightedSum(tl, tr, bl, br, w);
    store(conv, d, row_out + x);
  }

  template <size_t kSizeModN, class WrapRow>
  static SIMD_ATTR PIK_INLINE void ConvolveRow(
      LeftRightValid, const float* const PIK_RESTRICT row_m, const size_t xsize,
      const int64_t stride, const WrapRow& wrap_row,
      const Weights3x3& PIK_RESTRICT weights,
      float* const PIK_RESTRICT row_out) {
    const D d;
    // t, m, b = top, middle, bottom row;
    const float* const PIK_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const PIK_RESTRICT row_b = wrap_row(row_m + stride, stride);

    const V w = load_dup128(d, weights.tl);

    // l, c, r = left, center, right.
    for (size_t x = 0; x < xsize; x += d.N) {
      const V conv = ConvolveValid(row_t, row_b, x, w);
      store(conv, d, row_out + x);
    }
  }

 private:
  // Returns sum{x_i * w_i}.
  template <class V>
  static SIMD_ATTR PIK_INLINE V WeightedSum(const V tl, const V tr, const V bl,
                                            const V br, const V w) {
    const V sub_l = tl - bl;
    const V sub_r = br - tr;
    const V sum = sub_l + sub_r;
    return sum * w;
  }

  static SIMD_ATTR PIK_INLINE V ConvolveValid(const float* PIK_RESTRICT row_t,
                                              const float* PIK_RESTRICT row_b,
                                              const int64_t x, const V w) {
    const D d;
    const V tl = load_unaligned(d, row_t + x - 1);
    const V tr = load_unaligned(d, row_t + x + 1);
    const V bl = load_unaligned(d, row_b + x - 1);
    const V br = load_unaligned(d, row_b + x + 1);
    return WeightedSum(tl, tr, bl, br, w);
  }
};

// 3x3, NSEW = 1 and corners == 0.
class Laplacian3 {
  using D = SIMD_FULL(float);
  using V = D::V;

 public:
  static constexpr int64_t kRadius = 1;

  // Only accesses pixels in [0, xsize).
  template <size_t kSizeModN, class WrapRow>
  static SIMD_ATTR PIK_INLINE void ConvolveRow(
      LeftRightInvalid, const float* const PIK_RESTRICT row_m,
      const size_t xsize, const int64_t stride, const WrapRow& wrap_row,
      const Weights3x3& weights, float* const PIK_RESTRICT row_out) {
    const D d;
    // t, m, b = top, middle, bottom row;
    const float* const PIK_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const PIK_RESTRICT row_b = wrap_row(row_m + stride, stride);
    // p, c, n = previous, current, next register.
    auto tp = load_unaligned(d, row_t);
    auto tc = load_unaligned(d, row_t + d.N);
    auto mp = load_unaligned(d, row_m);
    auto mc = load_unaligned(d, row_m + d.N);
    auto bp = load_unaligned(d, row_b);
    auto bc = load_unaligned(d, row_b + d.N);

    const V w0 = load_dup128(d, weights.mc);

    // Leftmost vector: "previous" is actually center.
    const V sum_lr = Neighbors::FirstL1(mp) + Neighbors::R1(mp, mc);
    const V conv = mul_add(mp, w0, tp + bp + sum_lr);
    store(conv, d, row_out + 0);

    // Loop while at least 1 value to load (higher lanes may be uninitialized)
    size_t x = d.N;
    for (; x + d.N + 1 <= xsize; x += d.N) {
      const V tn = load_unaligned(d, row_t + x + d.N);
      const V mn = load_unaligned(d, row_m + x + d.N);
      const V bn = load_unaligned(d, row_b + x + d.N);

      // Here, this is faster than the unaligned loads in Symmetric3!
      const V sum_lr = Neighbors::L1(mc, mp) + Neighbors::R1(mc, mn);
      const V conv = mul_add(mc, w0, (tc + bc) + sum_lr);
      store(conv, d, row_out + x);

      tp = tc;
      tc = tn;
      mp = mc;
      mc = mn;
      bp = bc;
      bc = bn;
    }

#if SIMD_TARGET_VALUE == SIMD_NONE
#else
    // Not a whole vector => need to pad "center" via mirroring.
    if ((xsize % d.N) != 0) {
      // Last valid value in uppermost lane.
      const V t_last = load_unaligned(d, row_t + xsize - d.N);
      const V m_last = load_unaligned(d, row_m + xsize - d.N);
      const V b_last = load_unaligned(d, row_b + xsize - d.N);

      const auto mirror = set_table_indices(d, MirrorLanes(xsize % d.N));
      tc = table_lookup_lanes(t_last, mirror);
      mc = table_lookup_lanes(m_last, mirror);
      bc = table_lookup_lanes(b_last, mirror);
    }
#endif

    // Write the last vector, of which [1, d.N] lanes are valid.
    {
      const V sum_lr = Neighbors::L1(mc, mp) + Neighbors::LastR1(mc);
      const V conv = mul_add(mc, w0, tc + bc + sum_lr);
      store(conv, d, row_out + x);
    }
  }

  // Weights: Manhattan distance 0 = center, 1 = 4-neighborhood, 2 = diagonal.
  template <size_t kSizeModN, class WrapRow>
  static SIMD_ATTR PIK_INLINE void ConvolveRow(
      LeftRightValid, const float* const PIK_RESTRICT row_m, const size_t xsize,
      const int64_t stride, const WrapRow& wrap_row, const Weights3x3& weights,
      float* const PIK_RESTRICT row_out) {
    const D d;
    // t, m, b = top, middle, bottom row;
    const float* const PIK_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const PIK_RESTRICT row_b = wrap_row(row_m + stride, stride);
    // p, c, n = previous, current, next register.
    auto tp = load_unaligned(d, row_t - d.N);
    auto tc = load_unaligned(d, row_t);
    auto mp = load_unaligned(d, row_m - d.N);
    auto mc = load_unaligned(d, row_m);
    auto bp = load_unaligned(d, row_b - d.N);
    auto bc = load_unaligned(d, row_b);

    const V w0 = load_dup128(d, weights.mc);

    // Loop until all output produced. WARNING: padding is uninitialized!
    for (size_t x = 0; x < xsize; x += d.N) {
      const V tn = load_unaligned(d, row_t + x + d.N);
      const V mn = load_unaligned(d, row_m + x + d.N);
      const V bn = load_unaligned(d, row_b + x + d.N);

      const V sum_lr = Neighbors::L1(mc, mp) + Neighbors::R1(mc, mn);
      const V conv = mul_add(mc, w0, (tc + bc) + sum_lr);
      store(conv, d, row_out + x);

      tp = tc;
      tc = tn;
      mp = mc;
      mc = mn;
      bp = bc;
      bc = bn;
    }
  }
};

// 5x5 convolution by separable kernel with a single scan through the input.
class Separable5 {
  using D = SIMD_FULL(float);
  using V = D::V;

 public:
  static constexpr int64_t kRadius = 2;

  template <size_t kSizeModN, class WrapRow>
  static SIMD_ATTR PIK_INLINE void ConvolveRow(
      LeftRightInvalid, const float* const PIK_RESTRICT row_m,
      const size_t xsize, const int64_t stride, const WrapRow& wrap_row,
      const WeightsSeparable5& weights, float* const PIK_RESTRICT row_out) {
    const D d;
    const int64_t neg_stride = -stride;  // allows LEA addressing.
    const float* const PIK_RESTRICT row_t2 =
        wrap_row(row_m + 2 * neg_stride, stride);
    const float* const PIK_RESTRICT row_t1 =
        wrap_row(row_m + 1 * neg_stride, stride);
    const float* const PIK_RESTRICT row_b1 =
        wrap_row(row_m + 1 * stride, stride);
    const float* const PIK_RESTRICT row_b2 =
        wrap_row(row_m + 2 * stride, stride);

    const V wh0 = load_dup128(d, weights.horz + 0 * 4);
    const V wh1 = load_dup128(d, weights.horz + 1 * 4);
    const V wh2 = load_dup128(d, weights.horz + 2 * 4);
    const V wv0 = load_dup128(d, weights.vert + 0 * 4);
    const V wv1 = load_dup128(d, weights.vert + 1 * 4);
    const V wv2 = load_dup128(d, weights.vert + 2 * 4);

    size_t x = 0;

    // Need to loop more than once for scalars (d.N == 1).
    for (; x < kRadius; x += d.N) {
      const V conv0 = HorzConvolveFirst(row_m, x, xsize, wh0, wh1, wh2) * wv0;

      const V conv1t = HorzConvolveFirst(row_t1, x, xsize, wh0, wh1, wh2);
      const V conv1b = HorzConvolveFirst(row_b1, x, xsize, wh0, wh1, wh2);
      const V conv1 = mul_add(conv1t + conv1b, wv1, conv0);

      const V conv2t = HorzConvolveFirst(row_t2, x, xsize, wh0, wh1, wh2);
      const V conv2b = HorzConvolveFirst(row_b2, x, xsize, wh0, wh1, wh2);
      const V conv2 = mul_add(conv2t + conv2b, wv2, conv1);
      store(conv2, d, row_out + x);
    }

    // Main loop: load inputs without padding
    for (; x + d.N + kRadius <= xsize; x += d.N) {
      const V conv0 = HorzConvolve(row_m + x, wh0, wh1, wh2) * wv0;

      const V conv1t = HorzConvolve(row_t1 + x, wh0, wh1, wh2);
      const V conv1b = HorzConvolve(row_b1 + x, wh0, wh1, wh2);
      const V conv1 = mul_add(conv1t + conv1b, wv1, conv0);

      const V conv2t = HorzConvolve(row_t2 + x, wh0, wh1, wh2);
      const V conv2b = HorzConvolve(row_b2 + x, wh0, wh1, wh2);
      const V conv2 = mul_add(conv2t + conv2b, wv2, conv1);
      store(conv2, d, row_out + x);
    }

    // Last full vector to write (the above loop handled mod >= 2)
#if SIMD_TARGET_VALUE == SIMD_NONE
    while (x < xsize) {
#else
    if (kSizeModN < 2) {
#endif
      const V conv0 =
          HorzConvolveLast<kSizeModN>(row_m, x, xsize, wh0, wh1, wh2) * wv0;

      const V conv1t =
          HorzConvolveLast<kSizeModN>(row_t1, x, xsize, wh0, wh1, wh2);
      const V conv1b =
          HorzConvolveLast<kSizeModN>(row_b1, x, xsize, wh0, wh1, wh2);
      const V conv1 = mul_add(conv1t + conv1b, wv1, conv0);

      const V conv2t =
          HorzConvolveLast<kSizeModN>(row_t2, x, xsize, wh0, wh1, wh2);
      const V conv2b =
          HorzConvolveLast<kSizeModN>(row_b2, x, xsize, wh0, wh1, wh2);
      const V conv2 = mul_add(conv2t + conv2b, wv2, conv1);
      store(conv2, d, row_out + x);
      x += d.N;
    }

    // If mod = 0, the above vector was the last.
    if (kSizeModN != 0) {
      for (; x < xsize; ++x) {
        float mul = 0.0f;
        for (int64_t dy = -kRadius; dy <= kRadius; ++dy) {
          const float wy = weights.vert[std::abs(dy) * 4];
          const float* clamped_row = wrap_row(row_m + dy * stride, stride);
          for (int64_t dx = -kRadius; dx <= kRadius; ++dx) {
            const float wx = weights.horz[std::abs(dx) * 4];
            const int64_t clamped_x = Mirror(x + dx, xsize);
            mul += clamped_row[clamped_x] * wx * wy;
          }
        }
        row_out[x] = mul;
      }
    }
  }

  template <size_t kSizeModN, class WrapRow>
  static SIMD_ATTR PIK_INLINE void ConvolveRow(
      LeftRightValid, const float* const PIK_RESTRICT row_m, const size_t xsize,
      const int64_t stride, const WrapRow& wrap_row,
      const WeightsSeparable5& weights, float* const PIK_RESTRICT row_out) {
    const D d;
    const int64_t neg_stride = -stride;  // allows LEA addressing.
    const float* const PIK_RESTRICT row_t2 =
        wrap_row(row_m + 2 * neg_stride, stride);
    const float* const PIK_RESTRICT row_t1 =
        wrap_row(row_m + 1 * neg_stride, stride);
    const float* const PIK_RESTRICT row_b1 =
        wrap_row(row_m + 1 * stride, stride);
    const float* const PIK_RESTRICT row_b2 =
        wrap_row(row_m + 2 * stride, stride);

    const V wh0 = load_dup128(d, weights.horz + 0 * 4);
    const V wh1 = load_dup128(d, weights.horz + 1 * 4);
    const V wh2 = load_dup128(d, weights.horz + 2 * 4);
    const V wv0 = load_dup128(d, weights.vert + 0 * 4);
    const V wv1 = load_dup128(d, weights.vert + 1 * 4);
    const V wv2 = load_dup128(d, weights.vert + 2 * 4);

    // Loop until all output produced. WARNING: padding is uninitialized!
    for (size_t x = 0; x < xsize; x += d.N) {
      const V conv0 = HorzConvolve(row_m + x, wh0, wh1, wh2) * wv0;

      const V conv1t = HorzConvolve(row_t1 + x, wh0, wh1, wh2);
      const V conv1b = HorzConvolve(row_b1 + x, wh0, wh1, wh2);
      const V conv1 = mul_add(conv1t + conv1b, wv1, conv0);

      const V conv2t = HorzConvolve(row_t2 + x, wh0, wh1, wh2);
      const V conv2b = HorzConvolve(row_b2 + x, wh0, wh1, wh2);
      const V conv2 = mul_add(conv2t + conv2b, wv2, conv1);

      store(conv2, d, row_out + x);
    }
  }

 private:
  // Same as HorzConvolve for the first/last vector in a row.
  static SIMD_ATTR PIK_INLINE V HorzConvolveFirst(
      const float* const PIK_RESTRICT row, const int64_t x, const int64_t xsize,
      const V wh0, const V wh1, const V wh2) {
    const D d;
    const V c = load_unaligned(d, row + x);
    const V mul0 = c * wh0;

#if SIMD_TARGET_VALUE == SIMD_NONE
    const V l1 = load_unaligned(d, row + Mirror(x - 1, xsize));
    const V l2 = load_unaligned(d, row + Mirror(x - 2, xsize));
#else
    const V l1 = Neighbors::FirstL1(c);
    const V l2 = Neighbors::FirstL2(c);
#endif

    const V r1 = load_unaligned(d, row + x + 1);
    const V r2 = load_unaligned(d, row + x + 2);

    const V mul1 = mul_add(l1 + r1, wh1, mul0);
    const V mul2 = mul_add(l2 + r2, wh2, mul1);
    return mul2;
  }

  template <size_t kSizeModN>
  static SIMD_ATTR PIK_INLINE V
  HorzConvolveLast(const float* const PIK_RESTRICT row, const int64_t x,
                   const int64_t xsize, const V wh0, const V wh1, const V wh2) {
    const D d;
    const V c = load_unaligned(d, row + x);
    const V mul0 = c * wh0;

    const V l1 = load_unaligned(d, row + x - 1);
    const V l2 = load_unaligned(d, row + x - 2);

    V r1, r2;
#if SIMD_TARGET_VALUE == SIMD_NONE
    r1 = load_unaligned(d, row + Mirror(x + 1, xsize));
    r2 = load_unaligned(d, row + Mirror(x + 2, xsize));
#else
    if (kSizeModN == 0) {
      r2 = table_lookup_lanes(c, set_table_indices(d, MirrorLanes(d.N - 2)));
      r1 = table_lookup_lanes(c, set_table_indices(d, MirrorLanes(d.N - 1)));
    } else {  // == 1
      const auto last = load_unaligned(d, row + xsize - d.N);
      r2 = table_lookup_lanes(last, set_table_indices(d, MirrorLanes(d.N - 1)));
      r1 = last;
    }
#endif

    // Sum of pixels with Manhattan distance i, multiplied by weights[i].
    const V sum1 = l1 + r1;
    const V mul1 = mul_add(sum1, wh1, mul0);
    const V sum2 = l2 + r2;
    const V mul2 = mul_add(sum2, wh2, mul1);
    return mul2;
  }

  // Requires kRadius valid pixels before/after pos.
  static SIMD_ATTR PIK_INLINE V
  HorzConvolve(const float* const PIK_RESTRICT pos, const V wh0, const V wh1,
               const V wh2) {
    const D d;
    const V c = load_unaligned(d, pos);
    const V mul0 = c * wh0;

    // Loading anew is faster than combining vectors.
    const V l1 = load_unaligned(d, pos - 1);
    const V r1 = load_unaligned(d, pos + 1);
    const V l2 = load_unaligned(d, pos - 2);
    const V r2 = load_unaligned(d, pos + 2);
    // Sum of pixels with Manhattan distance i, multiplied by weights[i].
    const V sum1 = l1 + r1;
    const V mul1 = mul_add(sum1, wh1, mul0);
    const V sum2 = l2 + r2;
    const V mul2 = mul_add(sum2, wh2, mul1);
    return mul2;
  }
};

}  // namespace strategy

// Avoids PadImage, but requires a strategy that supports LeftRightInvalid.
struct BorderNeverUsed {};

// Slow: Convolve calls PadImage and requires bounds checks.
struct BorderNeedsInit {};

// 3x3 kernels require inputs at least this wide - for the first vector, they
// load right neighbors (N lanes starting from x + 1).
static constexpr size_t kConvolveMinWidth = SIMD_FULL(float)::N + 1;

// Single entry point for convolution.
// "Strategy" (Direct*/Separable*) decides kernel size and how to evaluate it.
template <class Strategy>
class ConvolveT {
  static constexpr int64_t kRadius = Strategy::kRadius;

 public:
  // Uses default Border/Executor. "Image" is ImageF or Image3F.
  template <class Image, class Kernel>
  static SIMD_ATTR PIK_INLINE void Run(const Image& in, const Kernel& kernel,
                                       const Image* out) {
    Run(BorderNeverUsed(), ExecutorLoop(), in, kernel, out);
  }

  // "Border" is Border{NeverUsed/NeedsInit/AlreadyValid}.
  // "Executor": ExecutorPool uses a ThreadPool; ExecutorLoop just loops.
  // "Image" is ImageF or Image3F.
  template <class Border, class Executor, class Image, class Kernel>
  static SIMD_ATTR PIK_INLINE void Run(const Border border,
                                       const Executor executor, const Image& in,
                                       const Kernel& kernel, const Image* out) {
    PIK_CHECK(SameSize(in, *out));
    PIK_CHECK(in.xsize() >= kConvolveMinWidth);  // For BorderNeverUsed.
    RunImpl(border, executor, in, kernel, out);
  }

 private:
  template <size_t kSizeModN, class LeftRight, class WrapRow, class Kernel>
  static SIMD_ATTR PIK_INLINE void RunRow(const float* PIK_RESTRICT in,
                                          const size_t xsize,
                                          const int64_t stride,
                                          const WrapRow& wrap_row,
                                          const Kernel& kernel,
                                          const float* PIK_RESTRICT out) {
    // LeftRight value instead of template arg enables overload resolution.
    Strategy::template ConvolveRow<kSizeModN>(LeftRight(), in, xsize, stride,
                                              wrap_row, kernel.Weights(),
                                              const_cast<float*>(out));
  }

  template <size_t kSizeModN, class LeftRight, class Kernel>
  static SIMD_ATTR PIK_INLINE void RunBorder(const ImageF& in,
                                             const int64_t ybegin,
                                             const int64_t yend,
                                             const Kernel& kernel,
                                             const ImageF* out) {
    const int64_t stride = in.PixelsPerRow();
    const WrapRowMirror wrap_row(in, in.ysize());
    for (int64_t y = ybegin; y < yend; ++y) {
      RunRow<kSizeModN, LeftRight>(in.ConstRow(y), in.xsize(), stride, wrap_row,
                                   kernel, out->Row(y));
    }
  }

  // Image3F.
  template <size_t kSizeModN, class LeftRight, class Kernel>
  static SIMD_ATTR PIK_INLINE void RunBorder(const Image3F& in,
                                             const int64_t ybegin,
                                             const int64_t yend,
                                             const Kernel& kernel,
                                             const Image3F* out) {
    const int64_t stride = in.PixelsPerRow();
    for (int64_t y = ybegin; y < yend; ++y) {
      for (size_t c = 0; c < 3; ++c) {
        const WrapRowMirror wrap_row(in.Plane(c), in.ysize());
        RunRow<kSizeModN, LeftRight>(in.ConstPlaneRow(c, y), in.xsize(), stride,
                                     wrap_row, kernel, out->PlaneRow(c, y));
      }
    }
  }

  // Threaded.
  template <size_t kSizeModN, class LeftRight, class Kernel>
  static SIMD_ATTR PIK_INLINE void RunInterior(
      const ExecutorPool executor, const ImageF& in, const int64_t ybegin,
      const int64_t yend, const Kernel& kernel, const ImageF* out) {
    // There is no interior if ysize <= 2 * kRadius.
    if (ybegin >= yend) return;

    const int64_t stride = in.PixelsPerRow();
    executor.Run(
        ybegin, yend,
        [&in, stride, &kernel, out](const int y, const int thread) SIMD_ATTR {
          RunRow<kSizeModN, LeftRight>(in.ConstRow(y), in.xsize(), stride,
                                       WrapRowUnchanged(), kernel, out->Row(y));
        },
        "Convolve");
  }

  // Threaded, Image3.
  template <size_t kSizeModN, class LeftRight, class Kernel>
  static SIMD_ATTR PIK_INLINE void RunInterior(
      const ExecutorPool executor, const Image3F& in, const int64_t ybegin,
      const int64_t yend, const Kernel& kernel, const Image3F* out) {
    // There is no interior if ysize <= 2 * kRadius.
    if (ybegin >= yend) return;

    const int64_t stride = in.PixelsPerRow();
    executor.Run(
        ybegin, yend,
        [&in, stride, &kernel, out](const int y, const int thread) SIMD_ATTR {
          for (size_t c = 0; c < 3; ++c) {
            RunRow<kSizeModN, LeftRight>(in.ConstPlaneRow(c, y), in.xsize(),
                                         stride, WrapRowUnchanged(), kernel,
                                         out->PlaneRow(c, y));
          }
        },
        "Convolve3");
  }

  // Plain loop.
  template <size_t kSizeModN, class LeftRight, class Kernel>
  static SIMD_ATTR PIK_INLINE void RunInterior(ExecutorLoop, const ImageF& in,
                                               const int64_t ybegin,
                                               const int64_t yend,
                                               const Kernel& kernel,
                                               const ImageF* out) {
    const int64_t stride = in.PixelsPerRow();
    const float* row_in = in.ConstRow(ybegin);
    const float* row_out = out->Row(ybegin);  // RunRow casts to float*.
    for (int64_t y = ybegin; y < yend; ++y) {
      RunRow<kSizeModN, LeftRight>(row_in, in.xsize(), stride,
                                   WrapRowUnchanged(), kernel, row_out);
      row_in += in.PixelsPerRow();
      row_out += out->PixelsPerRow();
    }
  }

  // Plain loop, Image3.
  template <size_t kSizeModN, class LeftRight, class Kernel>
  static SIMD_ATTR PIK_INLINE void RunInterior(ExecutorLoop, const Image3F& in,
                                               const int64_t ybegin,
                                               const int64_t yend,
                                               const Kernel& kernel,
                                               const Image3F* out) {
    const int64_t stride = in.PixelsPerRow();
    for (int64_t y = ybegin; y < yend; ++y) {
      for (size_t c = 0; c < 3; ++c) {
        const float* row_in = in.ConstPlaneRow(c, ybegin);
        const float* row_out = out->PlaneRow(c, ybegin);  // Used as float*.
        RunRow<kSizeModN, LeftRight>(row_in, in.xsize(), stride,
                                     WrapRowUnchanged(), kernel, row_out);
        row_in += in.PixelsPerRow();
        row_out += out->PixelsPerRow();
      }
    }
  }

  template <size_t kSizeModN, class LeftRight, class Executor, class Image,
            class Kernel>
  static SIMD_ATTR PIK_INLINE void RunWithBoundsChecks(const Executor executor,
                                                       const Image& in,
                                                       const Kernel& kernel,
                                                       const Image* out) {
    const int64_t ysize = in.ysize();
    RunBorder<kSizeModN, LeftRight>(in, 0, kRadius, kernel, out);
    RunInterior<kSizeModN, LeftRight>(executor, in, kRadius, ysize - kRadius,
                                      kernel, out);
    RunBorder<kSizeModN, LeftRight>(in, ysize - kRadius, ysize, kernel, out);
  }

  // Ensures each row has an additional vector's worth of valid values on the
  // right AND left borders (residing in otherwise unused padding area reserved
  // by BytesPerRow), initialized via mirroring with replication.
  static void PadImage(const size_t xsize, const size_t ysize,
                       const ImageF* image) {
    using T = float;
    PIK_ASSERT(xsize > kRadius && ysize > kRadius);
    static_assert(kRadius * sizeof(T) <= kMaxVectorSize, "Not enough padding");

    for (size_t y = 0; y < ysize; ++y) {
      // Even if the image is const, we're allowed to overwrite its padding.
      T* const PIK_RESTRICT row = const_cast<T*>(image->ConstRow(y));

      for (int64_t i = 0; i < kRadius; ++i) {
        row[xsize + i] = row[Mirror(xsize + i, xsize)];
        row[-1 - i] = row[i];
      }
    }
  }

  // Same for Image3.
  static void PadImage(const size_t xsize, const size_t ysize,
                       const Image3F* image) {
    using T = float;
    PIK_ASSERT(xsize > kRadius && ysize > kRadius);
    static_assert(kRadius * sizeof(T) <= kMaxVectorSize, "Not enough padding");

    for (size_t y = 0; y < ysize; ++y) {
      for (size_t c = 0; c < 3; ++c) {
        // Even if the image is const, we're allowed to overwrite its padding.
        T* const PIK_RESTRICT row = const_cast<T*>(image->ConstPlaneRow(c, y));

        for (int64_t i = 0; i < kRadius; ++i) {
          row[xsize + i] = row[Mirror(xsize + i, xsize)];
          row[-1 - i] = row[i];
        }
      }
    }
  }

  // Slow path: padding and bounds checks.
  template <class Executor, class Image, class Kernel>
  static SIMD_ATTR PIK_INLINE void RunImpl(BorderNeedsInit,
                                           const Executor executor,
                                           const Image& in,
                                           const Kernel& kernel,
                                           const Image* out) {
    PROFILER_ZONE("Convolve slow");
    // Each RunRow requires that 2*kRadius+1 rows already be padded. Padding
    // the entire image pollutes the cache. We could pre-pad 2*kRadius rows and
    // then one row per RunRow, but callers who care about speed should anyway
    // use the other, faster Border modes.
    PadImage(in.xsize(), in.ysize(), &in);

    switch (in.xsize() % SIMD_FULL(float)::N) {
      case 0:
        return RunWithBoundsChecks<0, LeftRightValid>(executor, in, kernel,
                                                      out);
      case 1:
        return RunWithBoundsChecks<1, LeftRightValid>(executor, in, kernel,
                                                      out);
      default:  // Only need <= kRadius
        return RunWithBoundsChecks<2, LeftRightValid>(executor, in, kernel,
                                                      out);
    }
  }

  // Fast: no padding, but bounds checks.
  template <class Image, class Executor, class Kernel>
  static SIMD_ATTR PIK_INLINE void RunImpl(BorderNeverUsed,
                                           const Executor executor,
                                           const Image& in, const Kernel kernel,
                                           const Image* out) {
    PROFILER_ZONE("Convolve fast");

    switch (in.xsize() % SIMD_FULL(float)::N) {
      case 0:
        return RunWithBoundsChecks<0, LeftRightInvalid>(executor, in, kernel,
                                                        out);
      case 1:
        return RunWithBoundsChecks<1, LeftRightInvalid>(executor, in, kernel,
                                                        out);
      default:  // Only need <= kRadius
        return RunWithBoundsChecks<2, LeftRightInvalid>(executor, in, kernel,
                                                        out);
    }
  }
};

}  // namespace pik

#endif  // PIK_CONVOLVE_H_
