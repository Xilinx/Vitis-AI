// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// Fast SIMD cubic upsampling.

#ifndef PIK_RESAMPLE_H_
#define PIK_RESAMPLE_H_

#include <stddef.h>
#include <atomic>
#include <cmath>

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/data_parallel.h"
#include "pik/image.h"
#include "pik/image_ops.h"
#include "pik/profiler.h"
#include "pik/simd/simd.h"
#include "pik/status.h"

namespace pik {

// Main entry points. Executor{Loop/Pool} are from data_parallel.h.
// Results are better if input pixels are non-linear (e.g. gamma-compressed).

// (Possibly) multithreaded, single channel: called by all other Upsample.
template <class Upsampler, class Executor, class Kernel>
SIMD_ATTR PIK_INLINE void Upsample(const Executor executor, const ImageF& in,
                                   const Kernel& kernel, ImageF* out) {
  Upsampler::Run(executor, in, kernel, out);
}

// (Possibly) multithreaded, RGB.
template <class Upsampler, class Executor, class Kernel>
SIMD_ATTR PIK_INLINE void Upsample(const Executor executor, const Image3F& in,
                                   const Kernel& kernel, Image3F* out) {
  Upsampler::Run(executor, in, kernel, out);
}

// Single-thread, single channel.
template <class Upsampler, class Kernel>
SIMD_ATTR PIK_INLINE void Upsample(const ImageF& in, const Kernel& kernel,
                                   ImageF* PIK_RESTRICT out) {
  Upsample<Upsampler>(ExecutorLoop(), in, kernel, out);
}

// Single-thread, RGB.
template <class Upsampler, class Kernel>
SIMD_ATTR PIK_INLINE void Upsample(const Image3F& in, const Kernel& kernel,
                                   Image3F* PIK_RESTRICT out) {
  Upsample<Upsampler>(ExecutorLoop(), in, kernel, out);
}

namespace kernel {

// Arbitrary, possibly non-separable kernel.
template <int kRadiusArg>
class Custom {
 public:
  static constexpr int kRadius = kRadiusArg;
  static constexpr int kWidth = 2 * kRadius;

  // Derive from impulse response image.
  static Custom FromResult(const ImageF& result) {
    PIK_ASSERT(result.xsize() % 8 == 0);
    PIK_ASSERT(result.ysize() % 8 == 0);
    PIK_ASSERT(result.xsize() / 8 >= kWidth);
    PIK_ASSERT(result.ysize() / 8 >= kWidth);
    const int off_x = result.xsize() / 8 / 2 + kRadius;
    const int off_y = result.ysize() / 8 / 2 + kRadius;
    Custom kernel;
    int idx = 0;
    for (int mod_y = 0; mod_y < 8; mod_y++) {
      for (int tap_y = 0; tap_y < kWidth; tap_y++) {
        for (int tap_x = 0; tap_x < kWidth; tap_x++) {
          for (int mod_x = 0; mod_x < 8; mod_x++) {
            const int wrap_mod_x = mod_x >= 4 ? mod_x - 8 : mod_x;
            const int wrap_mod_y = mod_y >= 4 ? mod_y - 8 : mod_y;
            const int x = (off_x - tap_x) * 8 + wrap_mod_x;
            const int y = (off_y - tap_y) * 8 + wrap_mod_y;
            kernel.weights_[idx++] = result.Row(y)[x];
          }
        }
      }
    }
    return kernel;
  }

  const char* Name() const { return "Custom"; }
  const float* Weights2D() const { return weights_; }

 private:
  SIMD_ALIGN float weights_[8 * kWidth * kWidth * 8];
};

class CatmullRom {
  // constexpr functions for precomputing Weights_i() at compile time.

  static constexpr float Abs(const float x) { return x < 0.0f ? -x : x; }
  static constexpr int Ceil(const float x) {
    return (static_cast<float>(static_cast<int>(x)) == x)
               ? static_cast<int>(x)
               : static_cast<int>(x) + ((x > 0.0f) ? 1 : 0);
  }

  static constexpr float F0_1(float u) {
    return ((-1.5f * u + 2.0f) * u + 0.5f) * u;
  }
  static constexpr float F1_2(float u) { return ((0.5f * u - 0.5f) * u) * u; }

  static constexpr float EvalNonNegative(const float x) {
    return x > 2.0f ? 0.0f : x > 1.0f ? F1_2(2.0f - x) : F0_1(1.0f - x);
  }
  static constexpr float Eval(const float x) { return EvalNonNegative(Abs(x)); }

  static constexpr float InX(const float out_x) {
    return (out_x + 0.5f) / 8 - 0.5f;
  }

  // template enables static_assert.
  template <int tap, int mod>
  static constexpr float Weight() {
    static_assert(0 <= tap && tap < 2 * kRadius, "Invalid tap");
    static_assert(0 <= mod && mod < 8, "Invalid mod");
    return Eval((InX(mod) - (Ceil(InX(mod) - kRadius) + tap)));
  }

  template <int tap_y, int tap_x, int mod_y, int mod_x>
  static constexpr float Weight2D() {
    return Weight<tap_y, mod_y>() * Weight<tap_x, mod_x>();
  }

 public:
  static constexpr int kRadius = 2;  // cubic

  const char* Name() const { return "CatmullRom"; }

  constexpr float operator()(const float x) const { return Eval(x); }

  // Precomputed weights for upscalers with separate X/Y muls.

#define PIK_FOREACH_MOD_X(mod_y, tap_x)                           \
  Weight<tap_x, 0>(), Weight<tap_x, 1>(), Weight<tap_x, 2>(),     \
      Weight<tap_x, 3>(), Weight<tap_x, 4>(), Weight<tap_x, 5>(), \
      Weight<tap_x, 6>(), Weight<tap_x, 7>()

#define PIK_FOREACH_TAP_X_AND_Y(mod_y)                                  \
  /* [tap_x=4][mod_x=8]: */                                             \
  PIK_FOREACH_MOD_X(mod_y, 0), PIK_FOREACH_MOD_X(mod_y, 1),             \
      PIK_FOREACH_MOD_X(mod_y, 2),                                      \
      PIK_FOREACH_MOD_X(mod_y, 3), /* [tap_y=4][4]: */                  \
      SIMD_REP4((Weight<0, mod_y>())), SIMD_REP4((Weight<1, mod_y>())), \
      SIMD_REP4((Weight<2, mod_y>())), SIMD_REP4((Weight<3, mod_y>()))

#define PIK_FOREACH_MOD_Y                                     \
  PIK_FOREACH_TAP_X_AND_Y(0), PIK_FOREACH_TAP_X_AND_Y(1),     \
      PIK_FOREACH_TAP_X_AND_Y(2), PIK_FOREACH_TAP_X_AND_Y(3), \
      PIK_FOREACH_TAP_X_AND_Y(4), PIK_FOREACH_TAP_X_AND_Y(5), \
      PIK_FOREACH_TAP_X_AND_Y(6), PIK_FOREACH_TAP_X_AND_Y(7)

  PIK_INLINE const float* WeightsSeparated() const {
    // Memory layout required for SIMD (we load 4..8 consecutive mod_x):
    // For each mod_y(8): [tap_x=4][mod_x=8], [4x tap_y=4] = 384
    // (repeating the tap_x/mod_x for every mod_y is wasteful but avoids
    // needing two separate weight pointers/GetWeights)
    SIMD_ALIGN static constexpr float weights[8 * 48] = {PIK_FOREACH_MOD_Y};
    return weights;
  }

#undef PIK_FOREACH_MOD_Y
#undef PIK_FOREACH_TAP_X_AND_Y
#undef PIK_FOREACH_MOD_X

  // Precomputed weights for non-separable upscalers.

#define PIK_FOREACH_MOD_X(mod_y, tap_y, tap_x)                                \
  Weight2D<tap_y, tap_x, mod_y, 0>(), Weight2D<tap_y, tap_x, mod_y, 1>(),     \
      Weight2D<tap_y, tap_x, mod_y, 2>(), Weight2D<tap_y, tap_x, mod_y, 3>(), \
      Weight2D<tap_y, tap_x, mod_y, 4>(), Weight2D<tap_y, tap_x, mod_y, 5>(), \
      Weight2D<tap_y, tap_x, mod_y, 6>(), Weight2D<tap_y, tap_x, mod_y, 7>()

#define PIK_FOREACH_TAP_X(mod_y, tap_y)                                   \
  PIK_FOREACH_MOD_X(mod_y, tap_y, 0), PIK_FOREACH_MOD_X(mod_y, tap_y, 1), \
      PIK_FOREACH_MOD_X(mod_y, tap_y, 2), PIK_FOREACH_MOD_X(mod_y, tap_y, 3)

#define PIK_FOREACH_TAP_Y(mod_y)                            \
  PIK_FOREACH_TAP_X(mod_y, 0), PIK_FOREACH_TAP_X(mod_y, 1), \
      PIK_FOREACH_TAP_X(mod_y, 2), PIK_FOREACH_TAP_X(mod_y, 3)

#define PIK_FOREACH_MOD_Y                                               \
  PIK_FOREACH_TAP_Y(0), PIK_FOREACH_TAP_Y(1), PIK_FOREACH_TAP_Y(2),     \
      PIK_FOREACH_TAP_Y(3), PIK_FOREACH_TAP_Y(4), PIK_FOREACH_TAP_Y(5), \
      PIK_FOREACH_TAP_Y(6), PIK_FOREACH_TAP_Y(7)

  PIK_INLINE const float* Weights2D() const {
    // Memory layout required for SIMD (we load 4..8 consecutive mod_x):
    // 4D array: [mod_y=8][tap_y=4][tap_x=4][mod_x=8] = 1024 entries
    SIMD_ALIGN static constexpr float weights[1024] = {PIK_FOREACH_MOD_Y};
    return weights;
  }

#undef PIK_FOREACH_MOD_Y
#undef PIK_FOREACH_TAP_Y
#undef PIK_FOREACH_TAP_X
#undef PIK_FOREACH_MOD_X
};

// 6-tap
class Lanczos3 {
  static constexpr float Abs(const float x) { return x < 0.0f ? -x : x; }

  static /*constexpr*/ float Sinc(const float x) {
    const float t = x * 3.1415926536f;
    return x == 0.0f ? 1.0f : sin(t) / t;
  }

  static /*constexpr*/ float EvalNonNegative(const float x) {
    return x > 3.0f ? 0.0f : Sinc(x) * Sinc(x * 0.333333333f);
  }
  static /*constexpr*/ float Eval(const float x) {
    return EvalNonNegative(Abs(x));
  }

 public:
  static constexpr int kRadius = 3;

  const char* Name() const { return "Lanczos3"; }

  /*constexpr*/ float operator()(const float x) const { return Eval(x); }
};

}  // namespace kernel

namespace slow {

// For verifying Upsampler8. Supports any scale factor and kernel size, but slow
// (cache thrashing) due to separate X/Y passes through the entire image.
class Upsampler {
 public:
  // TODO(janwas): add ExecutorPool overload
  template <class Executor, class Kernel>
  static void Run(const Executor executor, const ImageF& in,
                  const Kernel& kernel, ImageF* PIK_RESTRICT out) {
    const size_t in_xsize = in.xsize();
    const size_t in_ysize = in.ysize();
    const size_t out_xsize = out->xsize();
    const size_t out_ysize = out->ysize();
    ImageF resampled_rows(out_xsize, in_ysize);
    PROFILER_ZONE("slow::Upsampler");

    for (size_t y = 0; y < in_ysize; ++y) {
      const float* PIK_RESTRICT in_row = in.ConstRow(y);
      float* PIK_RESTRICT out_row = resampled_rows.Row(y);
      Upsample1D(in_row, in_xsize, 1, kernel, out_row, out_xsize, 1);
    }

    const size_t in_stride = resampled_rows.PixelsPerRow();
    const size_t out_stride = out->PixelsPerRow();
    for (size_t out_x = 0; out_x < out_xsize; ++out_x) {
      const float* PIK_RESTRICT in_col = resampled_rows.Row(0) + out_x;
      float* PIK_RESTRICT out_col = out->Row(0) + out_x;
      Upsample1D(in_col, in_ysize, in_stride, kernel, out_col, out_ysize,
                 out_stride);
    }
  }

  template <class Executor, class Kernel>
  static void Run(const Executor executor, const Image3F& in,
                  const Kernel& kernel, Image3F* PIK_RESTRICT out) {
    // Unoptimized: separate planes (additional fork/join)
    for (int c = 0; c < 3; ++c) {
      Run(executor, in.Plane(c), kernel, const_cast<ImageF*>(&out->Plane(c)));
    }
  }

 private:
  template <class Kernel>
  static void Upsample1D(const float* PIK_RESTRICT in, const size_t in_size,
                         const size_t in_stride, const Kernel& kernel,
                         float* PIK_RESTRICT out, const int64_t out_size,
                         const size_t out_stride) {
    for (int64_t idx_out = 0; idx_out < out_size; idx_out++) {
      // Position in input/output, [0, 1].
      const float x = (idx_out + 0.5f) / out_size;
      const float in_x = x * in_size - 0.5f;
      // Leftmost sample index.
      const int in_min_x = static_cast<int>(std::ceil(in_x - Kernel::kRadius));

      float sum = 0.0f;
      for (int i = in_min_x; i < in_min_x + 2 * Kernel::kRadius; i++) {
        const int64_t mirror = Mirror(i, in_size);
        sum += in[mirror * in_stride] * kernel(in_x - i);
      }
      out[idx_out * out_stride] = sum;
    }
  }
};

// For verifying GeneralUpsampler8 using a kernel that is actually separable.
// Computes tensor product using Kernel::operator(). Supports any scale factor
// and kernel size.
class GeneralUpsamplerFromSeparable {
 public:
  // TODO(janwas): add ExecutorPool overload
  template <class Executor, class Kernel>
  static void Run(const Executor executor, const ImageF& in,
                  const Kernel& kernel, ImageF* PIK_RESTRICT out) {
    const size_t in_xsize = in.xsize();
    const size_t in_ysize = in.ysize();
    const size_t out_xsize = out->xsize();
    const size_t out_ysize = out->ysize();
    PROFILER_ZONE("slow::GeneralUpsamplerFromSeparable");

    const int64_t kWidth = 2 * Kernel::kRadius;  // even

    for (size_t out_y = 0; out_y < out_ysize; ++out_y) {
      float* PIK_RESTRICT out_row = out->Row(out_y);
      const float in_fy = ((out_y + 0.5f) / out_ysize) * in_ysize - 0.5f;
      const int64_t top = std::ceil(in_fy - Kernel::kRadius);

      const float* PIK_RESTRICT in_rows[kWidth];
      float wy[kWidth];
      for (int64_t i = 0; i < kWidth; ++i) {
        in_rows[i] = in.ConstRow(Mirror(top + i, in_ysize));
        wy[i] = kernel(in_fy - (top + i));
      }

      for (int64_t out_x = 0; out_x < out_xsize; out_x++) {
        const float in_fx = ((out_x + 0.5f) / out_xsize) * in_xsize - 0.5f;
        const int64_t left = std::ceil(in_fx - Kernel::kRadius);

        int64_t in_x[kWidth];
        float wx[kWidth];
        for (int64_t i = 0; i < kWidth; ++i) {
          in_x[i] = Mirror(left + i, in_xsize);
          wx[i] = kernel(in_fx - (left + i));
        }

        float sum = 0.0f;
        for (size_t r = 0; r < kWidth; ++r) {
          const float* PIK_RESTRICT in_row = in_rows[r];
          for (size_t c = 0; c < kWidth; ++c) {
            sum += in_row[in_x[c]] * wy[r] * wx[c];
          }
        }

        out_row[out_x] = sum;
      }
    }
  }

  template <class Executor, class Kernel>
  static void Run(const Executor executor, const Image3F& in,
                  const Kernel& kernel, Image3F* PIK_RESTRICT out) {
    // Unoptimized: separate planes (additional fork/join)
    for (int c = 0; c < 3; ++c) {
      Run(executor, in.Plane(c), kernel, const_cast<ImageF*>(&out->Plane(c)));
    }
  }
};

// Supports any kernel size. Requires known kScale and Kernel::Weights2D.
template <int64_t kScale>
class GeneralUpsampler {
 public:
  // TODO(janwas): add ExecutorPool overload
  template <class Executor, class Kernel>
  static void Run(const Executor executor, const ImageF& in,
                  const Kernel& kernel, ImageF* PIK_RESTRICT out) {
    const size_t in_xsize = in.xsize();
    const size_t in_ysize = in.ysize();
    const size_t out_xsize = out->xsize();
    const size_t out_ysize = out->ysize();
    PROFILER_ZONE("slow::GeneralUpsampler");

    const int64_t kWidth = 2 * Kernel::kRadius;  // even
    const float* PIK_RESTRICT weights = kernel.Weights2D();

    for (size_t out_y = 0; out_y < out_ysize; ++out_y) {
      float* PIK_RESTRICT out_row = out->Row(out_y);
      const float in_fy = ((out_y + 0.5f) / out_ysize) * in_ysize - 0.5f;
      const int64_t top = std::ceil(in_fy - Kernel::kRadius);

      const float* PIK_RESTRICT in_rows[kWidth];
      for (int64_t i = 0; i < kWidth; ++i) {
        in_rows[i] = in.ConstRow(Mirror(top + i, in_ysize));
      }

      for (int64_t out_x = 0; out_x < out_xsize; out_x++) {
        const float in_fx = ((out_x + 0.5f) / out_xsize) * in_xsize - 0.5f;
        const int64_t left = std::ceil(in_fx - Kernel::kRadius);

        int64_t in_x[kWidth];
        for (int64_t i = 0; i < kWidth; ++i) {
          in_x[i] = Mirror(left + i, in_xsize);
        }

        float sum = 0.0f;
        for (size_t r = 0; r < kWidth; ++r) {
          const float* PIK_RESTRICT in_row = in_rows[r];
          for (size_t c = 0; c < kWidth; ++c) {
            size_t idx_weight = out_y % kScale;
            idx_weight *= kWidth;
            idx_weight += r;
            idx_weight *= kWidth;
            idx_weight += c;
            idx_weight *= kScale;
            idx_weight += out_x % kScale;
            sum += in_row[in_x[c]] * weights[idx_weight];
          }
        }

        out_row[out_x] = sum;
      }
    }
  }

  template <class Executor, class Kernel>
  static void Run(const Executor executor, const Image3F& in,
                  const Kernel& kernel, Image3F* PIK_RESTRICT out) {
    // Unoptimized: separate planes (additional fork/join)
    for (int c = 0; c < 3; ++c) {
      Run(executor, in.Plane(c), kernel, const_cast<ImageF*>(&out->Plane(c)));
    }
  }
};

}  // namespace slow

// Shared code factored out of *Upsample8. CRTP: Derived needs kScale etc. and
// implements ProducePair.
template <int64_t kRadiusArg, class Derived>
class Upsampler8Base {
 public:
  // Called by Upsample function templates. Image = Image[3]F.
  template <class Executor, class Image, class Kernel>
  static SIMD_ATTR PIK_INLINE void Run(const Executor executor, const Image& in,
                                       const Kernel& kernel, Image* out) {
    PROFILER_ZONE("Upsampler8");
    PIK_CHECK(in.xsize() * kScale == out->xsize());
    PIK_CHECK(in.ysize() * kScale == out->ysize());

    const float* PIK_RESTRICT weights = Derived::GetWeights(kernel);

    if (out->xsize() >= kBorder) {
      RunImpl(HorzSplit(), executor, in, weights, out);
    } else {
      RunImpl(HorzLoop(), executor, in, weights, out);
    }
  }

 protected:
  using D = SIMD_FULL(float);
  using V = D::V;

  static constexpr int64_t kRadius = kRadiusArg;
  static constexpr int64_t kWidth = 2 * kRadius;

  static constexpr int kLogScale = 3;
  static constexpr int64_t kScale = 1 << kLogScale;

  static constexpr int64_t kBorder = kRadius * kScale;

  // Returns first (left/top) input x/y for the given output x/y. "out_mod" is
  // "out" % kScale.
  static PIK_INLINE int64_t InFromOut(const size_t out, const size_t out_mod) {
    // Shifted by 0.5 (dual grid).
    return (out >> kLogScale) + (out_mod >> (kLogScale - 1)) - kRadius;
  }

 private:
  // Policies for iterating in X direction:

  // Wide enough to skip bounds checks in the interior.
  struct HorzSplit {
    SIMD_ATTR PIK_INLINE void operator()(
        const float* PIK_RESTRICT row_t3, const float* PIK_RESTRICT row_t2,
        const float* PIK_RESTRICT row_t, const float* PIK_RESTRICT row_m,
        const float* PIK_RESTRICT row_b, const float* PIK_RESTRICT row_b2,
        const size_t in_xsize, const float* PIK_RESTRICT weights,
        float* PIK_RESTRICT row_out, const size_t out_xsize) const {
      size_t out_x = 0;
      for (; out_x < kBorder; out_x += 2 * D::N) {
        Derived::template ProducePair(out_x, row_t3, row_t2, row_t, row_m,
                                      row_b, row_b2, in_xsize, WrapMirror(),
                                      weights, row_out);
      }
      // (One more than kRadius because ProducePair reads offsets [-r, r+1])
      for (; out_x < out_xsize - (kRadius + 1) * kScale; out_x += 2 * D::N) {
        Derived::template ProducePair(out_x, row_t3, row_t2, row_t, row_m,
                                      row_b, row_b2, in_xsize, WrapUnchanged(),
                                      weights, row_out);
      }
      for (; out_x < out_xsize; out_x += 2 * D::N) {
        Derived::template ProducePair(out_x, row_t3, row_t2, row_t, row_m,
                                      row_b, row_b2, in_xsize, WrapMirror(),
                                      weights, row_out);
      }
    }
  };

  // Narrow, only a single loop with X bounds checks.
  struct HorzLoop {
    SIMD_ATTR PIK_INLINE void operator()(
        const float* PIK_RESTRICT row_t3, const float* PIK_RESTRICT row_t2,
        const float* PIK_RESTRICT row_t, const float* PIK_RESTRICT row_m,
        const float* PIK_RESTRICT row_b, const float* PIK_RESTRICT row_b2,
        const size_t in_xsize, const float* PIK_RESTRICT weights,
        float* PIK_RESTRICT row_out, const size_t out_xsize) const {
      for (size_t out_x = 0; out_x < out_xsize; out_x += 2 * D::N) {
        Derived::template ProducePair(out_x, row_t3, row_t2, row_t, row_m,
                                      row_b, row_b2, in_xsize, WrapMirror(),
                                      weights, row_out);
      }
    }
  };

  // Produces a row of output using a single pass through the input rows.
  template <class Horz, class WrapY>
  static SIMD_ATTR void ProduceRow(const Horz horz, const size_t out_y,
                                   const ImageF& in, const WrapY wrap_y,
                                   const float* PIK_RESTRICT weights,
                                   float* PIK_RESTRICT out_row,
                                   size_t out_xsize) {
    const size_t in_xsize = in.xsize();
    const size_t in_ysize = in.ysize();

    const size_t mod_y = out_y % kScale;
    // Coordinate of the top input row (possibly out of bounds).
    const int64_t in_y = InFromOut(out_y, mod_y);

    const float* PIK_RESTRICT row_t3 = in.ConstRow(wrap_y(in_y + 0, in_ysize));
    const float* PIK_RESTRICT row_t2 = in.ConstRow(wrap_y(in_y + 1, in_ysize));
    const float* PIK_RESTRICT row_t = in.ConstRow(wrap_y(in_y + 2, in_ysize));
    const float* PIK_RESTRICT row_m = in.ConstRow(wrap_y(in_y + 3, in_ysize));
    // Avoid out-of-bounds access - these two rows are unused for r=2 anyway.
    const float* PIK_RESTRICT row_b =
        kRadius == 2 ? nullptr : in.ConstRow(wrap_y(in_y + 4, in_ysize));
    const float* PIK_RESTRICT row_b2 =
        kRadius == 2 ? nullptr : in.ConstRow(wrap_y(in_y + 5, in_ysize));

    weights += mod_y * Derived::kWeightsPerModY;
    horz(row_t3, row_t2, row_t, row_m, row_b, row_b2, in_xsize, weights,
         out_row, out_xsize);
  }

  template <class Horz, class Executor>
  static SIMD_ATTR void RunImpl(const Horz horz, const Executor executor,
                                const ImageF& in,
                                const float* PIK_RESTRICT weights,
                                ImageF* PIK_RESTRICT out) {
    const size_t out_xsize = out->xsize();
    const size_t out_ysize = out->ysize();

    // Short: single loop (ignore pool - not worthwhile).
    if (out_ysize <= 2 * kBorder) {
      for (size_t out_y = 0; out_y < out_ysize; ++out_y) {
        ProduceRow(horz, out_y, in, WrapMirror(), weights, out->Row(out_y),
                   out_xsize);
      }
      return;
    }

    // Tall: skip bounds checks for middle rows.
    for (size_t out_y = 0; out_y < kBorder; ++out_y) {
      ProduceRow(horz, out_y, in, WrapMirror(), weights, out->Row(out_y),
                 out_xsize);
    }
    executor.Run(
        kBorder, out_ysize - kBorder,
        [horz, &in, weights, out, out_xsize](const int task, const int thread) {
          const int64_t out_y = task;
          ProduceRow(horz, out_y, in, WrapUnchanged(), weights, out->Row(out_y),
                     out_xsize);
        },
        "Resample");
    for (size_t out_y = out_ysize - kBorder; out_y < out_ysize; ++out_y) {
      ProduceRow(horz, out_y, in, WrapMirror(), weights, out->Row(out_y),
                 out_xsize);
    }
  }

  template <class Horz, class Executor>
  static SIMD_ATTR void RunImpl(const Horz horz, const Executor executor,
                                const Image3F& in,
                                const float* PIK_RESTRICT weights,
                                Image3F* PIK_RESTRICT out) {
    const size_t out_xsize = out->xsize();
    const size_t out_ysize = out->ysize();

    // Short: single loop (ignore pool - not worthwhile).
    if (out_ysize <= 2 * kBorder) {
      for (int c = 0; c < 3; ++c) {
        for (size_t out_y = 0; out_y < out_ysize; ++out_y) {
          ProduceRow(horz, out_y, in.Plane(c), WrapMirror(), weights,
                     out->PlaneRow(c, out_y), out_xsize);
        }
      }
      return;
    }

    // Tall: skip bounds checks for middle rows.
    for (int c = 0; c < 3; ++c) {
      for (size_t out_y = 0; out_y < kBorder; ++out_y) {
        ProduceRow(horz, out_y, in.Plane(c), WrapMirror(), weights,
                   out->PlaneRow(c, out_y), out_xsize);
      }
    }
    executor.Run(
        kBorder, out_ysize - kBorder,
        [horz, &in, weights, out, out_xsize](const int task, const int thread) {
          const int64_t out_y = task;
          for (int c = 0; c < 3; ++c) {
            ProduceRow(horz, out_y, in.Plane(c), WrapUnchanged(), weights,
                       out->PlaneRow(c, out_y), out_xsize);
          }
        },
        "Resample3");
    for (int c = 0; c < 3; ++c) {
      for (size_t out_y = out_ysize - kBorder; out_y < out_ysize; ++out_y) {
        ProduceRow(horz, out_y, in.Plane(c), WrapMirror(), weights,
                   out->PlaneRow(c, out_y), out_xsize);
      }
    }
  }
};

// Single-pass 8x cubic upsampling for separable 4x4 kernels. Unused: slower
// than GeneralUpsampler8(!) and we need 6x6 for sufficient quality.
class Upsampler8 : public Upsampler8Base<2, Upsampler8> {
 public:
  static constexpr size_t kWeightsPerModY = kWidth * kScale + kWidth * 4;

  // Returns contiguous storage: x[tap_x=4][mod_x=8], 4x-broadcasted y[tap_y=4].
  // Extracts the required kind of weights from Kernel. Type-safe: compile error
  // if kernel is unable to precompute non-separated weights.
  template <class Kernel>
  static PIK_INLINE const float* PIK_RESTRICT GetWeights(const Kernel& kernel) {
    return kernel.WeightsSeparated();
  }

  // Stores 2 vectors of upsampled pixels to row_out + out_x. "weights" are
  // the return value of GetWeights. About 104 uops.
  template <class WrapX>
  static SIMD_ATTR PIK_INLINE void ProducePair(
      const size_t out_x, const float* PIK_RESTRICT row_t3,
      const float* PIK_RESTRICT row_t2, const float* PIK_RESTRICT row_t,
      const float* PIK_RESTRICT row_m, const float* PIK_RESTRICT row_b,
      const float* PIK_RESTRICT row_b2, const size_t in_xsize,
      const WrapX wrap_x, const float* PIK_RESTRICT weights,
      float* PIK_RESTRICT row_out) {
    const D d;
    const V wy0 = load_dup128(d, weights + 4 * kScale + 0 * 4);
    const V wy1 = load_dup128(d, weights + 4 * kScale + 1 * 4);
    const V wy2 = load_dup128(d, weights + 4 * kScale + 2 * 4);
    const V wy3 = load_dup128(d, weights + 4 * kScale + 3 * 4);

    // Accumulators for upsampled output, i.e. sum(horz * wy).
    V u0 = setzero(d);
    V u1 = setzero(d);
    // t3 is our first valid row; ignore b/b2.
    MulAddHorzConv(out_x, row_t3, in_xsize, wrap_x, weights, wy0, &u0, &u1);
    MulAddHorzConv(out_x, row_t2, in_xsize, wrap_x, weights, wy1, &u0, &u1);
    MulAddHorzConv(out_x, row_t, in_xsize, wrap_x, weights, wy2, &u0, &u1);
    MulAddHorzConv(out_x, row_m, in_xsize, wrap_x, weights, wy3, &u0, &u1);
    // stream is slightly slower (for both in-L3 and larger outputs)
    store(u0, d, row_out + out_x);
    store(u1, d, row_out + out_x + d.N);
  }

 private:
#if SIMD_TARGET_VALUE == SIMD_NONE
  // Without SIMD, there's no point in the pair unrolling because we cannot
  // reuse anything between them. This function produces a single output pixel.
  template <class WrapX>
  static SIMD_ATTR PIK_INLINE void MulAddHorzConv1(
      const size_t out_x, const float* PIK_RESTRICT row_in,
      const size_t in_xsize, const WrapX wrap_x,
      const float* PIK_RESTRICT weights, const V wy, V* PIK_RESTRICT out) {
    const D d;
    const size_t mod_x = out_x % kScale;
    const V wx0 = load(d, weights + mod_x + 0 * kScale);
    const V wx1 = load(d, weights + mod_x + 1 * kScale);
    const V wx2 = load(d, weights + mod_x + 2 * kScale);
    const V wx3 = load(d, weights + mod_x + 3 * kScale);

    // We'll load 4 input values from these (clamped) coordinates.
    const int64_t in_x = InFromOut(out_x, mod_x);
    const int64_t in_x0 = wrap_x(in_x + 0, in_xsize);
    const int64_t in_x1 = wrap_x(in_x + 1, in_xsize);
    const int64_t in_x2 = wrap_x(in_x + 2, in_xsize);
    const int64_t in_x3 = wrap_x(in_x + 3, in_xsize);

    const V v0 = set1(d, row_in[in_x0]);
    const V v1 = set1(d, row_in[in_x1]);
    const V v2 = set1(d, row_in[in_x2]);
    const V v3 = set1(d, row_in[in_x3]);

    const V m0 = v0 * wx0;
    const V m1 = v1 * wx1;

    const V m2 = mul_add(v2, wx2, m0);
    const V m3 = mul_add(v3, wx3, m1);

    *out = mul_add(m2 + m3, wy, *out);
  }
#endif

  // Computes two vectors of horizontal 1D convolution results, multiplies them
  // with the Y weight "wy" and accumulates into out0/1.
  template <class WrapX>
  static SIMD_ATTR PIK_INLINE void MulAddHorzConv(
      const size_t out_x, const float* PIK_RESTRICT row_in,
      const size_t in_xsize, const WrapX wrap_x,
      const float* PIK_RESTRICT weights, const V wy, V* PIK_RESTRICT out0,
      V* PIK_RESTRICT out1) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
    const D d;
    const size_t mod_x = 0;  // because 2 * d.N == 2 * kScale

    // Load weights for all mod_x values.
    const V wx0 = load(d, weights + mod_x + 0 * kScale);
    const V wx1 = load(d, weights + mod_x + 1 * kScale);
    const V wx2 = load(d, weights + mod_x + 2 * kScale);
    const V wx3 = load(d, weights + mod_x + 3 * kScale);

    // We'll load 6 input values from these (clamped) coordinates.
    const int64_t in_x = InFromOut(out_x, mod_x);
    const int64_t in_x0 = wrap_x(in_x + 0, in_xsize);
    const int64_t in_x1 = wrap_x(in_x + 1, in_xsize);
    const int64_t in_x2 = wrap_x(in_x + 2, in_xsize);
    const int64_t in_x3 = wrap_x(in_x + 3, in_xsize);
    const int64_t in_x4 = wrap_x(in_x + 4, in_xsize);
    const int64_t in_x5 = wrap_x(in_x + 5, in_xsize);

    const V in0 = set1(d, row_in[in_x0]);
    const V in1 = set1(d, row_in[in_x1]);
    const V in2 = set1(d, row_in[in_x2]);
    const V in3 = set1(d, row_in[in_x3]);
    // Upper half = in1, lower half = in0.
    const V v0 = concat_hi_lo(in1, in0);
    const V v1 = concat_hi_lo(in2, in1);
    const V v2 = concat_hi_lo(in3, in2);
    const V m0 = v0 * wx0;
    const V m1 = v1 * wx1;
    // out1 is the result for out_x + kScale, basically unrolling the caller's
    // loop once. This gives a 1.25x overall speedup because we reuse the
    // weights and v1..3, and hide the multiplication latency.
    const V n0 = v1 * wx0;
    const V n1 = v2 * wx1;

    const V in4 = set1(d, row_in[in_x4]);
    const V in5 = set1(d, row_in[in_x5]);
    const V v3 = concat_hi_lo(in4, in3);
    const V v4 = concat_hi_lo(in5, in4);
    const V m2 = mul_add(v2, wx2, m0);
    const V m3 = mul_add(v3, wx3, m1);
    const V n2 = mul_add(v3, wx2, n0);
    const V n3 = mul_add(v4, wx3, n1);

    *out0 = mul_add(m2 + m3, wy, *out0);
    *out1 = mul_add(n2 + n3, wy, *out1);
#elif SIMD_TARGET_VALUE != SIMD_NONE
    const D d;

    // Load first two weights for the first and second vectors.
    constexpr size_t mod_x = 0;  // because 2 * d.N == kScale
    const V wx0 = load(d, weights + mod_x + 0 * kScale);
    const V wx1 = load(d, weights + mod_x + 1 * kScale);
    const V wx0H = load(d, weights + mod_x + 0 * kScale + 4);
    const V wx1H = load(d, weights + mod_x + 1 * kScale + 4);

    // We'll load 5 input values from these (clamped) coordinates.
    const int64_t in_x = InFromOut(out_x, mod_x);
    const int64_t in_x0 = wrap_x(in_x + 0, in_xsize);
    const int64_t in_x1 = wrap_x(in_x + 1, in_xsize);
    const int64_t in_x2 = wrap_x(in_x + 2, in_xsize);
    const int64_t in_x3 = wrap_x(in_x + 3, in_xsize);
    const int64_t in_x4 = wrap_x(in_x + 4, in_xsize);

    const V v0 = set1(d, row_in[in_x0]);
    const V v1 = set1(d, row_in[in_x1]);
    const V v2 = set1(d, row_in[in_x2]);
    const V m0 = v0 * wx0;
    const V m1 = v1 * wx1;
    const V n0 = v1 * wx0H;
    const V n1 = v2 * wx1H;

    const V wx2 = load(d, weights + mod_x + 2 * kScale);
    const V wx3 = load(d, weights + mod_x + 3 * kScale);
    const V wx2H = load(d, weights + mod_x + 2 * kScale + 4);
    const V wx3H = load(d, weights + mod_x + 3 * kScale + 4);

    const V v3 = set1(d, row_in[in_x3]);
    const V v4 = set1(d, row_in[in_x4]);
    const V m2 = mul_add(v2, wx2, m0);
    const V m3 = mul_add(v3, wx3, m1);
    const V n2 = mul_add(v3, wx2H, n0);
    const V n3 = mul_add(v4, wx3H, n1);

    *out0 = mul_add(m2 + m3, wy, *out0);
    *out1 = mul_add(n2 + n3, wy, *out1);
#else
    MulAddHorzConv1(out_x + 0, row_in, in_xsize, wrap_x, weights, wy, out0);
    MulAddHorzConv1(out_x + 1, row_in, in_xsize, wrap_x, weights, wy, out1);
#endif
  }
};

// Single-pass 8x cubic upsampling for not necessarily separable 4x4 kernels.
// Unused: we need 6x6 for sufficient quality.
class GeneralUpsampler8 : public Upsampler8Base<2, GeneralUpsampler8> {
 public:
  static constexpr size_t kWeightsPerModY = kWidth * kWidth * kScale;

  // Extracts the required kind of weights from Kernel. Type-safe: compile error
  // if kernel is unable to precompute non-separated weights.
  template <class Kernel>
  static PIK_INLINE const float* PIK_RESTRICT GetWeights(const Kernel& kernel) {
    return kernel.Weights2D();
  }

  // Stores 2 vectors of upsampled pixels to row_out + out_x. "weights" are
  // the return value of GetWeights.
  template <class WrapX>
  static SIMD_ATTR PIK_INLINE void ProducePair(
      const size_t out_x, const float* PIK_RESTRICT row_t3,
      const float* PIK_RESTRICT row_t2, const float* PIK_RESTRICT row_t,
      const float* PIK_RESTRICT row_m, const float* PIK_RESTRICT row_b,
      const float* PIK_RESTRICT row_b2, const size_t in_xsize,
      const WrapX wrap_x, const float* PIK_RESTRICT weights,
      float* PIK_RESTRICT row_out) {
#if SIMD_TARGET_VALUE == SIMD_AVX2
    const D d;
    const int64_t mod_x = 0;  // because 2 * d.N == 2 * kScale

    // We'll load 6 input values from these (clamped) coordinates.
    const int64_t in_x = InFromOut(out_x, mod_x);
    const int64_t in_x0 = wrap_x(in_x + 0, in_xsize);
    const int64_t in_x1 = wrap_x(in_x + 1, in_xsize);
    const int64_t in_x2 = wrap_x(in_x + 2, in_xsize);
    const int64_t in_x3 = wrap_x(in_x + 3, in_xsize);
    const int64_t in_x4 = wrap_x(in_x + 4, in_xsize);
    const int64_t in_x5 = wrap_x(in_x + 5, in_xsize);

    V in0, in1, in2, in3, in4, in5;
    V v0, v1, v2, v3, v4;
    V w0, w1, w2, w3;

    // (broadcastss only requires load ports, not port5.)
    // Start at t3, our top row; ignore b/b2.
    in0 = set1(d, row_t3[in_x0]);
    in1 = set1(d, row_t3[in_x1]);
    in2 = set1(d, row_t3[in_x2]);
    in3 = set1(d, row_t3[in_x3]);
    in4 = set1(d, row_t3[in_x4]);
    in5 = set1(d, row_t3[in_x5]);
    // v := upper half = next X, lower half = current X. Note that port5
    // is underutilized; blendps should use ports 015 but IACA only shows 01.
    // However, using concat_lo_lo for some of these is actually slower.
    v0 = concat_hi_lo(in1, in0);
    v1 = concat_hi_lo(in2, in1);
    v2 = concat_hi_lo(in3, in2);
    v3 = concat_hi_lo(in4, in3);
    v4 = concat_hi_lo(in5, in4);
    // wyx[i] is the weight for tap_x=x, tap_y=y and mod_x=i (mod_y was
    // already used to select the "weights" range). Reused once.
    w0 = load(d, weights + 0 * kScale);
    w1 = load(d, weights + 1 * kScale);
    w2 = load(d, weights + 2 * kScale);
    w3 = load(d, weights + 3 * kScale);
    // 4 separate accumulators for inputs * weights; their sum is the result.
    const V m00 = v0 * w0;
    const V m01 = v1 * w1;
    const V m02 = v2 * w2;
    const V m03 = v3 * w3;
    // For the second output vector, use same weights but skip 1 input vector.
    const V n00 = v1 * w0;
    const V n01 = v2 * w1;
    const V n02 = v3 * w2;
    const V n03 = v4 * w3;

    // Prevents clang from doing 13 successive broadcast+blend; it's about 2%
    // faster to separate them into groups of 5.
    std::atomic_thread_fence(std::memory_order_release);

    in0 = set1(d, row_t2[in_x0]);
    in1 = set1(d, row_t2[in_x1]);
    in2 = set1(d, row_t2[in_x2]);
    in3 = set1(d, row_t2[in_x3]);
    in4 = set1(d, row_t2[in_x4]);
    in5 = set1(d, row_t2[in_x5]);
    v0 = concat_hi_lo(in1, in0);
    v1 = concat_hi_lo(in2, in1);
    v2 = concat_hi_lo(in3, in2);
    v3 = concat_hi_lo(in4, in3);
    v4 = concat_hi_lo(in5, in4);
    w0 = load(d, weights + 4 * kScale);
    w1 = load(d, weights + 5 * kScale);
    w2 = load(d, weights + 6 * kScale);
    w3 = load(d, weights + 7 * kScale);
    const V m10 = mul_add(v0, w0, m00);
    const V m11 = mul_add(v1, w1, m01);
    const V m12 = mul_add(v2, w2, m02);
    const V m13 = mul_add(v3, w3, m03);
    const V n10 = mul_add(v1, w0, n00);
    const V n11 = mul_add(v2, w1, n01);
    const V n12 = mul_add(v3, w2, n02);
    const V n13 = mul_add(v4, w3, n03);

    in0 = set1(d, row_t[in_x0]);
    in1 = set1(d, row_t[in_x1]);
    in2 = set1(d, row_t[in_x2]);
    in3 = set1(d, row_t[in_x3]);
    in4 = set1(d, row_t[in_x4]);
    in5 = set1(d, row_t[in_x5]);
    v0 = concat_hi_lo(in1, in0);
    v1 = concat_hi_lo(in2, in1);
    v2 = concat_hi_lo(in3, in2);
    v3 = concat_hi_lo(in4, in3);
    v4 = concat_hi_lo(in5, in4);
    w0 = load(d, weights + 8 * kScale);
    w1 = load(d, weights + 9 * kScale);
    w2 = load(d, weights + 10 * kScale);
    w3 = load(d, weights + 11 * kScale);
    const V m20 = mul_add(v0, w0, m10);
    const V m21 = mul_add(v1, w1, m11);
    const V m22 = mul_add(v2, w2, m12);
    const V m23 = mul_add(v3, w3, m13);
    const V n20 = mul_add(v1, w0, n10);
    const V n21 = mul_add(v2, w1, n11);
    const V n22 = mul_add(v3, w2, n12);
    const V n23 = mul_add(v4, w3, n13);

    in0 = set1(d, row_m[in_x0]);
    in1 = set1(d, row_m[in_x1]);
    in2 = set1(d, row_m[in_x2]);
    in3 = set1(d, row_m[in_x3]);
    in4 = set1(d, row_m[in_x4]);
    in5 = set1(d, row_m[in_x5]);
    v0 = concat_hi_lo(in1, in0);
    v1 = concat_hi_lo(in2, in1);
    v2 = concat_hi_lo(in3, in2);
    v3 = concat_hi_lo(in4, in3);
    v4 = concat_hi_lo(in5, in4);
    w0 = load(d, weights + 12 * kScale);
    w1 = load(d, weights + 13 * kScale);
    w2 = load(d, weights + 14 * kScale);
    w3 = load(d, weights + 15 * kScale);
    const V m30 = mul_add(v0, w0, m20);
    const V m31 = mul_add(v1, w1, m21);
    const V m32 = mul_add(v2, w2, m22);
    const V m33 = mul_add(v3, w3, m23);
    const V n30 = mul_add(v1, w0, n20);
    const V n31 = mul_add(v2, w1, n21);
    const V n32 = mul_add(v3, w2, n22);
    const V n33 = mul_add(v4, w3, n23);
    const V k1 = set1(d, 1.0f);
    const V sum0_01 = mul_add(m30, k1, m31);
    const V sum0_23 = mul_add(m32, k1, m33);
    const V sum1_01 = mul_add(n30, k1, n31);
    const V sum1_23 = mul_add(n32, k1, n33);
    const V sum0 = mul_add(sum0_01, k1, sum0_23);
    const V sum1 = mul_add(sum1_01, k1, sum1_23);
    store(sum0, d, row_out + out_x);
    store(sum1, d, row_out + out_x + d.N);
#elif SIMD_TARGET_VALUE != SIMD_NONE
    const D d;
    const int64_t mod_x = 0;  // because 2 * d.N == kScale.
    // wyx[i] is the weight for tap_x=x, tap_y=y and mod_x=i (mod_y was already
    // used to select the "weights" range). Reused once.
    const V w00 = load(d, weights + 0 * kScale);
    const V w01 = load(d, weights + 1 * kScale);
    const V w02 = load(d, weights + 2 * kScale);
    const V w03 = load(d, weights + 3 * kScale);
    const V w10 = load(d, weights + 4 * kScale);
    const V w11 = load(d, weights + 5 * kScale);
    const V w12 = load(d, weights + 6 * kScale);
    const V w13 = load(d, weights + 7 * kScale);
    const V w20 = load(d, weights + 8 * kScale);
    const V w21 = load(d, weights + 9 * kScale);
    const V w22 = load(d, weights + 10 * kScale);
    const V w23 = load(d, weights + 11 * kScale);
    const V w30 = load(d, weights + 12 * kScale);
    const V w31 = load(d, weights + 13 * kScale);
    const V w32 = load(d, weights + 14 * kScale);
    const V w33 = load(d, weights + 15 * kScale);

    // Same, but wyxH has mod_x=i+4.
    const V w00H = load(d, weights + 0 * kScale + d.N);
    const V w01H = load(d, weights + 1 * kScale + d.N);
    const V w02H = load(d, weights + 2 * kScale + d.N);
    const V w03H = load(d, weights + 3 * kScale + d.N);
    const V w10H = load(d, weights + 4 * kScale + d.N);
    const V w11H = load(d, weights + 5 * kScale + d.N);
    const V w12H = load(d, weights + 6 * kScale + d.N);
    const V w13H = load(d, weights + 7 * kScale + d.N);
    const V w20H = load(d, weights + 8 * kScale + d.N);
    const V w21H = load(d, weights + 9 * kScale + d.N);
    const V w22H = load(d, weights + 10 * kScale + d.N);
    const V w23H = load(d, weights + 11 * kScale + d.N);
    const V w30H = load(d, weights + 12 * kScale + d.N);
    const V w31H = load(d, weights + 13 * kScale + d.N);
    const V w32H = load(d, weights + 14 * kScale + d.N);
    const V w33H = load(d, weights + 15 * kScale + d.N);

    // We'll load 5 input values from these (clamped) coordinates.
    const int64_t in_x = InFromOut(out_x, mod_x);
    const int64_t in_x0 = wrap_x(in_x + 0, in_xsize);
    const int64_t in_x1 = wrap_x(in_x + 1, in_xsize);
    const int64_t in_x2 = wrap_x(in_x + 2, in_xsize);
    const int64_t in_x3 = wrap_x(in_x + 3, in_xsize);
    const int64_t in_x4 = wrap_x(in_x + 4, in_xsize);

    const V v00 = set1(d, row_t3[in_x0]);
    const V v01 = set1(d, row_t3[in_x1]);
    const V v02 = set1(d, row_t3[in_x2]);
    const V v03 = set1(d, row_t3[in_x3]);
    const V v04 = set1(d, row_t3[in_x4]);
    const V v10 = set1(d, row_t2[in_x0]);
    const V v11 = set1(d, row_t2[in_x1]);
    const V v12 = set1(d, row_t2[in_x2]);
    const V v13 = set1(d, row_t2[in_x3]);
    const V v14 = set1(d, row_t2[in_x4]);
    const V v20 = set1(d, row_t[in_x0]);
    const V v21 = set1(d, row_t[in_x1]);
    const V v22 = set1(d, row_t[in_x2]);
    const V v23 = set1(d, row_t[in_x3]);
    const V v24 = set1(d, row_t[in_x4]);
    const V v30 = set1(d, row_m[in_x0]);
    const V v31 = set1(d, row_m[in_x1]);
    const V v32 = set1(d, row_m[in_x2]);
    const V v33 = set1(d, row_m[in_x3]);
    const V v34 = set1(d, row_m[in_x4]);

    const V m00 = v00 * w00;
    const V m01 = v01 * w01;
    const V m02 = v02 * w02;
    const V m03 = v03 * w03;
    const V n00 = v01 * w00H;
    const V n01 = v02 * w01H;
    const V n02 = v03 * w02H;
    const V n03 = v04 * w03H;

    const V m10 = mul_add(v10, w10, m00);
    const V m11 = mul_add(v11, w11, m01);
    const V m12 = mul_add(v12, w12, m02);
    const V m13 = mul_add(v13, w13, m03);
    const V n10 = mul_add(v11, w10H, n00);
    const V n11 = mul_add(v12, w11H, n01);
    const V n12 = mul_add(v13, w12H, n02);
    const V n13 = mul_add(v14, w13H, n03);

    const V m20 = mul_add(v20, w20, m10);
    const V m21 = mul_add(v21, w21, m11);
    const V m22 = mul_add(v22, w22, m12);
    const V m23 = mul_add(v23, w23, m13);
    const V n20 = mul_add(v21, w20H, n10);
    const V n21 = mul_add(v22, w21H, n11);
    const V n22 = mul_add(v23, w22H, n12);
    const V n23 = mul_add(v24, w23H, n13);

    const V m30 = mul_add(v30, w30, m20);
    const V m31 = mul_add(v31, w31, m21);
    const V m32 = mul_add(v32, w32, m22);
    const V m33 = mul_add(v33, w33, m23);
    const V n30 = mul_add(v31, w30H, n20);
    const V n31 = mul_add(v32, w31H, n21);
    const V n32 = mul_add(v33, w32H, n22);
    const V n33 = mul_add(v34, w33H, n23);

    const V sum0 = (m30 + m31) + (m32 + m33);
    const V sum1 = (n30 + n31) + (n32 + n33);
    store(sum0, d, row_out + out_x);
    store(sum1, d, row_out + out_x + d.N);
#else
    weights += (out_x % kScale);
    ProduceSingle(out_x + 0, row_t3, row_t2, row_t, row_m, in_xsize, wrap_x,
                  weights + 0, row_out);
    ProduceSingle(out_x + 1, row_t3, row_t2, row_t, row_m, in_xsize, wrap_x,
                  weights + 1, row_out);
#endif
  }

 private:
#if SIMD_TARGET_VALUE == SIMD_NONE
  template <class WrapX>
  static SIMD_ATTR PIK_INLINE void ProduceSingle(
      const size_t out_x, const float* PIK_RESTRICT row_t2,
      const float* PIK_RESTRICT row_t, const float* PIK_RESTRICT row_m,
      const float* PIK_RESTRICT row_b, const size_t in_xsize,
      const WrapX wrap_x, const float* PIK_RESTRICT weights,
      float* PIK_RESTRICT row_out) {
    const D d;
    const int64_t mod_x = out_x % kScale;

    // We'll load 4 input values from these (clamped) coordinates.
    const int64_t in_x = InFromOut(out_x, mod_x);
    const int64_t in_x0 = wrap_x(in_x + 0, in_xsize);
    const int64_t in_x1 = wrap_x(in_x + 1, in_xsize);
    const int64_t in_x2 = wrap_x(in_x + 2, in_xsize);
    const int64_t in_x3 = wrap_x(in_x + 3, in_xsize);

    const V v00 = set1(d, row_t2[in_x0]);
    const V v01 = set1(d, row_t2[in_x1]);
    const V v02 = set1(d, row_t2[in_x2]);
    const V v03 = set1(d, row_t2[in_x3]);
    // mod_y and mod_x have determined weights.
    const V w00 = load(d, weights + 0 * kScale);
    const V w01 = load(d, weights + 1 * kScale);
    const V w02 = load(d, weights + 2 * kScale);
    const V w03 = load(d, weights + 3 * kScale);
    const V m00 = v00 * w00;
    const V m01 = v01 * w01;
    const V m02 = v02 * w02;
    const V m03 = v03 * w03;

    const V v10 = set1(d, row_t[in_x0]);
    const V v11 = set1(d, row_t[in_x1]);
    const V v12 = set1(d, row_t[in_x2]);
    const V v13 = set1(d, row_t[in_x3]);
    const V w10 = load(d, weights + 4 * kScale);
    const V w11 = load(d, weights + 5 * kScale);
    const V w12 = load(d, weights + 6 * kScale);
    const V w13 = load(d, weights + 7 * kScale);
    const V m10 = mul_add(v10, w10, m00);
    const V m11 = mul_add(v11, w11, m01);
    const V m12 = mul_add(v12, w12, m02);
    const V m13 = mul_add(v13, w13, m03);

    const V v20 = set1(d, row_m[in_x0]);
    const V v21 = set1(d, row_m[in_x1]);
    const V v22 = set1(d, row_m[in_x2]);
    const V v23 = set1(d, row_m[in_x3]);
    const V w20 = load(d, weights + 8 * kScale);
    const V w21 = load(d, weights + 9 * kScale);
    const V w22 = load(d, weights + 10 * kScale);
    const V w23 = load(d, weights + 11 * kScale);
    const V m20 = mul_add(v20, w20, m10);
    const V m21 = mul_add(v21, w21, m11);
    const V m22 = mul_add(v22, w22, m12);
    const V m23 = mul_add(v23, w23, m13);

    const V v30 = set1(d, row_b[in_x0]);
    const V v31 = set1(d, row_b[in_x1]);
    const V v32 = set1(d, row_b[in_x2]);
    const V v33 = set1(d, row_b[in_x3]);
    const V w30 = load(d, weights + 12 * kScale);
    const V w31 = load(d, weights + 13 * kScale);
    const V w32 = load(d, weights + 14 * kScale);
    const V w33 = load(d, weights + 15 * kScale);
    const V m30 = mul_add(v30, w30, m20);
    const V m31 = mul_add(v31, w31, m21);
    const V m32 = mul_add(v32, w32, m22);
    const V m33 = mul_add(v33, w33, m23);
    const V sum = (m30 + m31) + (m32 + m33);
    store(sum, d, row_out + out_x);
  }
#endif
};

// Single-pass 8x cubic upsampling for not necessarily separable 6x6 kernels.
class GeneralUpsampler8_6x6 : public Upsampler8Base<3, GeneralUpsampler8_6x6> {
 public:
  static constexpr size_t kWeightsPerModY = kWidth * kWidth * kScale;

  // Extracts the required kind of weights from Kernel. Type-safe: compile error
  // if kernel is unable to precompute non-separated weights.
  template <class Kernel>
  static PIK_INLINE const float* PIK_RESTRICT GetWeights(const Kernel& kernel) {
    static_assert(Kernel::kRadius == kRadius, "kRadius mismatch");
    return kernel.Weights2D();
  }

  // Stores 2 vectors of upsampled pixels to row_out + out_x. "weights" are
  // the return value of GetWeights.
  template <class WrapX>
  static SIMD_ATTR PIK_INLINE void ProducePair(
      const size_t out_x, const float* PIK_RESTRICT row_t3,
      const float* PIK_RESTRICT row_t2, const float* PIK_RESTRICT row_t,
      const float* PIK_RESTRICT row_m, const float* PIK_RESTRICT row_b,
      const float* PIK_RESTRICT row_b2, const size_t in_xsize,
      const WrapX wrap_x, const float* PIK_RESTRICT weights,
      float* PIK_RESTRICT row_out) {
    const D d;
#if SIMD_TARGET_VALUE == SIMD_AVX2
    const int64_t mod_x = 0;  // because 2 * d.N == 2 * kScale

    // We'll load 8 input values from each row at these (wrapped) coordinates.
    const int64_t in_x = InFromOut(out_x, mod_x);
    const int64_t in_x0 = wrap_x(in_x + 0, in_xsize);
    const int64_t in_x1 = wrap_x(in_x + 1, in_xsize);
    const int64_t in_x2 = wrap_x(in_x + 2, in_xsize);
    const int64_t in_x3 = wrap_x(in_x + 3, in_xsize);
    const int64_t in_x4 = wrap_x(in_x + 4, in_xsize);
    const int64_t in_x5 = wrap_x(in_x + 5, in_xsize);
    const int64_t in_x6 = wrap_x(in_x + 6, in_xsize);
    const int64_t in_x7 = wrap_x(in_x + 7, in_xsize);

    // First row (tap_y = 0): multiply 6*8 weights by broadcasted inputs and
    // begin adding them together:

    // (set1 is faster than broadcast because it uses load ports, not port5.)
    V in0 = set1(d, row_t3[in_x0]);
    V in1 = set1(d, row_t3[in_x1]);
    V in2 = set1(d, row_t3[in_x2]);
    // w#[i] := weight for tap_x=# and mod_x=i (mod_y was already used to select
    // the "weights" range). Reused once.
    V w0 = load(d, weights + 0 * kScale);
    // q := upper four = next pixel replicated 4x, lower four = current 4x.
    // (port5 is underutilized; blendps should use ports 015 but IACA only
    // shows 01. Using concat_lo_lo for some of these is still slower.)
    V q0 = concat_hi_lo(in1, in0);
    V q1 = concat_hi_lo(in2, in1);
    // s := accumulators for inputs * weights; their sum is the first result.
    V s0 = q0 * w0;
    // t := accumulators for second output vector (same weights, shifted in#).
    V t0 = q1 * w0;
    V in3 = set1(d, row_t3[in_x3]);
    V w1 = load(d, weights + 1 * kScale);
    V q2 = concat_hi_lo(in3, in2);
    V s1 = q1 * w1;
    V t1 = q2 * w1;
    V in4 = set1(d, row_t3[in_x4]);
    V w2 = load(d, weights + 2 * kScale);
    V q3 = concat_hi_lo(in4, in3);
    V s2 = q2 * w2;
    V t2 = q3 * w2;
    V in5 = set1(d, row_t3[in_x5]);
    V w3 = load(d, weights + 3 * kScale);
    V q4 = concat_hi_lo(in5, in4);
    V s3 = q3 * w3;
    V t3 = q4 * w3;
    V in6 = set1(d, row_t3[in_x6]);
    V w4 = load(d, weights + 4 * kScale);
    V q5 = concat_hi_lo(in6, in5);
    V s4 = q4 * w4;
    V t4 = q5 * w4;
    V in7 = set1(d, row_t3[in_x7]);
    V w5 = load(d, weights + 5 * kScale);
    V q6 = concat_hi_lo(in7, in6);
    // s/t0 already finished, take advantage of free add.
    V s5 = mul_add(q5, w5, s0);
    V t5 = mul_add(q6, w5, t0);

    // This prevents spills, leading to a 1.8x speedup.
    std::atomic_thread_fence(std::memory_order_release);

    // Last 5 rows: multiply and accumulate into existing s/t.
#define PIK_MUL_WEIGHTS_ACCUMULATE(p_row, p_weights) \
  in0 = set1(d, p_row[in_x0]);                       \
  in1 = set1(d, p_row[in_x1]);                       \
  in2 = set1(d, p_row[in_x2]);                       \
  w0 = load(d, p_weights + 0 * kScale);              \
  q0 = concat_hi_lo(in1, in0);                       \
  q1 = concat_hi_lo(in2, in1);                       \
  s0 = mul_add(q0, w0, s1);                          \
  t0 = mul_add(q1, w0, t1);                          \
  in3 = set1(d, p_row[in_x3]);                       \
  w1 = load(d, p_weights + 1 * kScale);              \
  q2 = concat_hi_lo(in3, in2);                       \
  s1 = mul_add(q1, w1, s2);                          \
  t1 = mul_add(q2, w1, t2);                          \
  in4 = set1(d, p_row[in_x4]);                       \
  w2 = load(d, p_weights + 2 * kScale);              \
  q3 = concat_hi_lo(in4, in3);                       \
  s2 = mul_add(q2, w2, s3);                          \
  t2 = mul_add(q3, w2, t3);                          \
  in5 = set1(d, p_row[in_x5]);                       \
  w3 = load(d, p_weights + 3 * kScale);              \
  q4 = concat_hi_lo(in5, in4);                       \
  s3 = mul_add(q3, w3, s4);                          \
  t3 = mul_add(q4, w3, t4);                          \
  in6 = set1(d, p_row[in_x6]);                       \
  w4 = load(d, p_weights + 4 * kScale);              \
  q5 = concat_hi_lo(in6, in5);                       \
  s4 = mul_add(q4, w4, s5);                          \
  t4 = mul_add(q5, w4, t5);                          \
  in7 = set1(d, p_row[in_x7]);                       \
  w5 = load(d, p_weights + 5 * kScale);              \
  q6 = concat_hi_lo(in7, in6);                       \
  s5 = mul_add(q5, w5, s0);                          \
  t5 = mul_add(q6, w5, t0)

    PIK_MUL_WEIGHTS_ACCUMULATE(row_t2, weights + 6 * kScale);
    PIK_MUL_WEIGHTS_ACCUMULATE(row_t, weights + 12 * kScale);
    PIK_MUL_WEIGHTS_ACCUMULATE(row_m, weights + 18 * kScale);
    PIK_MUL_WEIGHTS_ACCUMULATE(row_b, weights + 24 * kScale);
    PIK_MUL_WEIGHTS_ACCUMULATE(row_b2, weights + 30 * kScale);
#undef PIK_MUL_WEIGHTS_ACCUMULATE

    // (s/t5 already include s/t0)
    const V sum0 = (s1 + s2) + (s3 + s4) + s5;
    const V sum1 = (t1 + t2) + (t3 + t4) + t5;
    store(sum0, d, row_out + out_x);
    store(sum1, d, row_out + out_x + d.N);
#else
    ProduceVector(out_x, row_t3, row_t2, row_t, row_m, row_b, row_b2, in_xsize,
                  wrap_x, weights, row_out);
    ProduceVector(out_x + d.N, row_t3, row_t2, row_t, row_m, row_b, row_b2,
                  in_xsize, wrap_x, weights, row_out);
#endif
  }

 private:
#if SIMD_TARGET_VALUE != SIMD_AVX2
  // If less than 8 lanes, produce a single output vector at a time because
  // there is not much benefit from pairwise unrolling.
  template <class WrapX>
  static SIMD_ATTR PIK_INLINE void ProduceVector(
      const size_t out_x, const float* PIK_RESTRICT row_t3,
      const float* PIK_RESTRICT row_t2, const float* PIK_RESTRICT row_t,
      const float* PIK_RESTRICT row_m, const float* PIK_RESTRICT row_b,
      const float* PIK_RESTRICT row_b2, const size_t in_xsize,
      const WrapX wrap_x, const float* PIK_RESTRICT weights,
      float* PIK_RESTRICT row_out) {
    const D d;
    const int64_t mod_x = out_x % kScale;

    // We'll load 6 input values from these (clamped) coordinates.
    const int64_t in_x = InFromOut(out_x, mod_x);
    const int64_t in_x0 = wrap_x(in_x + 0, in_xsize);
    const int64_t in_x1 = wrap_x(in_x + 1, in_xsize);
    const int64_t in_x2 = wrap_x(in_x + 2, in_xsize);
    const int64_t in_x3 = wrap_x(in_x + 3, in_xsize);
    const int64_t in_x4 = wrap_x(in_x + 4, in_xsize);
    const int64_t in_x5 = wrap_x(in_x + 5, in_xsize);

    // wyx[i] is the weight for tap_x=x, tap_y=y and mod_x=i (mod_y was already
    // used to select the "weights" range). Reused once.
    V w0 = load(d, weights + mod_x + 0 * kScale);
    V w1 = load(d, weights + mod_x + 1 * kScale);
    V w2 = load(d, weights + mod_x + 2 * kScale);
    V w3 = load(d, weights + mod_x + 3 * kScale);
    V w4 = load(d, weights + mod_x + 4 * kScale);
    V w5 = load(d, weights + mod_x + 5 * kScale);

    V q0 = set1(d, row_t3[in_x0]);
    V q1 = set1(d, row_t3[in_x1]);
    V q2 = set1(d, row_t3[in_x2]);
    V q3 = set1(d, row_t3[in_x3]);
    V q4 = set1(d, row_t3[in_x4]);
    V q5 = set1(d, row_t3[in_x5]);

    V s0 = q0 * w0;
    V s1 = q1 * w1;
    V s2 = q2 * w2;
    V s3 = q3 * w3;
    V s4 = q4 * w4;
    V s5 = q5 * w5;

#define PIK_MUL_WEIGHTS_ACCUMULATE(p_row, p_weights) \
  q0 = set1(d, p_row[in_x0]);                        \
  q1 = set1(d, p_row[in_x1]);                        \
  q2 = set1(d, p_row[in_x2]);                        \
  q3 = set1(d, p_row[in_x3]);                        \
  q4 = set1(d, p_row[in_x4]);                        \
  q5 = set1(d, p_row[in_x5]);                        \
  w0 = load(d, p_weights + mod_x + 0 * kScale);      \
  w1 = load(d, p_weights + mod_x + 1 * kScale);      \
  w2 = load(d, p_weights + mod_x + 2 * kScale);      \
  w3 = load(d, p_weights + mod_x + 3 * kScale);      \
  w4 = load(d, p_weights + mod_x + 4 * kScale);      \
  w5 = load(d, p_weights + mod_x + 5 * kScale);      \
  s0 += q0 * w0;                                     \
  s1 += q1 * w1;                                     \
  s2 += q2 * w2;                                     \
  s3 += q3 * w3;                                     \
  s4 += q4 * w4;                                     \
  s5 += q5 * w5

    PIK_MUL_WEIGHTS_ACCUMULATE(row_t2, weights + 6 * kScale);
    PIK_MUL_WEIGHTS_ACCUMULATE(row_t, weights + 12 * kScale);
    PIK_MUL_WEIGHTS_ACCUMULATE(row_m, weights + 18 * kScale);
    PIK_MUL_WEIGHTS_ACCUMULATE(row_b, weights + 24 * kScale);
    PIK_MUL_WEIGHTS_ACCUMULATE(row_b2, weights + 30 * kScale);
#undef PIK_MUL_WEIGHTS_ACCUMULATE

    const V sum = (s0 + s1) + (s2 + s3) + (s4 + s5);
    store(sum, d, row_out + out_x);
  }
#endif
};

// (Possibly) multithreaded, variable scale.
template <size_t N, class Image, class Executor, class Kernel>
PIK_NOINLINE SIMD_ATTR void Upsample(const Executor executor, const Image& in,
                                     const Kernel& kernel, Image* out) {
  if (N == 8) {
    Upsample<GeneralUpsampler8_6x6>(executor, in, kernel, out);
  } else {
    Upsample<slow::GeneralUpsampler<N> >(executor, in, kernel, out);
  }
}

}  // namespace pik

#endif  // PIK_RESAMPLE_H_
