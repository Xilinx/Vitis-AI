// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/gaborish.h"

#include "pik/convolve.h"
#include "pik/pik_params.h"

namespace pik {

namespace kernel {

struct Gaborish3_1000 {
  PIK_INLINE const Weights3x3& Weights() const {
    // Unnormalized.
    constexpr float wu0 = 1.0f;
    const float wu1 = static_cast<float>(0.11501538179658321);
    const float wu2 = static_cast<float>(0.089979079587015454);
    const float mul = 1.0 / (wu0 + 4 * (wu1 + wu2));
    const float w0 = wu0 * mul;
    const float w1 = wu1 * mul;
    const float w2 = wu2 * mul;
    static const Weights3x3 weights = {
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)},
        {SIMD_REP4(w1)}, {SIMD_REP4(w0)}, {SIMD_REP4(w1)},
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)}};
    return weights;
  }
};

struct Gaborish3_875 {
  PIK_INLINE const Weights3x3& Weights() const {
    // Unnormalized.
    constexpr float wu0 = 1.0f;
    const float x = 0.875;
    const float wu1 = static_cast<float>(x * 0.11501538179658321);
    const float wu2 = static_cast<float>(x * 0.089979079587015454);
    const float mul = 1.0 / (wu0 + 4 * (wu1 + wu2));
    const float w0 = wu0 * mul;
    const float w1 = wu1 * mul;
    const float w2 = wu2 * mul;
    static const Weights3x3 weights = {
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)},
        {SIMD_REP4(w1)}, {SIMD_REP4(w0)}, {SIMD_REP4(w1)},
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)}};
    return weights;
  }
};

struct Gaborish3_750 {
  PIK_INLINE const Weights3x3& Weights() const {
    // Unnormalized.
    constexpr float wu0 = 1.0f;
    const float x = 0.75;
    const float wu1 = static_cast<float>(x * 0.13959942428275746);
    const float wu2 = static_cast<float>(x * 0.074240717189152386);
    const float mul = 1.0 / (wu0 + 4 * (wu1 + wu2));
    const float w0 = wu0 * mul;
    const float w1 = wu1 * mul;
    const float w2 = wu2 * mul;
    static const Weights3x3 weights = {
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)},
        {SIMD_REP4(w1)}, {SIMD_REP4(w0)}, {SIMD_REP4(w1)},
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)}};
    return weights;
  }
};

struct Gaborish3_500 {
  PIK_INLINE const Weights3x3& Weights() const {
    // Unnormalized.
    constexpr float wu0 = 1.0f;
    const float wu1 = static_cast<float>(0.056007960760453189);
    const float wu2 = static_cast<float>(0.045899074552453879);
    const float mul = 1.0 / (wu0 + 4 * (wu1 + wu2));
    const float w0 = wu0 * mul;
    const float w1 = wu1 * mul;
    const float w2 = wu2 * mul;
    static const Weights3x3 weights = {
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)},
        {SIMD_REP4(w1)}, {SIMD_REP4(w0)}, {SIMD_REP4(w1)},
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)}};
    return weights;
  }
};

}  // namespace kernel

Image3F GaborishInverse(const Image3F& in, double mul) {
  PIK_ASSERT(mul > 0.0);
  PROFILER_FUNC;

  // Only an approximation. One or even two 3x3, and rank-1 (separable) 5x5
  // are insufficient.
  static const double kGaborish[5] = {
      -0.092359145662814029,  -0.039253623634014627, 0.016176494530216929,
      0.00083458437774987476, 0.004512465323949319,
  };
  const float smooth_weights5[9] = {
      1.0f,
      static_cast<float>(mul * kGaborish[0]),
      static_cast<float>(mul * kGaborish[2]),

      static_cast<float>(mul * kGaborish[0]),
      static_cast<float>(mul * kGaborish[1]),
      static_cast<float>(mul * kGaborish[3]),

      static_cast<float>(mul * kGaborish[2]),
      static_cast<float>(mul * kGaborish[3]),
      static_cast<float>(mul * kGaborish[4]),
  };
  Image3F sharpened(in.xsize(), in.ysize());
  slow::SymmetricConvolution<2, WrapClamp>::Run(in, in.xsize(), in.ysize(),
                                                smooth_weights5, &sharpened);
  return sharpened;
}

namespace {
template <typename Kernel, typename Executor>
// Assumes `Kernel` is symmetric.
SIMD_ATTR void RunConv(const Executor executor, const Image3F& in,
                       const Kernel& kernel, Image3F* out) {
  const BorderNeverUsed border;
  if (in.xsize() < kConvolveMinWidth) {
    using Convolution = slow::General3x3Convolution<1, WrapMirror>;
    Convolution::Run(in, in.xsize(), in.ysize(), kernel, out);
  } else {
    using Conv3 = ConvolveT<strategy::Symmetric3>;
    Conv3::Run(border, executor, in, kernel, out);
  }
}
}  // namespace

SIMD_ATTR Status ConvolveGaborish(const Image3F& in,
                                  GaborishStrength strength, ThreadPool* pool,
                                  Image3F* PIK_RESTRICT out) {
  // Since kOff would want us to return the `in` parameter as `out`, which would
  // lead to either a copy or a new memory handling strategy, we disallow it and
  // require callers to avoid it.
  PIK_CHECK(strength != GaborishStrength::kOff);
  PROFILER_FUNC;
  const ExecutorPool executor(pool);
  *out = Image3F(in.xsize(), in.ysize());
  if (strength == GaborishStrength::k1000) {
    RunConv(executor, in, kernel::Gaborish3_1000(), out);
  } else if (strength == GaborishStrength::k875) {
    RunConv(executor, in, kernel::Gaborish3_875(), out);
  } else if (strength == GaborishStrength::k750) {
    RunConv(executor, in, kernel::Gaborish3_750(), out);
  } else if (strength == GaborishStrength::k500) {
    RunConv(executor, in, kernel::Gaborish3_500(), out);
  } else {
    return PIK_FAILURE("Invalid strength argument");
  }
  return true;
}

}  // namespace pik
