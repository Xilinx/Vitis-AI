// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/convolve.h"

#include <stdlib.h>
#include <random>

#include "gtest/gtest.h"
#include "pik/compiler_specific.h"
#include "pik/data_parallel.h"
#include "pik/image_test_utils.h"

#define VERBOSE 1

namespace pik {

namespace kernel {

struct TestCorner3 {
  PIK_INLINE const Weights3x3& Weights() const {
    constexpr float tl = 0.016356047330531105f;
    constexpr float tr = -0.016356047330531105f;
    constexpr float br = 0.016356047330531105f;
    constexpr float bl = -0.016356047330531105f;
    static constexpr Weights3x3 weights = {
        {SIMD_REP4(tl)}, {SIMD_REP4(0.)}, {SIMD_REP4(tr)},
        {SIMD_REP4(0.)}, {SIMD_REP4(0.)}, {SIMD_REP4(0.)},
        {SIMD_REP4(bl)}, {SIMD_REP4(0.)}, {SIMD_REP4(br)}};
    return weights;
  }
};

struct TestGradY3 {
  PIK_INLINE const Weights3x3& Weights() const {
    constexpr float tl = 0.017367273579512846;
    constexpr float tc = 0.13753865853650102;
    constexpr float tr = 0.017367273579512846;
    constexpr float bl = -0.017367273579512846;
    constexpr float bc = -0.13753865853650102;
    constexpr float br = -0.017367273579512846;
    static constexpr Weights3x3 weights = {
        {SIMD_REP4(tl)}, {SIMD_REP4(tc)}, {SIMD_REP4(tr)},
        {SIMD_REP4(0.)}, {SIMD_REP4(0.)}, {SIMD_REP4(0.)},
        {SIMD_REP4(bl)}, {SIMD_REP4(bc)}, {SIMD_REP4(br)}};
    return weights;
  }
};

struct TestGradX3 {
  PIK_INLINE const Weights3x3& Weights() const {
    constexpr float tl = 0.017367273579512846;
    constexpr float tr = -0.017367273579512846;
    constexpr float ml = 0.13753865853650102;
    constexpr float mr = -0.13753865853650102;
    constexpr float bl = 0.017367273579512846;
    constexpr float br = -0.017367273579512846;
    static constexpr Weights3x3 weights = {
        {SIMD_REP4(tl)}, {SIMD_REP4(0.)}, {SIMD_REP4(tr)},
        {SIMD_REP4(ml)}, {SIMD_REP4(0.)}, {SIMD_REP4(mr)},
        {SIMD_REP4(bl)}, {SIMD_REP4(0.)}, {SIMD_REP4(br)}};
    return weights;
  }
};

}  // namespace kernel

namespace {

template <class Slow, class ImageOrView, class Kernel>
ImageF SlowConvolve(const ImageOrView& in, const size_t xsize,
                    const size_t ysize, const Kernel& kernel) {
  ImageF out(xsize, ysize);
  Slow::Run(in, xsize, ysize, kernel, &out);
  return out;
}

// Compares ConvolveT<> against SlowConvolve.
template <template <int64_t, class> class SlowT, class Strategy, class Border,
          class Kernel, class Executor, class Random>
SIMD_ATTR void Verify(const size_t xsize, const size_t ysize,
                      const Kernel& kernel, const Executor executor,
                      Random* rng) {
  constexpr size_t kRadius = Strategy::kRadius;
  static_assert(kRadius <= kConvolveMaxRadius, "Update max radius");

  ImageF in(xsize, ysize);
  GenerateImage(GeneratorRandom<float, Random>(rng, 1.0f), &in);

  using Slow = SlowT<kRadius, WrapMirror>;
  const ImageF out_expected = SlowConvolve<Slow>(in, xsize, ysize, kernel);

  ImageF out_actual(xsize, ysize);
  ConvolveT<Strategy>::Run(Border(), executor, in, kernel, &out_actual);
  VerifyRelativeError(out_expected, out_actual, 1E-5f, 1E-5f);
}

// Calls Verify for all border/executor combinations.
template <template <int64_t, class> class SlowT, class Strategy, class Kernel,
          class Random>
SIMD_ATTR void VerifyAll(const size_t xsize, const size_t ysize,
                         const Kernel& kernel, Random* rng, ThreadPool* pool) {
#if VERBOSE
  printf("needsInit loop\n");
#endif
  Verify<SlowT, Strategy, BorderNeedsInit>(xsize, ysize, kernel, ExecutorLoop(),
                                           rng);
#if VERBOSE
  printf("neverUsed loop\n");
#endif
  Verify<SlowT, Strategy, BorderNeverUsed>(xsize, ysize, kernel, ExecutorLoop(),
                                           rng);

  const ExecutorPool exec0(/*pool=*/nullptr);
#if VERBOSE
  printf("needsInit pool0\n");
#endif
  Verify<SlowT, Strategy, BorderNeedsInit>(xsize, ysize, kernel, exec0, rng);
#if VERBOSE
  printf("neverUsed pool0\n");
#endif
  Verify<SlowT, Strategy, BorderNeverUsed>(xsize, ysize, kernel, exec0, rng);

  const ExecutorPool executor(pool);
#if VERBOSE
  printf("needsInit pool\n");
#endif
  Verify<SlowT, Strategy, BorderNeedsInit>(xsize, ysize, kernel, executor, rng);
#if VERBOSE
  printf("neverUsed pool\n");
#endif
  Verify<SlowT, Strategy, BorderNeverUsed>(xsize, ysize, kernel, executor, rng);
}

// Compares slow::Symmetric3x3Convolution against slow::SymmetricConvolution.
template <class Random>
SIMD_ATTR void VerifyConvolveSymmetric3x3(const size_t xsize,
                                          const size_t ysize, Random* rng) {
  const size_t kRadius = 1;
  PIK_CHECK(xsize > kRadius);
  PIK_CHECK(ysize > kRadius);

  ImageF in(xsize, ysize);
  GenerateImage(GeneratorRandom<float, Random>(rng, 1.0f), &in);

  ImageF out_expected(xsize, ysize);
  ImageF out_actual(xsize, ysize);

  slow::Symmetric3x3Convolution<1, WrapMirror>::Run(
      in, xsize, ysize, kernel::Lowpass3(), &out_expected);

  // Expanded form of kernel::Lowpass3: lower-right quadrant.
  const float weights_symmetric[4] = {0.36208932f, 0.12820096f,  //
                                      0.12820096f, 0.03127668f};
  slow::SymmetricConvolution<kRadius, WrapClamp>::Run(
      in, xsize, ysize, weights_symmetric, &out_actual);

  VerifyRelativeError(out_expected, out_actual, 1E-5f, 1E-5f);
}

// Compares ConvolveT<> against slow::ConvolveSymmetric.
template <class Random>
SIMD_ATTR void VerifyConvolveSymmetric5x5(const size_t xsize,
                                          const size_t ysize, Random* rng) {
  const size_t kRadius = 2;
  PIK_CHECK(xsize > kRadius);
  PIK_CHECK(ysize > kRadius);

  ImageF in(xsize, ysize);
  GenerateImage(GeneratorRandom<float, Random>(rng, 1.0f), &in);

  ImageF out_expected(xsize, ysize);
  ImageF out_actual(xsize, ysize);

  ConvolveT<strategy::Separable5>::Run(in, kernel::Lowpass5(), &out_expected);

  // Expanded form of kernel::Lowpass5: lower-right quadrant.
  const float weights_symmetric[9] = {0.1740135f, 0.1065369f, 0.0150310f,  //
                                      0.1065369f, 0.0652254f, 0.0092025f,  //
                                      0.0150310f, 0.0092025f, 0.0012984f};
  slow::SymmetricConvolution<kRadius, WrapMirror>::Run(
      in, xsize, ysize, weights_symmetric, &out_actual);

  VerifyRelativeError(out_expected, out_actual, 1E-5f, 1E-5f);
}

// For all xsize/ysize and kernels:
SIMD_ATTR void TestSameResultsImpl() {
  ThreadPool pool(8);
  const size_t min_xsize = SIMD_FULL(float)::N + kConvolveMaxRadius;
  pool.Run(min_xsize, 40, [](const int task, const int thread) {
    const size_t xsize = task;
    std::mt19937_64 rng(129 + 13 * xsize);

    ThreadPool pool3(3);
    for (size_t ysize = 4; ysize < 16; ++ysize) {
#if VERBOSE
      printf("%zu x %zu=====================================\n", xsize, ysize);
#endif

#if VERBOSE
      printf("Sym3x3------------------\n");
#endif
      VerifyAll<slow::Symmetric3x3Convolution, strategy::Symmetric3>(
          xsize, ysize, kernel::Lowpass3(), &rng, &pool3);

#if VERBOSE
      printf("GradX3---------------\n");
#endif
      VerifyAll<slow::General3x3Convolution, strategy::GradX3>(
          xsize, ysize, kernel::TestGradX3(), &rng, &pool3);

#if VERBOSE
      printf("GradY3---------------\n");
#endif
      VerifyAll<slow::General3x3Convolution, strategy::GradY3>(
          xsize, ysize, kernel::TestGradY3(), &rng, &pool3);

#if VERBOSE
      printf("Corner3-----------------\n");
#endif
      VerifyAll<slow::General3x3Convolution, strategy::Corner3>(
          xsize, ysize, kernel::TestCorner3(), &rng, &pool3);

#if VERBOSE
      printf("Lapl3-------------------\n");
#endif
      VerifyAll<slow::Symmetric3x3Convolution, strategy::Laplacian3>(
          xsize, ysize, kernel::Laplacian3(), &rng, &pool3);
#if VERBOSE
      printf("Sep5x5------------------\n");
#endif
      VerifyAll<slow::SeparableConvolution, strategy::Separable5>(
          xsize, ysize, kernel::Lowpass5(), &rng, &pool3);

      VerifyConvolveSymmetric3x3(xsize, ysize, &rng);
      VerifyConvolveSymmetric5x5(xsize, ysize, &rng);
    }
  });
}

TEST(ConvolveTest, TestSameResults) { TestSameResultsImpl(); }

}  // namespace
}  // namespace pik
