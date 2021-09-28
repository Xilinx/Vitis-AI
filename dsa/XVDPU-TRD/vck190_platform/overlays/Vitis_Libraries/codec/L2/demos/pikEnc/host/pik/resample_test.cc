// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/resample.h"

#include <stdio.h>

#include "gtest/gtest.h"
#include "pik/codec.h"
#include "pik/data_parallel.h"
#include "pik/image.h"
#include "pik/testdata_path.h"

namespace pik {
namespace {

#define ENABLE_SLOW 1
#define ENABLE_SEPARATED 1
#define ENABLE_GENERAL 1

Image3F TestImage() {
  const std::string path =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecContext context;
  CodecInOut io(&context);
  PIK_CHECK(io.SetFromFile(path));
  Image3F srgb;
  PIK_CHECK(io.CopyTo(Rect(io.color()), context.c_srgb[io.IsGray()], &srgb));
  srgb = ScaleImage(255.f, srgb);
  return srgb;
}

SIMD_ATTR void CustomImpl() {
  ImageF impulse(10, 10);
  ZeroFillImage(&impulse);
  impulse.Row(5)[5] = 1.0f;
  ImageF expected(impulse.xsize() * 8, impulse.ysize() * 8);
  Upsample<slow::Upsampler>(ExecutorLoop(), impulse, kernel::CatmullRom(),
                            &expected);

  ImageF out(expected.xsize(), expected.ysize());

  printf("custom r=2\n");
  auto kernel2 = kernel::Custom<2>::FromResult(expected);
  Upsample<GeneralUpsampler8>(ExecutorLoop(), impulse, kernel2, &out);
  VerifyRelativeError(expected, out, 2E-5f, 2E-5f);

  printf("custom r=3\n");
  auto kernel3 = kernel::Custom<3>::FromResult(expected);
  Upsample<GeneralUpsampler8_6x6>(ExecutorLoop(), impulse, kernel3, &out);
  VerifyRelativeError(expected, out, 2E-5f, 2E-5f);
}
TEST(ResampleTest, Custom) { CustomImpl(); }

template <class Kernel>
SIMD_ATTR void TestStepEdge(const Kernel kernel) {
  const size_t in_size = 2 * Kernel::kRadius + 1;

  // Vertical step image
  ImageF in(in_size, in_size);
  for (size_t y = 0; y < in_size; ++y) {
    float* PIK_RESTRICT row = in.Row(y);
    for (size_t x = 0; x < in_size; ++x) {
      row[x] = x < in_size / 2 ? 1.0f : 100.0f;
    }
  }

  ImageF out(in_size * 8, in_size * 8);
  Upsample<slow::Upsampler>(in, kernel, &out);

  for (size_t x = 2 * in_size; x < 6 * in_size; ++x) {
    printf("%.2f\n", out.ConstRow(in_size * 4)[x]);
  }
}

TEST(ResampleTest, Test1D) { TestStepEdge(kernel::Lanczos3()); }

// Ensures slow::Upsampler and Upsampler8 return the same result.
template <class Executor, class Kernel, class Image>
SIMD_ATTR void VerifyUpsampler8(const Executor executor, const Image& in,
                                const Kernel kernel) {
// Reference for any comparisons below
#if ENABLE_GENERAL || ENABLE_SEPARATED || ENABLE_SLOW
  Image out(in.xsize() * 8, in.ysize() * 8);
  Upsample<slow::Upsampler>(in, kernel, &out);
#endif

#if ENABLE_SEPARATED
  Image out_x8(in.xsize() * 8, in.ysize() * 8);
  Upsample<Upsampler8>(executor, in, kernel, &out_x8);
#endif

#if ENABLE_SLOW
  Image out_from(in.xsize() * 8, in.ysize() * 8);
  Upsample<slow::GeneralUpsamplerFromSeparable>(in, kernel, &out_from);
  Image out_gen(in.xsize() * 8, in.ysize() * 8);
  Upsample<slow::GeneralUpsampler<8>>(in, kernel, &out_gen);
#endif

#if ENABLE_GENERAL
  Image out_gen_x8(in.xsize() * 8, in.ysize() * 8);
  Upsample<GeneralUpsampler8>(in, kernel, &out_gen_x8);
#endif

#if ENABLE_SEPARATED
  VerifyRelativeError(out, out_x8, 2E-5, 2E-5);
#endif
#if ENABLE_SLOW
  VerifyRelativeError(out, out_from, 2E-5, 2E-5);
  VerifyRelativeError(out, out_gen, 2E-5, 2E-5);
#endif
#if ENABLE_GENERAL
  VerifyRelativeError(out, out_gen_x8, 2E-5, 2E-5);
#endif
}

template <class Executor>
SIMD_ATTR void VerifyUpsampler8Subset(Executor executor, const Image3F& whole,
                                      const size_t in_xsize,
                                      const size_t in_ysize) {
  Image3F in3 = CopyImage(whole);
  in3.ShrinkTo(in_xsize, in_ysize);

  printf("---------------cr %zu %zu\n", in_xsize, in_ysize);
  for (int c = 0; c < 3; ++c) {
    VerifyUpsampler8(executor, in3.Plane(c), kernel::CatmullRom());
  }
  printf("---------------cr3 %zu %zu\n", in_xsize, in_ysize);
  VerifyUpsampler8(executor, in3, kernel::CatmullRom());
}

SIMD_ATTR void TestX8SameOnRealImageImpl() {
  const Image3F& in3 = TestImage();

  // Test wide/narrow and short/tall cases.
  VerifyUpsampler8Subset(ExecutorLoop(), in3, 8, 32);
  VerifyUpsampler8Subset(ExecutorLoop(), in3, 32, 8);
  VerifyUpsampler8Subset(ExecutorLoop(), in3, 63, 63);

  // Larger image, parallel.
  ThreadPool pool(4);
  VerifyUpsampler8Subset(ExecutorPool(&pool), in3, 80, 96);
}
TEST(ResampleTest, TestX8SameOnRealImage) { TestX8SameOnRealImageImpl(); }

SIMD_ATTR void TestX8SameOnSyntheticImpl() {
  // Horizontal/vertical/diagonal lines for verifying ringing.
  Image3F synth(128, 128);
  ZeroFillImage(&synth);

  for (size_t x = 20; x < 60; ++x) {
    synth.PlaneRow(0, 20)[x] = 1.0f;
  }

  for (size_t y = 20; y < 60; ++y) {
    synth.PlaneRow(0, y)[30] = 1.0f;
  }

  for (size_t y = 80; y < 100; ++y) {
    synth.PlaneRow(0, y)[y] = 1.0f;
  }

  VerifyUpsampler8Subset(ExecutorLoop(), synth, synth.xsize(), synth.ysize());

  ThreadPool pool(4);
  VerifyUpsampler8Subset(ExecutorPool(&pool), synth, synth.xsize(),
                         synth.ysize());
}
TEST(ResampleTest, TestX8SameOnSynthetic) { TestX8SameOnSyntheticImpl(); }

}  // namespace
}  // namespace pik
