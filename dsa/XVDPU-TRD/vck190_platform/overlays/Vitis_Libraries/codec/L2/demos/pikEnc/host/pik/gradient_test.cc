// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/pik.h"

#include "gtest/gtest.h"
#include "pik/codec.h"

namespace pik {
namespace {

// Returns distance of point p to line p0..p1, the result is signed and is not
// normalized.
double PointLineDist(double x0, double y0, double x1, double y1, double x,
                     double y) {
  return (y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0;
}

// Generates a test image with a gradient from one color to another.
// Angle in degrees, colors can be given in hex as 0xRRGGBB. The angle is the
// angle in which the change direction happens.
Image3F GenerateTestGradient(uint32_t color0, uint32_t color1, double angle,
                             size_t xsize, size_t ysize) {
  Image3F image(xsize, ysize);

  double x0 = xsize / 2;
  double y0 = ysize / 2;
  double x1 = x0 + std::sin(angle / 360.0 * 2.0 * M_PI);
  double y1 = y0 + std::cos(angle / 360.0 * 2.0 * M_PI);

  double maxdist =
      std::max<double>(fabs(PointLineDist(x0, y0, x1, y1, 0, 0)),
                       fabs(PointLineDist(x0, y0, x1, y1, xsize, 0)));

  for (size_t c = 0; c < 3; ++c) {
    float c0 = ((color0 >> (8 * (2 - c))) & 255);
    float c1 = ((color1 >> (8 * (2 - c))) & 255);
    for (size_t y = 0; y < ysize; ++y) {
      float* row = image.PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        double dist = PointLineDist(x0, y0, x1, y1, x, y);
        double v = ((dist / maxdist) + 1.0) / 2.0;
        float color = c0 * (1.0 - v) + c1 * v;
        row[x] = color;
      }
    }
  }

  return image;
}

// Computes the max of the horizontal and vertical second derivative for each
// pixel, where second derivative means absolute value of difference of left
// delta and right delta (top/bottom for vertical direction).
// The radius over which the derivative is computed is only 1 pixel and it only
// checks two angles (hor and ver), but this approximation works well enough.
static ImageF Gradient2(const ImageF& image) {
  size_t xsize = image.xsize();
  size_t ysize = image.ysize();
  ImageF image2(image.xsize(), image.ysize());
  for (size_t y = 1; y + 1 < ysize; y++) {
    const auto* PIK_RESTRICT row0 = image.Row(y - 1);
    const auto* PIK_RESTRICT row1 = image.Row(y);
    const auto* PIK_RESTRICT row2 = image.Row(y + 1);
    auto* row_out = image2.Row(y);
    for (int x = 1; x + 1 < xsize; x++) {
      float ddx = (row1[x] - row1[x - 1]) - (row1[x + 1] - row1[x]);
      float ddy = (row1[x] - row0[x]) - (row2[x] - row1[x]);
      row_out[x] = std::max(fabsf(ddx), fabsf(ddy));
    }
  }
  // Copy to the borders
  if (ysize > 2) {
    auto* PIK_RESTRICT row0 = image2.Row(0);
    const auto* PIK_RESTRICT row1 = image2.Row(1);
    const auto* PIK_RESTRICT row2 = image2.Row(ysize - 2);
    auto* PIK_RESTRICT row3 = image2.Row(ysize - 1);
    for (size_t x = 1; x + 1 < xsize; x++) {
      row0[x] = row1[x];
      row3[x] = row2[x];
    }
  } else {
    const auto* row0_in = image.Row(0);
    const auto* row1_in = image.Row(ysize - 1);
    auto* row0_out = image2.Row(0);
    auto* row1_out = image2.Row(ysize - 1);
    for (size_t x = 1; x + 1 < xsize; x++) {
      // Image too narrow, take first derivative instead
      row0_out[x] = row1_out[x] = fabsf(row0_in[x] - row1_in[x]);
    }
  }
  if (xsize > 2) {
    for (size_t y = 0; y < ysize; y++) {
      auto* row = image2.Row(y);
      row[0] = row[1];
      row[xsize - 1] = row[xsize - 2];
    }
  } else {
    for (size_t y = 0; y < ysize; y++) {
      const auto* PIK_RESTRICT row_in = image.Row(y);
      auto* row_out = image2.Row(y);
      // Image too narrow, take first derivative instead
      row_out[0] = row_out[xsize - 1] = fabsf(row_in[0] - row_in[xsize - 1]);
    }
  }
  return image2;
}

static Image3F Gradient2(const Image3F& image) {
  return Image3F(Gradient2(image.Plane(0)), Gradient2(image.Plane(1)),
                 Gradient2(image.Plane(2)));
}

/*
Tests if roundtrip with pik on a gradient image doesn't cause banding.
Only tests if use_gradient is true. Set to false for debugging to see the
distance values.
Angle in degrees, colors can be given in hex as 0xRRGGBB.
*/
void TestGradient(ThreadPool* pool, uint32_t color0, uint32_t color1,
                  size_t xsize, size_t ysize, float angle, bool fast_mode,
                  float butteraugli_distance, bool use_gradient = true) {
  CompressParams cparams;
  cparams.butteraugli_distance = butteraugli_distance;
  cparams.gradient = use_gradient ? Override::kOn : Override::kOff;
  cparams.fast_mode = fast_mode;
  DecompressParams dparams;

  Image3F gradient = GenerateTestGradient(color0, color1, angle, xsize, ysize);

  CodecContext codec_context;
  CodecInOut io(&codec_context);
  io.SetFromImage(std::move(gradient), codec_context.c_srgb[0]);
  io.SetOriginalBitsPerSample(8);

  CodecInOut io2(&codec_context);

  PaddedBytes compressed;
  PikInfo* aux = nullptr;
  EXPECT_TRUE(PixelsToPik(cparams, &io, &compressed, aux, pool));
  EXPECT_TRUE(PikToPixels(dparams, compressed, &io2, aux, pool));
  EXPECT_TRUE(io2.TransformTo(io2.dec_c_original, pool));

  if (use_gradient) {
    // Test that the gradient map worked. For that, we take a second derivative
    // of the image with Gradient2 to measure how linear the change is in x and
    // y direction. For a well handled gradient, we expect max values around
    // 0.1, while if there is noticeable banding, which means the gradient map
    // failed, the values are around 0.5-1.0 (regardless of
    // butteraugli_distance).
    Image3F gradient2 = Gradient2(io2.color());

    std::array<float, 3> image_max;
    Image3Max(gradient2, &image_max);
    EXPECT_LE(image_max[0], 0.5);
    EXPECT_LE(image_max[1], 0.5);
    EXPECT_LE(image_max[2], 0.5);
  }
}

static constexpr bool fast_mode = true;

TEST(GradientTest, SteepGradient) {
  ThreadPool pool(8);
  // Relatively steep gradients, colors from the sky of stp.png
  TestGradient(&pool, 0xd99d58, 0x889ab1, 512, 512, 90, fast_mode, 4.0);
}

TEST(GradientTest, SubtleGradient) {
  ThreadPool pool(8);
  // Very subtle gradient
  TestGradient(&pool, 0xb89b7b, 0xa89b8d, 512, 512, 90, fast_mode, 4.0);
}

}  // namespace
}  // namespace pik
