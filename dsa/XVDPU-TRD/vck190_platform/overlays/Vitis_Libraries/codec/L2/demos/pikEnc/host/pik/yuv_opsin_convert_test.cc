// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/yuv_opsin_convert.h"

#include <string>

#include "gtest/gtest.h"
#include "pik/butteraugli_distance.h"
#include "pik/codec.h"
#include "pik/image.h"
#include "pik/testdata_path.h"

namespace pik {
namespace {

template <typename T>
void VerifyEqualWithTolerance(const Image3<T>& expected,
                              const Image3<T>& actual, int tolerance0,
                              int tolerance1, int tolerance2) {
  ASSERT_EQ(expected.xsize(), actual.xsize());
  ASSERT_EQ(expected.ysize(), actual.ysize());
  int max_diff[3] = {0};
  const int kTolerance[3] = {tolerance0, tolerance1, tolerance2};
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < expected.ysize(); ++y) {
      const T* PIK_RESTRICT row_expected = expected.PlaneRow(c, y);
      const T* PIK_RESTRICT row_actual = actual.PlaneRow(c, y);
      for (size_t x = 0; x < expected.xsize(); ++x) {
        const int error = std::abs(row_expected[x] - row_actual[x]);
        max_diff[c] = std::max<int>(max_diff[c], error);
        ASSERT_LE(error, kTolerance[c])
            << "x: " << x << " y: " << y << " c: " << c
            << " rgb_in: " << row_expected[x] << " rgb_out: " << row_actual[x]
            << "\n";
      }
    }
  }
  printf("max diff: %d, %d, %d\n", max_diff[0], max_diff[1], max_diff[2]);
}

void VerifyYUVRoundtripWithTolerance(const Image3U& yuv_in, const Image3U& rgb,
                                     const Image3U& yuv_out, int tolerance) {
  ASSERT_EQ(yuv_in.xsize(), yuv_out.xsize());
  ASSERT_EQ(yuv_in.ysize(), yuv_out.ysize());
  ASSERT_EQ(yuv_in.xsize(), rgb.xsize());
  ASSERT_EQ(yuv_in.ysize(), rgb.ysize());
  int max_diff = 0;
  for (size_t y = 0; y < yuv_in.ysize(); ++y) {
    for (size_t x = 0; x < yuv_in.xsize(); ++x) {
      // Skip roundtrip comparison for this pixel when at least one color
      // coordinate was potentially truncated in the yuv->rgb step, i.e. the
      // yuv value was outside the rgb color cube.
      bool skip = false;
      for (int c = 0; c < 3; ++c) {
        const uint16_t* PIK_RESTRICT row_rgb = rgb.PlaneRow(c, y);
        skip |= (row_rgb[x] == 0 || row_rgb[x] == 65535);
      }

      if (skip) continue;
      for (int c = 0; c < 3; ++c) {
        const uint16_t* PIK_RESTRICT row_in = yuv_in.PlaneRow(c, y);
        const uint16_t* PIK_RESTRICT row_rgb = rgb.PlaneRow(c, y);
        const uint16_t* PIK_RESTRICT row_out = yuv_out.PlaneRow(c, y);

        max_diff = std::max<int>(max_diff, std::abs(row_in[x] - row_out[x]));
        const int error = std::abs(row_in[x] - row_out[x]);
        ASSERT_LE(error, tolerance)
            << "x: " << x << " y: " << y << " c: " << c
            << " yuv_in: " << row_in[x] << " rgb: " << row_rgb[x]
            << " yuv_out: " << row_out[x] << "\n";
      }
    }
  }
  printf("max diff: %d\n", max_diff);
}

Image3U GenerateRandomYUVImage(int num_samples, int bits, int seed) {
  const uint16_t yuv_min = 16 << (bits - 8);
  const uint16_t y_max = 235 << (bits - 8);
  const uint16_t uv_max = 240 << (bits - 8);
  ImageU yplane(num_samples, 1);
  ImageU uplane(num_samples, 1);
  ImageU vplane(num_samples, 1);
  RandomFillImage(&yplane, yuv_min, y_max, seed);
  RandomFillImage(&uplane, yuv_min, uv_max, seed + 1);
  RandomFillImage(&vplane, yuv_min, uv_max, seed + 2);
  Image3U out(std::move(yplane), std::move(uplane), std::move(vplane));
  return out;
}

TEST(YUVOpsinConvertTest, RGB8ToYUVToRGB8RoundTrip) {
  const std::string path =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  Image3B rgb8_in;
  CodecContext context;
  CodecInOut io(&context);
  ASSERT_TRUE(io.SetFromFile(path));
  ASSERT_TRUE(
      io.CopyTo(Rect(io.color()), context.c_srgb[io.IsGray()], &rgb8_in));
  {
    Image3U yuv = YUVOpsinImageFromRGB8(rgb8_in, 8);
    Image3B rgb8_out = RGB8ImageFromYUVOpsin(yuv, 8);
    VerifyEqualWithTolerance(rgb8_in, rgb8_out, 30, 11, 30);
  }
  {
    Image3U yuv = YUVOpsinImageFromRGB8(rgb8_in, 10);
    Image3B rgb8_out = RGB8ImageFromYUVOpsin(yuv, 10);
    VerifyEqualWithTolerance(rgb8_in, rgb8_out, 11, 3, 11);
  }
  {
    Image3U yuv = YUVOpsinImageFromRGB8(rgb8_in, 12);
    Image3B rgb8_out = RGB8ImageFromYUVOpsin(yuv, 12);
    VerifyEqualWithTolerance(rgb8_in, rgb8_out, 3, 1, 3);
  }
}

TEST(YUVOpsinConvertTest, RGB16ToYUVToRGB16RoundTrip) {
  const std::string path =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  Image3U rgb16_in;
  CodecContext context;
  CodecInOut io(&context);
  ASSERT_TRUE(io.SetFromFile(path));
  ASSERT_TRUE(
      io.CopyTo(Rect(io.color()), context.c_srgb[io.IsGray()], &rgb16_in));
  {
    Image3U yuv = YUVOpsinImageFromRGB16(rgb16_in, 8);
    Image3U rgb16_out = RGB16ImageFromYUVOpsin(yuv, 8);
    VerifyEqualWithTolerance(rgb16_in, rgb16_out, 7710, 2831, 7673);
  }
  {
    Image3U yuv = YUVOpsinImageFromRGB16(rgb16_in, 10);
    Image3U rgb16_out = RGB16ImageFromYUVOpsin(yuv, 10);
    VerifyEqualWithTolerance(rgb16_in, rgb16_out, 2939, 694, 2827);
  }
  {
    Image3U yuv = YUVOpsinImageFromRGB16(rgb16_in, 12);
    Image3U rgb16_out = RGB16ImageFromYUVOpsin(yuv, 12);
    VerifyEqualWithTolerance(rgb16_in, rgb16_out, 716, 144, 724);
  }
}

TEST(YUVOpsinConvertTest, YUVToRGB16ToYUVRoundTrip) {
  const int kNumSamples = 1 << 20;
  {
    const int bits = 8;
    Image3U yuv_in = GenerateRandomYUVImage(kNumSamples, bits, 7);
    Image3U rgb = RGB16ImageFromYUVOpsin(yuv_in, bits);
    Image3U yuv_out = YUVOpsinImageFromRGB16(rgb, bits);
    VerifyYUVRoundtripWithTolerance(yuv_in, rgb, yuv_out, 0);
  }
  {
    const int bits = 10;
    Image3U yuv_in = GenerateRandomYUVImage(kNumSamples, bits, 77);
    Image3U rgb = RGB16ImageFromYUVOpsin(yuv_in, bits);
    Image3U yuv_out = YUVOpsinImageFromRGB16(rgb, bits);
    VerifyYUVRoundtripWithTolerance(yuv_in, rgb, yuv_out, 0);
  }
  {
    const int bits = 12;
    Image3U yuv_in = GenerateRandomYUVImage(kNumSamples, bits, 777);
    Image3U rgb = RGB16ImageFromYUVOpsin(yuv_in, bits);
    Image3U yuv_out = YUVOpsinImageFromRGB16(rgb, bits);
    VerifyYUVRoundtripWithTolerance(yuv_in, rgb, yuv_out, 0);
  }
}

TEST(YUVOpsinConvertTest, RGBLinearToYUVToRGBLinearRoundTrip) {
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecContext codec_context;
  CodecInOut io(&codec_context);
  ThreadPool pool(4);
  PIK_CHECK(io.SetFromFile(pathname, &pool));
  PIK_CHECK(io.TransformTo(codec_context.c_linear_srgb[io.IsGray()], &pool));

  CodecInOut io2(&codec_context);

  {
    Image3U yuv = YUVOpsinImageFromRGBLinear(io.color(), 8);
    io2.SetFromImage(RGBLinearImageFromYUVOpsin(yuv, 8),
                     codec_context.c_linear_srgb[0]);
    float distance =
        ButteraugliDistance(&io, &io2, 1.0, /*distmap=*/nullptr, &pool);
    printf("distance = %.4f\n", distance);
    EXPECT_LT(distance, 1.70);
  }
  {
    Image3U yuv = YUVOpsinImageFromRGBLinear(io.color(), 10);
    io2.SetFromImage(RGBLinearImageFromYUVOpsin(yuv, 10),
                     codec_context.c_linear_srgb[0]);
    float distance =
        ButteraugliDistance(&io, &io2, 1.0, /*distmap=*/nullptr, &pool);
    printf("distance = %.4f\n", distance);
    EXPECT_LT(distance, 0.35);
  }
  {
    Image3U yuv = YUVOpsinImageFromRGBLinear(io.color(), 12);
    io2.SetFromImage(RGBLinearImageFromYUVOpsin(yuv, 12),
                     codec_context.c_linear_srgb[0]);
    float distance =
        ButteraugliDistance(&io, &io2, 1.0, /*distmap=*/nullptr, &pool);
    printf("distance = %.4f\n", distance);
    EXPECT_LT(distance, 0.3);
  }
}

}  // namespace
}  // namespace pik
