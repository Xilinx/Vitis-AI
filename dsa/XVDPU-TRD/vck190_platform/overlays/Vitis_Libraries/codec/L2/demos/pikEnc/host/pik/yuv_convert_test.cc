// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/yuv_convert.h"

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
                              const Image3<T>& actual, int tolerance) {
  ASSERT_EQ(expected.xsize(), actual.xsize());
  ASSERT_EQ(expected.ysize(), actual.ysize());
  int max_diff = 0;
  for (int c = 0; c < 3; ++c) {
    for (size_t y = 0; y < expected.ysize(); ++y) {
      const T* PIK_RESTRICT row_expected = expected.PlaneRow(c, y);
      const T* PIK_RESTRICT row_actual = actual.PlaneRow(c, y);
      for (size_t x = 0; x < expected.xsize(); ++x) {
        max_diff =
            std::max<int>(max_diff, std::abs(row_expected[x] - row_actual[x]));
        ASSERT_LE(row_expected[x], row_actual[x] + tolerance)
            << c << " " << x << " " << y;
        ASSERT_GE(row_expected[x], row_actual[x] - tolerance)
            << c << " " << x << " " << y;
      }
    }
  }
  printf("max diff: %d\n", max_diff);
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

TEST(YUVConvertTest, RGB8ToYUVToRGB8RoundTrip) {
  const std::string path =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  Image3B rgb8_in;
  CodecContext context;
  CodecInOut io(&context);
  ASSERT_TRUE(io.SetFromFile(path));
  ASSERT_TRUE(
      io.CopyTo(Rect(io.color()), context.c_srgb[io.IsGray()], &rgb8_in));
  {
    Image3U yuv = YUVRec709ImageFromRGB8(rgb8_in, 8);
    Image3B rgb8_out = RGB8ImageFromYUVRec709(yuv, 8);
    VerifyEqualWithTolerance(rgb8_in, rgb8_out, 2);
  }
  {
    Image3U yuv = YUVRec709ImageFromRGB8(rgb8_in, 10);
    Image3B rgb8_out = RGB8ImageFromYUVRec709(yuv, 10);
    VerifyEqualWithTolerance(rgb8_in, rgb8_out, 0);
  }
  {
    Image3U yuv = YUVRec709ImageFromRGB8(rgb8_in, 12);
    Image3B rgb8_out = RGB8ImageFromYUVRec709(yuv, 12);
    VerifyEqualWithTolerance(rgb8_in, rgb8_out, 0);
  }
}

TEST(YUVConvertTest, RGB16ToYUVToRGB16RoundTrip) {
  const std::string path =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  Image3U rgb16_in;
  CodecContext context;
  CodecInOut io(&context);
  ASSERT_TRUE(io.SetFromFile(path));
  ASSERT_TRUE(
      io.CopyTo(Rect(io.color()), context.c_srgb[io.IsGray()], &rgb16_in));
  {
    Image3U yuv = YUVRec709ImageFromRGB16(rgb16_in, 8);
    Image3U rgb16_out = RGB16ImageFromYUVRec709(yuv, 8);
    VerifyEqualWithTolerance(rgb16_in, rgb16_out, 420);
  }
  {
    Image3U yuv = YUVRec709ImageFromRGB16(rgb16_in, 10);
    Image3U rgb16_out = RGB16ImageFromYUVRec709(yuv, 10);
    VerifyEqualWithTolerance(rgb16_in, rgb16_out, 105);
  }
  {
    Image3U yuv = YUVRec709ImageFromRGB16(rgb16_in, 12);
    Image3U rgb16_out = RGB16ImageFromYUVRec709(yuv, 12);
    VerifyEqualWithTolerance(rgb16_in, rgb16_out, 26);
  }
}

TEST(YUVConvertTest, YUVToRGB16ToYUVRoundTrip) {
  const int kNumSamples = 1 << 22;
  {
    const int bits = 8;
    Image3U yuv_in = GenerateRandomYUVImage(kNumSamples, bits, 7);
    Image3U rgb = RGB16ImageFromYUVRec709(yuv_in, bits);
    Image3U yuv_out = YUVRec709ImageFromRGB16(rgb, bits);
    VerifyYUVRoundtripWithTolerance(yuv_in, rgb, yuv_out, 0);
  }
  {
    const int bits = 10;
    Image3U yuv_in = GenerateRandomYUVImage(kNumSamples, bits, 77);
    Image3U rgb = RGB16ImageFromYUVRec709(yuv_in, bits);
    Image3U yuv_out = YUVRec709ImageFromRGB16(rgb, bits);
    VerifyYUVRoundtripWithTolerance(yuv_in, rgb, yuv_out, 0);
  }
  {
    const int bits = 12;
    Image3U yuv_in = GenerateRandomYUVImage(kNumSamples, bits, 777);
    Image3U rgb = RGB16ImageFromYUVRec709(yuv_in, bits);
    Image3U yuv_out = YUVRec709ImageFromRGB16(rgb, bits);
    VerifyYUVRoundtripWithTolerance(yuv_in, rgb, yuv_out, 0);
  }
}

TEST(YUVConvertTest, RGBLinearToYUVToRGBLinearRoundTrip) {
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecContext codec_context;
  ThreadPool pool(4);
  CodecInOut io(&codec_context);
  PIK_CHECK(io.SetFromFile(pathname, &pool));
  PIK_CHECK(io.TransformTo(codec_context.c_linear_srgb[io.IsGray()], &pool));

  CodecInOut io2(&codec_context);

  {
    Image3U yuv = YUVRec709ImageFromRGBLinear(io.color(), 8);
    io2.SetFromImage(RGBLinearImageFromYUVRec709(yuv, 8),
                     codec_context.c_linear_srgb[0]);
    float distance =
        ButteraugliDistance(&io, &io2, 1.0, /*distmap=*/nullptr, &pool);
    printf("distance = %.4f\n", distance);
    EXPECT_LT(distance, 0.49);
  }
  {
    Image3U yuv = YUVRec709ImageFromRGBLinear(io.color(), 10);
    io2.SetFromImage(RGBLinearImageFromYUVRec709(yuv, 10),
                     codec_context.c_linear_srgb[0]);
    float distance =
        ButteraugliDistance(&io, &io2, 1.0, /*distmap=*/nullptr, &pool);
    printf("distance = %.4f\n", distance);
    EXPECT_LT(distance, 0.2);
  }
  {
    Image3U yuv = YUVRec709ImageFromRGBLinear(io.color(), 12);
    io2.SetFromImage(RGBLinearImageFromYUVRec709(yuv, 12),
                     codec_context.c_linear_srgb[0]);
    float distance =
        ButteraugliDistance(&io, &io2, 1.0, /*distmap=*/nullptr, &pool);
    printf("distance = %.4f\n", distance);
    EXPECT_LT(distance, 0.2);
  }
}

TEST(YUVConvertTest, SubSampleSuperSampleRoundTrip) {
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/cvo9xd_keong_macan_srgb8.png");
  CodecContext codec_context;
  ThreadPool pool(4);
  CodecInOut io(&codec_context);
  PIK_CHECK(io.SetFromFile(pathname, &pool));
  PIK_CHECK(io.TransformTo(codec_context.c_linear_srgb[io.IsGray()], &pool));

  CodecInOut io_in(&codec_context);
  CodecInOut io_out(&codec_context);

  io.ShrinkTo(128, 129);
  for (int bits = 8; bits <= 12; bits += 2) {
    Image3U yuv_in = YUVRec709ImageFromRGBLinear(io.color(), bits);
    ImageU yplane, uplane, vplane;
    SubSampleChroma(yuv_in, bits, &yplane, &uplane, &vplane);
    EXPECT_EQ(uplane.xsize(), 64);
    EXPECT_EQ(vplane.xsize(), 64);
    EXPECT_EQ(uplane.ysize(), 65);
    EXPECT_EQ(vplane.ysize(), 65);
    Image3U yuv_out = SuperSampleChroma(yplane, uplane, vplane, bits);
    EXPECT_EQ(yuv_out.xsize(), 128);
    EXPECT_EQ(yuv_out.ysize(), 129);
    io_in.SetFromImage(RGBLinearImageFromYUVRec709(yuv_in, bits),
                       codec_context.c_linear_srgb[0]);
    io_out.SetFromImage(RGBLinearImageFromYUVRec709(yuv_out, bits),
                        codec_context.c_linear_srgb[0]);
    float distance =
        ButteraugliDistance(&io_in, &io_out, 1.0, /*distmap=*/nullptr, &pool);
    printf("bits = %d distance = %.4f\n", bits, distance);
    EXPECT_LT(distance, 2.8);
  }
}

}  // namespace
}  // namespace pik
