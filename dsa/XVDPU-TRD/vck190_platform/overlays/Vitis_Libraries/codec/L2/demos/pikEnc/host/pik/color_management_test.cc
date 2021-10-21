// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/color_management.h"
#include "gtest/gtest.h"

#include "pik/common.h"
#include "pik/data_parallel.h"

namespace pik {
namespace {

// Small enough to be fast. If changed, must update Generate*.
static constexpr size_t kWidth = 16;

struct Globals {
  static constexpr size_t kNumThreads = 0;  // only have a single row.

  Globals() : pool(kNumThreads) {
    in_gray = GenerateGray();
    in_color = GenerateColor();
    out_gray = ImageF(kWidth, 1);
    out_color = ImageF(kWidth * 3, 1);

    c_native.transfer_function = TransferFunction::kLinear;
    PIK_CHECK(ColorManagement::SetProfileFromFields(&c_native));

    c_gray.color_space = ColorSpace::kGray;
    c_gray.transfer_function = TransferFunction::kLinear;
    PIK_CHECK(ColorManagement::SetProfileFromFields(&c_gray));
  }

 private:
  static ImageF GenerateGray() {
    ImageF gray(kWidth, 1);
    float* PIK_RESTRICT row = gray.Row(0);
    // Increasing left to right
    for (int32_t x = 0; x < kWidth; ++x) {
      row[x] = x * 1.0f / (kWidth - 1);  // [0, 1]
    }
    return gray;
  }

  static ImageF GenerateColor() {
    ImageF image(kWidth * 3, 1);
    float* PIK_RESTRICT interleaved = image.Row(0);
    std::fill(interleaved, interleaved + kWidth * 3, 0.0f);

    // [0, 4): neutral
    for (int32_t x = 0; x < 4; ++x) {
      interleaved[3 * x + 0] = x * 1.0f / 3;  // [0, 1]
      interleaved[3 * x + 2] = interleaved[3 * x + 1] = interleaved[3 * x + 0];
    }

    // [4, 13): pure RGB with low/medium/high saturation
    for (int32_t c = 0; c < 3; ++c) {
      interleaved[3 * (4 + c) + c] = 0.08f + c * 0.01f;
      interleaved[3 * (7 + c) + c] = 0.75f + c * 0.01f;
      interleaved[3 * (10 + c) + c] = 1.0f;
    }

    // [13, 16): impure, not quite saturated RGB
    interleaved[3 * 13 + 0] = 0.86f;
    interleaved[3 * 13 + 2] = interleaved[3 * 13 + 1] = 0.16f;
    interleaved[3 * 14 + 1] = 0.87f;
    interleaved[3 * 14 + 2] = interleaved[3 * 14 + 0] = 0.16f;
    interleaved[3 * 15 + 2] = 0.88f;
    interleaved[3 * 15 + 1] = interleaved[3 * 15 + 0] = 0.16f;

    return image;
  }

 public:
  ThreadPool pool;

  // ImageF so we can use VerifyRelativeError; all are interleaved RGB.
  ImageF in_gray;
  ImageF in_color;
  ImageF out_gray;
  ImageF out_color;
  ColorEncoding c_native;
  ColorEncoding c_gray;
};
static Globals* g;

class ColorManagementTest : public ::testing::TestWithParam<ColorEncoding> {
 public:
  static void SetUpTestSuite() { g = new Globals; }
  static void TearDownTestSuite() { delete g; }

  static void VerifySameFields(const ColorEncoding& c,
                               const ColorEncoding& c2) {
    ASSERT_EQ(c.rendering_intent, c2.rendering_intent);
    ASSERT_EQ(c.color_space, c2.color_space);
    // XYZ doesn't have a white point.
    if (c.color_space != ColorSpace::kXYZ) {
      ASSERT_EQ(c.white_point, c2.white_point);
    }
    // Gray and (absolute) XYZ don't have primaries.
    if (c.color_space != ColorSpace::kGray &&
        c.color_space != ColorSpace::kXYZ) {
      ASSERT_EQ(c.primaries, c2.primaries);
    }
    ASSERT_EQ(c.transfer_function, c2.transfer_function);
  }

  // "Same" pixels after converting g->c_native -> c -> g->c_native.
  static void VerifyPixelRoundTrip(const ColorEncoding& c) {
    const ColorEncoding& c_native = c.IsGray() ? g->c_gray : g->c_native;
    ColorSpaceTransform xform_fwd;
    ColorSpaceTransform xform_rev;
    ASSERT_TRUE(xform_fwd.Init(c_native, c, kWidth, g->pool.NumThreads()));
    ASSERT_TRUE(xform_rev.Init(c, c_native, kWidth, g->pool.NumThreads()));

    const size_t thread = 0;
    const ImageF& in = c.IsGray() ? g->in_gray : g->in_color;
    ImageF* PIK_RESTRICT out = c.IsGray() ? &g->out_gray : &g->out_color;
    xform_fwd.Run(thread, in.Row(0), xform_fwd.BufDst(thread));
    xform_rev.Run(thread, xform_fwd.BufDst(thread), out->Row(0));

    double max_l1 = 5E-5;
    // Most are lower; reached 3E-7 with D60 AP0.
    double max_rel = 4E-7;
    if (c.IsGray()) max_rel = 2E-5;
    VerifyRelativeError(in, *out, max_l1, max_rel);
  }
};
INSTANTIATE_TEST_CASE_P(ColorManagementTestInstantiation, ColorManagementTest,
                        ::testing::ValuesIn(AllEncodings()));

// Exercises the ColorManagement interface for ALL ColorEncoding synthesizable
// via enums.
TEST_P(ColorManagementTest, VerifyAllProfiles) {
  ColorEncoding c = GetParam();
  const std::string& description = Description(c);
  printf("%s\n", description.c_str());

  // Can create profile.
  ASSERT_TRUE(ColorManagement::SetProfileFromFields(&c));

  // Can set an equivalent ColorEncoding from ProfileParams.
  ColorEncoding c2;
  ProfileParams pp;
  ASSERT_TRUE(ColorEncodingToParams(c, &pp));
  ASSERT_TRUE(ColorManagement::SetFromParams(pp, &c2));
  VerifySameFields(c, c2);

  // Can set an equivalent ColorEncoding from the generated ICC profile.
  ColorEncoding c3;
  ASSERT_TRUE(ColorManagement::SetFromProfile(std::move(c.icc), &c3));
  VerifySameFields(c, c3);
  // (need a profile for VerifyPixelRoundTrip and MaybeRemoveProfile.)
  c.icc = std::move(c3.icc);

  VerifyPixelRoundTrip(c);

  // MaybeRemoveProfile actually removes the profile.
  ASSERT_TRUE(ColorManagement::MaybeRemoveProfile(&c));
  EXPECT_TRUE(c.icc.empty());
}

}  // namespace
}  // namespace pik
