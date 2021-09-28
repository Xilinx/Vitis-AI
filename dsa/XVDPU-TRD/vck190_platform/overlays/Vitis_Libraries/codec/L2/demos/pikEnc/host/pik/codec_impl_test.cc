// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/codec.h"

#include <stddef.h>
#include <stdio.h>

#include "gtest/gtest.h"
#include "pik/color_management.h"
#include "pik/external_image.h"
#include "pik/file_io.h"
#include "pik/testdata_path.h"

namespace pik {
namespace {

// Ensures reading a newly written file leads to the same image pixels.
void TestRoundTrip(Codec codec, const size_t xsize, const size_t ysize,
                   const bool is_gray, const bool add_alpha,
                   const size_t bits_per_sample,
                   const CodecContext& codec_context, ThreadPool* pool) {
  if (codec == Codec::kPNM && add_alpha) return;
  printf("Codec %s sb:%zu gr:%d al:%d\n", ExtensionFromCodec(codec).c_str(),
         bits_per_sample, is_gray, add_alpha);

  ColorEncoding c_native;
  c_native.color_space = is_gray ? ColorSpace::kGray : ColorSpace::kRGB;
  c_native.primaries = Primaries::k2020;
  c_native.transfer_function = TransferFunction::kLinear;
  PIK_CHECK(ColorManagement::SetProfileFromFields(&c_native));

  // Generally store same color space to reduce round trip errors..
  ColorEncoding c_external = c_native;
  // .. except for higher bit depths where we can afford to convert.
  if (bits_per_sample >= 16) {
    c_external.primaries = Primaries::kAP1;
    c_external.transfer_function = TransferFunction::kSRGB;
  }
  PIK_CHECK(ColorManagement::SetProfileFromFields(&c_external));

  Image3F image(xsize, ysize);
  std::mt19937_64 rng(129);
  std::uniform_real_distribution<float> dist(0.0f, 255.0f);
  if (is_gray) {
    for (size_t y = 0; y < ysize; ++y) {
      float* PIK_RESTRICT row0 = image.PlaneRow(0, y);
      float* PIK_RESTRICT row1 = image.PlaneRow(1, y);
      float* PIK_RESTRICT row2 = image.PlaneRow(2, y);
      for (size_t x = 0; x < xsize; ++x) {
        row0[x] = row1[x] = row2[x] = dist(rng);
      }
    }
  } else {
    RandomFillImage(&image, 255.0f);
  }
  CodecInOut io(&codec_context);
  io.SetFromImage(std::move(image), c_native);
  if (add_alpha) {
    ImageU alpha(xsize, ysize);
    const size_t alpha_bits = bits_per_sample <= 8 ? 8 : 16;
    const uint16_t max = (1U << alpha_bits) - 1;
    RandomFillImage(&alpha, max);
    io.SetAlpha(std::move(alpha), alpha_bits);
  }

  PaddedBytes encoded;
  PIK_CHECK(io.Encode(codec, c_external, bits_per_sample, &encoded, pool));

  CodecInOut io2(&codec_context);
  // Avoid warnings about PNG ignoring them.
  if (codec == Codec::kPNM) {
    io2.dec_hints.Add("color_space", Description(c_external));
  }
  PIK_CHECK(io2.SetFromBytes(encoded, pool));

  EXPECT_EQ(io.enc_bits_per_sample, io2.original_bits_per_sample());
  EXPECT_EQ(Description(c_external), Description(io2.dec_c_original));

  // See c_external above - for low bits_per_sample the encoded space is
  // already the same.
  if (bits_per_sample < 16) {
    EXPECT_EQ(Description(io.c_current()), Description(io2.c_current()));
  }

  if (add_alpha) {
    EXPECT_TRUE(SamePixels(io.alpha(), io2.alpha()));
  }

  PIK_CHECK(io2.TransformTo(io.c_current(), pool));

  double max_l1, max_rel;
  // Round-trip tolerances must be higher than in external_image_test because
  // codecs do not support unbounded ranges.
  if (io.enc_bits_per_sample <= 12) {
    max_l1 = 0.5;
    max_rel = 6E-3;
  } else if (io.enc_bits_per_sample == 16) {
    max_l1 = 2E-3;
    max_rel = 1E-4;
  } else {
    max_l1 = 1E-7;
    max_rel = 1E-5;
  }

  VerifyRelativeError(io.color(), io2.color(), max_l1, max_rel);
}

TEST(CodecImplTest, TestRoundTrip) {
  const CodecContext codec_context;
  ThreadPool pool(12);

  const size_t xsize = 7;
  const size_t ysize = 4;

  for (Codec codec : AllValues<Codec>()) {
    for (size_t bits_per_sample : {8, 10, 12, 16, 32}) {
      for (bool is_gray : {false, true}) {
        for (bool add_alpha : {false, true}) {
          TestRoundTrip(codec, xsize, ysize, is_gray, add_alpha,
                        bits_per_sample, codec_context, &pool);
        }
      }
    }
  }
}

CodecInOut DecodeRoundtrip(const std::string& pathname, Codec expected_codec,
                           const CodecContext& codec_context, ThreadPool* pool,
                           const DecoderHints& dec_hints = DecoderHints()) {
  CodecInOut io(&codec_context);
  io.dec_hints = dec_hints;
  PIK_CHECK(io.SetFromFile(pathname, pool));

  // Encode/Decode again to make sure Encode carries through all metadata.
  PaddedBytes encoded;
  PIK_CHECK(io.Encode(expected_codec, io.dec_c_original,
                  io.original_bits_per_sample(), &encoded, pool));

  CodecInOut io2(&codec_context);
  io2.dec_hints = dec_hints;
  PIK_CHECK(io2.SetFromBytes(encoded, pool));
  EXPECT_EQ(Description(io.dec_c_original), Description(io2.dec_c_original));
  EXPECT_EQ(Description(io.c_current()), Description(io2.c_current()));

  // "Same" pixels?
  double max_l1 = io.enc_bits_per_sample <= 12 ? 1.3 : 2E-3;
  double max_rel = io.enc_bits_per_sample <= 12 ? 6E-3 : 1E-4;
  if (io.dec_c_original.IsGray()) {
    max_rel *= 2.0;
  } else if (io.dec_c_original.primaries != Primaries::kSRGB) {
    // Need more tolerance for large gamuts (anything but sRGB)
    max_l1 *= 1.5;
    max_rel *= 3.0;
  }
  VerifyRelativeError(io.color(), io2.color(), max_l1, max_rel);

  // Simulate the encoder removing profile and decoder restoring it.
  if (ColorManagement::MaybeRemoveProfile(&io2.dec_c_original)) {
    PIK_CHECK(ColorManagement::SetProfileFromFields(&io2.dec_c_original));
  }

  return io2;
}

TEST(CodecImplTest, TestMetadataSRGB) {
  const CodecContext codec_context;
  ThreadPool pool(12);

  const char* paths[] = {"raw.pixls/DJI-FC6310-16bit_srgb8_v4_krita.png",
                         "raw.pixls/Google-Pixel2XL-16bit_srgb8_v4_krita.png",
                         "raw.pixls/HUAWEI-EVA-L09-16bit_srgb8_dt.png",
                         "raw.pixls/Nikon-D300-12bit_srgb8_dt.png",
                         "raw.pixls/Sony-DSC-RX1RM2-14bit_srgb8_v4_krita.png"};
  for (const char* relative_pathname : paths) {
    const CodecInOut io = DecodeRoundtrip(GetTestDataPath(relative_pathname),
                                          Codec::kPNG, codec_context, &pool);
    EXPECT_EQ(8, io.original_bits_per_sample());

    EXPECT_EQ(64, io.xsize());
    EXPECT_EQ(64, io.ysize());
    EXPECT_FALSE(io.HasAlpha());

    EXPECT_FALSE(io.dec_c_original.icc.empty());
    EXPECT_EQ(ColorSpace::kRGB, io.dec_c_original.color_space);
    EXPECT_EQ(WhitePoint::kD65, io.dec_c_original.white_point);
    EXPECT_EQ(Primaries::kSRGB, io.dec_c_original.primaries);
    EXPECT_EQ(TransferFunction::kSRGB, io.dec_c_original.transfer_function);
  }
}

TEST(CodecImplTest, TestMetadataLinear) {
  const CodecContext codec_context;
  ThreadPool pool(12);

  const char* paths[3] = {
      "raw.pixls/Google-Pixel2XL-16bit_acescg_g1_v4_krita.png",
      "raw.pixls/HUAWEI-EVA-L09-16bit_709_g1_dt.png",
      "raw.pixls/Nikon-D300-12bit_2020_g1_dt.png",
  };
  const WhitePoint white_points[3] = {WhitePoint::kD60, WhitePoint::kD65,
                                      WhitePoint::kD65};
  const Primaries primaries[3] = {Primaries::kAP1, Primaries::kSRGB,
                                  Primaries::k2020};

  for (size_t i = 0; i < 3; ++i) {
    const CodecInOut io = DecodeRoundtrip(GetTestDataPath(paths[i]),
                                          Codec::kPNG, codec_context, &pool);
    EXPECT_EQ(16, io.original_bits_per_sample());

    EXPECT_EQ(64, io.xsize());
    EXPECT_EQ(64, io.ysize());
    EXPECT_FALSE(io.HasAlpha());

    EXPECT_FALSE(io.dec_c_original.icc.empty());
    EXPECT_EQ(ColorSpace::kRGB, io.dec_c_original.color_space);
    EXPECT_EQ(white_points[i], io.dec_c_original.white_point);
    EXPECT_EQ(primaries[i], io.dec_c_original.primaries);
    EXPECT_EQ(TransferFunction::kLinear, io.dec_c_original.transfer_function);
  }
}

TEST(CodecImplTest, TestMetadataICC) {
  const CodecContext codec_context;
  ThreadPool pool(12);

  const char* paths[] = {
      "raw.pixls/DJI-FC6310-16bit_709_v4_krita.png",
      "raw.pixls/Sony-DSC-RX1RM2-14bit_709_v4_krita.png",
  };
  for (const char* relative_pathname : paths) {
    const CodecInOut io = DecodeRoundtrip(GetTestDataPath(relative_pathname),
                                          Codec::kPNG, codec_context, &pool);
    EXPECT_EQ(16, io.original_bits_per_sample());

    EXPECT_EQ(64, io.xsize());
    EXPECT_EQ(64, io.ysize());
    EXPECT_FALSE(io.HasAlpha());

    EXPECT_FALSE(io.dec_c_original.icc.empty());
    EXPECT_EQ(RenderingIntent::kPerceptual, io.dec_c_original.rendering_intent);
    EXPECT_EQ(ColorSpace::kRGB, io.dec_c_original.color_space);
    EXPECT_EQ(WhitePoint::kD65, io.dec_c_original.white_point);
    EXPECT_EQ(Primaries::kSRGB, io.dec_c_original.primaries);
    EXPECT_EQ(TransferFunction::k709, io.dec_c_original.transfer_function);
  }
}

TEST(CodecImplTest, TestPNGSuite) {
  const CodecContext codec_context;
  ThreadPool pool(12);

  // Ensure we can load PNG with text, japanese UTF-8, compressed text.
  (void)DecodeRoundtrip(GetTestDataPath("pngsuite/ct1n0g04.png"), Codec::kPNG,
                        codec_context, &pool);
  (void)DecodeRoundtrip(GetTestDataPath("pngsuite/ctjn0g04.png"), Codec::kPNG,
                        codec_context, &pool);
  (void)DecodeRoundtrip(GetTestDataPath("pngsuite/ctzn0g04.png"), Codec::kPNG,
                        codec_context, &pool);

  // Extract gAMA
  const CodecInOut b1 =
      DecodeRoundtrip(GetTestDataPath("pngsuite/g10n3p04.png"), Codec::kPNG,
                      codec_context, &pool);
  EXPECT_EQ(TransferFunction::kLinear, b1.dec_c_original.transfer_function);

  // Extract cHRM
  const CodecInOut b_p =
      DecodeRoundtrip(GetTestDataPath("pngsuite/ccwn2c08.png"), Codec::kPNG,
                      codec_context, &pool);
  EXPECT_EQ(Primaries::kSRGB, b_p.dec_c_original.primaries);
  EXPECT_EQ(WhitePoint::kD65, b_p.dec_c_original.white_point);

  // Extract EXIF from (new-style) dedicated chunk
  const CodecInOut b_exif =
      DecodeRoundtrip(GetTestDataPath("pngsuite/exif2c08.png"), Codec::kPNG,
                      codec_context, &pool);
  EXPECT_EQ(978, b_exif.metadata.exif.size());
}

void VerifyWideGamutMetadata(const std::string& relative_pathname,
                             const Primaries primaries,
                             const CodecContext& codec_context,
                             ThreadPool* pool) {
  const CodecInOut io = DecodeRoundtrip(GetTestDataPath(relative_pathname),
                                        Codec::kPNG, codec_context, pool);

  EXPECT_EQ(8, io.original_bits_per_sample());

  EXPECT_FALSE(io.dec_c_original.icc.empty());
  EXPECT_EQ(RenderingIntent::kAbsolute, io.dec_c_original.rendering_intent);
  EXPECT_EQ(ColorSpace::kRGB, io.dec_c_original.color_space);
  EXPECT_EQ(WhitePoint::kD65, io.dec_c_original.white_point);
  EXPECT_EQ(primaries, io.dec_c_original.primaries);
}

TEST(CodecImplTest, TestWideGamut) {
  const CodecContext codec_context;
  ThreadPool pool(12);
  VerifyWideGamutMetadata("wide-gamut-tests/P3-sRGB-color-bars.png",
                          Primaries::kP3, codec_context, &pool);
  VerifyWideGamutMetadata("wide-gamut-tests/P3-sRGB-color-ring.png",
                          Primaries::kP3, codec_context, &pool);
  VerifyWideGamutMetadata("wide-gamut-tests/R2020-sRGB-color-bars.png",
                          Primaries::k2020, codec_context, &pool);
  VerifyWideGamutMetadata("wide-gamut-tests/R2020-sRGB-color-ring.png",
                          Primaries::k2020, codec_context, &pool);
}

}  // namespace
}  // namespace pik
