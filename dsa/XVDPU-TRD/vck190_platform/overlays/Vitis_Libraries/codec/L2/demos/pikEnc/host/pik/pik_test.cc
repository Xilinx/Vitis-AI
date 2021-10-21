// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/pik.h"

#include <string>

#include "gtest/gtest.h"
#include "pik/butteraugli_distance.h"
#include "pik/codec.h"
#include "pik/gaborish.h"
#include "pik/testdata_path.h"

namespace pik {
namespace {

// Returns compressed size [bytes].
size_t Roundtrip(CodecInOut* io, const CompressParams& cparams,
                 const DecompressParams& dparams, ThreadPool* pool,
                 CodecInOut* PIK_RESTRICT io2) {
  PikInfo* aux = nullptr;
  PaddedBytes compressed;

  // Remember prior dec_c_original, will be returned by decoder.
  const Primaries ext_pr = io->dec_c_original.primaries;
  const TransferFunction ext_tf = io->dec_c_original.transfer_function;
  // c_current should not change during encoding.
  const Primaries cur_pr = io->c_current().primaries;
  const TransferFunction cur_tf = io->c_current().transfer_function;

  EXPECT_TRUE(PixelsToPik(cparams, io, &compressed, aux, pool));

  // Should still be in the same color space after encoding.
  EXPECT_EQ(cur_pr, io->c_current().primaries);
  EXPECT_EQ(cur_tf, io->c_current().transfer_function);

  EXPECT_TRUE(PikToPixels(dparams, compressed, io2, aux, pool));

  if (!cparams.lossless_mode) {
    // Non-lossless PIK returns linear sRGB.
    EXPECT_EQ(Primaries::kSRGB, io2->c_current().primaries);
    EXPECT_TRUE(IsLinear(io2->c_current().transfer_function));
  } else {
    // Lossless PIK returns in the original color space.
    EXPECT_EQ(io->c_current().primaries, io2->c_current().primaries);
    EXPECT_EQ(io->c_current().transfer_function,
              io2->c_current().transfer_function);
  }

  // Decoder returns the original dec_c_original passed to the encoder.
  EXPECT_EQ(ext_pr, io2->dec_c_original.primaries);
  EXPECT_EQ(ext_tf, io2->dec_c_original.transfer_function);

  return compressed.size();
}

TEST(PikTest, RoundtripSinglePixel) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;
  Image3F image(1, 1);
  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  DecompressParams dparams;

  image.PlaneRow(0, 0)[0] = 0.0f;
  image.PlaneRow(1, 0)[0] = 0.0f;
  image.PlaneRow(2, 0)[0] = 0.0f;

  CodecInOut io(&codec_context);
  io.SetFromImage(std::move(image), codec_context.c_srgb[0]);
  io.dec_c_original = codec_context.c_srgb[0];
  io.SetOriginalBitsPerSample(32);
  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, pool, &io2);
}

TEST(PikTest, RoundtripTinyFast) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, pool));
  io.ShrinkTo(16, 16);

  CompressParams cparams;
  cparams.fast_mode = true;
  DecompressParams dparams;

  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, pool, &io2);
}

TEST(PikTest, RoundtripSmallD1) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, pool));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  DecompressParams dparams;

  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, pool, &io2);
  EXPECT_LE(ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            1.5);
}

TEST(PikTest, RoundtripSmallProgressive) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, pool));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  cparams.progressive_mode = true;
  DecompressParams dparams;

  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, pool, &io2);
  EXPECT_LE(ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            1.5);
}

TEST(PikTest, RoundtripUnalignedD2) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, pool));
  io.ShrinkTo(io.xsize() / 12, io.ysize() / 7);

  CompressParams cparams;
  cparams.butteraugli_distance = 2.0;
  DecompressParams dparams;

  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, pool, &io2);
  EXPECT_LE(ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            3.2);
}

TEST(PikTest, RoundtripUnalignedProgressive) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, pool));
  io.ShrinkTo(io.xsize() / 12, io.ysize() / 7);

  CompressParams cparams;
  cparams.butteraugli_distance = 2.0;
  cparams.progressive_mode = true;
  DecompressParams dparams;

  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, pool, &io2);
  EXPECT_LE(ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            3.2);
}

TEST(PikTest, RoundtripMultiGroup) {
  ThreadPool pool(4);
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, &pool));
  io.ShrinkTo(600, 1024);  // partial X, full Y group

  CompressParams cparams;
  DecompressParams dparams;

  cparams.butteraugli_distance = 1.0f;
  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, &pool, &io2);
  EXPECT_LE(ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, &pool),
            1.8f);

  cparams.butteraugli_distance = 2.0f;
  CodecInOut io3(&codec_context);
  Roundtrip(&io, cparams, dparams, &pool, &io3);
  EXPECT_LE(ButteraugliDistance(&io, &io3, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, &pool),
            3.0f);
}

TEST(PikTest, RoundtripMultiGroupProgressive) {
  ThreadPool pool(4);
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, &pool));
  io.ShrinkTo(600, 1024);  // partial X, full Y group

  CompressParams cparams;
  DecompressParams dparams;

  cparams.butteraugli_distance = 1.0f;
  cparams.progressive_mode = true;
  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, &pool, &io2);
  EXPECT_LE(ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, &pool),
            1.8f);

  cparams.butteraugli_distance = 2.0f;
  CodecInOut io3(&codec_context);
  Roundtrip(&io, cparams, dparams, &pool, &io3);
  EXPECT_LE(ButteraugliDistance(&io, &io3, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, &pool),
            3.0f);
}

TEST(PikTest, RoundtripLargeFast) {
  ThreadPool pool(8);
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, &pool));

  CompressParams cparams;
  cparams.fast_mode = true;
  DecompressParams dparams;

  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, &pool, &io2);
}

TEST(PikTest, RoundtripLargeFastProgressive) {
  ThreadPool pool(8);
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, &pool));

  CompressParams cparams;
  cparams.fast_mode = true;
  cparams.progressive_mode = true;
  DecompressParams dparams;

  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, &pool, &io2);
}

// Checks for differing size/distance in two consecutive runs of distance 2,
// which involves additional processing including adaptive reconstruction.
// Failing this may be a sign of race conditions or invalid memory accesses.
TEST(PikTest, RoundtripD2Consistent) {
  ThreadPool pool(8);
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, &pool));

  CompressParams cparams;
  cparams.fast_mode = true;
  cparams.butteraugli_distance = 2.0;
  DecompressParams dparams;

  // Try each xsize mod kBlockDim to verify right border handling.
  for (size_t xsize = 48; xsize > 40; --xsize) {
    io.ShrinkTo(xsize, 15);

    CodecInOut io2(&codec_context);
    const size_t size2 = Roundtrip(&io, cparams, dparams, &pool, &io2);

    CodecInOut io3(&codec_context);
    const size_t size3 = Roundtrip(&io, cparams, dparams, &pool, &io3);

    // Exact same compressed size.
    EXPECT_EQ(size2, size3);

    // Exact same distance.
    const float dist2 = ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                            /*distmap=*/nullptr, &pool);
    const float dist3 = ButteraugliDistance(&io, &io3, cparams.hf_asymmetry,
                                            /*distmap=*/nullptr, &pool);
    EXPECT_EQ(dist2, dist3);
  }
}

TEST(PikTest, RoundtripProgressiveConsistent) {
  ThreadPool pool(8);
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, &pool));

  CompressParams cparams;
  cparams.fast_mode = true;
  cparams.progressive_mode = true;
  cparams.butteraugli_distance = 2.0;
  DecompressParams dparams;

  // Try each xsize mod kBlockDim to verify right border handling.
  for (size_t xsize = 48; xsize > 40; --xsize) {
    io.ShrinkTo(xsize, 15);

    CodecInOut io2(&codec_context);
    const size_t size2 = Roundtrip(&io, cparams, dparams, &pool, &io2);

    CodecInOut io3(&codec_context);
    const size_t size3 = Roundtrip(&io, cparams, dparams, &pool, &io3);

    // Exact same compressed size.
    EXPECT_EQ(size2, size3);

    // Exact same distance.
    const float dist2 = ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                            /*distmap=*/nullptr, &pool);
    const float dist3 = ButteraugliDistance(&io, &io3, cparams.hf_asymmetry,
                                            /*distmap=*/nullptr, &pool);
    EXPECT_EQ(dist2, dist3);
  }
}

TEST(PikTest, ProgressiveDecoding) {
  ThreadPool pool(8);
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut input(&codec_context);
  ASSERT_TRUE(input.SetFromFile(pathname, &pool));

  PaddedBytes compressed;
  PikInfo aux;

  CompressParams cparams;
  cparams.fast_mode = true;
  cparams.progressive_mode = true;
  cparams.butteraugli_distance = 2.0;
  ASSERT_TRUE(PixelsToPik(cparams, &input, &compressed, &aux, &pool));

  // The default progressive encoding scheme should make all these downsampling
  // factors achievable.
  for (const size_t downsampling : {1, 2, 4, 8}) {
    DecompressParams dparams;
    dparams.max_downsampling = downsampling;
    CodecInOut output(&codec_context);
    ASSERT_TRUE(PikToPixels(dparams, compressed, &output, &aux, &pool));
    EXPECT_EQ(aux.downsampling, downsampling);
    EXPECT_EQ(output.xsize(), DivCeil(input.xsize(), aux.downsampling))
        << "downsampling = " << downsampling;
    EXPECT_EQ(output.ysize(), DivCeil(input.ysize(), aux.downsampling))
        << "downsampling = " << downsampling;
  }
}

TEST(PikTest, NonProgressiveDCImage) {
  ThreadPool pool(8);
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut input(&codec_context);
  ASSERT_TRUE(input.SetFromFile(pathname, &pool));

  PaddedBytes compressed;
  PikInfo aux;

  CompressParams cparams;
  cparams.fast_mode = true;
  cparams.progressive_mode = false;
  cparams.butteraugli_distance = 2.0;
  ASSERT_TRUE(PixelsToPik(cparams, &input, &compressed, &aux, &pool));

  // Even in non-progressive mode, it should be possible to return a DC-only
  // image.
  DecompressParams dparams;
  dparams.max_downsampling = 100;
  CodecInOut output(&codec_context);
  ASSERT_TRUE(PikToPixels(dparams, compressed, &output, &aux, &pool));
  constexpr decltype(output.xsize()) expected_downscale = 8;
  EXPECT_EQ(aux.downsampling, expected_downscale);
  EXPECT_EQ(output.xsize(), DivCeil(input.xsize(), expected_downscale));
  EXPECT_EQ(output.ysize(), DivCeil(input.ysize(), expected_downscale));
}

TEST(PikTest, RoundtripSmallNoGaborish) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, pool));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  CompressParams cparams;
  cparams.gaborish = int(GaborishStrength::kOff);
  cparams.butteraugli_distance = 1.0;
  DecompressParams dparams;

  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, pool, &io2);
  EXPECT_LE(ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            1.5);
}

TEST(PikTest, RoundtripSmallNoGaborishProgressive) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, pool));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  CompressParams cparams;
  cparams.gaborish = int(GaborishStrength::kOff);
  cparams.butteraugli_distance = 1.0;
  cparams.progressive_mode = true;
  DecompressParams dparams;

  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, pool, &io2);
  EXPECT_LE(ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            1.5);
}

TEST(PikTest, RoundtripGrayscale) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/cvo9xd_keong_macan_grayscale.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, pool));
  ASSERT_NE(io.xsize(), 0);
  io.ShrinkTo(128, 128);
  EXPECT_TRUE(io.IsGray());
  EXPECT_EQ(8, io.original_bits_per_sample());
  EXPECT_EQ(TransferFunction::kSRGB, io.dec_c_original.transfer_function);
  PikInfo* aux = nullptr;

  {
    CompressParams cparams;
    cparams.butteraugli_distance = 1.0;
    DecompressParams dparams;

    PaddedBytes compressed;
    EXPECT_TRUE(PixelsToPik(cparams, &io, &compressed, aux, pool));
    CodecInOut io2(&codec_context);
    EXPECT_TRUE(PikToPixels(dparams, compressed, &io2, aux, pool));
    EXPECT_TRUE(io2.IsGray());

    EXPECT_LE(ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                  /*distmap=*/nullptr, pool),
              1.5);
  }

  // Test with larger butteraugli distance and other settings enabled so
  // different pik codepaths trigger.
  {
    CompressParams cparams;
    cparams.butteraugli_distance = 8.0;
    cparams.gradient = Override::kOn;
    DecompressParams dparams;

    PaddedBytes compressed;
    EXPECT_TRUE(PixelsToPik(cparams, &io, &compressed, aux, pool));
    CodecInOut io2(&codec_context);
    EXPECT_TRUE(PikToPixels(dparams, compressed, &io2, aux, pool));
    EXPECT_TRUE(io2.IsGray());

    EXPECT_LE(ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                  /*distmap=*/nullptr, pool),
              9.0);
  }
}

TEST(PikTest, RoundtripAlpha) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/tmshre_riaphotographs_alpha.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, pool));

  ASSERT_NE(io.xsize(), 0);
  ASSERT_TRUE(io.HasAlpha());
  io.ShrinkTo(128, 128);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  DecompressParams dparams;

  EXPECT_EQ(8, io.original_bits_per_sample());
  EXPECT_EQ(TransferFunction::kSRGB, io.dec_c_original.transfer_function);
  PikInfo* aux = nullptr;
  PaddedBytes compressed;
  EXPECT_TRUE(PixelsToPik(cparams, &io, &compressed, aux, pool));
  CodecInOut io2(&codec_context);
  EXPECT_TRUE(PikToPixels(dparams, compressed, &io2, aux, pool));

  // TODO(robryk): Fix the following line in presence of different alpha_bits in
  // the two contexts.
  // EXPECT_TRUE(SamePixels(io.alpha(), io2.alpha()));
  // TODO(robryk): Fix the distance estimate used in the encoder.
  EXPECT_LE(ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            6.3);
}

TEST(PikTest, RoundtripAlphaNonMultipleOf8) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/tmshre_riaphotographs_alpha.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, pool));

  ASSERT_NE(io.xsize(), 0);
  ASSERT_TRUE(io.HasAlpha());
  io.ShrinkTo(12, 12);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  DecompressParams dparams;

  EXPECT_EQ(8, io.original_bits_per_sample());
  EXPECT_EQ(TransferFunction::kSRGB, io.dec_c_original.transfer_function);
  PikInfo* aux = nullptr;
  PaddedBytes compressed;
  EXPECT_TRUE(PixelsToPik(cparams, &io, &compressed, aux, pool));
  CodecInOut io2(&codec_context);
  EXPECT_TRUE(PikToPixels(dparams, compressed, &io2, aux, pool));

  // TODO(robryk): Fix the following line in presence of different alpha_bits in
  // the two contexts.
  // EXPECT_TRUE(SamePixels(io.alpha(), io2.alpha()));
  // TODO(robryk): Fix the distance estimate used in the encoder.
  EXPECT_LE(ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            6.3);
}

TEST(PikTest, RoundtripAlpha16) {
  ThreadPool pool(4);
  CodecContext codec_context;

  size_t xsize = 1200, ysize = 160;
  std::vector<uint16_t> pixels(xsize * ysize * 4);
  // Generate 16-bit pattern that uses various colors and alpha values.
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      size_t i = y * xsize + x;
      pixels[i * 4 + 0] = y * 65535 / ysize;
      pixels[i * 4 + 1] = x * 65535 / xsize;
      pixels[i * 4 + 2] = (y + x) * 65535 / (xsize + ysize);
      pixels[i * 4 + 3] = 65535 * y / ysize;
    }
  }
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromSRGB(xsize, ysize, /*is_gray=*/false,
                             /*has_alpha=*/true, pixels.data(),
                             pixels.data() + pixels.size(), &pool));

  // The image is wider than 512 pixels to ensure multiple groups are tested.

  ASSERT_NE(io.xsize(), 0);
  ASSERT_TRUE(io.HasAlpha());

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  // Prevent the test to be too slow, does not affect alpha
  cparams.fast_mode = true;
  DecompressParams dparams;

  EXPECT_EQ(16, io.original_bits_per_sample());
  EXPECT_EQ(TransferFunction::kSRGB, io.dec_c_original.transfer_function);
  PikInfo* aux = nullptr;
  PaddedBytes compressed;
  EXPECT_TRUE(PixelsToPik(cparams, &io, &compressed, aux, &pool));
  CodecInOut io2(&codec_context);
  EXPECT_TRUE(PikToPixels(dparams, compressed, &io2, aux, &pool));

  EXPECT_TRUE(SamePixels(io.alpha(), io2.alpha()));
}

TEST(PikTest, RoundtripLossless8) {
  ThreadPool pool(8);
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, &pool));

  CompressParams cparams;
  cparams.lossless_mode = true;
  DecompressParams dparams;

  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, &pool, &io2);
  // If this test fails with a very close to 0.0 but not exactly 0.0 butteraugli
  // distance, then there is likely a floating point issue, that could be
  // happening either in io or io2. The values of io are generated by
  // external_image.cc, and those in io2 by the pik decoder. If they use
  // slightly different floating point operations (say, one casts int to float
  // while other divides the int through 255.0f and later multiplies it by
  // 255 again) they will get slightly different values. To fix, ensure both
  // sides do the following formula for converting integer range 0-255 to
  // floating point range 0.0f-255.0f: static_cast<float>(i)
  // without any further intermediate operations.
  // Note that this precision issue is not a problem in practice if the values
  // are equal when rounded to 8-bit int, but currently full exact precision is
  // tested.
  EXPECT_EQ(0.0, ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                     /*distmap=*/nullptr, &pool));
}

TEST(PikTest, RoundtripLossless8Alpha) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/tmshre_riaphotographs_alpha.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, pool));
  EXPECT_EQ(8, io.AlphaBits());
  EXPECT_EQ(8, io.original_bits_per_sample());

  CompressParams cparams;
  cparams.lossless_mode = true;
  DecompressParams dparams;

  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, pool, &io2);
  // If fails, see note about floating point in RoundtripLossless8.
  EXPECT_EQ(0.0, ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                     /*distmap=*/nullptr, pool));
  EXPECT_TRUE(SamePixels(io.alpha(), io2.alpha()));
  EXPECT_EQ(8, io2.AlphaBits());
  EXPECT_EQ(8, io2.original_bits_per_sample());
}

TEST(PikTest, RoundtripLossless16Alpha) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;

  size_t xsize = 1200, ysize = 160;
  std::vector<uint16_t> pixels(xsize * ysize * 4);
  // Generate 16-bit pattern that uses various colors and alpha values.
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      size_t i = y * xsize + x;
      pixels[i * 4 + 0] = y * 65535 / ysize;
      pixels[i * 4 + 1] = x * 65535 / xsize;
      pixels[i * 4 + 2] = (y + x) * 65535 / (xsize + ysize);
      pixels[i * 4 + 3] = 65535 * y / ysize;
    }
  }
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromSRGB(xsize, ysize, /*is_gray=*/false,
                             /*has_alpha=*/true, pixels.data(),
                             pixels.data() + pixels.size(), pool));

  EXPECT_EQ(16, io.AlphaBits());
  EXPECT_EQ(16, io.original_bits_per_sample());

  CompressParams cparams;
  cparams.lossless_mode = true;
  DecompressParams dparams;

  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, pool, &io2);
  // If this test fails with a very close to 0.0 but not exactly 0.0 butteraugli
  // distance, then there is likely a floating point issue, that could be
  // happening either in io or io2. The values of io are generated by
  // external_image.cc, and those in io2 by the pik decoder. If they use
  // slightly different floating point operations (say, one does "i / 257.0f"
  // while the other does "i * (1.0f / 257)" they will get slightly different
  // values. To fix, ensure both sides do the following formula for converting
  // integer range 0-65535 to Image3F floating point range 0.0f-255.0f:
  // "i * (1.0f / 257)".
  // Note that this precision issue is not a problem in practice if the values
  // are equal when rounded to 16-bit int, but currently full exact precision is
  // tested.
  EXPECT_EQ(0.0, ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                     /*distmap=*/nullptr, pool));
  EXPECT_TRUE(SamePixels(io.alpha(), io2.alpha()));
  EXPECT_EQ(16, io2.AlphaBits());
  EXPECT_EQ(16, io2.original_bits_per_sample());
}

TEST(PikTest, RoundtripLossless16AlphaNotMisdetectedAs8Bit) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;

  size_t xsize = 128, ysize = 128;
  std::vector<uint16_t> pixels(xsize * ysize * 4);
  // All 16-bit values, both color and alpha, of this image are below 64.
  // This allows testing if a code path wrongly concludes it's an 8-bit instead
  // of 16-bit image (or even 6-bit).
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      size_t i = y * xsize + x;
      pixels[i * 4 + 0] = y * 64 / ysize;
      pixels[i * 4 + 1] = x * 64 / xsize;
      pixels[i * 4 + 2] = (y + x) * 64 / (xsize + ysize);
      pixels[i * 4 + 3] = 64 * y / ysize;
    }
  }
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromSRGB(xsize, ysize, /*is_gray=*/false,
                             /*has_alpha=*/true, pixels.data(),
                             pixels.data() + pixels.size(), pool));

  EXPECT_EQ(16, io.AlphaBits());
  EXPECT_EQ(16, io.original_bits_per_sample());

  CompressParams cparams;
  cparams.lossless_mode = true;
  DecompressParams dparams;

  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, pool, &io2);
  EXPECT_EQ(16, io2.AlphaBits());
  EXPECT_EQ(16, io2.original_bits_per_sample());
  // If fails, see note about floating point in RoundtripLossless8.
  EXPECT_EQ(0.0, ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                     /*distmap=*/nullptr, pool));
  EXPECT_TRUE(SamePixels(io.alpha(), io2.alpha()));
}

TEST(PikTest, RoundtripLossless8Gray) {
  ThreadPool* pool = nullptr;
  CodecContext codec_context;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/cvo9xd_keong_macan_grayscale.png");
  CodecInOut io(&codec_context);
  ASSERT_TRUE(io.SetFromFile(pathname, pool));

  CompressParams cparams;
  cparams.lossless_mode = true;
  DecompressParams dparams;

  EXPECT_TRUE(io.IsGray());
  EXPECT_EQ(8, io.original_bits_per_sample());
  CodecInOut io2(&codec_context);
  Roundtrip(&io, cparams, dparams, pool, &io2);
  // If fails, see note about floating point in RoundtripLossless8.
  EXPECT_EQ(0.0, ButteraugliDistance(&io, &io2, cparams.hf_asymmetry,
                                     /*distmap=*/nullptr, pool));
  EXPECT_TRUE(io2.IsGray());
  EXPECT_EQ(8, io2.original_bits_per_sample());
}
}  // namespace
}  // namespace pik
