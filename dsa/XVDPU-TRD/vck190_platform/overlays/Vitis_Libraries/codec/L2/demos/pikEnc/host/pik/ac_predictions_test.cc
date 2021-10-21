// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/ac_predictions.h"

#include "gtest/gtest.h"
#include "pik/codec.h"
#include "pik/common.h"
#include "pik/pik.h"
#include "pik/status.h"
#include "pik/testdata_path.h"

namespace pik {
namespace {

struct AcPredictionsTestParams {
  explicit AcPredictionsTestParams(const double butteraugli_distance,
                                   const bool fast_mode = false,
                                   const bool shrink8 = false)
      : butteraugli_distance(butteraugli_distance),
        fast_mode(fast_mode),
        shrink8(shrink8) {}
  double butteraugli_distance;
  bool fast_mode;
  bool shrink8;
};

std::ostream& operator<<(std::ostream& os, AcPredictionsTestParams params) {
  auto previous_flags = os.flags();
  os << std::boolalpha;
  os << "AcPredictionsTestParams{/*butteraugli_distance=*/"
     << params.butteraugli_distance << ", /*fast_mode=*/" << params.fast_mode
     << ", /*shrink8=*/" << params.shrink8 << "}";
  os.flags(previous_flags);
  return os;
}

class AcPredictionsTest
    : public testing::TestWithParam<AcPredictionsTestParams> {
 protected:
  // Returns compressed size [bytes].
  static void Roundtrip(CodecInOut* io, const CompressParams& cparams,
                        const DecompressParams& dparams, ThreadPool* pool) {
    PaddedBytes compressed;

    const size_t xsize_blocks = DivCeil(io->xsize(), kBlockDim);
    const size_t ysize_blocks = DivCeil(io->ysize(), kBlockDim);

    Image3F encoding_ac_prediction(xsize_blocks * kDCTBlockSize, ysize_blocks);
    PikInfo encoding_info;
    encoding_info.testing_aux.ac_prediction = &encoding_ac_prediction;

    EXPECT_TRUE(PixelsToPik(cparams, io, &compressed, &encoding_info, pool));

    Image3F decoding_ac_prediction;
    PikInfo decoding_info;
    // WARNING: the code that fills this is not thread-safe, and will only
    // work for a single group, which is OK for the current test image.
    PIK_CHECK(xsize_blocks <= kGroupDimInBlocks &&
              ysize_blocks <= kGroupDimInBlocks);
    decoding_info.testing_aux.ac_prediction = &decoding_ac_prediction;

    EXPECT_TRUE(PikToPixels(dparams, compressed, io, &decoding_info, pool));

    const float kErrorThreshold = 1e-6;

    EXPECT_LE(
        VerifyRelativeError(decoding_ac_prediction, encoding_ac_prediction,
                            kErrorThreshold, kErrorThreshold),
        kErrorThreshold);
  }
};

INSTANTIATE_TEST_SUITE_P(
    AcPredictionsTestInstantiation, AcPredictionsTest,
    testing::Values(AcPredictionsTestParams{/*butteraugli_distance=*/1.0,
                                            /*fast_mode=*/false,
                                            /*shrink8=*/true},
                    // No noise, no gradient
                    AcPredictionsTestParams{/*butteraugli_distance=*/1.0},
                    // Noise, no gradient
                    AcPredictionsTestParams{/*butteraugli_distance=*/1.5},
                    // Noise, gradient
                    AcPredictionsTestParams{/*butteraugli_distance=*/2.0},
                    // No noise, no gradient
                    AcPredictionsTestParams{/*butteraugli_distance=*/1.0,
                                            /*fast_mode=*/true},
                    // Noise, no gradient
                    AcPredictionsTestParams{/*butteraugli_distance=*/1.5,
                                            /*fast_mode=*/true},
                    // Noise, gradient
                    AcPredictionsTestParams{/*butteraugli_distance=*/2.0,
                                            /*fast_mode=*/true}));

TEST_P(AcPredictionsTest, Roundtrip) {
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecContext codec_context;
  CodecInOut io(&codec_context);
  ThreadPool pool(8);
  ASSERT_TRUE(io.SetFromFile(pathname, &pool));

  const AcPredictionsTestParams& params = GetParam();

  if (params.shrink8) {
    io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);
  }

  CompressParams cparams;
  cparams.butteraugli_distance = params.butteraugli_distance;
  cparams.fast_mode = params.fast_mode;
  DecompressParams dparams;

  Roundtrip(&io, cparams, dparams, &pool);
}

}  // namespace
}  // namespace pik
