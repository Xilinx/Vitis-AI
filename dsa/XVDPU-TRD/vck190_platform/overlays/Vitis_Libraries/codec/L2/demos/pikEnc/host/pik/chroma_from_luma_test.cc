// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/chroma_from_luma.h"
#include "gtest/gtest.h"

#include "pik/block.h"
#include "pik/butteraugli_distance.h"
#include "pik/codec.h"
#include "pik/common.h"
#include "pik/dct_util.h"
#include "pik/descriptive_statistics.h"
#include "pik/image.h"
#include "pik/opsin_image.h"
#include "pik/opsin_inverse.h"
#include "pik/quantizer.h"
#include "pik/testdata_path.h"

namespace pik {
namespace {

void RoundtripDC(const Image3F& in, Image3F* PIK_RESTRICT residuals,
                 Image3F* PIK_RESTRICT restored) {
  const Rect rect(in);
  DequantMatrices dequant(/*need_inv_matrices=*/true);
  Quantizer quantizer(&dequant, in.xsize(), in.ysize());
  quantizer.SetQuant(5.0f);

  CFL_Stats stats;
  *residuals = Image3F(in.xsize(), in.ysize());
  *restored = Image3F(in.xsize(), in.ysize());
  DecorrelateDC(in.Plane(1), in, rect, quantizer, 0, residuals, restored,
                &stats);

  Image3F restored2(restored->xsize(), restored->ysize());
  RestoreDC(in.Plane(1), *residuals, rect, &restored2, &stats);

  // Restored images also need a Y channel for comparisons below.
  for (size_t by = 0; by < in.ysize(); ++by) {
    memcpy(restored->PlaneRow(1, by), in.ConstPlaneRow(1, by), in.xsize() * 4);
    memcpy(restored2.PlaneRow(1, by), in.ConstPlaneRow(1, by), in.xsize() * 4);
  }

  VerifyRelativeError(*restored, restored2, 1E-6, 1E-6);
}

TEST(ColorCorrelationTest, RoundtripFlatDC) {
  const size_t xsize_blocks = 5;
  const size_t ysize_blocks = 3;
  Image3F in(xsize_blocks, ysize_blocks);

  // Different values in each channel
  GenerateImage(
      [](size_t x, size_t y, int c) {
        return 2 * c + 0.5;  // 0.5, 2.5, 4.5
      },
      &in);

  Image3F residuals, restored;
  RoundtripDC(in, &residuals, &restored);

  // Residuals are zero
  for (size_t c = 0; c < 3; c += 2) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float* PIK_RESTRICT row_res = residuals.ConstPlaneRow(c, by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        if (bx == 0 && by == 0) continue;
        if (std::abs(row_res[bx]) > 1E-6f) {
          PIK_ABORT("c=%zu %zu %zu: %f\n", c, bx, by, row_res[bx]);
        }
      }
    }
  }

  // Near-exact reconstruction
  VerifyRelativeError(in, restored, 1E-5, 1E-4);
}

TEST(ColorCorrelationTest, RoundtripVertGradientDC) {
  const size_t xsize_blocks = 5;
  const size_t ysize_blocks = 3;
  Image3F in(xsize_blocks, ysize_blocks);

  // Different values in each channel
  GenerateImage([](size_t x, size_t y, int c) { return 0.5 * y + 1; }, &in);

  Image3F residuals, restored;
  RoundtripDC(in, &residuals, &restored);

  // Residuals are nearly zero
  double sum_abs_res = 0.0;
  for (size_t c = 0; c < 3; c += 2) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float* PIK_RESTRICT row_res = residuals.ConstPlaneRow(c, by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        if (bx == 0 && by == 0) continue;
        sum_abs_res += std::abs(row_res[bx]);
      }
    }
  }
  PIK_CHECK(sum_abs_res < 3E-3);

  // Near-exact reconstruction
  VerifyRelativeError(in, restored, 5E-4, 5E-4);
}

TEST(ColorCorrelationTest, RoundtripRandomDC) {
  const size_t xsize_blocks = 9;
  const size_t ysize_blocks = 7;
  Image3F in(xsize_blocks, ysize_blocks);

  RandomFillImage(&in, 255.0f);

  Image3F residuals, restored;
  RoundtripDC(in, &residuals, &restored);

  // Nonzero residuals
  double sum_residuals = 0.0;
  for (size_t c = 0; c < 3; c += 2) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float* PIK_RESTRICT row_res = residuals.ConstPlaneRow(c, by);
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        if (bx == 0 && by == 0) continue;
        sum_residuals += std::abs(row_res[bx]);
      }
    }
  }
  PIK_ASSERT(sum_residuals > xsize_blocks * ysize_blocks * 255 / 2);

  // Reasonable reconstruction
  VerifyRelativeError(in, restored, 5E-4, 6E-5);
}

void QuantizeBlock(const size_t c, const Quantizer& quantizer,
                   const int32_t quant_ac, const float* PIK_RESTRICT from,
                   const size_t from_stride, float* PIK_RESTRICT to,
                   const size_t to_stride) {
  const AcStrategy acs(AcStrategy::Type::DCT, 0);
  PIK_ASSERT(acs.IsFirstBlock());
  quantizer.QuantizeRoundtripBlockAC(
      c, 0, quant_ac, acs.GetQuantKind(), acs.covered_blocks_x(),
      acs.covered_blocks_y(), from, from_stride, to, to_stride);

  // Always use DCT8 quantization kind for DC
  const float mul = quantizer.DequantMatrix(0, kQuantKindDCT8, c)[0] *
                    quantizer.inv_quant_dc();
  to[0] = quantizer.QuantizeDC(c, from[0]) * mul;
}

void QuantizePlaneRow(const Image3F& from, const size_t c, const size_t by,
                      const Quantizer& quantizer, Image3F* PIK_RESTRICT to) {
  PIK_ASSERT(SameSize(from, *to));
  const size_t xsize = from.xsize();
  PIK_ASSERT(xsize % kDCTBlockSize == 0);
  const float* row_from = from.ConstPlaneRow(c, by);
  float* row_to = to->PlaneRow(c, by);

  const int32_t* PIK_RESTRICT row_quant = quantizer.RawQuantField().Row(by);
  for (size_t bx = 0; bx < xsize / kDCTBlockSize; ++bx) {
    QuantizeBlock(c, quantizer, row_quant[bx], row_from + bx * kDCTBlockSize,
                  from.PixelsPerRow(), row_to + bx * kDCTBlockSize,
                  to->PixelsPerRow());
  }
}

TEST(ColorCorrelationTest, RoundtripQuantized) {
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecContext codec_context;
  CodecInOut io_in(&codec_context);
  PIK_CHECK(io_in.SetFromFile(pathname, /*pool=*/nullptr));
  io_in.ShrinkTo(io_in.xsize() & ~(kBlockDim - 1),
                 io_in.ysize() & ~(kBlockDim - 1));
  const size_t xsize_blocks = io_in.xsize() / kBlockDim;
  const size_t ysize_blocks = io_in.ysize() / kBlockDim;
  const Rect rect(0, 0, xsize_blocks, ysize_blocks);

  Image3F opsin = OpsinDynamicsImage(&io_in, Rect(io_in.color()));

  DequantMatrices dequant(/*need_inv_matrices=*/true);
  Quantizer quantizer(&dequant, xsize_blocks, ysize_blocks);
  quantizer.SetQuant(5.0f);

  Image3F dct(xsize_blocks * kDCTBlockSize, ysize_blocks);
  TransposedScaledDCT(opsin, &dct);

  const size_t cY = 1;

  // Input Y must be pre-quantized.
  Image3F quantized(dct.xsize(), dct.ysize());
  for (size_t c = 0; c < 3; ++c) {
    for (size_t by = 0; by < ysize_blocks; ++by) {
      QuantizePlaneRow(dct, c, by, quantizer, &quantized);
    }
  }
  VerifyRelativeError(dct.Plane(cY), quantized.Plane(cY), 2.2E-2, 1E-3);

  // ------------------ DC

  const Image3F dc = DCImage(dct);
  const Image3F qdc = DCImage(quantized);
  Image3F residuals_dc(xsize_blocks, ysize_blocks);
  Image3F restored_dc(xsize_blocks, ysize_blocks);
  CFL_Stats stats_dc;
  DecorrelateDC(qdc.Plane(cY), dc, rect, quantizer, 0, &residuals_dc,
                &restored_dc, &stats_dc);
  printf("DC:\n");
  stats_dc.Print();

  Image3F restored2_dc(xsize_blocks, ysize_blocks);
  CFL_Stats stats2_dc;
  RestoreDC(qdc.Plane(cY), residuals_dc, rect, &restored2_dc, &stats2_dc);

  // Restored images also need a Y channel for comparisons below.
  for (size_t by = 0; by < ysize_blocks; ++by) {
    memcpy(restored_dc.PlaneRow(cY, by), qdc.ConstPlaneRow(cY, by),
           qdc.xsize() * sizeof(float));
    memcpy(restored2_dc.PlaneRow(cY, by), qdc.ConstPlaneRow(cY, by),
           qdc.xsize() * sizeof(float));
  }
  VerifyRelativeError(restored_dc, restored2_dc, 3E-4, 1E-5);
  VerifyRelativeError(dc, restored_dc, 2E-2, 1E-3);

  // ------------------ AC

  Image3F residuals(dct.xsize(), dct.ysize());
  Image3F restored(dct.xsize(), dct.ysize());
  CFL_Stats stats;
  DecorrelateAC(quantized.Plane(cY), dct, rect, quantizer, 0, &residuals,
                &restored, &stats);
  printf("AC:\n");
  stats.Print();

  FillDC(restored_dc, &restored);

  Image3F restored2(dct.xsize(), dct.ysize());
  CFL_Stats stats2;
  RestoreAC(quantized.Plane(cY), residuals, rect, &restored2, &stats2);
  FillDC(restored_dc, &restored2);

  // Restored images also need a Y channel for comparisons below.
  for (size_t by = 0; by < ysize_blocks; ++by) {
    memcpy(restored.PlaneRow(cY, by), quantized.ConstPlaneRow(cY, by),
           dct.xsize() * sizeof(float));
    memcpy(restored2.PlaneRow(cY, by), quantized.ConstPlaneRow(cY, by),
           dct.xsize() * sizeof(float));
  }
  VerifyRelativeError(restored, restored2, 3E-4, 1E-5);
  VerifyRelativeError(dct, restored, 2E-2, 1E-3);

  Image3F idct_restored(xsize_blocks * kBlockDim, ysize_blocks * kBlockDim);
  TransposedScaledIDCT(restored, &idct_restored);

  OpsinToLinear(&idct_restored, /*pool=*/nullptr);

  CodecInOut io_restored(&codec_context);
  io_restored.SetFromImage(std::move(idct_restored),
                           codec_context.c_linear_srgb[0]);
  (void)io_restored.EncodeToFile(codec_context.c_srgb[0], 8,
                                 "/tmp/dct_idct.png");

  const float dist_restored =
      ButteraugliDistance(&io_in, &io_restored, 1.0f,
                          /*distmap=*/nullptr, /*pool=*/nullptr);
  printf("dist restored %.2f\n", dist_restored);
}

}  // namespace
}  // namespace pik
