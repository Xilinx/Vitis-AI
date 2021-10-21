// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/adaptive_reconstruction.h"

#include "gtest/gtest.h"
#include "pik/common.h"
#include "pik/entropy_coder.h"
#include "pik/epf.h"
#include "pik/quant_weights.h"
#include "pik/single_image_handler.h"

namespace pik {
namespace {

const size_t xsize = 8;
const size_t ysize = 8;

void GenerateFlat(const float background, const float foreground,
                  std::vector<Image3F>* images) {
  for (int c = 0; c < Image3F::kNumPlanes; ++c) {
    Image3F in(xsize, ysize);
    // Plane c = foreground, all others = background.
    for (size_t y = 0; y < ysize; ++y) {
      float* rows[3] = {in.PlaneRow(0, y), in.PlaneRow(1, y),
                        in.PlaneRow(2, y)};
      for (size_t x = 0; x < xsize; ++x) {
        rows[0][x] = rows[1][x] = rows[2][x] = background;
        rows[c][x] = foreground;
      }
    }
    images->push_back(std::move(in));
  }
}

// Single foreground point at any position in any channel
void GeneratePoints(const float background, const float foreground,
                    std::vector<Image3F>* images) {
  for (int c = 0; c < Image3F::kNumPlanes; ++c) {
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        Image3F in(xsize, ysize);
        FillImage(background, &in);
        in.PlaneRow(c, y)[x] = foreground;
        images->push_back(std::move(in));
      }
    }
  }
}

void GenerateHorzEdges(const float background, const float foreground,
                       std::vector<Image3F>* images) {
  for (int c = 0; c < Image3F::kNumPlanes; ++c) {
    // Begin of foreground rows
    for (size_t y = 1; y < ysize; ++y) {
      Image3F in(xsize, ysize);
      FillImage(background, &in);
      for (size_t iy = y; iy < ysize; ++iy) {
        std::fill(in.PlaneRow(c, iy), in.PlaneRow(c, iy) + xsize, foreground);
      }
      images->push_back(std::move(in));
    }
  }
}

void GenerateVertEdges(const float background, const float foreground,
                       std::vector<Image3F>* images) {
  for (int c = 0; c < Image3F::kNumPlanes; ++c) {
    // Begin of foreground columns
    for (size_t x = 1; x < xsize; ++x) {
      Image3F in(xsize, ysize);
      FillImage(background, &in);
      for (size_t iy = 0; iy < ysize; ++iy) {
        float* PIK_RESTRICT row = in.PlaneRow(c, iy);
        for (size_t ix = x; ix < xsize; ++ix) {
          row[ix] = foreground;
        }
      }
      images->push_back(std::move(in));
    }
  }
}

// Ensures input remains unchanged by filter - verifies the edge-preserving
// nature of the filter because inputs are piecewise constant.
void EnsureUnchanged(const float background, const float foreground) {
  std::vector<Image3F> images;
  GenerateFlat(background, foreground, &images);
  GeneratePoints(background, foreground, &images);
  GenerateHorzEdges(background, foreground, &images);
  GenerateVertEdges(background, foreground, &images);

  DequantMatrices dequant(/*need_inv_matrices=*/false);
  Quantizer quantizer(&dequant, DivCeil(xsize, kBlockDim),
                      DivCeil(ysize, kBlockDim));
  (void)quantizer.SetQuant(0.2f);

  AcStrategyImage ac_strategy(xsize / kBlockDim, ysize / kBlockDim);

  ImageB dequant_cf(DivCeil(xsize, kTileDim), DivCeil(ysize, kTileDim));
  ZeroFillImage(&dequant_cf);
  uint8_t dequant_map[kMaxQuantControlFieldValue][256] = {};

  const EpfParams epf_params;

  for (size_t idx_image = 0; idx_image < images.size(); ++idx_image) {
    const Image3F& in = images[idx_image];
    Image3F ar_input = CopyImage(in);
    ImageB lut_ids(xsize / kBlockDim, ysize / kBlockDim);
    ZeroFillImage(&lut_ids);

    Image3F out = AdaptiveReconstruction(
        ar_input, in, quantizer, quantizer.RawQuantField(), dequant_cf,
        dequant_map, lut_ids, ac_strategy, epf_params, /*pool=*/nullptr);

    for (int c = 0; c < Image3F::kNumPlanes; ++c) {
      for (size_t y = 0; y < ysize; ++y) {
        const float* PIK_RESTRICT in_row = in.PlaneRow(c, y);
        const float* PIK_RESTRICT out_row = out.PlaneRow(c, y);

        for (size_t x = 0; x < xsize; ++x) {
          const bool near_zero = std::abs(in_row[x]) < 1E-6;
          EXPECT_NEAR(in_row[x], out_row[x], near_zero ? 1.5E-4 : 3E-4)
              << "img " << idx_image << " c " << c << " x " << x << " y " << y;
        }
      }
    }
  }
}

TEST(AdaptiveReconstructionTest, TestBright) { EnsureUnchanged(0.0f, 255.0f); }
TEST(AdaptiveReconstructionTest, TestDark) { EnsureUnchanged(255.0f, 0.0f); }

}  // namespace
}  // namespace pik
