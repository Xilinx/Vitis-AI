// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_COMPRESSED_IMAGE_FWD_H_
#define PIK_COMPRESSED_IMAGE_FWD_H_

#include "pik/ac_strategy.h"
#include "pik/common.h"
#include "pik/data_parallel.h"
#include "pik/gauss_blur.h"
#include "pik/image.h"

namespace pik {

struct GradientMap {
  Image3F gradient;  // corners of the gradient map tiles
  Image3B apply;     // gradient application mask.

  // Size of the DC image
  size_t xsize_dc;
  size_t ysize_dc;

  // Size of the gradient map (amount of corner points of tiles, one larger than
  // amount of tiles in x and y direction)
  size_t xsize;
  size_t ysize;

  bool grayscale;
};

// Contains global information that are computed once per pass.
struct FrameEncCache {
  // DCT coefficients for the full image
  Image3F coeffs;

  Image3F dc_dec;
  Image3S dc;

  // Enable new Lossless codec for DC. This flag exists only temporarily
  // as long as both old and new implementation co-exist, and eventually
  // only the new implementation should remain.
  bool use_new_dc = false;

  bool use_gradient;
  bool grayscale_opt = false;

  // Gradient map, if used.
  GradientMap gradient;

  DequantMatrices matrices{/*need_inv_table=*/true};

  // Control field for dequant matrix selection.
  ImageB dequant_control_field;

  // Map of dequant control field and adaptive quantization level to
  // dequantization table.
  uint8_t dequant_map[kMaxQuantControlFieldValue][256] = {};

  // AC strategy.
  AcStrategyImage ac_strategy;

  // Per-block indices into LUT for adaptive reconstruction's blur strength.
  ImageB ar_sigma_lut_ids;
};

// Working area for ComputeCoefficients
struct EncCache {
  bool initialized = false;

  bool grayscale_opt = false;

  size_t xsize_blocks;
  size_t ysize_blocks;

  // ComputePredictionResiduals
  Image3F dc_dec;

  // Working value, copied from coeffs_init.
  Image3F coeffs;

  // AC strategy.
  AcStrategyImage ac_strategy;

  // Every cell with saliency > threshold will be considered as 'salient'.
  float saliency_threshold;
  // Debug parameter: If true, drop non-salient AC part in progressive encoding.
  bool saliency_debug_skip_nonsalient;

  // Enable/disable predictions. Set in ComputeInitialCoefficients from the
  // pass header. Current usage is only in progressive mode.
  bool predict_lf;
  bool predict_hf;

  // Output values
  Image3S ac;          // 64 coefs per block, first (DC) is ignored.
  ImageI quant_field;  // Final values, to be encoded in stream.
};

// Information that is used at the pass level. All the images here should be
// accessed through a group rect (either with block units or pixel units).
struct FrameDecCache {
  // Enable new Lossless codec for DC. This flag exists only temporarily
  // as long as both old and new implementation co-exist, and eventually
  // only the new implementation should remain.
  bool use_new_dc = false;

  bool grayscale;

  // Full DC of the pass. Note that this will be split in *AC* group sized
  // chunks for AC predictions (DC group size != AC group size).
  Image3F dc;

  GradientMap gradient;

  // Raw quant field to be used for adaptive reconstruction.
  ImageI raw_quant_field;

  AcStrategyImage ac_strategy;

  DequantMatrices matrices{/*need_inv_table=*/false};

  // Control field for dequant matrix selection.
  ImageB dequant_control_field;

  // Map of dequant control field and adaptive quantization level to
  // dequantization table.
  uint8_t dequant_map[kMaxQuantControlFieldValue][256] = {};

  // Per-block indices into LUT for adaptive reconstruction's blur strength.
  ImageB ar_sigma_lut_ids;
};

// Temp images required for decoding a single group. Reduces memory allocations
// for large images because we only initialize min(#threads, #groups) instances.
struct GroupDecCache {
  // Separate from InitOnce because the caller only knows the DC group size.
  void InitDecodeDCGroup(size_t xsize_blocks, size_t ysize_blocks) {
    if (quantized_dc.xsize() == 0) {
      quantized_dc = Image3S(kDcGroupDimInBlocks, kDcGroupDimInBlocks);
      dc_y = ImageS(kDcGroupDimInBlocks, kDcGroupDimInBlocks);
      dc_xz_residuals = ImageS(kDcGroupDimInBlocks * 2, kDcGroupDimInBlocks);
      dc_xz_expanded = ImageS(kDcGroupDimInBlocks * 2, kDcGroupDimInBlocks);
    }

    quantized_dc.ShrinkTo(xsize_blocks, ysize_blocks);
    dc_y.ShrinkTo(xsize_blocks, ysize_blocks);
    dc_xz_residuals.ShrinkTo(xsize_blocks * 2, ysize_blocks);
    dc_xz_expanded.ShrinkTo(xsize_blocks * 2, ysize_blocks);
    ac_strategy_raw = ImageB(kDcGroupDimInBlocks, kDcGroupDimInBlocks);
  }

  void InitOnce(size_t xsize_blocks, size_t ysize_blocks) {
    if (num_nzeroes.xsize() == 0) {
      // Allocate enough for a whole tile - partial tiles on the right/bottom
      // border just use a subset. The valid size is passed via Rect.

      ac = Image3F(kGroupDimInBlocks * kDCTBlockSize, kGroupDimInBlocks);
      dc = Image3F(kGroupDimInBlocks + 2, kGroupDimInBlocks + 2);

      quantized_ac =
          Image3S(kTileDimInBlocks * kDCTBlockSize, kTileDimInBlocks);
      num_nzeroes = Image3I(kTileDimInBlocks, kTileDimInBlocks);

      const size_t xsize_tiles = DivCeil(kGroupDimInBlocks, kTileDimInBlocks);
      const size_t ysize_tiles = DivCeil(kGroupDimInBlocks, kTileDimInBlocks);
      tile_stage = ImageB(xsize_tiles + 1, ysize_tiles + 1);

      const size_t kWidth2x2 = (kGroupDimInBlocks + 2) * 2;
      const size_t kHeight2x2 = (kGroupDimInBlocks + 2) * 2;

      // TODO(user): do not allocate when !predict_hf
      pred2x2 = Image3F(kWidth2x2, kHeight2x2);
      // TODO(user): do not allocate when !predict_lf
      lf2x2 = Image3F(kWidth2x2, kHeight2x2);
      llf = Image3F(kGroupDimInBlocks + 2, kGroupDimInBlocks + 2);

      blur_x = ImageF(kGroupDimInBlocks * 8, kGroupDimInBlocks * 2 + 2);
    }

    // These images need to have correct sizes (used as loop bounds):

    // Ensure ShrinkTo is safe.
    PIK_ASSERT(xsize_blocks <= kGroupDimInBlocks);
    PIK_ASSERT(ysize_blocks <= kGroupDimInBlocks);

    dc.ShrinkTo(xsize_blocks + 2, ysize_blocks + 2);
    ac.ShrinkTo(xsize_blocks * kDCTBlockSize, ysize_blocks);

    const size_t xsize2x2 = (xsize_blocks + 2) * 2;
    const size_t ysize2x2 = (ysize_blocks + 2) * 2;

    pred2x2.ShrinkTo(xsize2x2, ysize2x2);
    llf.ShrinkTo(xsize_blocks + 2, ysize_blocks + 2);
    lf2x2.ShrinkTo(xsize2x2, ysize2x2);

    blur_x.ShrinkTo(xsize_blocks * 8, ysize_blocks * 2 + 2);
  }

  // Dequantized output produced by DecodeFromBitstream, DequantImage or
  // ExtractGroupDC.
  // TODO(veluca): replace the DC with a pointer + a rect to avoid copies.
  Image3F dc;
  Image3F ac;

  // Decode
  Image3S quantized_ac;
  // DequantAC
  Image3I num_nzeroes;

  // DecodeDCGroup
  Image3S quantized_dc;
  // TODO(janwas): remove these after use_new_dc
  ImageS dc_y;
  ImageS dc_xz_residuals;
  ImageS dc_xz_expanded;

  ImageB ac_strategy_raw;

  // ReconOpsinImage
  Image3F pred2x2;
  Image3F llf;
  Image3F lf2x2;
  ImageB tile_stage;

  // AddPredictions
  ImageF blur_x;
};

template <size_t N>
std::vector<float> DCfiedGaussianKernel(float sigma) {
  std::vector<float> result(3, 0.0);
  std::vector<float> hires = GaussianKernel<float>(N, sigma);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < hires.size(); j++) {
      result[(i + j) / N] += hires[j] / N;
    }
  }
  return result;
}

}  // namespace pik

#endif  // PIK_COMPRESSED_IMAGE_FWD_H_
