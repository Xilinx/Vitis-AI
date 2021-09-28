// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/ac_predictions.h"
#include <cstdint>
#include "pik/ac_strategy.h"
#include "pik/codec.h"
#include "pik/color_correlation.h"
#include "pik/compressed_image_fwd.h"
#include "pik/data_parallel.h"
#include "pik/opsin_inverse.h"
#include "pik/quant_weights.h"

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/bits.h"
#include "pik/block.h"
#include "pik/common.h"
#include "pik/convolve.h"
#include "pik/dct.h"
#include "pik/dct_util.h"
#include "pik/deconvolve.h"
#include "pik/entropy_coder.h"
#include "pik/image.h"
#include "pik/profiler.h"
#include "pik/quantizer.h"
#include "pik/resample.h"
#include "pik/upscaler.h"

namespace pik {
namespace {
// Adds or subtracts block to/from "add_to", except low and lowest frequencies.
// `block` is assumed to be contiguous, `add_to` has `ysize` slices of
// `xsize`*kBlockDim*kBlockDim coefficients with a stride of `stride`.
template <bool add>
SIMD_ATTR void AddBlockExceptLFAndLLFTo(const float* PIK_RESTRICT block,
                                        size_t xsize, size_t ysize,
                                        float* PIK_RESTRICT add_to,
                                        size_t stride) {
  // TODO(veluca): SIMD-fy
  PIK_ASSERT(ysize <= 4);
  // Rows with LF and LLF coefficients.
  for (size_t y = 0; y < 2 * ysize; y++) {
    for (size_t x = 2 * xsize; x < xsize * kBlockDim; x++) {
      if (add) {
        add_to[y * xsize * kBlockDim + x] += block[y * xsize * kBlockDim + x];
      } else {
        add_to[y * xsize * kBlockDim + x] -= block[y * xsize * kBlockDim + x];
      }
    }
  }
  size_t block_shift =
      NumZeroBitsBelowLSBNonzero(kBlockDim * kBlockDim * xsize);
  for (size_t y = 2 * ysize; y < ysize * kBlockDim; y++) {
    size_t line_start = y * xsize * kBlockDim;
    size_t block_off = line_start >> block_shift;
    size_t block_idx = line_start & (xsize * kBlockDim * kBlockDim - 1);
    line_start = block_off * stride + block_idx;
    for (size_t x = 0; x < xsize * kBlockDim; x++) {
      if (add) {
        add_to[line_start + x] += block[y * xsize * kBlockDim + x];
      } else {
        add_to[line_start + x] -= block[y * xsize * kBlockDim + x];
      }
    }
  }
}

// Un-color-correlates, quantizes, dequantizes and color-correlates the
// specified coefficients inside the given block, using (c==0,2) or storing
// (c==1) the y-channel values in y_block. Used by predictors to compute the
// decoder-side values to compute predictions on. Coefficients are specified as
// a bit array. Assumes that `block` and `y_block` have the same stride.
template <size_t c>
SIMD_ATTR PIK_INLINE void ComputeDecoderCoefficients(
    const float cmap_factor, const Quantizer& quantizer, uint8_t quant_table,
    const int32_t quant_ac, const float inv_quant_ac, const uint8_t quant_kind,
    size_t xsize, size_t ysize, const float* block_src, size_t block_stride,
    uint64_t coefficients, float* block, size_t out_stride, float* y_block) {
  // TODO(janwas): restrict ptrs
#ifdef ADDRESS_SANITIZER
  PIK_ASSERT(coefficients < 0x1000);
  PIK_ASSERT(ysize <= 4);
#endif
  for (size_t y = 0; y < ysize; y++) {
    memcpy(block + out_stride * y, block_src + block_stride * y,
           sizeof(float) * xsize * kBlockDim * kBlockDim);
  }
  if (c != 1) {
    for (size_t y = 0; y < ysize; y++) {
      for (size_t i = 0; i < xsize * kBlockDim * kBlockDim; i++) {
        block[y * xsize * kBlockDim * kBlockDim + i] -=
            y_block[y * xsize * kBlockDim * kBlockDim + i] * cmap_factor;
      }
    }
  }
  quantizer.QuantizeRoundtripBlockCoefficients<c>(
      quant_table, quant_ac, quant_kind, xsize, ysize, block, out_stride, block,
      out_stride, coefficients);
  if (c != 1) {
    for (size_t y = 0; y < ysize; y++) {
      for (size_t i = 0; i < xsize * kBlockDim * kBlockDim; i++) {
        block[y * xsize * kBlockDim * kBlockDim + i] +=
            y_block[y * xsize * kBlockDim * kBlockDim + i] * cmap_factor;
      }
    }
  } else {
    for (size_t i = 0; i < ysize; i++) {
      memcpy(y_block + out_stride * i, block + out_stride * i,
             sizeof(float) * xsize * kBlockDim * kBlockDim);
    }
  }
}

static constexpr float k4x4BlurStrength = 2.0007879236394901;

namespace lf_kernel {
struct LFPredictionBlur {
  PIK_INLINE const Weights3x3& Weights() const {
    static constexpr float w0 = 0.41459272584128337;
    static constexpr float w1 = 0.25489157325704559;
    static constexpr float w2 = 0.046449679523692139;
    static const Weights3x3 weights = {
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)},
        {SIMD_REP4(w1)}, {SIMD_REP4(w0)}, {SIMD_REP4(w1)},
        {SIMD_REP4(w2)}, {SIMD_REP4(w1)}, {SIMD_REP4(w2)}};
    return weights;
  }
};
}  // namespace lf_kernel

// Subtract predictions, compute decoder-side coefficients, add predictions and
// compute 2x2 block.
template <size_t c>
SIMD_ATTR void ComputeDecoderBlockAnd2x2DC(
    bool is_border, bool predict_lf, bool predict_hf, AcStrategy acs,
    const size_t residuals_stride, const size_t pred_stride,
    const size_t lf2x2_stride, const size_t bx, const Quantizer& quantizer,
    uint8_t quant_table, int32_t quant_ac,
    const float* PIK_RESTRICT cmap_factor, const float* PIK_RESTRICT pred[3],
    float* PIK_RESTRICT residuals[3], float* PIK_RESTRICT lf2x2_row[3],
    const float* PIK_RESTRICT dc[3], float* PIK_RESTRICT y_residuals_dec) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  float* block_start = residuals[c] + N * N * (bx - 1);
  const float* pred_start = pred[c] + 2 * bx;
  SIMD_ALIGN float decoder_coeffs[AcStrategy::kMaxCoeffArea] = {};
  if (!is_border) {
    if (predict_lf) {
      // Remove prediction
      for (size_t y = 0; y < 2 * acs.covered_blocks_y(); y++) {
        size_t start = y < acs.covered_blocks_y() ? acs.covered_blocks_x() : 0;
        for (size_t x = start; x < 2 * acs.covered_blocks_x(); x++) {
          block_start[y * acs.covered_blocks_x() * kBlockDim + x] -=
              pred_start[y * pred_stride + x];
        }
      }
    }

    // Quantization roundtrip
    const size_t kind = acs.GetQuantKind();
    const float inv_quant_ac = quantizer.inv_quant_ac(quant_ac);
    // 0x302 has bits 1, 8, 9 set.
    ComputeDecoderCoefficients<c>(
        cmap_factor[c], quantizer, quant_table, quant_ac, inv_quant_ac, kind,
        acs.covered_blocks_x(), acs.covered_blocks_y(), block_start,
        residuals_stride, 0x302, decoder_coeffs,
        acs.covered_blocks_x() * kBlockDim * kBlockDim, y_residuals_dec);

    if (predict_lf) {
      // Add back prediction
      for (size_t y = 0; y < 2 * acs.covered_blocks_y(); y++) {
        size_t start = y < acs.covered_blocks_y() ? acs.covered_blocks_x() : 0;
        for (size_t x = 0; x < start; x++) {
          decoder_coeffs[y * acs.covered_blocks_x() * kBlockDim + x] =
              block_start[y * acs.covered_blocks_x() * kBlockDim + x];
        }
        for (size_t x = start; x < 2 * acs.covered_blocks_x(); x++) {
          decoder_coeffs[y * acs.covered_blocks_x() * kBlockDim + x] +=
              pred_start[y * pred_stride + x];
        }
      }
    }
  } else {
    decoder_coeffs[0] = dc[c][bx];
    if (predict_lf) {
      decoder_coeffs[1] = pred[c][2 * bx + 1];
      decoder_coeffs[N] = pred[c][pred_stride + 2 * bx];
      decoder_coeffs[N + 1] = pred[c][pred_stride + 2 * bx + 1];
    }
  }
  if (predict_hf) {
    acs.DC2x2FromLowFrequencies(decoder_coeffs, N * N * acs.covered_blocks_x(),
                                lf2x2_row[c] + 2 * bx, lf2x2_stride);
  }
}

// Copies the lowest-frequency coefficients from DC- to AC-sized image.
SIMD_ATTR void CopyLlf(const Image3F& llf, const AcStrategyImage& ac_strategy,
                       Image3F* PIK_RESTRICT ac64) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  const size_t xsize = llf.xsize() - 2;
  const size_t ysize = llf.ysize() - 2;
  const size_t llf_stride = llf.PixelsPerRow();

  // Copy (reinterpreted) DC values to 0-th block values.
  for (size_t c = 0; c < ac64->kNumPlanes; c++) {
    for (size_t by = 0; by < ysize; ++by) {
      const float* llf_row = llf.ConstPlaneRow(c, by + 1);
      float* ac_row = ac64->PlaneRow(c, by);
      AcStrategyRow strategy_row = ac_strategy.ConstRow(by);
      for (size_t bx = 0; bx < xsize; bx++) {
        AcStrategy strategy = strategy_row[bx];
        if (!strategy.IsFirstBlock()) continue;
        for (size_t y = 0; y < strategy.covered_blocks_y(); y++) {
          for (size_t x = 0; x < strategy.covered_blocks_x(); x++) {
            ac_row[block_size * bx +
                   strategy.covered_blocks_y() * kBlockDim * y + x] =
                llf_row[bx + 1 + llf_stride * y + x];
            std::cout<<"std_llf_index c="<<c<<" by="<<by<<" bx="<<bx<<" y="<<y<<" x="<<x<<std::endl;
          }
        }
      }
    }
  }
}

typedef std::array<std::array<float, 4>, 3> Ub4Kernel;

SIMD_ATTR void ComputeUb4Kernel(const float sigma, Ub4Kernel* out) {
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 4; ++k) {
      out->at(j)[k] = 0.0f;
    }
  }
  std::vector<float> kernel = GaussianKernel(4, sigma);
  for (int k = 0; k < 4; ++k) {
    const int split0 = 4 - k;
    const int split1 = 8 - k;
    for (int j = 0; j < split0; ++j) {
      out->at(0)[k] += kernel[j];
    }
    for (int j = split0; j < split1; ++j) {
      out->at(1)[k] += kernel[j];
    }
    for (int j = split1; j < kernel.size(); ++j) {
      out->at(2)[k] += kernel[j];
    }
  }
}

// Adds to "add_to" (DCT) an image defined by the following transformations:
//  1) Upsample image 4x4 with nearest-neighbor
//  2) Blur with a Gaussian kernel of radius 4 and given sigma
//  3) perform TransposedScaledDCT()
//  4) Zero out the top 2x2 corner of each DCT block
//  5) Negates the prediction if add is false (so the encoder subtracts, and
//  the decoder adds)
template <bool add>
SIMD_ATTR void UpSample4x4BlurDCT(const Rect& dc_rect, const ImageF& img,
                                  const Ub4Kernel& kernel,
                                  const AcStrategyImage& ac_strategy,
                                  const Rect& acs_rect,
                                  ImageF* PIK_RESTRICT blur_x,
                                  ImageF* PIK_RESTRICT add_to) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;

  // TODO(robryk): There's no good reason to compute the full DCT here. It's
  // fine if the output is in pixel space, we just need to zero out top 2x2
  // DCT coefficients. We can do that by computing a "partial DCT" and
  // subtracting (we can have two outputs: a positive pixel-space output and a
  // negative DCT-space output).

  // TODO(robryk): Failing that, merge the blur and DCT into a single linear
  // operation, if feasible.

  const size_t bx0 = dc_rect.x0();
  const size_t bxs = dc_rect.xsize();
  PIK_CHECK(bxs >= 1);
  const size_t bx1 = bx0 + bxs;
  const size_t bx_max = DivCeil(add_to->xsize(), block_size);
  const size_t by0 = dc_rect.y0();
  const size_t bys = dc_rect.ysize();
  PIK_CHECK(bys >= 1);
  const size_t by1 = by0 + bys;
  const size_t by_max = add_to->ysize();
  PIK_CHECK(bx1 <= bx_max && by1 <= by_max);
  const size_t xs = bxs * 2;
  const size_t ys = bys * 2;

  using D = SIMD_PART(float, SIMD_MIN(SIMD_FULL(float)::N, 8));
  using V = D::V;
  const D d;
  V vw0[4] = {set1(d, kernel[0][0]), set1(d, kernel[0][1]),
              set1(d, kernel[0][2]), set1(d, kernel[0][3])};
  V vw1[4] = {set1(d, kernel[1][0]), set1(d, kernel[1][1]),
              set1(d, kernel[1][2]), set1(d, kernel[1][3])};
  V vw2[4] = {set1(d, kernel[2][0]), set1(d, kernel[2][1]),
              set1(d, kernel[2][2]), set1(d, kernel[2][3])};

  PIK_ASSERT(blur_x->xsize() == xs * 4 && blur_x->ysize() == ys + 2);
  for (size_t y = 0; y < ys + 2; ++y) {
    const float* PIK_RESTRICT row = img.ConstRow(y + 1);
    float* const PIK_RESTRICT row_out = blur_x->Row(y);
    for (int x = 0; x < xs; ++x) {
      const float v0 = row[x + 1];
      const float v1 = row[x + 2];
      const float v2 = row[x + 3];
      for (int ix = 0; ix < 4; ++ix) {
        row_out[4 * x + ix] =
            v0 * kernel[0][ix] + v1 * kernel[1][ix] + v2 * kernel[2][ix];
      }
    }
  }

  {
    PROFILER_ZONE("dct upsample");
    for (size_t by = 0; by < bys; ++by) {
      const D d;
      SIMD_ALIGN float block[AcStrategy::kMaxCoeffArea];
      SIMD_ALIGN float coeffs[AcStrategy::kMaxCoeffArea];
      const size_t out_stride = add_to->PixelsPerRow();
      const size_t blur_stride = blur_x->PixelsPerRow();

      float* PIK_RESTRICT row_out = add_to->Row(by0 + by);
      AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(acs_rect, by0 + by);
      for (int bx = 0; bx < bxs; ++bx) {
        AcStrategy acs = ac_strategy_row[bx0 + bx];
        if (!acs.IsFirstBlock()) continue;
        if (!acs.PredictHF()) continue;
        for (int idy = 0; idy < acs.covered_blocks_y(); idy++) {
          const float* PIK_RESTRICT row0d = blur_x->ConstRow(2 * (by + idy));
          const float* PIK_RESTRICT row1d = row0d + blur_stride;
          const float* PIK_RESTRICT row2d = row1d + blur_stride;
          const float* PIK_RESTRICT row3d = row2d + blur_stride;
          for (int idx = 0; idx < acs.covered_blocks_x(); idx++) {
            float* PIK_RESTRICT block_ptr =
                block + AcStrategy::kMaxCoeffBlocks * block_size * idy +
                8 * idx;
            for (int ix = 0; ix < 8; ix += d.N) {
              const auto val0 = load(d, &row0d[(bx + idx) * 8 + ix]);
              const auto val1 = load(d, &row1d[(bx + idx) * 8 + ix]);
              const auto val2 = load(d, &row2d[(bx + idx) * 8 + ix]);
              const auto val3 = load(d, &row3d[(bx + idx) * 8 + ix]);
              for (int iy = 0; iy < 4; ++iy) {
                // A mul_add pair is faster but causes 1E-5 difference.
                const auto vala =
                    val0 * vw0[iy] + val1 * vw1[iy] + val2 * vw2[iy];
                const auto valb =
                    val1 * vw0[iy] + val2 * vw1[iy] + val3 * vw2[iy];
                store(vala, d, &block_ptr[iy * AcStrategy::kMaxBlockDim + ix]);
                store(valb, d,
                      &block_ptr[iy * AcStrategy::kMaxBlockDim +
                                 AcStrategy::kMaxBlockDim * 4 + ix]);
              }
            }
          }
        }

        acs.TransformFromPixels(block, AcStrategy::kMaxBlockDim, coeffs,
                                acs.covered_blocks_x() * kBlockDim * kBlockDim);
        AddBlockExceptLFAndLLFTo<add>(
            coeffs, acs.covered_blocks_x(), acs.covered_blocks_y(),
            row_out + block_size * (bx0 + bx), out_stride);
      }
    }
  }
}

}  // namespace

// Compute the lowest-frequency coefficients in the DCT block (1x1 for DCT8,
// 2x2 for DCT16, etc.)
SIMD_ATTR void ComputeLlf(const Image3F& dc, const AcStrategyImage& ac_strategy,
                          const Rect& acs_rect, Image3F* PIK_RESTRICT llf) {
  PROFILER_FUNC;
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();
  const size_t dc_stride = dc.PixelsPerRow();
  const size_t llf_stride = llf->PixelsPerRow();

  // Copy (reinterpreted) DC values to LLF image.
  for (size_t c = 0; c < llf->kNumPlanes; c++) {
    for (size_t by = 0; by < ysize; ++by) {
      const bool is_border_y = by == 0 || by == ysize - 1;
      AcStrategyRow ac_strategy_row =
          ac_strategy.ConstRow(acs_rect, is_border_y ? 0 : by - 1);
      const float* dc_row = dc.ConstPlaneRow(c, by);
      float* llf_row = llf->PlaneRow(c, by);
      for (size_t bx = 0; bx < xsize; bx++) {
        const bool is_border = is_border_y || (bx == 0 || bx == xsize - 1);
        AcStrategy acs = is_border ? AcStrategy(AcStrategy::Type::DCT, 0)
                                   : ac_strategy_row[bx - 1];
        acs.LowestFrequenciesFromDC(dc_row + bx, dc_stride, llf_row + bx,
                                    llf_stride);      
      }
    }
  }

  std::cout << "llf_acs:" << std::endl;
  for (size_t by = 0; by < ysize; ++by) {
      const bool is_border_y = by == 0 || by == ysize - 1;
      AcStrategyRow ac_strategy_row =
          ac_strategy.ConstRow(acs_rect, is_border_y ? 0 : by - 1);
      for (size_t bx = 0; bx < xsize; bx++) {
        const bool is_border = is_border_y || (bx == 0 || bx == xsize - 1);
        AcStrategy acs = is_border ? AcStrategy(AcStrategy::Type::DCT, 0)
                                   : ac_strategy_row[bx - 1];
        std::cout << (int)acs.RawStrategy() << ",";
    }
      std::cout << std::endl;
  }

  std::cout << "std_llf_x:" << std::endl;
  for (size_t by = 0; by < ysize; ++by) {
      float* llf_row = llf->PlaneRow(0, by);
      for (size_t bx = 0; bx < xsize; bx++) {
        std::cout << llf_row[bx] << ",";
    }
      std::cout << std::endl;
  }
}

// The LF prediction works as follows:
// - Blur the initial DC2x2 image (see ComputeSharpDc2x2FromLlf)
// - Compute the same-size DCT of the resulting blurred image
SIMD_ATTR void PredictLf(const AcStrategyImage& ac_strategy,
                         const Rect& acs_rect, const Image3F& llf,
                         ImageF* tmp2x2, Image3F* lf2x2) {
  PROFILER_FUNC;
  const size_t xsize = llf.xsize();
  const size_t ysize = llf.ysize();
  const size_t llf_stride = llf.PixelsPerRow();
  const size_t lf2x2_stride = lf2x2->PixelsPerRow();
  const size_t tmp2x2_stride = tmp2x2->PixelsPerRow();

  // Plane-wise transforms require 2*4DC*4 = 128KiB active memory. Would be
  // further subdivided into 2 or more stripes to reduce memory pressure.
  for (size_t c = 0; c < lf2x2->kNumPlanes; c++) {
    ImageF* PIK_RESTRICT lf2x2_plane = const_cast<ImageF*>(&lf2x2->Plane(c));

    // Computes the initial DC2x2 from the lowest-frequency coefficients.
    for (size_t by = 0; by < ysize; ++by) {
      const bool is_border_y = by == 0 || by == ysize - 1;
      AcStrategyRow ac_strategy_row =
          ac_strategy.ConstRow(acs_rect, is_border_y ? 0 : by - 1);
      float* tmp2x2_row = tmp2x2->Row(2 * by);
      const float* llf_row = llf.PlaneRow(c, by);
      for (size_t bx = 0; bx < xsize; bx++) {
        const bool is_border = is_border_y || (bx == 0 || bx == xsize - 1);
        AcStrategy acs = is_border ? AcStrategy(AcStrategy::Type::DCT, 0)
                                   : ac_strategy_row[bx - 1];
        acs.DC2x2FromLowestFrequencies(llf_row + bx, llf_stride,
                                       tmp2x2_row + 2 * bx, tmp2x2_stride);
      }
    }

    // Smooth out DC2x2.
    if (xsize * 2 < kConvolveMinWidth) {
      using Convolution = slow::General3x3Convolution<1, WrapMirror>;
      Convolution::Run(*tmp2x2, xsize * 2, ysize * 2,
                       lf_kernel::LFPredictionBlur(), lf2x2_plane);
    } else {
      const BorderNeverUsed border;
      // Parallel doesn't help here for moderate-sized images.
      const ExecutorLoop executor;
      ConvolveT<strategy::Symmetric3>::Run(border, executor, *tmp2x2,
                                           lf_kernel::LFPredictionBlur(),
                                           lf2x2_plane);
    }

    // Compute LF coefficients
    for (size_t by = 0; by < ysize; ++by) {
      const bool is_border_y = by == 0 || by == ysize - 1;
      AcStrategyRow ac_strategy_row =
          ac_strategy.ConstRow(acs_rect, is_border_y ? 0 : by - 1);
      float* lf2x2_row = lf2x2_plane->Row(2 * by);
      for (size_t bx = 0; bx < xsize; bx++) {
        const bool is_border = is_border_y || (bx == 0 || bx == xsize - 1);
        AcStrategy acs = is_border ? AcStrategy(AcStrategy::Type::DCT, 0)
                                   : ac_strategy_row[bx - 1];
        acs.LowFrequenciesFromDC2x2(lf2x2_row + 2 * bx, lf2x2_stride,
                                    lf2x2_row + 2 * bx, lf2x2_stride);
      }
    }
  }
}

// Predict dc2x2 from DC values.
// - Use the LF block (but not the lowest frequency block) as a predictor
// - Update those values with the actual residuals, and re-compute a 2x
//   upsampled image out of that as an input for HF predictions.
// Note: assumes that cmap and quant_cf have the same tile size.
SIMD_ATTR void PredictLfForEncoder(
    bool predict_lf, bool predict_hf, const Image3F& dc,
    const AcStrategyImage& ac_strategy, const ColorCorrelationMap& cmap,
    const Rect& cmap_rect, const Quantizer& quantizer, const ImageB& quant_cf,
    const uint8_t quant_cf_map[kMaxQuantControlFieldValue][256],
    Image3F* PIK_RESTRICT ac64, Image3F* dc2x2) {
  PROFILER_FUNC;
  const size_t xsize = dc.xsize();
  const size_t ysize = dc.ysize();
  const size_t ac_stride = ac64->PixelsPerRow();
  const size_t dc2x2_stride = dc2x2->PixelsPerRow();
  // TODO(user): should not be allocated, when predict_lf == false.
  Image3F lf2x2(xsize * 2, ysize * 2);
  {
    Image3F llf(xsize, ysize);
    ComputeLlf(dc, ac_strategy, Rect(ac_strategy.ConstRaw()), &llf);
    CopyLlf(llf, ac_strategy, ac64);
  }
}

// Similar to PredictLfForEncoder.
SIMD_ATTR void UpdateLfForDecoder(const Rect& tile, bool predict_lf,
                                  bool predict_hf,
                                  const AcStrategyImage& ac_strategy,
                                  const Rect& acs_rect, const Image3F& llf,
                                  Image3F* PIK_RESTRICT ac64,
                                  Image3F* PIK_RESTRICT dc2x2,
                                  Image3F* PIK_RESTRICT lf2x2, size_t c) {
  constexpr size_t N = kBlockDim;
  constexpr size_t block_size = N * N;
  const size_t bx0 = tile.x0();
  const size_t bx1 = bx0 + tile.xsize();
  const size_t by0 = tile.y0();
  const size_t by1 = by0 + tile.ysize();
  const size_t ac_stride = ac64->PixelsPerRow();
  const size_t dc2x2_stride = predict_hf ? dc2x2->PixelsPerRow() : 0;
  const size_t lf2x2_stride = predict_lf ? lf2x2->PixelsPerRow() : 0;
  const size_t llf_stride = llf.PixelsPerRow();

  for (size_t by = by0; by < by1; ++by) {
    const float* llf_row = llf.ConstPlaneRow(c, by + 1);
    float* ac_row = ac64->PlaneRow(c, by);
    AcStrategyRow acs_row = ac_strategy.ConstRow(acs_rect, by);
    for (size_t bx = bx0; bx < bx1; bx++) {
      AcStrategy acs = acs_row[bx];
      if (!acs.IsFirstBlock()) continue;
      for (size_t y = 0; y < acs.covered_blocks_y(); y++) {
        for (size_t x = 0; x < acs.covered_blocks_x(); x++) {
          ac_row[block_size * bx + acs.covered_blocks_x() * kBlockDim * y + x] =
              llf_row[bx + 1 + llf_stride * y + x];
        }
      }
    }
  }

  // Compute decoder-side coefficients, 2x scaled DC image, and subtract
  // predictions.
  // Add predictions and compute 2x scaled image to feed to HF predictor
  if (predict_lf) {
    for (size_t by = by0; by < by1; ++by) {
      AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(acs_rect, by);
      float* PIK_RESTRICT ac_row = ac64->PlaneRow(c, by);
      const float* PIK_RESTRICT lf2x2_row =
          lf2x2->ConstPlaneRow(c, 2 * (by + 1));
      for (size_t bx = bx0; bx < bx1; bx++) {
        AcStrategy acs = ac_strategy_row[bx];
        float* PIK_RESTRICT ac_pos = ac_row + bx * block_size;
        const float* PIK_RESTRICT lf2x2_pos = lf2x2_row + (bx + 1) * 2;
        if (!acs.IsFirstBlock()) continue;
        if (predict_lf) {
          for (size_t y = 0; y < 2 * acs.covered_blocks_y(); y++) {
            size_t start =
                y < acs.covered_blocks_y() ? acs.covered_blocks_x() : 0;
            for (size_t x = start; x < 2 * acs.covered_blocks_x(); x++) {
              ac_pos[y * acs.covered_blocks_x() * kBlockDim + x] +=
                  lf2x2_pos[y * lf2x2_stride + x];
            }
          }
        }
      }
    }
  }

  if (predict_hf) {
    for (size_t by = by0; by < by1; ++by) {
      AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(acs_rect, by);
      const float* PIK_RESTRICT ac_row = ac64->PlaneRow(c, by);
      float* PIK_RESTRICT dc2x2_row = dc2x2->PlaneRow(c, 2 * (by + 1));
      for (size_t bx = bx0; bx < bx1; bx++) {
        AcStrategy acs = ac_strategy_row[bx];
        if (!acs.IsFirstBlock()) continue;
        acs.DC2x2FromLowFrequencies(ac_row + block_size * bx, ac_stride,
                                    dc2x2_row + 2 * (bx + 1), dc2x2_stride);
      }
    }
  }
}

SIMD_ATTR void ComputePredictionResiduals(const Image3F& pred2x2,
                                          const AcStrategyImage& ac_strategy,
                                          Image3F* PIK_RESTRICT coeffs) {
  Rect dc_rect(0, 0, pred2x2.xsize() / 2 - 2, pred2x2.ysize() / 2 - 2);
  Rect acs_rect(0, 0, ac_strategy.xsize(), ac_strategy.ysize());
  Ub4Kernel kernel;
  ComputeUb4Kernel(k4x4BlurStrength, &kernel);
  ImageF blur_x(dc_rect.xsize() * 8, dc_rect.ysize() * 2 + 2);
  for (int c = 0; c < coeffs->kNumPlanes; ++c) {
    UpSample4x4BlurDCT</*add=*/false>(dc_rect, pred2x2.Plane(c), kernel,
                                      ac_strategy, acs_rect, &blur_x,
                                      const_cast<ImageF*>(&coeffs->Plane(c)));
  }
}

void AddPredictions(const Image3F& pred2x2, const AcStrategyImage& ac_strategy,
                    const Rect& acs_rect, ImageF* PIK_RESTRICT blur_x,
                    Image3F* PIK_RESTRICT dcoeffs) {
  PROFILER_FUNC;
  Rect dc_rect(0, 0, pred2x2.xsize() / 2 - 2, pred2x2.ysize() / 2 - 2);
  Ub4Kernel kernel;
  ComputeUb4Kernel(k4x4BlurStrength, &kernel);
  for (int c = 0; c < dcoeffs->kNumPlanes; ++c) {
    // Updates dcoeffs _except_ 0HVD.
    UpSample4x4BlurDCT</*add=*/true>(dc_rect, pred2x2.Plane(c), kernel,
                                     ac_strategy, acs_rect, blur_x,
                                     const_cast<ImageF*>(&dcoeffs->Plane(c)));
  }
}

}  // namespace pik
