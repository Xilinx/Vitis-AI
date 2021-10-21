// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/adaptive_reconstruction.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

#include "pik/ac_strategy.h"
#include "pik/block.h"
#include "pik/common.h"
#include "pik/data_parallel.h"
#include "pik/dct.h"
#include "pik/dct_util.h"
#include "pik/entropy_coder.h"
#include "pik/epf.h"
#include "pik/profiler.h"
#include "pik/quant_weights.h"
#include "pik/quantizer.h"
#include "pik/simd/simd.h"

#ifndef PIK_AR_PRINT_STATS
#define PIK_AR_PRINT_STATS 0
#endif

namespace pik {
namespace {

using DF = SIMD_FULL(float);
using DI = SIMD_FULL(int32_t);
using DU = SIMD_FULL(uint32_t);
using VF = DF::V;
using VI = DI::V;
using VU = DU::V;

struct ARStats {
  SIMD_ATTR void Assimilate(const ARStats& other) {
    const DU d;
    for (int c = 0; c < 3; ++c) {
      const auto low =
          load_unaligned(d, clamp_lo[c]) + load_unaligned(d, other.clamp_lo[c]);
      const auto high =
          load_unaligned(d, clamp_hi[c]) + load_unaligned(d, other.clamp_hi[c]);
      store_unaligned(low, d, clamp_lo[c]);
      store_unaligned(high, d, clamp_hi[c]);
    }
  }

  static SIMD_ATTR uint32_t Total(const uint32_t* PIK_RESTRICT from) {
    const DU d;
    const SIMD_PART(uint32_t, 1) d1;
    return get_part(d1, ext::sum_of_lanes(load_unaligned(d, from)));
  }

  static SIMD_ATTR void Add(const VU add, uint32_t* PIK_RESTRICT to) {
    const DU d;
    store_unaligned(load_unaligned(d, to) + add, d, to);
  }

  // Number of values (after aggregating across lanes).
  uint32_t clamp_lo[3][DU::N] = {{0}};
  uint32_t clamp_hi[3][DU::N] = {{0}};
};

// Clamp the difference between the coefficients of the filtered image and the
// coefficients of the original image (i.e. `correction`) to an interval whose
// size depends on the values in the non-smoothed image. The interval is
// scaled according to `interval_scale`.
template <int c>
SIMD_ATTR PIK_INLINE void SymmetricClamp(const VF interval_scale,
                                         float* PIK_RESTRICT block,
                                         float* PIK_RESTRICT min_ratio,
                                         ARStats* PIK_RESTRICT stats) {
  const DF df;
  const auto half = set1(df, 0.5);
  const auto upper_bound = half * interval_scale;

  const auto neghalf = set1(df, -0.5);
  const auto lower_bound = neghalf * interval_scale;

  const auto correction = load(df, block);
  // Note: this clamping is only for purposes of determining `min_ratio`.
  const auto clamped = min(max(lower_bound, correction), upper_bound);

  // Integer comparisons are faster than float.
  const SIMD_FULL(uint32_t) du;
  const auto correction_u = cast_to(du, correction);
  const auto zero_u = setzero(du);
  const auto correction_is_zero = cast_to(df, correction_u == zero_u);

  // Sanity checks
#ifdef ADDRESS_SANITIZER
  // clamped=0 can only happen if correction_is_zero.
  PIK_ASSERT(ext::all_zero(correction_is_zero) ||
             !ext::all_zero(cast_to(du, clamped) == zero_u));

  // clamped must never change sign vs. correction (else min_ratio is negative).
  const auto sign = cast_to(df, set1(du, 0x80000000u));
  const auto changed_sign = (clamped ^ correction) & sign;
  PIK_ASSERT(ext::all_zero(cast_to(du, changed_sign)));
#endif

  // ratio := clamped/correction: small if 'correction' was clamped a lot.
  // If correction == 0, ratio will be large and min_ratio not updated (fine
  // because zero definitely lies within the quantization interval.)
  const auto divisor = select(correction, set1(df, 1E-7f), correction_is_zero);

  const auto clamp_ratio = clamped / divisor;
  store(min(clamp_ratio, load(df, min_ratio)), df, min_ratio);

#if PIK_AR_PRINT_STATS
  const auto one = set1(du, uint32_t(1));
  const auto is_low = correction < clamped;
  ARStats::Add(cast_to(du, is_low) & one, stats->clamp_lo[c]);
  const auto is_high = correction > clamped;
  ARStats::Add(cast_to(du, is_high) & one, stats->clamp_hi[c]);
#endif
}

// Clamps a block of the filtered image, pointed to by `opsin`, ensuring that it
// does not get too far away from the values in the corresponding block of the
// original image, pointed to by `original`. Instead of computing the difference
// of the DCT of the two images, we compute the DCT of the difference as DCT is
// a linear operator and this saves some work.
template <int c>
SIMD_ATTR PIK_INLINE void UpdateMinRatioOfClampToOriginalDCT(
    const float* PIK_RESTRICT original, size_t stride,
    const float* PIK_RESTRICT dequant_matrix, const float inv_quant_ac,
    const float dc_mul, AcStrategy acs, const float* PIK_RESTRICT filt,
    float* PIK_RESTRICT min_ratio, float* PIK_RESTRICT block,
    ARStats* PIK_RESTRICT stats) {
  const SIMD_FULL(float) df;
  const SIMD_FULL(uint32_t) du;

  const size_t block_width = kBlockDim * acs.covered_blocks_x();
  const size_t block_height = kBlockDim * acs.covered_blocks_y();

  for (size_t iy = 0; iy < block_height; iy++) {
    for (size_t ix = 0; ix < block_width; ix += df.N) {
      const auto filt_v = load(df, filt + stride * iy + ix);
      const auto original_v = load(df, original + stride * iy + ix);
      store(filt_v - original_v, df, block + block_width * iy + ix);
    }
  }

  size_t covered_blocks = acs.covered_blocks_x() * acs.covered_blocks_y();

  {
    SIMD_ALIGN float temp_block[AcStrategy::kMaxCoeffArea];
    acs.TransformFromPixels(block, block_width, temp_block,
                            block_width * kBlockDim);
    memcpy(block, temp_block,
           covered_blocks * kBlockDim * kBlockDim * sizeof(float));
  }

  const uint32_t* only_llf_bits = nullptr;
  if (acs.covered_blocks_x() == 1) {
    SIMD_ALIGN static const uint32_t only_llf_b[AcStrategy::kLLFMaskDim] = {
        ~0u};
    only_llf_bits = only_llf_b;
  }
  if (acs.covered_blocks_x() == 2) {
    SIMD_ALIGN static const uint32_t only_llf_b[AcStrategy::kLLFMaskDim] = {
        ~0u, ~0u};
    only_llf_bits = only_llf_b;
  }
  if (acs.covered_blocks_x() == 4) {
    SIMD_ALIGN static const uint32_t only_llf_b[AcStrategy::kLLFMaskDim] = {
        ~0u, ~0u, ~0u, ~0u};
    only_llf_bits = only_llf_b;
  }
  PIK_ASSERT(only_llf_bits != nullptr);
  PIK_ASSERT(acs.covered_blocks_y() <= kBlockDim);
  static_assert(kBlockDim % SIMD_FULL(float)::N == 0,
                "Block dimension is not a multiple of lane size!");

  // TODO(janwas): template, make covered_blocks* constants
  // Handle lowest-frequencies and corresponding rows.
  for (size_t y = 0; y < acs.covered_blocks_y(); y++) {
    const size_t row_offset = y * block_width;
    const float* llf_scales = acs.ARLowestFrequencyScales(y);
    for (size_t k = 0; k < std::min(AcStrategy::kLLFMaskDim, block_width);
         k += df.N) {
      const size_t ofs = row_offset + k;
      const auto only_llf = cast_to(df, load(du, only_llf_bits + k));
      const auto cur_dc_mul = load(df, llf_scales + k) * set1(df, dc_mul);
      const auto ac_mul =
          load(df, dequant_matrix + ofs) * set1(df, inv_quant_ac);
      const auto interval_scale = select(ac_mul, cur_dc_mul, only_llf);
      SymmetricClamp<c>(interval_scale, block + ofs, min_ratio + ofs, stats);
    }
    for (size_t k = AcStrategy::kLLFMaskDim; k < block_width; k += df.N) {
      const size_t ofs = row_offset + k;
      const auto interval_scale =
          load(df, dequant_matrix + ofs) * set1(df, inv_quant_ac);
      SymmetricClamp<c>(interval_scale, block + ofs, min_ratio + ofs, stats);
    }
  }

  // All other coefficients
  for (size_t k = covered_blocks * kBlockDim;
       k < covered_blocks * kBlockDim * kBlockDim; k += df.N) {
    const auto interval_scale =
        load(df, dequant_matrix + k) * set1(df, inv_quant_ac);
    SymmetricClamp<c>(interval_scale, block + k, min_ratio + k, stats);
  }
}

// Clamp by multiplying block[k] by min_ratio[k], then IDCT.
// DoMul allows disabling the scaling for X as an experiment (disabled).
template <bool DoMul>
SIMD_ATTR PIK_INLINE void ClampAndIDCT(
    float* PIK_RESTRICT block, const size_t block_width,
    const size_t block_height, const float* PIK_RESTRICT min_ratio,
    const AcStrategy acs, const float* PIK_RESTRICT original,
    float* PIK_RESTRICT filt, size_t stride) {
#ifdef ADDRESS_SANITIZER
  for (size_t k = 0; k < block_width * block_height; ++k) {
    PIK_ASSERT(min_ratio[k] >= 0.0f);
  }
#endif

  const SIMD_FULL(float) df;
  for (size_t k = 0; k < block_width * block_height; k += df.N) {
    const auto mul = DoMul ? load(df, min_ratio + k) : set1(df, 1.0f);
    const auto scaled = load(df, block + k) * mul;
    store(scaled, df, block + k);
  }

  // IDCT
  SIMD_ALIGN float pixels[AcStrategy::kMaxCoeffArea];
  acs.TransformToPixels(block, block_width * kBlockDim, pixels, block_width);

  for (size_t iy = 0; iy < block_height; iy++) {
    for (size_t ix = 0; ix < block_width; ix += df.N) {
      const auto block_v = load(df, pixels + block_width * iy + ix);
      const auto original_v = load(df, original + stride * iy + ix);
      store(block_v + original_v, df, filt + stride * iy + ix);
    }
  }
}

void ComputeResidualSlow(const Image3F& in, const Image3F& smoothed,
                         Image3F* PIK_RESTRICT residual) {
  for (int c = 0; c < in.kNumPlanes; ++c) {
    for (size_t y = 0; y < in.ysize(); ++y) {
      const float* row_in = in.PlaneRow(c, y);
      const float* row_smoothed = smoothed.PlaneRow(c, y);
      float* PIK_RESTRICT row_out = residual->PlaneRow(c, y);
      for (size_t x = 0; x < in.xsize(); ++x) {
        row_out[x] = std::abs(row_in[x] - row_smoothed[x]);
      }
    }
  }
}

// TODO(janwas): template, use actual max_coefs from acs as size.
struct ARBlocks {
  float x[AcStrategy::kMaxCoeffArea];
  uint8_t pad1[CacheAligned::kAlignment];
  float y[AcStrategy::kMaxCoeffArea];
  uint8_t pad2[CacheAligned::kAlignment];
  float b[AcStrategy::kMaxCoeffArea];
  uint8_t pad3[CacheAligned::kAlignment];
  float min_ratio[AcStrategy::kMaxCoeffArea];
};

}  // namespace

Image3F DoDenoise(const Image3F& opsin, const Image3F& opsin_sharp,
                  const Quantizer& quantizer, const ImageI& raw_quant_field,
                  const ImageB& sigma_lut_ids,
                  const AcStrategyImage& ac_strategy,
                  const EpfParams& epf_params,
                  AdaptiveReconstructionAux* aux) {
  if (aux != nullptr) {
    aux->quant_scale = quantizer.Scale();
  }

  Image3F smoothed(opsin.xsize(), opsin.ysize());

  const float quant_scale = quantizer.Scale();
  if (epf_params.enable_adaptive) {
    Dispatch(TargetBitfield().Best(), EdgePreservingFilter(), opsin,
             opsin_sharp, &raw_quant_field, quant_scale, sigma_lut_ids,
             ac_strategy, epf_params, &smoothed,
             aux ? &aux->epf_stats : nullptr);
  } else {
    float stretch;
    Dispatch(TargetBitfield().Best(), EdgePreservingFilter(), opsin,
             opsin_sharp, epf_params, aux ? &aux->stretch : &stretch,
             &smoothed);
  }
  return smoothed;
}

void AdaptiveDCReconstruction(Image3F& dc, const Quantizer& quantizer) {
  for (size_t c = 0; c < dc.kNumPlanes; c++) {
    const float half_step = quantizer.inv_quant_dc() *
                            quantizer.DequantMatrix(0, kQuantKindDCT8, c)[0] *
                            0.5f;
    for (size_t y = 0; y < dc.ysize(); y++) {
      float* PIK_RESTRICT dc_row = dc.PlaneRow(c, y);
      for (size_t x = 0; x < dc.xsize(); x++) {
        dc_row[x] = std::max(dc_row[x] - half_step,
                             std::min(dc_row[x], dc_row[x] + half_step));
      }
    }
  }
}

SIMD_ATTR Image3F AdaptiveReconstruction(
    const Image3F& in, const Image3F& non_smoothed, const Quantizer& quantizer,
    const ImageI& raw_quant_field, const ImageB& quant_cf,
    const uint8_t quant_cf_map[kMaxQuantControlFieldValue][256],
    const ImageB& sigma_lut_ids, const AcStrategyImage& ac_strategy,
    const EpfParams& epf_params,
    AdaptiveReconstructionAux* aux) {
  PROFILER_FUNC;
  PIK_ASSERT(in.xsize() / 8 == sigma_lut_ids.xsize() &&
             in.ysize() / 8 == sigma_lut_ids.ysize());
  // Input image should have an integer number of blocks.
  PIK_ASSERT(in.xsize() % kBlockDim == 0 && in.ysize() % kBlockDim == 0);
  const size_t xsize_blocks = in.xsize() / kBlockDim;
  const size_t ysize_blocks = in.ysize() / kBlockDim;

  // Dequantization matrices.
  const float* PIK_RESTRICT dequant_matrices =
      quantizer.DequantMatrix(0, kQuantKindDCT8, 0);
  float dc_mul[3];
  for (size_t c = 0; c < 3; c++) {
    dc_mul[c] =
        quantizer.inv_quant_dc() *
        dequant_matrices[quantizer.DequantMatrixOffset(0, kQuantKindDCT8, c)];
  }

  // Modified below (clamped).
  Image3F filt = DoDenoise(in, non_smoothed, quantizer, raw_quant_field,
                           sigma_lut_ids, ac_strategy, epf_params, aux);
  if (aux != nullptr && aux->filtered != nullptr) {
    *aux->filtered = CopyImage(filt);
  }

  const size_t stride = filt.PlaneRow(0, 1) - filt.PlaneRow(0, 0);
  PIK_ASSERT(stride == in.PlaneRow(0, 1) - in.PlaneRow(0, 0));

  ARStats stats;

	  for(int task = 0; task < ysize_blocks; ++task) {
        const size_t by = task;
        const int32_t* PIK_RESTRICT row_quant = raw_quant_field.ConstRow(by);
        const AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(by);

        const float* PIK_RESTRICT row_original_x =
            non_smoothed.ConstPlaneRow(0, by * kBlockDim);
        const float* PIK_RESTRICT row_original_y =
            non_smoothed.ConstPlaneRow(1, by * kBlockDim);
        const float* PIK_RESTRICT row_original_b =
            non_smoothed.ConstPlaneRow(2, by * kBlockDim);

        float* PIK_RESTRICT row_filt_x = filt.PlaneRow(0, by * kBlockDim);
        float* PIK_RESTRICT row_filt_y = filt.PlaneRow(1, by * kBlockDim);
        float* PIK_RESTRICT row_filt_b = filt.PlaneRow(2, by * kBlockDim);

        const size_t ty = by / kTileDimInBlocks;
        const uint8_t* row_quant_cf = quant_cf.ConstRow(ty);

        for (size_t bx = 0; bx < xsize_blocks; ++bx) {
          const int32_t qac = row_quant[bx];
          const float inv_quant_ac = quantizer.inv_quant_ac(qac);
          const AcStrategy acs = ac_strategy_row[bx];
          if (!acs.IsFirstBlock()) continue;

          const size_t tx = bx / kTileDimInBlocks;

          // TODO(janwas): hoist/precompute
          uint8_t quant_table = quant_cf_map[row_quant_cf[tx]][qac - 1];
          const float* dequant_matrix_x =
              dequant_matrices + quantizer.DequantMatrixOffset(
                                     quant_table, acs.GetQuantKind(), /*c=*/0);
          const float* dequant_matrix_y =
              dequant_matrices + quantizer.DequantMatrixOffset(
                                     quant_table, acs.GetQuantKind(), /*c=*/1);
          const float* dequant_matrix_b =
              dequant_matrices + quantizer.DequantMatrixOffset(
                                     quant_table, acs.GetQuantKind(), /*c=*/2);

          const size_t block_ofs = bx * kBlockDim;
          const float* PIK_RESTRICT pos_original_x = row_original_x + block_ofs;
          const float* PIK_RESTRICT pos_original_y = row_original_y + block_ofs;
          const float* PIK_RESTRICT pos_original_b = row_original_b + block_ofs;
          float* PIK_RESTRICT pos_filt_x = row_filt_x + block_ofs;
          float* PIK_RESTRICT pos_filt_y = row_filt_y + block_ofs;
          float* PIK_RESTRICT pos_filt_b = row_filt_b + block_ofs;

          const size_t block_width = kBlockDim * acs.covered_blocks_x();
          const size_t block_height = kBlockDim * acs.covered_blocks_y();

          SIMD_ALIGN ARBlocks blocks;
          const SIMD_FULL(float) df;
          for (size_t k = 0; k < block_width * block_height; k += df.N) {
            store(set1(df, 1.0f), df, blocks.min_ratio + k);
          }

          UpdateMinRatioOfClampToOriginalDCT<0>(
              pos_original_x, stride, dequant_matrix_x, inv_quant_ac, dc_mul[0],
              acs, pos_filt_x, blocks.min_ratio, blocks.x, &stats);
          UpdateMinRatioOfClampToOriginalDCT<1>(
              pos_original_y, stride, dequant_matrix_y, inv_quant_ac, dc_mul[1],
              acs, pos_filt_y, blocks.min_ratio, blocks.y, &stats);
          UpdateMinRatioOfClampToOriginalDCT<2>(
              pos_original_b, stride, dequant_matrix_b, inv_quant_ac, dc_mul[2],
              acs, pos_filt_b, blocks.min_ratio, blocks.b, &stats);

          ClampAndIDCT<true>(blocks.b, block_width, block_height,
                             blocks.min_ratio, acs, pos_original_b, pos_filt_b,
                             stride);

          ClampAndIDCT<true>(blocks.x, block_width, block_height,
                             blocks.min_ratio, acs, pos_original_x, pos_filt_x,
                             stride);
          ClampAndIDCT<true>(blocks.y, block_width, block_height,
                             blocks.min_ratio, acs, pos_original_y, pos_filt_y,
                             stride);
        }  // bx
      }  // by

#if PIK_AR_PRINT_STATS
  printf("Lo/Hi clamped: %5u %5u; %5u %5u; %5u %5u (pixels: %zu)\n",
         ARStats::Total(stats.clamp_lo[0]), ARStats::Total(stats.clamp_hi[0]),
         ARStats::Total(stats.clamp_lo[1]), ARStats::Total(stats.clamp_hi[1]),
         ARStats::Total(stats.clamp_lo[2]), ARStats::Total(stats.clamp_hi[2]),
         in.xsize() * in.ysize());
#endif

  if (aux != nullptr) {
    if (aux->residual != nullptr) {
      ComputeResidualSlow(in, filt, aux->residual);
    }
    if (aux->ac_quant != nullptr) {
      CopyImageTo(raw_quant_field, aux->ac_quant);
    }
    if (aux->ac_quant != nullptr) {
      CopyImageTo(ac_strategy.ConstRaw(), aux->ac_strategy);
    }
  }
  return filt;
}

}  // namespace pik
