// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/single_image_handler.h"
#include "pik/ac_strategy.h"
#include "pik/adaptive_quantization.h"
#include "pik/codec.h"
#include "pik/color_correlation.h"
#include "pik/common.h"
#include "pik/image.h"
#include "pik/opsin_image.h"
#include "pik/opsin_inverse.h"
#include "pik/pik_params.h"
#include "pik/profiler.h"
#include "pik/quantizer.h"

namespace pik {

MultipassHandler* SingleImageManager::GetGroupHandler(size_t group_id,
                                                      const Rect& group_rect) {
  if (group_handlers_.size() <= group_id) {
    group_handlers_.resize(group_id + 1);
  }
  if (!group_handlers_[group_id]) {
    group_handlers_[group_id].reset(
        new SingleImageHandler(this, group_rect, mode_));
  }
  return group_handlers_[group_id].get();
}

float SingleImageManager::BlockSaliency(size_t row, size_t col) const {
  auto saliency_map = saliency_map_.get();
  if (saliency_map == nullptr) return 0.0f;
  return saliency_map->Row(row)[col];
}

void SingleImageManager::GetColorCorrelationMap(const Image3F& opsin,
                                                const DequantMatrices& dequant,
                                                ColorCorrelationMap* cmap) {
  if (!has_cmap_) {
    cmap_ = std::move(*cmap);
    FindBestColorCorrelationMap(opsin, dequant, &cmap_);
    has_cmap_ = true;
  }
  *cmap = cmap_.Copy();
}

BlockDictionary SingleImageManager::GetBlockDictionary(
    double butteraugli_target, const Image3F& opsin) {
  return FindBestBlockDictionary(butteraugli_target, opsin);
}

void SingleImageManager::GetAcStrategy(float butteraugli_target,
                                       const ImageF* quant_field,
                                       const DequantMatrices& dequant,
                                       const Image3F& src, ThreadPool* pool,
                                       AcStrategyImage* ac_strategy,
                                       PikInfo* aux_out) {
  if (!has_ac_strategy_) {
    FindBestAcStrategy(butteraugli_target, quant_field, dequant, src, pool,
                       &ac_strategy_, aux_out);
    has_ac_strategy_ = true;
  }
  *ac_strategy = ac_strategy_.Copy();
}

std::shared_ptr<Quantizer> SingleImageManager::GetQuantizer(
    const CompressParams& cparams, size_t xsize_blocks, size_t ysize_blocks,
    const Image3F& opsin_orig, const Image3F& opsin,
    const FrameHeader& frame_header, const GroupHeader& header,
    const ColorCorrelationMap& cmap, const BlockDictionary& block_dictionary,
    const AcStrategyImage& ac_strategy, const ImageB& ar_sigma_lut_ids,
    const DequantMatrices* dequant, const ImageB& dequant_control_field,
    const uint8_t dequant_map[kMaxQuantControlFieldValue][256],
    ImageF& quant_field, PikInfo* aux_out) {
  if (!has_quantizer_) {
    FrameHeader hdr = frame_header;
    if (use_adaptive_reconstruction_) {
      hdr.have_adaptive_reconstruction = true;
    }
    quantizer_ = FindBestQuantizer(
        cparams, xsize_blocks, ysize_blocks, opsin_orig, opsin, hdr, header,
        cmap, block_dictionary, ac_strategy, ar_sigma_lut_ids, dequant,
        dequant_control_field, dequant_map, quant_field, aux_out, this);
    has_quantizer_ = true;
  }
  return quantizer_;
}

std::shared_ptr<Quantizer> SingleImageManager::GetQuantizerAvg(float avg, float absavg,
    const CompressParams& cparams, size_t xsize_blocks, size_t ysize_blocks,
    const Image3F& opsin_orig, const Image3F& opsin,
    const FrameHeader& frame_header, const GroupHeader& header,
    const ColorCorrelationMap& cmap, const BlockDictionary& block_dictionary,
    const AcStrategyImage& ac_strategy, const ImageB& ar_sigma_lut_ids,
    const DequantMatrices* dequant, const ImageB& dequant_control_field,
    const uint8_t dequant_map[kMaxQuantControlFieldValue][256],
    ImageF& quant_field, PikInfo* aux_out) {
  if (!has_quantizer_) {
    FrameHeader hdr = frame_header;
    if (use_adaptive_reconstruction_) {
      hdr.have_adaptive_reconstruction = true;
    }
    quantizer_ = FindBestQuantizerAvg(avg,absavg,
        cparams, xsize_blocks, ysize_blocks, opsin_orig, opsin, hdr, header,
        cmap, block_dictionary, ac_strategy, ar_sigma_lut_ids, dequant,
        dequant_control_field, dequant_map, quant_field, aux_out, this);
    has_quantizer_ = true;
  }
  return quantizer_;
}

std::vector<Image3S> SingleImageHandler::SplitACCoefficients(
    Image3S&& ac, const AcStrategyImage& ac_strategy) {
  if (mode_.num_passes == 1) {
    PIK_ASSERT(mode_.passes[0].num_coefficients == 8);
    PIK_ASSERT(!mode_.passes[0].salient_only);
    std::vector<Image3S> ret;
    ret.push_back(std::move(ac));
    return ret;
  }

  size_t xsize_blocks = ac.xsize() / (kBlockDim * kBlockDim);
  size_t ysize_blocks = ac.ysize();

  size_t last_ncoeff = 1;
  size_t last_salient_only = false;
  std::vector<Image3S> ac_split;

  // TODO(veluca): handle saliency.
  for (size_t i = 0; i < mode_.num_passes; i++) {
    ac_split.emplace_back(ac.xsize(), ac.ysize());
    Image3S* current = &ac_split.back();
    ZeroFillImage(current);
    size_t stride = current->PixelsPerRow();
    size_t frame_coeffs = mode_.passes[i].num_coefficients;
    for (size_t c = 0; c < ac.kNumPlanes; c++) {
      for (size_t by = 0; by < ysize_blocks; by++) {
        const int16_t* PIK_RESTRICT row_in = ac.ConstPlaneRow(c, by);
        AcStrategyRow row_strategy = ac_strategy.ConstRow(by);
        int16_t* PIK_RESTRICT row_out = current->PlaneRow(c, by);
        for (size_t bx = 0; bx < xsize_blocks; bx++) {
          AcStrategy strategy = row_strategy[bx];
          if (!strategy.IsFirstBlock()) continue;
          size_t xsize = strategy.covered_blocks_x();
          size_t ysize = strategy.covered_blocks_y();
          size_t block_shift =
              NumZeroBitsBelowLSBNonzero(kBlockDim * kBlockDim * xsize);
          for (size_t y = 0; y < ysize * frame_coeffs; y++) {
            size_t line_start = y * xsize * kBlockDim;
            size_t block_off = line_start >> block_shift;
            size_t block_idx = line_start & (xsize * kBlockDim * kBlockDim - 1);
            line_start = block_off * stride + block_idx;
            for (size_t x = 0; x < xsize * frame_coeffs; x++) {
              if (x < xsize * last_ncoeff && y < ysize * last_ncoeff) continue;
              row_out[bx * kBlockDim * kBlockDim + line_start + x] =
                  row_in[bx * kBlockDim * kBlockDim + line_start + x];
            }
          }
        }
      }
    }
    last_ncoeff = frame_coeffs;
    last_salient_only = mode_.passes[i].salient_only;
  }
  PIK_ASSERT(last_ncoeff == 8);
  PIK_ASSERT(last_salient_only == false);

  // Saved saliency code. TODO(veluca): integrate
  return ac_split;
}

MultipassManager* SingleImageHandler::Manager() { return manager_; }
}  // namespace pik
