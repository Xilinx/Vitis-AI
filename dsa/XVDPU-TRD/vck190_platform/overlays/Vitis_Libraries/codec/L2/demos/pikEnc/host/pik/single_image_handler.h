// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_SINGLE_PASS_HANDLER_H_
#define PIK_SINGLE_PASS_HANDLER_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "pik/ac_strategy.h"
#include "pik/adaptive_quantization.h"
#include "pik/codec.h"
#include "pik/color_correlation.h"
#include "pik/image.h"
#include "pik/multipass_handler.h"

// A multipass handler/manager to encode single images. It will run heuristics
// for quantization, AC strategy and color correlation map only the first time
// we want to encode a lossy pass, and will then re-use the existing heuristics
// for further passes. All the passes of a single image are added together.

namespace pik {

constexpr size_t kMaxNumPasses = 8;
constexpr size_t kNoDownsamplingFactor = std::numeric_limits<size_t>::max();

struct PassDefinition {
  // Side of the square of the coefficients that should be kept in each 8x8
  // block. Must be greater than 1, and at most 8. Should be in non-decreasing
  // order.
  size_t num_coefficients;
  // Whether or not we should include only salient blocks.
  // TODO(veluca): ignored for now.
  bool salient_only;

  // If specified, this indicates that if the requested downsampling factor is
  // sufficiently high, then it is fine to stop decoding after this pass.
  // By default, passes are not marked as being suitable for any downsampling.
  size_t suitable_for_downsampling_factor_of_at_least;
};

struct ProgressiveMode {
  size_t num_passes = 1;
  PassDefinition passes[kMaxNumPasses] = {
      PassDefinition{/*num_coefficients=*/8, /*salient_only=*/false,
                     /*suitable_for_downsampling_factor_of_at_least=*/1}};

  ProgressiveMode() {}

  template <size_t nump>
  ProgressiveMode(const PassDefinition (&p)[nump]) {
    PIK_ASSERT(nump <= kMaxNumPasses);
    num_passes = nump;
    PassDefinition previous_pass{
        /*num_coefficients=*/1,
        /*salient_only=*/false,
        /*suitable_for_downsampling_factor_of_at_least=*/kNoDownsamplingFactor};
    size_t last_downsampling_factor = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < nump; i++) {
      PIK_ASSERT(p[i].num_coefficients > previous_pass.num_coefficients ||
                 (p[i].num_coefficients == previous_pass.num_coefficients &&
                  !p[i].salient_only && previous_pass.salient_only));
      PIK_ASSERT(p[i].suitable_for_downsampling_factor_of_at_least ==
                     std::numeric_limits<size_t>::max() ||
                 p[i].suitable_for_downsampling_factor_of_at_least <=
                     last_downsampling_factor);
      if (p[i].suitable_for_downsampling_factor_of_at_least !=
          std::numeric_limits<size_t>::max()) {
        last_downsampling_factor =
            p[i].suitable_for_downsampling_factor_of_at_least;
      }
      previous_pass = passes[i] = p[i];
    }
  }
};

class SingleImageManager;

class SingleImageHandler : public MultipassHandler {
 public:
  SingleImageHandler(SingleImageManager* manager, const Rect& group_rect,
                     ProgressiveMode mode)
      : manager_(manager),
        group_rect_(group_rect),
        padded_group_rect_(group_rect.x0(), group_rect.y0(),
                           DivCeil(group_rect.xsize(), kBlockDim) * kBlockDim,
                           DivCeil(group_rect.ysize(), kBlockDim) * kBlockDim),
        mode_(mode) {}

  const Rect& GroupRect() override { return group_rect_; }
  const Rect& PaddedGroupRect() override { return padded_group_rect_; };

  std::vector<Image3S> SplitACCoefficients(
      Image3S&& ac, const AcStrategyImage& ac_strategy) override;

  MultipassManager* Manager() override;

 private:
  SingleImageManager* manager_;
  const Rect group_rect_;
  const Rect padded_group_rect_;
  ProgressiveMode mode_;
};

// A MultipassManager for single images.
class SingleImageManager : public MultipassManager {
 public:
  SingleImageManager() { group_handlers_.reserve(16); }

  void StartPass(const FrameHeader& frame_header) override {
    current_header_ = frame_header;
  }

  void SetDecodedPass(const Image3F& opsin) override {}
  void SetDecodedPass(CodecInOut* io) override {}
  void DecorrelateOpsin(Image3F* img) override {}
  void RestoreOpsin(Image3F* img) override {}

  void SetProgressiveMode(ProgressiveMode mode) { mode_ = mode; }

  void SetSaliencyMap(std::shared_ptr<ImageF> saliency_map) {
    saliency_map_ = saliency_map;
  }

  void UseAdaptiveReconstruction() override {
    use_adaptive_reconstruction_ = true;
  }

  MultipassHandler* GetGroupHandler(size_t group_id,
                                    const Rect& group_rect) override;

  BlockDictionary GetBlockDictionary(double butteraugli_target,
                                     const Image3F& opsin) override;

  void GetColorCorrelationMap(const Image3F& opsin,
                              const DequantMatrices& dequant,
                              ColorCorrelationMap* cmap) override;

  void GetAcStrategy(float butteraugli_target, const ImageF* quant_field,
                     const DequantMatrices& dequant, const Image3F& src,
                     ThreadPool* pool, AcStrategyImage* ac_strategy,
                     PikInfo* aux_out) override;

  std::shared_ptr<Quantizer> GetQuantizer(
      const CompressParams& cparams, size_t xsize_blocks, size_t ysize_blocks,
      const Image3F& opsin_orig, const Image3F& opsin,
      const FrameHeader& frame_header, const GroupHeader& header,
      const ColorCorrelationMap& cmap, const BlockDictionary& block_dictionary,
      const AcStrategyImage& ac_strategy, const ImageB& ar_sigma_lut_ids,
      const DequantMatrices* dequant, const ImageB& dequant_control_field,
      const uint8_t dequant_map[kMaxQuantControlFieldValue][256],
      ImageF& quant_field, PikInfo* aux_out) override;

  std::shared_ptr<Quantizer> GetQuantizerAvg(float avg, float absavg,
      const CompressParams& cparams, size_t xsize_blocks, size_t ysize_blocks,
      const Image3F& opsin_orig, const Image3F& opsin,
      const FrameHeader& frame_header, const GroupHeader& header,
      const ColorCorrelationMap& cmap, const BlockDictionary& block_dictionary,
      const AcStrategyImage& ac_strategy, const ImageB& ar_sigma_lut_ids,
      const DequantMatrices* dequant, const ImageB& dequant_control_field,
      const uint8_t dequant_map[kMaxQuantControlFieldValue][256],
      ImageF& quant_field, PikInfo* aux_out) override;

  size_t GetNumPasses() override { return mode_.num_passes; }
  std::vector<std::pair<uint32_t, uint32_t>> GetDownsamplingToNumPasses()
      override {
    std::vector<std::pair<uint32_t, uint32_t>> result;
    for (int i = 0; i < mode_.num_passes - 1; ++i) {
      const auto min_downsampling_factor =
          mode_.passes[i].suitable_for_downsampling_factor_of_at_least;
      if (1 < min_downsampling_factor &&
          min_downsampling_factor < std::numeric_limits<size_t>::max()) {
        result.emplace_back(min_downsampling_factor, i);
      }
    }
    return result;
  }

 private:
  friend class SingleImageHandler;

  float BlockSaliency(size_t row, size_t col) const;

  FrameHeader current_header_;
  ProgressiveMode mode_;
  bool use_adaptive_reconstruction_ = false;

  std::shared_ptr<ImageF> saliency_map_;

  std::shared_ptr<Quantizer> quantizer_;
  bool has_quantizer_ = false;
  ColorCorrelationMap cmap_;
  bool has_cmap_ = false;
  AcStrategyImage ac_strategy_;
  bool has_ac_strategy_ = false;

  std::vector<std::unique_ptr<SingleImageHandler>> group_handlers_;
};

}  // namespace pik

#endif  // PIK_SINGLE_PASS_HANDLER_H_
