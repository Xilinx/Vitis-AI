// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_ADAPTIVE_QUANTIZATION_H_
#define PIK_ADAPTIVE_QUANTIZATION_H_

#include <stddef.h>

#include "pik/block_dictionary.h"
#include "pik/color_correlation.h"
#include "pik/headers.h"
#include "pik/image.h"
#include "pik/multipass_handler.h"
#include "pik/pik_params.h"
#include "pik/quantizer.h"

// Heuristics to find a good quantizer for a given image. InitialQuantField
// produces a quantization field (i.e. relative quantization amounts for each
// block) out of an opsin-space image. `InitialQuantField` uses heuristics,
// `FindBestQuantizer` (in non-fast mode) will run multiple encoding-decoding
// steps and try to improve the given quant field.

namespace pik {

// Returns an image subsampled by kBlockDim in each direction. If the value
// at pixel (x,y) in the returned image is greater than 1.0, it means that
// more fine-grained quantization should be used in the corresponding block
// of the input image, while a value less than 1.0 indicates that less
// fine-grained quantization should be enough.
ImageF InitialQuantField(double butteraugli_target, double intensity_multiplier,
                         const Image3F& opsin_orig,
                         const CompressParams& cparams, ThreadPool* pool,
                         double rescale);

float InitialQuantDC(float butteraugli_target, float intensity_multiplier);

// Returns a quantizer that uses an adjusted version of the provided
// quant_field.
std::shared_ptr<Quantizer> FindBestQuantizer(
    const CompressParams& cparams, size_t xsize_blocks, size_t ysize_blocks,
    const Image3F& opsin_orig, const Image3F& opsin,
    const FrameHeader& frame_header, const GroupHeader& header,
    const ColorCorrelationMap& cmap, const BlockDictionary& block_dictionary,
    const AcStrategyImage& ac_strategy, const ImageB& ar_sigma_lut_ids,
    const DequantMatrices* dequant, const ImageB& dequant_control_field,
    const uint8_t dequant_map[kMaxQuantControlFieldValue][256],
    ImageF& quant_field, PikInfo* aux_out,
    MultipassManager* multipass_manager, double rescale = 1.0);

std::shared_ptr<Quantizer> FindBestQuantizerAvg(float avg, float absavg,
    const CompressParams& cparams, size_t xsize_blocks, size_t ysize_blocks,
    const Image3F& opsin_orig, const Image3F& opsin,
    const FrameHeader& frame_header, const GroupHeader& header,
    const ColorCorrelationMap& cmap, const BlockDictionary& block_dictionary,
    const AcStrategyImage& ac_strategy, const ImageB& ar_sigma_lut_ids,
    const DequantMatrices* dequant, const ImageB& dequant_control_field,
    const uint8_t dequant_map[kMaxQuantControlFieldValue][256],
    ImageF& quant_field, PikInfo* aux_out,
    MultipassManager* multipass_manager, double rescale = 1.0);

}  // namespace pik

#endif  // PIK_ADAPTIVE_QUANTIZATION_H_
