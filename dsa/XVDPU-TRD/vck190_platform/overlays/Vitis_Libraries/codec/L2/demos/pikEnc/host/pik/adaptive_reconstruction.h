// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_ADAPTIVE_RECONSTRUCTION_H_
#define PIK_ADAPTIVE_RECONSTRUCTION_H_

// "In-loop" filter: edge-preserving filter + adaptive clamping to DCT interval.

#include "pik/adaptive_reconstruction_fwd.h"
#include "pik/data_parallel.h"
#include "pik/epf.h"
#include "pik/image.h"
#include "pik/multipass_handler.h"
#include "pik/quantizer.h"

namespace pik {

// Edge-preserving smoothing plus clamping the result to the quantized interval
// (which requires `quantizer` to reconstruct the values that
// were actually quantized). `in` is the image to filter:  opsin AFTER gaborish.
// `non_smoothed` is BEFORE gaborish.
SIMD_ATTR Image3F AdaptiveReconstruction(
    const Image3F& in, const Image3F& non_smoothed, const Quantizer& quantizer,
    const ImageI& raw_quant_field, const ImageB& quant_cf,
    const uint8_t quant_cf_map[kMaxQuantControlFieldValue][256],
    const ImageB& sigma_lut_ids, const AcStrategyImage& ac_strategy,
    const EpfParams& epf_params,
    AdaptiveReconstructionAux* aux = nullptr);

// Edge-preserving smoothing plus clamping the result to quantized interval,
// done on the DC image in pixel space.
void AdaptiveDCReconstruction(Image3F& dc, const Quantizer& quantizer);

// Calls the edge-preserving filter using proper target dispatching.
Image3F DoDenoise(const Image3F& opsin, const Image3F& opsin_sharp,
                  const Quantizer& quantizer, const ImageI& raw_quant_field,
                  const ImageB& sigma_lut_ids,
                  const AcStrategyImage& ac_strategy,
                  const EpfParams& epf_params,
                  AdaptiveReconstructionAux* aux = nullptr);

}  // namespace pik

#endif  // PIK_ADAPTIVE_RECONSTRUCTION_H_
