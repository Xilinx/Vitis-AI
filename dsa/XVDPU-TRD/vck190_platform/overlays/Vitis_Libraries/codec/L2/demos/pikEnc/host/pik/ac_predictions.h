// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_AC_PREDICTIONS_H_
#define PIK_AC_PREDICTIONS_H_

#include "pik/color_correlation.h"
#include "pik/compressed_image_fwd.h"
#include "pik/quantizer.h"
#include "pik/simd/simd.h"

// Encoder-side and decoder-side functions to predict low frequency coefficients
// from lowest-frequency coefficients, and high frequency coefficients from low
// frequency coefficients.
// Given a block of size N, the top (N/8)*(N/8) block of coefficients are the
// lowest-frequency coefficients. Other coefficients in the top (N/4)*(N/4)
// block are the low frequency coefficients, and the rest of the block are high
// frequency coefficients.
// Lowest frequency coefficients are encoded in DC (as an (N/8)*(N/8) IDCT). We
// obtain a 2x upsampled image out of this, by computing (N/4)*(N/4) IDCTS of
// the LLF coefficients (other coefficients are set to 0). Then we smooth this
// image with a convolution, DCT it to obtain LF coefficients, and use that as a
// prediction.
// The process for HF predictions is similar: LF coefficients are IDCT-ed back
// into a 4x downsampled image, which is 4x upsampled and smoothed with a radius
// 4 gaussian blur. NxN blocks in the resulting image are then used to predict
// the HF coefficients, after a DCT.

namespace pik {

// All the `acs_rect`s here define which area of the ac_strategy image should be
// used to obtain the strategy of the current block from, and are specified in
// block coordinates.

// Common utilities.
SIMD_ATTR void ComputeLlf(const Image3F& dc, const AcStrategyImage& ac_strategy,
                          const Rect& acs_rect, Image3F* PIK_RESTRICT llf);
SIMD_ATTR void PredictLf(const AcStrategyImage& ac_strategy,
                         const Rect& acs_rect, const Image3F& llf,
                         ImageF* tmp2x2, Image3F* lf2x2);

// Encoder API.
SIMD_ATTR void PredictLfForEncoder(
    bool predict_lf, bool predict_hf, const Image3F& dc,
    const AcStrategyImage& ac_strategy, const ColorCorrelationMap& cmap,
    const Rect& cmap_rect, const Quantizer& quantizer, const ImageB& quant_cf,
    const uint8_t quant_cf_map[kMaxQuantControlFieldValue][256],
    Image3F* PIK_RESTRICT ac64, Image3F* dc2x2);

void ComputePredictionResiduals(const Image3F& pred2x2,
                                const AcStrategyImage& ac_strategy,
                                Image3F* PIK_RESTRICT coeffs);

// Decoder API. Encoder-decoder API is currently not symmetric. Ideally both
// should allow tile-wise processing.
SIMD_ATTR void UpdateLfForDecoder(const Rect& tile, bool predict_lf,
                                  bool predict_hf,
                                  const AcStrategyImage& ac_strategy,
                                  const Rect& acs_rect, const Image3F& llf,
                                  Image3F* PIK_RESTRICT ac64,
                                  Image3F* PIK_RESTRICT dc2x2,
                                  Image3F* PIK_RESTRICT lf2x2, size_t c);

// `blur_x` is preallocated by GroupDecCache.
void AddPredictions(const Image3F& pred2x2, const AcStrategyImage& ac_strategy,
                    const Rect& acs_rect, ImageF* PIK_RESTRICT blur_x,
                    Image3F* PIK_RESTRICT dcoeffs);

}  // namespace pik

#endif  // PIK_AC_PREDICTIONS_H_
