// Copyright 2019 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_CHROMA_FROM_LUMA_H_
#define PIK_CHROMA_FROM_LUMA_H_

// Chroma-from-luma without side information.

#include "pik/chroma_from_luma_fwd.h"  // CFL_Stats
#include "pik/common.h"
#include "pik/descriptive_statistics.h"
#include "pik/image.h"
#include "pik/quantizer.h"

namespace pik {

// Sets X and B channels while leaving Y unchanged:
//   `residual_xb` to quantize(`exact_xb` - r * `quantized_y`),
//   `restored_xb` to `residual_xb` + r * `quantized_y`,
// r is computed from previous (in scan order) parts of `restored_xb`. All
// images are in DCT layout, and `rect` is in units of blocks (must be the
// same as a subsequent call to RestoreAC). To skip statistics gathering, set
// `stats` = nullptr.
void DecorrelateAC(const ImageF& quantized_y, const Image3F& exact_xb,
                   const Rect& rect, const Quantizer& quantizer,
                   uint8_t quant_table, Image3F* PIK_RESTRICT residual_xb,
                   Image3F* PIK_RESTRICT restored_xb, CFL_Stats* stats);

// Sets X and B channels while leaving Y unchanged:
//   `restored_xb` to `residual_xb` + r * `quantized_y`. Thus, `restored_xb`
// matches the image returned by DecorrelateAC. All images are in DCT layout,
// and `rect` is in blocks. `residual_xb` may alias `restored_xb`. To skip
// statistics gathering, set `stats` = nullptr.
SIMD_ATTR void RestoreAC(const ImageF& quantized_y, const Image3F& residual_xb,
                         const Rect& rect, Image3F* restored_xb,
                         CFL_Stats* stats);

// As above, but all images contain only DC coefficients.
SIMD_ATTR void DecorrelateDC(const ImageF& quantized_y, const Image3F& exact_xb,
                             const Rect& rect, const Quantizer& quantizer,
                             uint8_t quant_table,
                             Image3F* PIK_RESTRICT residual_xb,
                             Image3F* PIK_RESTRICT restored_xb,
                             CFL_Stats* stats);

SIMD_ATTR void RestoreDC(const ImageF& quantized_y, const Image3F& residual_xb,
                         const Rect& rect, Image3F* restored_xb,
                         CFL_Stats* stats);

}  // namespace pik

#endif  // PIK_CHROMA_FROM_LUMA_H_
