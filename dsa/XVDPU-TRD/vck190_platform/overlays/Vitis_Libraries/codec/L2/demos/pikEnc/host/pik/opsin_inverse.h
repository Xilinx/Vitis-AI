// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_OPSIN_INVERSE_H_
#define PIK_OPSIN_INVERSE_H_

// XYB -> linear sRGB.

#include "pik/data_parallel.h"
#include "pik/image.h"
#include "pik/simd/simd.h"

namespace pik {

// Converts `inout` from opsin to linear sRGB in-place. Called from per-pass
// postprocessing, hence parallelized.
SIMD_ATTR void OpsinToLinear(Image3F* PIK_RESTRICT inout);

// Converts to linear sRGB, writing to linear:rect_out.
SIMD_ATTR void OpsinToLinear(const Image3F& opsin, const Rect& rect_out,
                             Image3F* PIK_RESTRICT linear);

}  // namespace pik

#endif  // PIK_OPSIN_INVERSE_H_
