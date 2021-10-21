// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_ALPHA_H_
#define PIK_ALPHA_H_

// Encodes/decodes alpha image to/from its compressed representation.

#include "pik/headers.h"
#include "pik/image.h"
#include "pik/pik_params.h"
#include "pik/status.h"

namespace pik {

Status EncodeAlpha(const CompressParams& params, const ImageU& plane,
                   const Rect& rect, int bit_depth, Alpha* alpha);

// "plane" must be pre-allocated (FileHeader knows the size).
Status DecodeAlpha(const DecompressParams& params, const Alpha& alpha,
                   ImageU* plane, const Rect& rect);

}  // namespace pik

#endif  // PIK_ALPHA_H_
