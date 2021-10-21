// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// @author Alexander Rhatushnyak

#ifndef PIK_LOSSLESS16_H_
#define PIK_LOSSLESS16_H_

#include "pik/image.h"
#include "pik/padded_bytes.h"

namespace pik {

// *compress appends to `bytes`.
// *decompress starts at byte offset `*pos` and sets `*pos` to point to the
// first unconsumed byte.
bool Grayscale16bit_compress(const ImageU& img, PaddedBytes* bytes);
bool Grayscale16bit_decompress(const PaddedBytes& bytes, size_t* pos,
                               ImageU* result);

bool Colorful16bit_compress(const Image3U& img, PaddedBytes* bytes);
bool Colorful16bit_decompress(const PaddedBytes& bytes, size_t* pos,
                              Image3U* result);
}  // namespace pik

#endif  // PIK_LOSSLESS16_H_
