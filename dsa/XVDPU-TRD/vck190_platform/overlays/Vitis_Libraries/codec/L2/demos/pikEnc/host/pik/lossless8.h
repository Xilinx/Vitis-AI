// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// @author Alexander Rhatushnyak

#ifndef PIK_LOSSLESS8_H_
#define PIK_LOSSLESS8_H_

#include "pik/image.h"
#include "pik/padded_bytes.h"

namespace pik {

bool Grayscale8bit_compress(const ImageB& img, PaddedBytes* bytes);
bool Grayscale8bit_decompress(const PaddedBytes& bytes, size_t* pos,
                              ImageB* result);

bool Colorful8bit_compress(const Image3B& img, PaddedBytes* bytes);
bool Colorful8bit_decompress(const PaddedBytes& bytes, size_t* pos,
                              Image3B* result);
}  // namespace pik

#endif  // PIK_LOSSLESS8_H_
