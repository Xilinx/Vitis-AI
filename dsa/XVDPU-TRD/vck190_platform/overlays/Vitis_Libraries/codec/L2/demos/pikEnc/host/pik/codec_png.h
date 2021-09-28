// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_CODEC_PNG_H_
#define PIK_CODEC_PNG_H_

// Encodes/decodes PNG pixels and metadata in memory.

#include "pik/codec.h"
#include "pik/color_management.h"
#include "pik/padded_bytes.h"

namespace pik {

// Decodes "bytes" and transforms to io->c_current color space. io->dec_hints
// may specify "color_space" and "range" (defaults are sRGB and full-range).
Status DecodeImagePNG(const PaddedBytes& bytes, ThreadPool* pool,
                      CodecInOut* io);

// Transforms from io->c_current to io->c_external and encodes into "bytes".
Status EncodeImagePNG(const CodecInOut* io, const ColorEncoding& c_desired,
                      size_t bits_per_sample, ThreadPool* pool,
                      PaddedBytes* bytes);

}  // namespace pik

#endif  // PIK_CODEC_PNG_H_
