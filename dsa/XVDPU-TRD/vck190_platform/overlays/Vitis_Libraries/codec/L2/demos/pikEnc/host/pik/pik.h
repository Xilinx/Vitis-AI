// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_PIK_H_
#define PIK_PIK_H_

// Top-level interface for PIK encoding/decoding.

#include "pik/codec.h"
#include "pik/data_parallel.h"
#include "pik/padded_bytes.h"
#include "pik/pik_info.h"
#include "pik/pik_params.h"
#include "pik/status.h"

namespace pik {

// Compresses pixels from `io` (given in any ColorEncoding).
// `io` must have original_bits_per_sample and dec_c_original fields set.
Status PixelsToPik(const CompressParams &params, std::string xclbinPath,
                   const CodecInOut *io, PaddedBytes *compressed,
                   PikInfo *aux_out = nullptr, ThreadPool *pool = nullptr);

// Implementation detail: currently decodes to linear sRGB. The contract is:
// `io` appears 'identical' (modulo compression artifacts) to the encoder input
// in a color-aware viewer. Note that `io`->dec_c_original identifies the color
// space that was passed to the encoder; clients that need that encoding must
// call `io`->TransformTo afterwards.
Status PikToPixels(const DecompressParams &params,
                   const PaddedBytes &compressed, CodecInOut *io,
                   PikInfo *aux_out = nullptr, ThreadPool *pool = nullptr);

} // namespace pik

#endif // PIK_PIK_H_
