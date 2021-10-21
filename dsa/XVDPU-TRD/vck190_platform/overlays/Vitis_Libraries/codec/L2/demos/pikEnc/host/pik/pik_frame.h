// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_PIK_PASS_H_
#define PIK_PIK_PASS_H_

#include "pik/codec.h"
#include "pik/compressed_image.h"
#include "pik/data_parallel.h"
#include "pik/headers.h"
#include "pik/multipass_handler.h"
#include "pik/padded_bytes.h"
#include "pik/pik_info.h"
#include "pik/pik_params.h"
#include "pik/quantizer.h"
#include "pik/status.h"

// Encode and decode a single pass of an image. A pass can be either a
// decomposition of an image (eg. DC-only pass), or a frame in an animation.
// The behaviour of the (en/de)coder is defined by the given multipass_manager.

namespace pik {

struct FrameParams {
  FrameInfo frame_info;
};

// These process each group in parallel.

// Encodes an input image `io` in a byte stream, without adding a file header.
// `pos` represents the bit position in the output data that we should
// start writing to.
Status PixelsToPikPass(CompressParams params, const FrameParams &frame_params,
                       const CodecInOut *io, ThreadPool *pool,
                       PaddedBytes *compressed, size_t &pos, PikInfo *aux_out,
                       MultipassManager *multipass_manager);

Status hls_PixelsToPikPass(CompressParams params, std::string xclbinPath,
                           const FrameParams &frame_params,
                           const CodecInOut *io, ThreadPool *pool,
                           PaddedBytes *compressed, size_t &pos,
                           PikInfo *aux_out,
                           MultipassManager *multipass_manager);

// Decodes an input image from a byte stream, using `file_header`.
// See PikToPixels for explanation of `io` color space.
Status PikPassToPixels(DecompressParams params, const PaddedBytes &compressed,
                       const FileHeader &file_header, ThreadPool *pool,
                       BitReader *reader, CodecInOut *io, PikInfo *aux_out,
                       MultipassManager *multipass_manager);

} // namespace pik

#endif // PIK_PIK_PASS_H_
