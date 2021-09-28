// Copyright 2019 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_COMPRESSED_DC_H_
#define PIK_COMPRESSED_DC_H_

#include <vector>
#include "pik/color_correlation.h"
#include "pik/compressed_image_fwd.h"
#include "pik/data_parallel.h"
#include "pik/headers.h"
#include "pik/multipass_handler.h"
#include "pik/padded_bytes.h"
#include "pik/pik_info.h"
#include "pik/quantizer.h"

// DC handling functions: encoding and decoding of DC to and from bitstream, and
// related function to initialize the per-group decoder cache.

namespace pik {

// Encodes the DC-related information from frame_enc_cache: quantized dc itself
// and gradient map.
PaddedBytes EncodeDCGroups(const Quantizer& quantizer,
                           const FrameEncCache& frame_enc_cache,
                           const AcStrategyImage& ac_strategy,
                           MultipassManager* manager, PikImageSizeInfo* dc_info,
                           PikImageSizeInfo* cfield_info);

// Decodes and dequantizes DC, and optionally decodes and applies the
// gradient map if requested.
Status DecodeDCGroups(BitReader* reader, const PaddedBytes& compressed,
                      const FrameHeader& frame_header, size_t xsize_blocks,
                      size_t ysize_blocks, const Quantizer& quantizer,
                      const ColorCorrelationMap& cmap, ThreadPool* pool,
                      MultipassManager* manager,
                      FrameDecCache* PIK_RESTRICT frame_dec_cache,
                      std::vector<GroupDecCache>* group_dec_caches,
                      PikInfo* aux_out);

// Clamps the input coordinate `candidate` to the [0, size) interval, using 1 px
// of border (extended by cloning, not mirroring).
PIK_INLINE size_t SourceCoord(size_t candidate, size_t size) {
  return candidate == 0 ? 0
                        : (candidate == size + 1 ? size - 1 : candidate - 1);
}

// Initializes the dec_cache for decoding the `rect` part of the image (in pixel
// units) from the pass decoder cache.
void InitializeDecCache(const FrameDecCache& frame_dec_cache, const Rect& rect,
                        GroupDecCache* PIK_RESTRICT group_dec_cache);

}  // namespace pik

#endif  // PIK_COMPRESSED_DC_H_
