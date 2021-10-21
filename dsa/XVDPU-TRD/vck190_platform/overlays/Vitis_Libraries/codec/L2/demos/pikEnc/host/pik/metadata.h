// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_METADATA_H_
#define PIK_METADATA_H_

// Image metadata stored in FileHeader and CodecInOut.

#include <stdint.h>

#include "pik/color_encoding.h"
#include "pik/field_encodings.h"
#include "pik/padded_bytes.h"
#include "pik/pik_params.h"
#include "pik/status.h"

namespace pik {

// Optional metadata about the original image source.
struct Transcoded {
  Transcoded();
  static const char* Name() { return "Transcoded"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    if (visitor->AllDefault(*this, &all_default)) return true;

    visitor->U32(0x05A09088, 8, &original_bit_depth);
    PIK_RETURN_IF_ERROR(visitor->VisitNested(&original_color_encoding));
    visitor->U32(0x84828180u, 0, &original_bytes_per_alpha);

    return true;
  }

  bool all_default;

  uint32_t original_bit_depth;            // = CodecInOut.dec_bit_depth
  ColorEncoding original_color_encoding;  // = io->dec_c_original in the encoder
  // TODO(lode): This should use bits instead of bytes, 1-bit alpha channel
  //             images exist and may be desired by users using this feature.
  // Alpha bytes per channel of original image (not necessarily the same as
  // the encoding used in the pik file).
  uint32_t original_bytes_per_alpha = 0;
};

struct Metadata {
  Metadata();
  static const char* Name() { return "Metadata"; }

  template <class Visitor>
  Status VisitFields(Visitor* PIK_RESTRICT visitor) {
    if (visitor->AllDefault(*this, &all_default)) return true;

    PIK_RETURN_IF_ERROR(visitor->VisitNested(&transcoded));

    // 100, 250, 4000 are common; don't anticipate more than 10,000.
    visitor->U32(0x08D08582, kDefaultIntensityTarget / 50, &target_nits_div50);

    visitor->Bytes(BytesEncoding::kBrotli, &exif);
    visitor->Bytes(BytesEncoding::kBrotli, &iptc);
    visitor->Bytes(BytesEncoding::kBrotli, &xmp);

    return true;
  }

  bool all_default;

  Transcoded transcoded;

  uint32_t target_nits_div50;

  PaddedBytes exif;
  PaddedBytes iptc;
  PaddedBytes xmp;
};

}  // namespace pik

#endif  // PIK_METADATA_H_
