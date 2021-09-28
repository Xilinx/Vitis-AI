// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_FIELD_ENCODINGS_H_
#define PIK_FIELD_ENCODINGS_H_

// Constants needed to encode/decode fields; avoids including the full fields.h.

#include <stdint.h>

namespace pik {

// kU32RawBits + x => send x raw bits. This value is convenient because x <= 32
// and ~32u + 32 == ~0u, which ensures RawBits can never exceed 32 and also
// allows the values to be sign-extended from an 8-bit immediate.
static constexpr uint32_t kU32RawBits = ~32u;

// Four direct values [0, 4).
static constexpr uint32_t kU32Direct0To3 = 0x83828180u;

// Three direct values 0, 1, 2 or 2 extra bits for [3, 6].
static constexpr uint32_t kU32Direct3Plus4 = 0x51828180u;

// Three direct values 0, 1, 2 or 3 extra bits for [3, 10].
static constexpr uint32_t kU32Direct3Plus8 = 0x52828180u;

// Four direct values 2, 3, 4, 8 or 1, 2, 4, 8.
static constexpr uint32_t kU32Direct2348 = 0x88848382u;
static constexpr uint32_t kU32Direct1248 = 0x88848281u;

enum class BytesEncoding {
  // Values are determined by kU32Direct3Plus8.
  kNone = 0,  // Not present, don't write size
  kRaw,
  kBrotli  // Only if smaller, otherwise kRaw.
  // Future extensions: [3, 10].
};

}  // namespace pik

#endif  // PIK_FIELD_ENCODINGS_H_
