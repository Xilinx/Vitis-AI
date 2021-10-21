// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_SIZE_CODER_H_
#define PIK_SIZE_CODER_H_

#include <cstddef>
#include <cstdint>
#include "pik/fields.h"
#include "pik/status.h"

namespace pik {

template <uint32_t kDistribution>
class SizeCoderT {
 public:
  static size_t MaxSize(const size_t num_sizes) {
    const size_t bits = U32Coder::MaxEncodedBits(kDistribution) * num_sizes;
    return DivCeil(bits, kBitsPerByte);
  }

  static void Encode(const size_t size, size_t* PIK_RESTRICT pos,
                     uint8_t* storage) {
    PIK_CHECK(U32Coder::Write(kDistribution, size, pos, storage));
  }

  static size_t Decode(BitReader* reader) {
    return U32Coder::Read(kDistribution, reader);
  }
};

}  // namespace pik

#endif  // PIK_SIZE_CODER_H_
