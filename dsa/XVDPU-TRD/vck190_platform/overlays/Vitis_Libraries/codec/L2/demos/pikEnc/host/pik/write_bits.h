// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_WRITE_BITS_H_
#define PIK_WRITE_BITS_H_

// Unbuffered writes to the bitstream using unaligned 64-bit stores.

#include <stdint.h>
#include <string.h>  // memcpy
#include <cstddef>

#include "pik/arch_specific.h"
#include "pik/byte_order.h"
#include "pik/compiler_specific.h"
#include "pik/status.h"

namespace pik {

// This function writes bits into bytes in increasing addresses, and within
// a byte least-significant-bit first.
//
// The function can write up to 56 bits in one go with WriteBits
// Example: let's assume that 3 bits (Rs below) have been written already:
//
// BYTE-0     BYTE+1       BYTE+2
//
// 0000 0RRR    0000 0000    0000 0000
//
// Now, we could write 5 or less bits in MSB by just shifting by 3
// and OR'ing to BYTE-0.
//
// For n bits, we take the last 5 bits, OR that with high bits in BYTE-0,
// and locate the rest in BYTE+1, BYTE+2, etc.
PIK_INLINE void WriteBits(const size_t n_bits, uint64_t bits,
                          size_t* PIK_RESTRICT pos,
                          uint8_t* PIK_RESTRICT array) {
  PIK_ASSERT((bits >> n_bits) == 0);
  PIK_ASSERT(n_bits <= 56);
#if PIK_BYTE_ORDER_LITTLE
  // This branch of the code can write up to 56 bits at a time,
  // 7 bits are lost by being perhaps already in *p and at least
  // 1 bit is needed to initialize the bit-stream ahead (i.e. if 7
  // bits are in *p and we write 57 bits, then the next write will
  // access a byte that was never initialized).
  uint8_t* p = &array[*pos >> 3];
  uint64_t v = *p;
  v |= bits << (*pos & 7);
  memcpy(p, &v, sizeof(v));  // Write bytes: possibly more than n_bits/8
  *pos += n_bits;
#else
  // implicit & 0xff is assumed for uint8_t arithmetics
  uint8_t* array_pos = &array[*pos >> 3];
  const size_t bits_reserved_in_first_byte = (*pos & 7);
  bits <<= bits_reserved_in_first_byte;
  *array_pos++ |= static_cast<uint8_t>(bits);
  for (size_t bits_left_to_write = n_bits + bits_reserved_in_first_byte;
       bits_left_to_write >= 9; bits_left_to_write -= 8) {
    bits >>= 8;
    *array_pos++ = static_cast<uint8_t>(bits);
  }
  *array_pos = 0;
  *pos += n_bits;
#endif
}

PIK_INLINE void WriteZeroesToByteBoundary(size_t* PIK_RESTRICT pos,
                                          uint8_t* PIK_RESTRICT array) {
  const size_t nbits = ((*pos + 7) & ~7) - *pos;
  WriteBits(nbits, 0, pos, array);
  PIK_ASSERT(*pos % 8 == 0);
}

PIK_INLINE void WriteBitsPrepareStorage(size_t pos, uint8_t* array) {
  PIK_ASSERT((pos & 7) == 0);
  array[pos >> 3] = 0;
}

PIK_INLINE void RewindStorage(const size_t pos0, size_t* PIK_RESTRICT pos,
                              uint8_t* PIK_RESTRICT array) {
  PIK_ASSERT(pos0 <= *pos);
  *pos = pos0;
  static const uint8_t kRewindMasks[8] = {0x0, 0x1,  0x3,  0x7,
                                          0xf, 0x1f, 0x3f, 0x7f};
  array[pos0 >> 3] &= kRewindMasks[pos0 & 7];
}

class BitWriter {
 public:
  BitWriter(size_t* storage_ix, uint8_t* storage)
      : storage_ix_(storage_ix), storage_(storage) {}

  void VisitBits(size_t nbits, uint64_t bits) {
    WriteBits(nbits, bits, storage_ix_, storage_);
  }

 protected:
  size_t* storage_ix_;
  uint8_t* storage_;
};

struct BitCounter {
  void VisitBits(size_t nbits, uint64_t bits) { num_bits += nbits; }
  size_t num_bits = 0;
};

}  // namespace pik

#endif  // PIK_WRITE_BITS_H_
