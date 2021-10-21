// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_BIT_READER_H_
#define PIK_BIT_READER_H_

// Bounds-checked bit reader; 64-bit buffer with support for deferred refills.

#include <stddef.h>
#include <stdint.h>
#include <string.h>  // memcpy
#include <algorithm>

#include "pik/compiler_specific.h"
#include "pik/status.h"

namespace pik {

// Reads bits previously written to memory by WriteBits. Reads 4 bytes (or
// len % 4 at the end) of input at a time into its 64-bit buffer. Performs
// bounds-checking, returns all-zero values after the memory buffer is depleted.
class BitReader {
 public:
  // data is not necessarily 4-byte aligned nor padded to RoundUp(len, 4).
  BitReader(const uint8_t* const PIK_RESTRICT data, const size_t len)
      : data32_(reinterpret_cast<const uint32_t*>(data)),
        len32_(len >> 2),
        len_mod4_(len % 4),
        buf_(0),
        pos32_(0),
        bit_pos_(64) {
    FillBitBuffer();
  }

  void FillBitBuffer() {
    if (PIK_UNLIKELY(bit_pos_ >= 32)) {
      bit_pos_ -= 32;
      buf_ >>= 32;

      if (PIK_LIKELY(pos32_ < len32_)) {
        // Read unaligned (memcpy avoids ubsan warning)
        uint32_t next;
        memcpy(&next, data32_ + pos32_, sizeof(next));
        buf_ |= static_cast<uint64_t>(next) << 32;
      } else if (pos32_ == len32_) {
        // Only read the valid bytes.
        const uint8_t* bytes =
            reinterpret_cast<const uint8_t*>(data32_ + pos32_);
        uint64_t next = 0;
        for (size_t i = 0; i < len_mod4_; ++i) {
          // Pre-shifted by 32 so we can inject into buf_ directly.
          // Assumes little-endian byte order.
          next |= static_cast<uint64_t>(bytes[i]) << (i * 8 + 32);
        }
        buf_ |= next;
      }
      ++pos32_;
    }
  }

  void Advance(size_t num_bits) {
    PIK_ASSERT(num_bits + bit_pos_ <= 64);
    bit_pos_ += num_bits;
  }

  template <size_t N>
  int PeekFixedBits() const {
    static_assert(N <= 32, "At most 32 bits may be read.");
    PIK_ASSERT(N + bit_pos_ <= 64);
    return (buf_ >> bit_pos_) & ((1ULL << N) - 1);
  }

  int PeekBits(size_t nbits) const {
    PIK_ASSERT(nbits <= 32);
    PIK_ASSERT(nbits + bit_pos_ <= 64);
    return (buf_ >> bit_pos_) & ((1ULL << nbits) - 1);
  }

  int ReadBits(size_t nbits) {
    FillBitBuffer();
    const int bits = PeekBits(nbits);
    bit_pos_ += nbits;
    return bits;
  }

  template <size_t N>
  int ReadFixedBits() {
    FillBitBuffer();
    const int bits = PeekFixedBits<N>();
    bit_pos_ += N;
    return bits;
  }

  uint16_t GetNextWord() { return static_cast<uint16_t>(ReadBits(16)); }

  void SkipBits(size_t skip) {
    // Satisfy from existing buffer
    const size_t consume_buffer = std::min(skip, 64 - bit_pos_);
    Advance(consume_buffer);
    PIK_ASSERT(bit_pos_ <= 64);
    skip -= consume_buffer;

    // Skip entire 32-bit words
    pos32_ += skip / 32;
    skip = skip % 32;

    FillBitBuffer();
    Advance(skip);
  }

  Status JumpToByteBoundary() {
    size_t rem = bit_pos_ % 8;
    if ((rem != 0) && (ReadBits(8 - rem) != 0)) {
      return PIK_FAILURE("Non-zero padding bits");
    }
    return true;
  }

  size_t BitsRead() const { return 32 * pos32_ + bit_pos_ - 64; }

  // Returns the (rounded up) number of bytes consumed so far.
  size_t Position() const { return (BitsRead() + 7) / 8; }

  bool Healthy() const { return Position() <= (len32_ << 2) + len_mod4_; }

 private:
  // *32 counters/pointers are in units of 4 bytes, or 32 bits.
  const uint32_t* const PIK_RESTRICT data32_;
  const size_t len32_;
  const size_t len_mod4_;
  uint64_t buf_;
  size_t pos32_;
  // Next bit == (buf_ >> bit_pos_) & 1. 64 => empty, 32 = upper 32 bits valid.
  size_t bit_pos_;
};

}  // namespace pik

#endif  // PIK_BIT_READER_H_
