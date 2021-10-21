// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_ANS_DECODE_H_
#define PIK_ANS_DECODE_H_

// Library to decode the ANS population counts from the bit-stream and build a
// decoding table from them.

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "pik/ans_params.h"
#include "pik/bit_reader.h"
#include "pik/byte_order.h"

namespace pik {
struct ANSSymbolInfo {
  uint16_t offset;
  uint16_t freq;
};

struct ANSCode {
  std::vector<uint16_t> map;
  // indexed by (entropy_code_id << ANS_LOG_TAB_SIZE) + symbol.
  std::vector<ANSSymbolInfo> info;
};

bool DecodeANSCodes(const size_t num_histograms, const size_t max_alphabet_size,
                    BitReader* in, ANSCode* result);

class ANSSymbolReader {
 public:
  ANSSymbolReader(const ANSCode* code) : code_(code) {}

  PIK_INLINE int ReadSymbol(const int histo_idx, BitReader* PIK_RESTRICT br) {
    if (PIK_UNLIKELY(symbols_left_ == 0)) {
      state_ = br->ReadBits(16);
      state_ = (state_ << 16) | br->ReadBits(16);
      br->FillBitBuffer();
      symbols_left_ = kANSBufferSize;
    }
    const uint32_t res = state_ & (ANS_TAB_SIZE - 1);
    const int histo_offset = histo_idx << ANS_LOG_TAB_SIZE;

#if PIK_BYTE_ORDER_LITTLE
    uint32_t s32;
    memcpy(&s32, &code_->map[histo_offset + res], sizeof(s32));
    const size_t symbol = s32 & 0xFFFF;

    memcpy(&s32, &code_->info[histo_offset + symbol], sizeof(s32));
    const uint32_t offset = s32 & 0xFFFF;
    const uint32_t freq = s32 >> 16;
    state_ = freq * (state_ >> ANS_LOG_TAB_SIZE) + res - offset;
#else
    const uint16_t symbol = code_->map[histo_offset + res];
    const ANSCode::ANSSymbolInfo s = code_->info[histo_offset + symbol];
    state_ = s.freq * (state_ >> ANS_LOG_TAB_SIZE) + res - s.offset;
#endif
    --symbols_left_;
    if (PIK_UNLIKELY(state_ < (1u << 16))) {
      state_ = (state_ << 16) | br->PeekFixedBits<16>();
      br->Advance(16);
    }
    return symbol;
  }

  bool CheckANSFinalState() { return state_ == (ANS_SIGNATURE << 16); }

 private:
  size_t symbols_left_ = 0;
  uint32_t state_ = ANS_SIGNATURE << 16;
  const ANSCode* code_;
};

}  // namespace pik

#endif  // PIK_ANS_DECODE_H_
