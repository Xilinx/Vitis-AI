// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_ANS_ENCODE_H_
#define PIK_ANS_ENCODE_H_

// Library to encode the ANS population counts to the bit-stream and encode
// symbols based on the respective distributions.

#include <stddef.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pik/ans_params.h"
#include "pik/compiler_specific.h"
#include "pik/write_bits.h"

namespace pik {

#define USE_MULT_BY_RECIPROCAL

// precision must be equal to:  #bits(state_) + #bits(freq)
#define RECIPROCAL_PRECISION 42

// Data structure representing one element of the encoding table built
// from a distribution.
struct ANSEncSymbolInfo {
  uint16_t freq_;
  uint16_t start_;
#ifdef USE_MULT_BY_RECIPROCAL
  uint64_t ifreq_;
#endif
};

void BuildAndStoreANSEncodingData(const int* histogram, int alphabet_size,
                                  ANSEncSymbolInfo* info, size_t* storage_ix,
                                  uint8_t* storage);

struct ANSEncodingData {
  void BuildAndStore(const int* histogram, size_t histo_size,
                     size_t* storage_ix, uint8_t* storage) {
    ans_table.resize(histo_size);
    BuildAndStoreANSEncodingData(histogram, histo_size, ans_table.data(),
                                 storage_ix, storage);
  }

  void BuildAndStore(const uint32_t* histogram, size_t histo_size,
                     size_t* storage_ix, uint8_t* storage) {
    std::vector<int> counts(histo_size);
    for (int i = 0; i < histo_size; ++i) {
      counts[i] = histogram[i];
    }
    BuildAndStore(counts.data(), counts.size(), storage_ix, storage);
  }

  std::vector<ANSEncSymbolInfo> ans_table;
};

// Returns an estimate of the number of bits required to encode the given
// histogram (header bits plus data bits).
float ANSPopulationCost(const int* data, int alphabet_size, int total_count);

class ANSCoder {
 public:
  ANSCoder() : state_(ANS_SIGNATURE << 16) {}

  uint32_t PutSymbol(const ANSEncSymbolInfo t, uint8_t* nbits) {
    uint32_t bits = 0;
    *nbits = 0;
    if ((state_ >> (32 - ANS_LOG_TAB_SIZE)) >= t.freq_) {
      bits = state_ & 0xffff;
      state_ >>= 16;
      *nbits = 16;
    }
#ifdef USE_MULT_BY_RECIPROCAL
    // We use mult-by-reciprocal trick, but that requires 64b calc.
    const uint32_t v = (state_ * t.ifreq_) >> RECIPROCAL_PRECISION;
    const uint32_t offset = state_ - v * t.freq_ + t.start_;
    state_ = (v << ANS_LOG_TAB_SIZE) + offset;
#else
    state_ = ((state_ / t.freq_) << ANS_LOG_TAB_SIZE) + (state_ % t.freq_) +
             t.start_;
#endif
    return bits;
  }

  uint32_t GetState() const { return state_; }

 private:
  uint32_t state_;
};

// Symbol visitor that collects symbols and raw bits to be encoded.
class ANSSymbolWriter {
 public:
  ANSSymbolWriter(const std::vector<ANSEncodingData>& codes,
                  const std::vector<uint8_t>& context_map, size_t* storage_ix,
                  uint8_t* storage)
      : idx_(0),
        symbol_idx_(0),
        code_words_(2 * kANSBufferSize),
        symbols_(kANSBufferSize),
        codes_(codes),
        context_map_(context_map),
        storage_ix_(storage_ix),
        storage_(storage) {
    num_extra_bits_[0] = num_extra_bits_[1] = num_extra_bits_[2] = 0;
  }

  void VisitBits(size_t nbits, uint64_t bits, int c) {
    PIK_ASSERT(nbits <= 16);
    PIK_ASSERT(idx_ < code_words_.size());
    if (nbits > 0) {
      code_words_[idx_++] = (bits << 16) + nbits;
    }
    num_extra_bits_[c] += nbits;
  }

  void VisitSymbol(int symbol, int ctx) {
    PIK_ASSERT(ctx < context_map_.size());
    PIK_ASSERT(context_map_[ctx] < codes_.size());
    PIK_ASSERT(symbol < codes_[context_map_[ctx]].ans_table.size());
    PIK_ASSERT(idx_ < code_words_.size());
    code_words_[idx_++] = 0xffff;  // Placeholder, to be encoded later.
    symbols_[symbol_idx_++] = (ctx << 16) + symbol;
    if (symbol_idx_ == kANSBufferSize) {
      FlushToBitStream();
    }
  }

  size_t num_extra_bits() const {
    return num_extra_bits_[0] + num_extra_bits_[1] + num_extra_bits_[2];
  }
  size_t num_extra_bits(int c) const { return num_extra_bits_[c]; }

  void FlushToBitStream() {
    const int num_codewords = idx_;
    ANSCoder ans;
    int first_symbol = num_codewords;
    // Replace placeholder code words with actual bits by feeding symbols to the
    // ANS encoder in a reverse order.
    for (int i = num_codewords - 1; i >= 0; --i) {
      const uint32_t cw = code_words_[i];
      if ((cw & 0xffff) == 0xffff) {
        const uint32_t sym = symbols_[--symbol_idx_];
        const uint32_t context = sym >> 16;
        const uint8_t histo_idx = context_map_[context];
        const uint32_t symbol = sym & 0xffff;
        const ANSEncSymbolInfo info = codes_[histo_idx].ans_table[symbol];
        uint8_t nbits = 0;
        uint32_t bits = ans.PutSymbol(info, &nbits);
        code_words_[i] = (bits << 16) + nbits;
        first_symbol = i;
      }
    }
    for (int i = 0; i < num_codewords; ++i) {
      if (i == first_symbol) {
        const uint32_t state = ans.GetState();
        WriteBits(16, (state >> 16) & 0xffff, storage_ix_, storage_);
        WriteBits(16, state & 0xffff, storage_ix_, storage_);
      }
      const uint32_t cw = code_words_[i];
      const uint32_t nbits = cw & 0xffff;
      const uint32_t bits = cw >> 16;
      WriteBits(nbits, bits, storage_ix_, storage_);
    }
    idx_ = 0;
    PIK_ASSERT(symbol_idx_ == 0);
  }

 private:
  int idx_;
  int symbol_idx_;
  // Vector of (bits, nbits) pairs to be encoded.
  std::vector<uint32_t> code_words_;
  // Vector of (context, symbol) pairs to be encoded.
  std::vector<uint32_t> symbols_;
  const std::vector<ANSEncodingData>& codes_;
  const std::vector<uint8_t>& context_map_;
  size_t num_extra_bits_[3];
  size_t* storage_ix_;
  uint8_t* storage_;
};

}  // namespace pik

#endif  // PIK_ANS_ENCODE_H_
