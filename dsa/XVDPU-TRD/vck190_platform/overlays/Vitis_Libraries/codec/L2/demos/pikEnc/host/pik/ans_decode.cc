// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/ans_decode.h"

#include <vector>

#include "pik/ans_common.h"
#include "pik/fast_log.h"

namespace pik {
namespace {

// Decodes a number in the range [0..65535], by reading 1 - 20 bits.
inline int DecodeVarLenUint16(BitReader* input) {
  if (input->ReadBits(1)) {
    int nbits = static_cast<int>(input->ReadBits(4));
    if (nbits == 0) {
      return 1;
    } else {
      return static_cast<int>(input->ReadBits(nbits)) + (1 << nbits);
    }
  }
  return 0;
}

Status ReadHistogram(int precision_bits, std::vector<int>* counts,
                     BitReader* input) {
  int simple_code = input->ReadBits(1);
  if (simple_code == 1) {
    int i;
    int symbols[2] = {0};
    int max_symbol = 0;
    const int num_symbols = input->ReadBits(1) + 1;
    for (i = 0; i < num_symbols; ++i) {
      symbols[i] = DecodeVarLenUint16(input);
      if (symbols[i] > max_symbol) max_symbol = symbols[i];
    }
    counts->resize(max_symbol + 1);
    if (num_symbols == 1) {
      (*counts)[symbols[0]] = 1 << precision_bits;
    } else {
      if (symbols[0] == symbols[1]) {  // corrupt data
        return false;
      }
      (*counts)[symbols[0]] = input->ReadBits(precision_bits);
      (*counts)[symbols[1]] = (1 << precision_bits) - (*counts)[symbols[0]];
    }
  } else {
    int is_flat = input->ReadBits(1);
    if (is_flat == 1) {
      int alphabet_size = input->ReadBits(precision_bits);
      if (alphabet_size == 0) {
        return PIK_FAILURE("Invalid alphabet size for flat histogram.");
      }
      *counts = CreateFlatHistogram(alphabet_size, 1 << precision_bits);
      return true;
    }
    int length = DecodeVarLenUint16(input) + 3;
    counts->resize(length);
    int total_count = 0;

    static const uint8_t huff[128][2] = {
        {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
        {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {5, 0},
        {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
        {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {6, 9},
        {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
        {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {5, 0},
        {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
        {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {7, 10},
        {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
        {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {5, 0},
        {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
        {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {6, 9},
        {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
        {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {5, 0},
        {2, 6}, {3, 7}, {3, 4}, {4, 1}, {2, 6}, {3, 8}, {3, 5}, {4, 3},
        {2, 6}, {3, 7}, {3, 4}, {4, 2}, {2, 6}, {3, 8}, {3, 5}, {7, 11},
    };
    std::vector<int> logcounts(counts->size());
    int omit_log = -1;
    int omit_pos = -1;
    // This array remembers which symbols have an RLE length.
    std::vector<int> same(counts->size(), 0);
    for (int i = 0; i < logcounts.size(); ++i) {
      input->FillBitBuffer();
      int idx = input->PeekFixedBits<7>();
      input->Advance(huff[idx][0]);
      logcounts[i] = huff[idx][1];
      // The RLE symbol.
      if (logcounts[i] == 11) {
        input->FillBitBuffer();
        int rle_length = input->PeekFixedBits<8>();
        input->Advance(8);
        same[i] = rle_length;
        i += rle_length - 2;
        continue;
      }
      if (logcounts[i] > omit_log) {
        omit_log = logcounts[i];
        omit_pos = i;
      }
    }
    // Invalid input, e.g. due to invalid usage of RLE.
    if (omit_pos < 0) return PIK_FAILURE("Invalid histogram.");
    int prev = 0;
    int numsame = 0;
    for (int i = 0; i < logcounts.size(); ++i) {
      if (same[i]) {
        // RLE sequence, let this loop output the same count for the next
        // iterations.
        numsame = same[i] - 1;
        prev = i > 0 ? (*counts)[i - 1] : 0;
      }
      if (numsame > 0) {
        (*counts)[i] = prev;
        numsame--;
      } else {
        int code = logcounts[i];
        if (i == omit_pos) {
          continue;
        } else if (code == 0) {
          continue;
        } else if (code == 1) {
          (*counts)[i] = 1;
        } else {
          int bitcount = GetPopulationCountPrecision(code - 1);
          (*counts)[i] = (1 << (code - 1)) +
                         (input->ReadBits(bitcount) << (code - 1 - bitcount));
        }
      }
      total_count += (*counts)[i];
    }
    (*counts)[omit_pos] = (1 << precision_bits) - total_count;
    if ((*counts)[omit_pos] <= 0) {
      // The histogram we've read sums to more than total_count (including at
      // least 1 for the omitted value).
      return PIK_FAILURE("Invalid histogram count.");
    }
  }
  return true;
}

}  // namespace

bool DecodeANSCodes(const size_t num_histograms, const size_t max_alphabet_size,
                    BitReader* in, ANSCode* result) {
  PIK_ASSERT(max_alphabet_size <= ANS_TAB_SIZE);
  result->map.resize((num_histograms << ANS_LOG_TAB_SIZE) + 1);
  result->info.resize(num_histograms << ANS_LOG_TAB_SIZE);
  for (size_t c = 0; c < num_histograms; ++c) {
    std::vector<int> counts;
    if (!ReadHistogram(ANS_LOG_TAB_SIZE, &counts, in)) {
      return PIK_FAILURE("Invalid histogram bitstream.");
    }
    if (counts.size() > max_alphabet_size) {
      return PIK_FAILURE("Alphabet size is too long.");
    }
    const size_t histo_offset = c << ANS_LOG_TAB_SIZE;
    uint32_t offset = 0;
    for (size_t i = 0, pos = 0; i < counts.size(); ++i) {
      const size_t symbol_idx = histo_offset + i;
      const uint32_t freq = counts[i];
#if PIK_BYTE_ORDER_LITTLE
      const uint32_t s32 = offset + (freq << 16);
      memcpy(&result->info[symbol_idx], &s32, sizeof(s32));
#else
      result->info[symbol_idx].offset = static_cast<uint16_t>(offset);
      result->info[symbol_idx].freq = static_cast<uint16_t>(freq);
#endif
      offset += counts[i];
      if (offset > ANS_TAB_SIZE) {
        return PIK_FAILURE("Invalid ANS histogram data.");
      }
      for (size_t j = 0; j < counts[i]; ++j, ++pos) {
        result->map[histo_offset + pos] = i;
      }
    }
  }
  return true;
}

}  // namespace pik
