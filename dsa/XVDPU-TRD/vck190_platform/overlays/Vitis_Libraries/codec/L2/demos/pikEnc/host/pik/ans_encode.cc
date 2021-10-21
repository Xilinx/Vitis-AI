// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/ans_encode.h"

#include <stdint.h>
#include <algorithm>
#include <vector>

#include "pik/ans_common.h"
#include "pik/fast_log.h"

namespace pik {

namespace {

static const int kMaxNumSymbolsForSmallCode = 4;

void ANSBuildInfoTable(const int* counts, int alphabet_size,
                       ANSEncSymbolInfo* info) {
  int total = 0;
  for (int s = 0; s < alphabet_size; ++s) {
    const uint32_t freq = counts[s];
    info[s].freq_ = counts[s];
    info[s].start_ = total;
    total += freq;
#ifdef USE_MULT_BY_RECIPROCAL
    if (freq != 0) {
      info[s].ifreq_ =
          ((1ull << RECIPROCAL_PRECISION) + info[s].freq_ - 1) / info[s].freq_;
    } else {
      info[s].ifreq_ = 1;  // shouldn't matter (symbol shoudln't occur), but...
    }
#endif
  }
}

int EstimateDataBits(const int* histogram, const int* counts, size_t len) {
  float sum = 0.0f;
  int total_histogram = 0;
  int total_counts = 0;
  for (int i = 0; i < len; ++i) {
    total_histogram += histogram[i];
    total_counts += counts[i];
    if (histogram[i] > 0) {
      PIK_ASSERT(counts[i] > 0);
      sum -= histogram[i] * FastLog2(counts[i]);
    }
  }
  if (total_histogram > 0) {
    PIK_ASSERT(total_counts == ANS_TAB_SIZE);
    const int log2_total_counts = ANS_LOG_TAB_SIZE;
    sum += total_histogram * log2_total_counts;
  }
  return static_cast<int>(sum + 1.0f);
}

int EstimateDataBitsFlat(const int* histogram, size_t len) {
  const float flat_bits = FastLog2(len);
  int total_histogram = 0;
  for (int i = 0; i < len; ++i) {
    total_histogram += histogram[i];
  }
  return static_cast<int>(total_histogram * flat_bits + 1.0);
}

// Static Huffman code for encoding logcounts. The last symbol is used as RLE
// sequence.
static const uint8_t kLogCountBitLengths[ANS_LOG_TAB_SIZE + 2] = {
    5, 4, 4, 4, 3, 3, 2, 3, 3, 6, 7, 7,
};
static const uint16_t kLogCountSymbols[ANS_LOG_TAB_SIZE + 2] = {
    15, 3, 11, 7, 2, 6, 0, 1, 5, 31, 63, 127,
};

// Returns the difference between largest count that can be represented and is
// smaller than "count" and smallest representable count larger than "count".
static int SmallestIncrement(int count) {
  int bits = Log2Floor(count);
  int drop_bits = bits - GetPopulationCountPrecision(bits);
  return (1 << drop_bits);
}

template <bool minimize_error_of_sum>
bool RebalanceHistogram(const float* targets, int max_symbol, int table_size,
                        int* omit_pos, int* counts) {
  int sum = 0;
  float sum_nonrounded = 0.0;
  int remainder_pos = 0;  // if all of them are handled in first loop
  int remainder_log = -1;
  for (int n = 0; n < max_symbol; ++n) {
    if (targets[n] > 0 && targets[n] < 1.0) {
      counts[n] = 1;
      sum_nonrounded += targets[n];
      sum += counts[n];
    }
  }
  const float discount_ratio =
      (table_size - sum) / (table_size - sum_nonrounded);
  PIK_ASSERT(discount_ratio > 0);
  PIK_ASSERT(discount_ratio <= 1.0);
  // Invariant for minimize_error_of_sum == true:
  // abs(sum - sum_nonrounded)
  //   <= SmallestIncrement(max(targets[])) + max_symbol
  for (int n = 0; n < max_symbol; ++n) {
    if (targets[n] >= 1.0) {
      sum_nonrounded += targets[n];
      counts[n] =
          static_cast<uint32_t>(targets[n] * discount_ratio);  // truncate
      if (counts[n] == 0) counts[n] = 1;
      if (counts[n] == table_size) counts[n] = table_size - 1;
      // Round the count to the closest nonzero multiple of SmallestIncrement
      // (when minimize_error_of_sum is false) or one of two closest so as to
      // keep the sum as close as possible to sum_nonrounded.
      int inc = SmallestIncrement(counts[n]);
      counts[n] -= counts[n] & (inc - 1);
      // TODO(robryk): Should we rescale targets[n]?
      const float target =
          minimize_error_of_sum ? (sum_nonrounded - sum) : targets[n];
      if (counts[n] == 0 ||
          (target > counts[n] + inc / 2 && counts[n] + inc < table_size)) {
        counts[n] += inc;
      }
      sum += counts[n];
      const int count_log = Log2FloorNonZero(counts[n]);
      if (count_log > remainder_log) {
        remainder_pos = n;
        remainder_log = count_log;
      }
    }
  }
  PIK_ASSERT(remainder_pos != -1);
  counts[remainder_pos] -= sum - table_size;
  *omit_pos = remainder_pos;
  return counts[remainder_pos] > 0;
}

bool NormalizeCounts(int* counts, int* omit_pos, const int length,
                     const int precision_bits, int* num_symbols, int* symbols) {
  const int table_size = 1 << precision_bits;  // target sum / table size
  uint64_t total = 0;
  int max_symbol = 0;
  int symbol_count = 0;
  for (int n = 0; n < length; ++n) {
    total += counts[n];
    if (counts[n] > 0) {
      if (symbol_count < kMaxNumSymbolsForSmallCode) {
        symbols[symbol_count] = n;
      }
      ++symbol_count;
      max_symbol = n + 1;
    }
  }
  *num_symbols = symbol_count;
  if (symbol_count == 0) {
    return true;
  }
  if (symbol_count == 1) {
    counts[symbols[0]] = table_size;
    return true;
  }
  if (symbol_count > table_size)
    return PIK_FAILURE("Too many entries in an ANS histogram");

  const float norm = 1.f * table_size / total;
  std::vector<float> targets(max_symbol);
  for (int n = 0; n < max_symbol; ++n) {
    targets[n] = norm * counts[n];
  }
  if (!RebalanceHistogram<false>(&targets[0], max_symbol, table_size, omit_pos,
                                 counts)) {
    // Use an alternative rebalancing mechanism if the one above failed
    // to create a histogram that is positive wherever the original one was.
    if (!RebalanceHistogram<true>(&targets[0], max_symbol, table_size, omit_pos,
                                  counts)) {
      return PIK_FAILURE("Logic error: couldn't rebalance a histogram");
    }
  }
  return true;
}

void StoreVarLenUint16(size_t n, size_t* storage_ix, uint8_t* storage) {
  if (n == 0) {
    WriteBits(1, 0, storage_ix, storage);
  } else {
    WriteBits(1, 1, storage_ix, storage);
    size_t nbits = Log2FloorNonZero(n);
    WriteBits(4, nbits, storage_ix, storage);
    WriteBits(nbits, n - (1ULL << nbits), storage_ix, storage);
  }
}

void EncodeCounts(const int* counts, const int alphabet_size,
                  const int omit_pos, const int num_symbols, const int* symbols,
                  size_t* storage_ix, uint8_t* storage) {
  if (num_symbols <= 2) {
    // Small tree marker to encode 1-2 symbols.
    WriteBits(1, 1, storage_ix, storage);
    if (num_symbols == 0) {
      WriteBits(1, 0, storage_ix, storage);
      StoreVarLenUint16(0, storage_ix, storage);
    } else {
      WriteBits(1, num_symbols - 1, storage_ix, storage);
      for (int i = 0; i < num_symbols; ++i) {
        StoreVarLenUint16(symbols[i], storage_ix, storage);
      }
    }
    if (num_symbols == 2) {
      WriteBits(ANS_LOG_TAB_SIZE, counts[symbols[0]], storage_ix, storage);
    }
  } else {
    // Mark non-small tree.
    WriteBits(1, 0, storage_ix, storage);
    // Mark non-flat histogram.
    WriteBits(1, 0, storage_ix, storage);

    // Precompute sequences for RLE encoding. Contains the number of identical
    // values starting at a given index. Only contains the value at the first
    // element of the series.
    std::vector<int> same(alphabet_size, 0);
    int last = 0;
    for (int i = 1; i < alphabet_size; i++) {
      // Store the sequence length once different symbol reached, or we're at
      // the end, or the length is longer than we can encode, or we are at
      // the omit_pos. We don't support including the omit_pos in an RLE
      // sequence because this value may use a different amoung of log2 bits
      // than standard, it is too complex to handle in the decoder.
      if (counts[i] != counts[last] || i + 1 == alphabet_size ||
          (i - last) >= 255 || i == omit_pos || i == omit_pos + 1) {
        same[last] = (i - last);
        last = i + 1;
      }
    }

    int length = 0;
    std::vector<int> logcounts(alphabet_size);
    int omit_log = 0;
    for (int i = 0; i < alphabet_size; ++i) {
      PIK_ASSERT(counts[i] <= ANS_TAB_SIZE);
      PIK_ASSERT(counts[i] >= 0);
      if (i == omit_pos) {
        length = i + 1;
      } else if (counts[i] > 0) {
        logcounts[i] = Log2FloorNonZero(counts[i]) + 1;
        length = i + 1;
        if (i < omit_pos) {
          omit_log = std::max(omit_log, logcounts[i] + 1);
        } else {
          omit_log = std::max(omit_log, logcounts[i]);
        }
      }
    }
    logcounts[omit_pos] = omit_log;
    // Since num_symbols >= 3, we know that length >= 3, therefore we encode
    // length - 3.
    StoreVarLenUint16(length - 3, storage_ix, storage);

    // The logcount values are encoded with a static Huffman code.
    static const size_t kMinReps = 4;
    size_t rep = ANS_LOG_TAB_SIZE + 1;
    for (int i = 0; i < length; ++i) {
      if (i > 0 && same[i - 1] > kMinReps) {
        // Encode the RLE symbol and skip the repeated ones.
        WriteBits(kLogCountBitLengths[rep], kLogCountSymbols[rep], storage_ix,
                  storage);
        WriteBits(8, same[i - 1], storage_ix, storage);
        i += same[i - 1] - 2;
        continue;
      }
      WriteBits(kLogCountBitLengths[logcounts[i]],
                kLogCountSymbols[logcounts[i]], storage_ix, storage);
    }
    for (int i = 0; i < length; ++i) {
      if (i > 0 && same[i - 1] > kMinReps) {
        // Skip symbols encoded by RLE.
        i += same[i - 1] - 2;
        continue;
      }
      if (logcounts[i] > 1 && i != omit_pos) {
        int bitcount = GetPopulationCountPrecision(logcounts[i] - 1);
        int drop_bits = logcounts[i] - 1 - bitcount;
        PIK_CHECK((counts[i] & ((1 << drop_bits) - 1)) == 0);
        WriteBits(bitcount, (counts[i] >> drop_bits) - (1 << bitcount),
                  storage_ix, storage);
      }
    }
  }
}

void EncodeFlatHistogram(const int alphabet_size, size_t* storage_ix,
                         uint8_t* storage) {
  // Mark non-small tree.
  WriteBits(1, 0, storage_ix, storage);
  // Mark uniform histogram.
  WriteBits(1, 1, storage_ix, storage);
  // Encode alphabet size.
  WriteBits(ANS_LOG_TAB_SIZE, alphabet_size, storage_ix, storage);
}

}  // namespace

void BuildAndStoreANSEncodingData(const int* histogram, int alphabet_size,
                                  ANSEncSymbolInfo* info, size_t* storage_ix,
                                  uint8_t* storage) {
  PIK_ASSERT(alphabet_size <= ANS_TAB_SIZE);
  int num_symbols;
  int symbols[kMaxNumSymbolsForSmallCode] = {0};
  std::vector<int> counts(histogram, histogram + alphabet_size);
  int omit_pos = 0;
  PIK_CHECK(NormalizeCounts(counts.data(), &omit_pos, alphabet_size,
                            ANS_LOG_TAB_SIZE, &num_symbols, symbols));
  ANSBuildInfoTable(counts.data(), alphabet_size, info);
  if (storage_ix != nullptr && storage != nullptr) {
    const int storage_ix0 = *storage_ix;
    EncodeCounts(counts.data(), alphabet_size, omit_pos, num_symbols, symbols,
                 storage_ix, storage);
    if (alphabet_size <= kMaxNumSymbolsForSmallCode) {
      return;
    }
    // Let's see if we can do better in terms of histogram size + data size.
    const int histo_bits = *storage_ix - storage_ix0;
    const int data_bits =
        EstimateDataBits(histogram, counts.data(), alphabet_size);
    const int histo_bits_flat = ANS_LOG_TAB_SIZE + 2;
    const int data_bits_flat = EstimateDataBitsFlat(histogram, alphabet_size);
    if (histo_bits_flat + data_bits_flat < histo_bits + data_bits) {
      counts = CreateFlatHistogram(alphabet_size, ANS_TAB_SIZE);
      ANSBuildInfoTable(counts.data(), alphabet_size, info);
      RewindStorage(storage_ix0, storage_ix, storage);
      EncodeFlatHistogram(alphabet_size, storage_ix, storage);
    }
  }
}

float ANSPopulationCost(const int* data, int alphabet_size, int total_count) {
  if (total_count == 0) {
    return 7;
  }

  float entropy_bits = total_count * ANS_LOG_TAB_SIZE;
  int histogram_bits = 0;
  int count = 0;
  int length = 0;
  if (total_count > ANS_TAB_SIZE) {
    uint64_t total = total_count;
    for (int i = 0; i < alphabet_size; ++i) {
      if (data[i] > 0) {
        ++count;
        length = i;
      }
    }
    if (count == 1) {
      return 7;
    }
    ++length;
    const uint64_t max0 = (total * length) >> ANS_LOG_TAB_SIZE;
    const uint64_t max1 = (max0 * length) >> ANS_LOG_TAB_SIZE;
    const uint32_t min_base = (total + max0 + max1) >> ANS_LOG_TAB_SIZE;
    total += min_base * count;
    const int64_t kFixBits = 32;
    const int64_t kFixOne = 1LL << kFixBits;
    const int64_t kDescaleBits = kFixBits - ANS_LOG_TAB_SIZE;
    const int64_t kDescaleOne = 1LL << kDescaleBits;
    const int64_t kDescaleMask = kDescaleOne - 1;
    const uint32_t mult = kFixOne / total;
    const uint32_t error = kFixOne % total;
    uint32_t cumul = error;
    if (error < kDescaleOne) {
      cumul += (kDescaleOne - error) >> 1;
    }
    if (data[0] > 0) {
      uint64_t c = (uint64_t)(data[0] + min_base) * mult + cumul;
      float log2count = FastLog2(c >> kDescaleBits);
      entropy_bits -= data[0] * log2count;
      cumul = c & kDescaleMask;
    }
    for (int i = 1; i < length; ++i) {
      if (data[i] > 0) {
        uint64_t c = (uint64_t)(data[i] + min_base) * mult + cumul;
        float log2count = FastLog2(c >> kDescaleBits);
        int log2floor = static_cast<int>(log2count);
        entropy_bits -= data[i] * log2count;
        histogram_bits += log2floor;
        histogram_bits += kLogCountBitLengths[log2floor + 1];
        cumul = c & kDescaleMask;
      } else {
        histogram_bits += kLogCountBitLengths[0];
      }
    }
  } else {
    float log2norm = ANS_LOG_TAB_SIZE - FastLog2(total_count);
    if (data[0] > 0) {
      float log2count = FastLog2(data[0]) + log2norm;
      entropy_bits -= data[0] * log2count;
      length = 0;
      ++count;
    }
    for (int i = 1; i < alphabet_size; ++i) {
      if (data[i] > 0) {
        float log2count = FastLog2(data[i]) + log2norm;
        int log2floor = static_cast<int>(log2count);
        entropy_bits -= data[i] * log2count;
        if (log2floor >= ANS_LOG_TAB_SIZE) {
          log2floor = ANS_LOG_TAB_SIZE - 1;
        }
        histogram_bits += GetPopulationCountPrecision(log2floor);
        histogram_bits += kLogCountBitLengths[log2floor + 1];
        length = i;
        ++count;
      } else {
        histogram_bits += kLogCountBitLengths[0];
      }
    }
    ++length;
  }

  if (count == 1) {
    return 7;
  }

  if (count == 2) {
    return static_cast<int>(entropy_bits) + 1 + 12 + ANS_LOG_TAB_SIZE;
  }

  int max_bits = 1 + Log2Floor(alphabet_size - 1);
  histogram_bits += max_bits;

  return histogram_bits + static_cast<int>(entropy_bits) + 1;
}

}  // namespace pik
