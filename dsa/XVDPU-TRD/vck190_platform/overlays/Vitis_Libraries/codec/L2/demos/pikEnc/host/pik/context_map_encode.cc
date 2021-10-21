// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// Library to encode the context map.

#include "pik/context_map_encode.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "pik/bits.h"
#include "pik/compiler_specific.h"
#include "pik/huffman_encode.h"
#include "pik/status.h"
#include "pik/write_bits.h"

namespace pik {

namespace {

void StoreVarLenUint8(size_t n, size_t* storage_ix, uint8_t* storage) {
  if (n == 0) {
    WriteBits(1, 0, storage_ix, storage);
  } else {
    WriteBits(1, 1, storage_ix, storage);
    size_t nbits = FloorLog2Nonzero(static_cast<uint64_t>(n));
    WriteBits(3, nbits, storage_ix, storage);
    WriteBits(nbits, n - (1ULL << nbits), storage_ix, storage);
  }
}

size_t IndexOf(const std::vector<uint8_t>& v, uint8_t value) {
  size_t i = 0;
  for (; i < v.size(); ++i) {
    if (v[i] == value) return i;
  }
  return i;
}

void MoveToFront(std::vector<uint8_t>* v, size_t index) {
  uint8_t value = (*v)[index];
  for (size_t i = index; i != 0; --i) {
    (*v)[i] = (*v)[i - 1];
  }
  (*v)[0] = value;
}

std::vector<uint8_t> MoveToFrontTransform(const std::vector<uint8_t>& v) {
  if (v.empty()) return v;
  uint8_t max_value = *std::max_element(v.begin(), v.end());
  std::vector<uint8_t> mtf(max_value + 1);
  for (size_t i = 0; i <= max_value; ++i) mtf[i] = i;
  std::vector<uint8_t> result(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    size_t index = IndexOf(mtf, v[i]);
    PIK_ASSERT(index < mtf.size());
    result[i] = static_cast<uint8_t>(index);
    MoveToFront(&mtf, index);
  }
  return result;
}

// Finds runs of zeros in v_in and replaces them with a prefix code of the run
// length plus extra bits in *v_out and *extra_bits. Non-zero values in v_in are
// shifted by *max_length_prefix. Will not create prefix codes bigger than the
// initial value of *max_run_length_prefix. The prefix code of run length L is
// simply Log2Floor(L) and the number of extra bits is the same as the prefix
// code.
void RunLengthCodeZeros(const std::vector<uint8_t>& v_in,
                        uint32_t* max_run_length_prefix,
                        std::vector<uint32_t>* v_out,
                        std::vector<uint32_t>* extra_bits) {
  uint32_t max_reps = 0;
  for (size_t i = 0; i < v_in.size();) {
    for (; i < v_in.size() && v_in[i] != 0; ++i) {
    }
    uint32_t reps = 0;
    for (; i < v_in.size() && v_in[i] == 0; ++i) {
      ++reps;
    }
    max_reps = std::max(reps, max_reps);
  }
  uint32_t max_prefix = max_reps > 0 ? FloorLog2Nonzero(max_reps) : 0;
  max_prefix = std::min(max_prefix, *max_run_length_prefix);
  *max_run_length_prefix = max_prefix;
  for (size_t i = 0; i < v_in.size();) {
    if (v_in[i] != 0) {
      v_out->push_back(v_in[i] + *max_run_length_prefix);
      extra_bits->push_back(0);
      ++i;
    } else {
      uint32_t reps = 1;
      for (size_t k = i + 1; k < v_in.size() && v_in[k] == 0; ++k) {
        ++reps;
      }
      i += reps;
      while (reps != 0) {
        if (reps < (2u << max_prefix)) {
          uint32_t run_length_prefix = FloorLog2Nonzero(reps);
          v_out->push_back(run_length_prefix);
          extra_bits->push_back(reps - (1u << run_length_prefix));
          break;
        } else {
          v_out->push_back(max_prefix);
          extra_bits->push_back((1u << max_prefix) - 1u);
          reps -= (2u << max_prefix) - 1u;
        }
      }
    }
  }
}

}  // namespace

void EncodeContextMap(const std::vector<uint8_t>& context_map,
                      size_t num_histograms, size_t* storage_ix,
                      uint8_t* storage) {
  StoreVarLenUint8(num_histograms - 1, storage_ix, storage);

  if (num_histograms == 1) {
    return;
  }
  // Alphabet size is 256 + 16 = 272. (We can have 256 clusters and 16 run
  // length codes).
  static const int kAlphabetSize = 272;

  std::vector<uint8_t> transformed_symbols = MoveToFrontTransform(context_map);
  std::vector<uint32_t> rle_symbols;
  std::vector<uint32_t> extra_bits;
  uint32_t max_run_length_prefix = 6;
  RunLengthCodeZeros(transformed_symbols, &max_run_length_prefix, &rle_symbols,
                     &extra_bits);
  uint32_t symbol_histogram[kAlphabetSize];
  memset(symbol_histogram, 0, sizeof(symbol_histogram));
  for (size_t i = 0; i < rle_symbols.size(); ++i) {
    ++symbol_histogram[rle_symbols[i]];
  }
  bool use_rle = max_run_length_prefix > 0;
  WriteBits(1, use_rle, storage_ix, storage);
  if (use_rle) {
    WriteBits(4, max_run_length_prefix - 1, storage_ix, storage);
  }
  uint8_t bit_depths[kAlphabetSize];
  uint16_t bit_codes[kAlphabetSize];
  memset(bit_depths, 0, sizeof(bit_depths));
  memset(bit_codes, 0, sizeof(bit_codes));
  BuildAndStoreHuffmanTree(symbol_histogram,
                           num_histograms + max_run_length_prefix, bit_depths,
                           bit_codes, storage_ix, storage);
  for (size_t i = 0; i < rle_symbols.size(); ++i) {
    WriteBits(bit_depths[rle_symbols[i]], bit_codes[rle_symbols[i]], storage_ix,
              storage);
    if (rle_symbols[i] > 0 && rle_symbols[i] <= max_run_length_prefix) {
      WriteBits(rle_symbols[i], extra_bits[i], storage_ix, storage);
    }
  }
  WriteBits(1, 1, storage_ix, storage);  // use move-to-front
}

}  // namespace pik
