// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/context_map_decode.h"

#include <cstring>
#include <vector>

#include "pik/huffman_decode.h"
#include "pik/status.h"

namespace pik {

namespace {

void MoveToFront(uint8_t* v, uint8_t index) {
  uint8_t value = v[index];
  uint8_t i = index;
  for (; i; --i) v[i] = v[i - 1];
  v[0] = value;
}

void InverseMoveToFrontTransform(uint8_t* v, int v_len) {
  uint8_t mtf[256];
  int i;
  for (i = 0; i < 256; ++i) {
    mtf[i] = static_cast<uint8_t>(i);
  }
  for (i = 0; i < v_len; ++i) {
    uint8_t index = v[i];
    v[i] = mtf[index];
    if (index) MoveToFront(mtf, index);
  }
}

// Decodes a number in the range [0..255], by reading 1 - 11 bits.
inline int DecodeVarLenUint8(BitReader* input) {
  if (input->ReadBits(1)) {
    int nbits = static_cast<int>(input->ReadBits(3));
    if (nbits == 0) {
      return 1;
    } else {
      return static_cast<int>(input->ReadBits(nbits)) + (1 << nbits);
    }
  }
  return 0;
}

bool VerifyContextMap(const std::vector<uint8_t>& context_map,
                      const size_t num_htrees) {
  std::vector<bool> have_htree(num_htrees);
  int num_found = 0;
  for (int i = 0; i < context_map.size(); ++i) {
    const int htree = context_map[i];
    if (htree >= num_htrees) {
      return PIK_FAILURE("Invalid histogram index in context map.");
    }
    if (!have_htree[htree]) {
      have_htree[htree] = true;
      ++num_found;
    }
  }
  if (num_found != num_htrees) {
    return PIK_FAILURE("Incomplete context map.");
  }
  return true;
}

}  // namespace

bool DecodeContextMap(std::vector<uint8_t>* context_map, size_t* num_htrees,
                      BitReader* input) {
  *num_htrees = DecodeVarLenUint8(input) + 1;

  if (*num_htrees <= 1) {
    memset(&(*context_map)[0], 0, context_map->size());
    return true;
  }

  int max_run_length_prefix = 0;
  int use_rle_for_zeros = input->ReadBits(1);
  if (use_rle_for_zeros) {
    max_run_length_prefix = input->ReadBits(4) + 1;
  }
  HuffmanDecodingData entropy;
  if (!entropy.ReadFromBitStream(input)) {
    return PIK_FAILURE("Invalid histogram data.");
  }
  HuffmanDecoder decoder;
  int i;
  for (i = 0; i < context_map->size();) {
    int code;
    code = decoder.ReadSymbol(entropy, input);
    if (code == 0) {
      (*context_map)[i] = 0;
      ++i;
    } else if (code <= max_run_length_prefix) {
      int reps = 1 + (1 << code) + input->ReadBits(code);
      while (--reps) {
        if (i >= context_map->size()) {
          return PIK_FAILURE("Invalid context map data.");
        }
        (*context_map)[i] = 0;
        ++i;
      }
    } else {
      (*context_map)[i] = static_cast<uint8_t>(code - max_run_length_prefix);
      ++i;
    }
  }
  if (input->ReadBits(1)) {
    InverseMoveToFrontTransform(&(*context_map)[0], context_map->size());
  }
  return VerifyContextMap(*context_map, *num_htrees);
}

}  // namespace pik
