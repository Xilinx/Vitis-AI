// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_HUFFMAN_DECODE_H_
#define PIK_HUFFMAN_DECODE_H_

// Library to decode the Huffman code lengths from the bit-stream and build a
// decoding table from them.

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <vector>

#include "pik/bit_reader.h"

namespace pik {

static const int kHuffmanMaxLength = 15;
static const int kHuffmanTableMask = 0xff;
static const int kHuffmanTableBits = 8;

typedef struct {
  uint8_t bits;   /* number of bits used for this symbol */
  uint16_t value; /* symbol value or table offset */
} HuffmanCode;

struct HuffmanDecodingData {
  HuffmanDecodingData() { table_.reserve(2048); }

  // Decodes the Huffman code lengths from the bit-stream and fills in the
  // pre-allocated table with the corresponding 2-level Huffman decoding table.
  // Returns false if the Huffman code lengths can not de decoded.
  bool ReadFromBitStream(BitReader* input);

  std::vector<HuffmanCode> table_;
};

struct HuffmanDecoder {
  // Decodes the next Huffman coded symbol from the bit-stream.
  int ReadSymbol(const HuffmanDecodingData& code, BitReader* input) {
    int nbits;
    const HuffmanCode* table = &code.table_[0];
    input->FillBitBuffer();
    table += input->PeekFixedBits<kHuffmanTableBits>();
    nbits = table->bits - kHuffmanTableBits;
    if (nbits > 0) {
      input->Advance(kHuffmanTableBits);
      table += table->value;
      table += input->PeekBits(nbits);
    }
    input->Advance(table->bits);
    return table->value;
  }
};

}  // namespace pik

#endif  // PIK_HUFFMAN_DECODE_H_
