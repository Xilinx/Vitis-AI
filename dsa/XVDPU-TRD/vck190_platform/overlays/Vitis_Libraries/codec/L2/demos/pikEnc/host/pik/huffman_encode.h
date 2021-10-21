// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_HUFFMAN_ENCODE_H_
#define PIK_HUFFMAN_ENCODE_H_

#include <stdint.h>
#include <cstddef>

namespace pik {

void BuildAndStoreHuffmanTree(const uint32_t* histogram, const size_t length,
                              uint8_t* depth, uint16_t* bits,
                              size_t* storage_ix, uint8_t* storage);

void BuildHuffmanTreeAndCountBits(const uint32_t* histogram,
                                  const size_t length, size_t* histogram_bits,
                                  size_t* data_bits);
}  // namespace pik

#endif  // PIK_HUFFMAN_ENCODE_H_
