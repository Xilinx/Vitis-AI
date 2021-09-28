// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/huffman_encode.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "pik/fast_log.h"
#include "pik/status.h"
#include "pik/write_bits.h"

namespace pik {

namespace {

static const int kCodeLengthCodes = 18;

// A node of a Huffman tree.
struct HuffmanTree {
  HuffmanTree(uint32_t count, int16_t left, int16_t right)
      : total_count(count), index_left(left), index_right_or_value(right) {}
  uint32_t total_count;
  int16_t index_left;
  int16_t index_right_or_value;
};

// Sort the root nodes, least popular first.
inline bool SortHuffmanTree(const HuffmanTree& v0, const HuffmanTree& v1) {
  return v0.total_count < v1.total_count;
}

void SetDepth(const HuffmanTree& p, HuffmanTree* pool, uint8_t* depth,
              uint8_t level) {
  if (p.index_left >= 0) {
    ++level;
    SetDepth(pool[p.index_left], pool, depth, level);
    SetDepth(pool[p.index_right_or_value], pool, depth, level);
  } else {
    depth[p.index_right_or_value] = level;
  }
}

// This function will create a Huffman tree.
//
// The (data,length) contains the population counts.
// The tree_limit is the maximum bit depth of the Huffman codes.
//
// The depth contains the tree, i.e., how many bits are used for
// the symbol.
//
// The catch here is that the tree cannot be arbitrarily deep.
// The format specifies a maximum depth of 15 bits for "code trees"
// and 7 bits for "code length code trees."
//
// count_limit is the value that is to be faked as the minimum value
// and this minimum value is raised until the tree matches the
// maximum length requirement.
//
// This algorithm is not of excellent performance for very long data blocks,
// especially when population counts are longer than 2**tree_limit, but
// we are not planning to use this with extremely long blocks.
//
// See http://en.wikipedia.org/wiki/Huffman_coding
void CreateHuffmanTree(const uint32_t* data, const size_t length,
                       const int tree_limit, uint8_t* depth) {
  // For block sizes below 64 kB, we never need to do a second iteration
  // of this loop. Probably all of our block sizes will be smaller than
  // that, so this loop is mostly of academic interest. If we actually
  // would need this, we would be better off with the Katajainen algorithm.
  for (uint32_t count_limit = 1;; count_limit = 2 * count_limit + 1) {
    std::vector<HuffmanTree> tree;
    tree.reserve(2 * length + 1);

    for (size_t i = length; i != 0;) {
      --i;
      if (data[i]) {
        const uint32_t count = std::max(data[i], count_limit);
        tree.push_back(HuffmanTree(count, -1, static_cast<int16_t>(i)));
      }
    }

    const size_t n = tree.size();
    if (n == 1) {
      depth[tree[0].index_right_or_value] = 1;  // Only one element.
      break;
    }

    std::stable_sort(tree.begin(), tree.end(), SortHuffmanTree);

    // The nodes are:
    // [0, n): the sorted leaf nodes that we start with.
    // [n]: we add a sentinel here.
    // [n + 1, 2n): new parent nodes are added here, starting from
    //              (n+1). These are naturally in ascending order.
    // [2n]: we add a sentinel at the end as well.
    // There will be (2n+1) elements at the end.
    const HuffmanTree sentinel(std::numeric_limits<uint32_t>::max(), -1, -1);
    tree.push_back(sentinel);
    tree.push_back(sentinel);

    size_t i = 0;      // Points to the next leaf node.
    size_t j = n + 1;  // Points to the next non-leaf node.
    for (size_t k = n - 1; k != 0; --k) {
      size_t left, right;
      if (tree[i].total_count <= tree[j].total_count) {
        left = i;
        ++i;
      } else {
        left = j;
        ++j;
      }
      if (tree[i].total_count <= tree[j].total_count) {
        right = i;
        ++i;
      } else {
        right = j;
        ++j;
      }

      // The sentinel node becomes the parent node.
      size_t j_end = tree.size() - 1;
      tree[j_end].total_count =
          tree[left].total_count + tree[right].total_count;
      tree[j_end].index_left = static_cast<int16_t>(left);
      tree[j_end].index_right_or_value = static_cast<int16_t>(right);

      // Add back the last sentinel node.
      tree.push_back(sentinel);
    }
    PIK_ASSERT(tree.size() == 2 * n + 1);
    SetDepth(tree[2 * n - 1], &tree[0], depth, 0);

    // We need to pack the Huffman tree in tree_limit bits.
    // If this was not successful, add fake entities to the lowest values
    // and retry.
    if (*std::max_element(&depth[0], &depth[length]) <= tree_limit) {
      break;
    }
  }
}

void Reverse(uint8_t* v, size_t start, size_t end) {
  --end;
  while (start < end) {
    uint8_t tmp = v[start];
    v[start] = v[end];
    v[end] = tmp;
    ++start;
    --end;
  }
}

void WriteHuffmanTreeRepetitions(const uint8_t previous_value,
                                 const uint8_t value, size_t repetitions,
                                 size_t* tree_size, uint8_t* tree,
                                 uint8_t* extra_bits_data) {
  PIK_ASSERT(repetitions > 0);
  if (previous_value != value) {
    tree[*tree_size] = value;
    extra_bits_data[*tree_size] = 0;
    ++(*tree_size);
    --repetitions;
  }
  if (repetitions == 7) {
    tree[*tree_size] = value;
    extra_bits_data[*tree_size] = 0;
    ++(*tree_size);
    --repetitions;
  }
  if (repetitions < 3) {
    for (size_t i = 0; i < repetitions; ++i) {
      tree[*tree_size] = value;
      extra_bits_data[*tree_size] = 0;
      ++(*tree_size);
    }
  } else {
    repetitions -= 3;
    size_t start = *tree_size;
    while (true) {
      tree[*tree_size] = 16;
      extra_bits_data[*tree_size] = repetitions & 0x3;
      ++(*tree_size);
      repetitions >>= 2;
      if (repetitions == 0) {
        break;
      }
      --repetitions;
    }
    Reverse(tree, start, *tree_size);
    Reverse(extra_bits_data, start, *tree_size);
  }
}

void WriteHuffmanTreeRepetitionsZeros(size_t repetitions, size_t* tree_size,
                                      uint8_t* tree, uint8_t* extra_bits_data) {
  if (repetitions == 11) {
    tree[*tree_size] = 0;
    extra_bits_data[*tree_size] = 0;
    ++(*tree_size);
    --repetitions;
  }
  if (repetitions < 3) {
    for (size_t i = 0; i < repetitions; ++i) {
      tree[*tree_size] = 0;
      extra_bits_data[*tree_size] = 0;
      ++(*tree_size);
    }
  } else {
    repetitions -= 3;
    size_t start = *tree_size;
    while (true) {
      tree[*tree_size] = 17;
      extra_bits_data[*tree_size] = repetitions & 0x7;
      ++(*tree_size);
      repetitions >>= 3;
      if (repetitions == 0) {
        break;
      }
      --repetitions;
    }
    Reverse(tree, start, *tree_size);
    Reverse(extra_bits_data, start, *tree_size);
  }
}

static void DecideOverRleUse(const uint8_t* depth, const size_t length,
                             bool* use_rle_for_non_zero,
                             bool* use_rle_for_zero) {
  size_t total_reps_zero = 0;
  size_t total_reps_non_zero = 0;
  size_t count_reps_zero = 1;
  size_t count_reps_non_zero = 1;
  for (size_t i = 0; i < length;) {
    const uint8_t value = depth[i];
    size_t reps = 1;
    for (size_t k = i + 1; k < length && depth[k] == value; ++k) {
      ++reps;
    }
    if (reps >= 3 && value == 0) {
      total_reps_zero += reps;
      ++count_reps_zero;
    }
    if (reps >= 4 && value != 0) {
      total_reps_non_zero += reps;
      ++count_reps_non_zero;
    }
    i += reps;
  }
  *use_rle_for_non_zero = total_reps_non_zero > count_reps_non_zero * 2;
  *use_rle_for_zero = total_reps_zero > count_reps_zero * 2;
}

// Write a Huffman tree from bit depths into the bitstream representation
// of a Huffman tree. The generated Huffman tree is to be compressed once
// more using a Huffman tree
void WriteHuffmanTree(const uint8_t* depth, size_t length, size_t* tree_size,
                      uint8_t* tree, uint8_t* extra_bits_data) {
  uint8_t previous_value = 8;

  // Throw away trailing zeros.
  size_t new_length = length;
  for (size_t i = 0; i < length; ++i) {
    if (depth[length - i - 1] == 0) {
      --new_length;
    } else {
      break;
    }
  }

  // First gather statistics on if it is a good idea to do rle.
  bool use_rle_for_non_zero = false;
  bool use_rle_for_zero = false;
  if (length > 50) {
    // Find rle coding for longer codes.
    // Shorter codes seem not to benefit from rle.
    DecideOverRleUse(depth, new_length, &use_rle_for_non_zero,
                     &use_rle_for_zero);
  }

  // Actual rle coding.
  for (size_t i = 0; i < new_length;) {
    const uint8_t value = depth[i];
    size_t reps = 1;
    if ((value != 0 && use_rle_for_non_zero) ||
        (value == 0 && use_rle_for_zero)) {
      for (size_t k = i + 1; k < new_length && depth[k] == value; ++k) {
        ++reps;
      }
    }
    if (value == 0) {
      WriteHuffmanTreeRepetitionsZeros(reps, tree_size, tree, extra_bits_data);
    } else {
      WriteHuffmanTreeRepetitions(previous_value, value, reps, tree_size, tree,
                                  extra_bits_data);
      previous_value = value;
    }
    i += reps;
  }
}

uint16_t ReverseBits(int num_bits, uint16_t bits) {
  static const size_t kLut[16] = {// Pre-reversed 4-bit values.
                                  0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
                                  0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf};
  size_t retval = kLut[bits & 0xf];
  for (int i = 4; i < num_bits; i += 4) {
    retval <<= 4;
    bits = static_cast<uint16_t>(bits >> 4);
    retval |= kLut[bits & 0xf];
  }
  retval >>= (-num_bits & 0x3);
  return static_cast<uint16_t>(retval);
}

// Get the actual bit values for a tree of bit depths.
void ConvertBitDepthsToSymbols(const uint8_t* depth, size_t len,
                               uint16_t* bits) {
  // In Brotli, all bit depths are [1..15]
  // 0 bit depth means that the symbol does not exist.
  const int kMaxBits = 16;  // 0..15 are values for bits
  uint16_t bl_count[kMaxBits] = {0};
  {
    for (size_t i = 0; i < len; ++i) {
      ++bl_count[depth[i]];
    }
    bl_count[0] = 0;
  }
  uint16_t next_code[kMaxBits];
  next_code[0] = 0;
  {
    int code = 0;
    for (int bits = 1; bits < kMaxBits; ++bits) {
      code = (code + bl_count[bits - 1]) << 1;
      next_code[bits] = static_cast<uint16_t>(code);
    }
  }
  for (size_t i = 0; i < len; ++i) {
    if (depth[i]) {
      bits[i] = ReverseBits(depth[i], next_code[depth[i]]++);
    }
  }
}

template <class BitVisitor>
void StoreHuffmanTreeOfHuffmanTreeToBitMask(const int num_codes,
                                            const uint8_t* code_length_bitdepth,
                                            BitVisitor* bit_visitor) {
  static const uint8_t kStorageOrder[kCodeLengthCodes] = {
      1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  // The bit lengths of the Huffman code over the code length alphabet
  // are compressed with the following static Huffman code:
  //   Symbol   Code
  //   ------   ----
  //   0          00
  //   1        1110
  //   2         110
  //   3          01
  //   4          10
  //   5        1111
  static const uint8_t kHuffmanBitLengthHuffmanCodeSymbols[6] = {0, 7, 3,
                                                                 2, 1, 15};
  static const uint8_t kHuffmanBitLengthHuffmanCodeBitLengths[6] = {2, 4, 3,
                                                                    2, 2, 4};

  // Throw away trailing zeros:
  size_t codes_to_store = kCodeLengthCodes;
  if (num_codes > 1) {
    for (; codes_to_store > 0; --codes_to_store) {
      if (code_length_bitdepth[kStorageOrder[codes_to_store - 1]] != 0) {
        break;
      }
    }
  }
  size_t skip_some = 0;  // skips none.
  if (code_length_bitdepth[kStorageOrder[0]] == 0 &&
      code_length_bitdepth[kStorageOrder[1]] == 0) {
    skip_some = 2;  // skips two.
    if (code_length_bitdepth[kStorageOrder[2]] == 0) {
      skip_some = 3;  // skips three.
    }
  }
  bit_visitor->VisitBits(2, skip_some);
  for (size_t i = skip_some; i < codes_to_store; ++i) {
    size_t l = code_length_bitdepth[kStorageOrder[i]];
    bit_visitor->VisitBits(kHuffmanBitLengthHuffmanCodeBitLengths[l],
                           kHuffmanBitLengthHuffmanCodeSymbols[l]);
  }
}

template <class BitVisitor>
void StoreHuffmanTreeToBitMask(const size_t huffman_tree_size,
                               const uint8_t* huffman_tree,
                               const uint8_t* huffman_tree_extra_bits,
                               const uint8_t* code_length_bitdepth,
                               const uint16_t* code_length_bitdepth_symbols,
                               BitVisitor* bit_visitor) {
  for (size_t i = 0; i < huffman_tree_size; ++i) {
    size_t ix = huffman_tree[i];
    bit_visitor->VisitBits(code_length_bitdepth[ix],
                           code_length_bitdepth_symbols[ix]);
    // Extra bits
    switch (ix) {
      case 16:
        bit_visitor->VisitBits(2, huffman_tree_extra_bits[i]);
        break;
      case 17:
        bit_visitor->VisitBits(3, huffman_tree_extra_bits[i]);
        break;
    }
  }
}

template <class BitVisitor>
void StoreVarLenUint16(size_t n, BitVisitor* visitor) {
  if (n == 0) {
    visitor->VisitBits(1, 0);
  } else {
    visitor->VisitBits(1, 1);
    size_t nbits = Log2FloorNonZero(n);
    visitor->VisitBits(4, nbits);
    visitor->VisitBits(nbits, n - (1ULL << nbits));
  }
}

template <class BitVisitor>
void StoreSimpleHuffmanTree(const uint8_t* depths, size_t symbols[4],
                            size_t num_symbols, BitVisitor* bit_visitor) {
  // value of 1 indicates a simple Huffman code
  bit_visitor->VisitBits(2, 1);
  bit_visitor->VisitBits(2, num_symbols - 1);  // NSYM - 1

  // Sort
  for (size_t i = 0; i < num_symbols; i++) {
    for (size_t j = i + 1; j < num_symbols; j++) {
      if (depths[symbols[j]] < depths[symbols[i]]) {
        std::swap(symbols[j], symbols[i]);
      }
    }
  }

  for (size_t i = 0; i < num_symbols; ++i) {
    StoreVarLenUint16(symbols[i], bit_visitor);
  }
  if (num_symbols == 4) {
    // tree-select
    bit_visitor->VisitBits(1, depths[symbols[0]] == 1 ? 1 : 0);
  }
}

// num = alphabet size
// depths = symbol depths
template <class BitVisitor>
void StoreHuffmanTree(const uint8_t* depths, size_t num,
                      BitVisitor* bit_visitor) {
  // Write the Huffman tree into the compact representation.
  std::vector<uint8_t> huffman_tree(num);
  std::vector<uint8_t> huffman_tree_extra_bits(num);
  size_t huffman_tree_size = 0;
  WriteHuffmanTree(depths, num, &huffman_tree_size, &huffman_tree[0],
                   &huffman_tree_extra_bits[0]);

  // Calculate the statistics of the Huffman tree in the compact representation.
  uint32_t huffman_tree_histogram[kCodeLengthCodes] = {0};
  for (size_t i = 0; i < huffman_tree_size; ++i) {
    ++huffman_tree_histogram[huffman_tree[i]];
  }

  int num_codes = 0;
  int code = 0;
  for (int i = 0; i < kCodeLengthCodes; ++i) {
    if (huffman_tree_histogram[i]) {
      if (num_codes == 0) {
        code = i;
        num_codes = 1;
      } else if (num_codes == 1) {
        num_codes = 2;
        break;
      }
    }
  }

  // Calculate another Huffman tree to use for compressing both the
  // earlier Huffman tree with.
  uint8_t code_length_bitdepth[kCodeLengthCodes] = {0};
  uint16_t code_length_bitdepth_symbols[kCodeLengthCodes] = {0};
  CreateHuffmanTree(&huffman_tree_histogram[0], kCodeLengthCodes, 5,
                    &code_length_bitdepth[0]);
  ConvertBitDepthsToSymbols(code_length_bitdepth, kCodeLengthCodes,
                            &code_length_bitdepth_symbols[0]);

  // Now, we have all the data, let's start storing it
  StoreHuffmanTreeOfHuffmanTreeToBitMask(num_codes, code_length_bitdepth,
                                         bit_visitor);

  if (num_codes == 1) {
    code_length_bitdepth[code] = 0;
  }

  // Store the real huffman tree now.
  StoreHuffmanTreeToBitMask(
      huffman_tree_size, &huffman_tree[0], &huffman_tree_extra_bits[0],
      &code_length_bitdepth[0], code_length_bitdepth_symbols, bit_visitor);
}

template <class BitVisitor>
void BuildAndVisitHuffmanTree(const uint32_t* histogram, const size_t length,
                              uint8_t* depth, uint16_t* bits,
                              BitVisitor* bit_visitor) {
  size_t count = 0;
  size_t s4[4] = {0};
  for (size_t i = 0; i < length; i++) {
    if (histogram[i]) {
      if (count < 4) {
        s4[count] = i;
      } else if (count > 4) {
        break;
      }
      count++;
    }
  }

  if (count <= 1) {
    bit_visitor->VisitBits(4, 1);
    StoreVarLenUint16(s4[0], bit_visitor);
    return;
  }

  CreateHuffmanTree(histogram, length, 15, depth);
  if (bits != nullptr) {
    ConvertBitDepthsToSymbols(depth, length, bits);
  }

  if (count <= 4) {
    StoreSimpleHuffmanTree(depth, s4, count, bit_visitor);
  } else {
    StoreHuffmanTree(depth, length, bit_visitor);
  }
}

}  // namespace

void BuildAndStoreHuffmanTree(const uint32_t* histogram, const size_t length,
                              uint8_t* depth, uint16_t* bits,
                              size_t* storage_ix, uint8_t* storage) {
  BitWriter bit_writer(storage_ix, storage);
  BuildAndVisitHuffmanTree(histogram, length, depth, bits, &bit_writer);
}

void BuildHuffmanTreeAndCountBits(const uint32_t* histogram,
                                  const size_t length, size_t* histogram_bits,
                                  size_t* data_bits) {
  BitCounter bit_counter;
  std::vector<uint8_t> depths(length);
  BuildAndVisitHuffmanTree(histogram, length, depths.data(), nullptr,
                           &bit_counter);
  *histogram_bits = bit_counter.num_bits;
  *data_bits = 0;
  for (int i = 0; i < length; ++i) {
    *data_bits += histogram[i] * depths[i];
  }
}

}  // namespace pik
