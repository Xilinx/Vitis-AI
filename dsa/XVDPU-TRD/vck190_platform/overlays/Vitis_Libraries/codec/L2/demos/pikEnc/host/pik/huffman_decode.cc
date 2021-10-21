// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/huffman_decode.h"

#include <stdint.h>
#include <cstring>
#include <vector>

#include "pik/compiler_specific.h"

namespace pik {

static const int kCodeLengthCodes = 18;
static const uint8_t kCodeLengthCodeOrder[kCodeLengthCodes] = {
    1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15,
};
static const uint8_t kDefaultCodeLength = 8;
static const uint8_t kCodeLengthRepeatCode = 16;

/* Returns reverse(reverse(key, len) + 1, len), where reverse(key, len) is the
   bit-wise reversal of the len least significant bits of key. */
static PIK_INLINE int GetNextKey(int key, int len) {
  int step = 1 << (len - 1);
  while (key & step) {
    step >>= 1;
  }
  return (key & (step - 1)) + step;
}

/* Stores code in table[0], table[step], table[2*step], ..., table[end] */
/* Assumes that end is an integer multiple of step */
static PIK_INLINE void ReplicateValue(HuffmanCode* table, int step, int end,
                                      HuffmanCode code) {
  do {
    end -= step;
    table[end] = code;
  } while (end > 0);
}

/* Returns the table width of the next 2nd level table. count is the histogram
   of bit lengths for the remaining symbols, len is the code length of the next
   processed symbol */
static PIK_INLINE int NextTableBitSize(const uint16_t* const count, int len,
                                       int root_bits) {
  int left = 1 << (len - root_bits);
  while (len < kHuffmanMaxLength) {
    left -= count[len];
    if (left <= 0) break;
    ++len;
    left <<= 1;
  }
  return len - root_bits;
}

/* Builds Huffman lookup table assuming code lengths are in symbol order. */
/* Returns false in case of error (invalid tree or memory error). */
void BuildHuffmanTable(std::vector<HuffmanCode>* table, int root_bits,
                       const uint8_t* const code_lengths, int code_lengths_size,
                       uint16_t* count) {
  HuffmanCode code; /* current table entry */
  int next;         /* next available space in table */
  int len;          /* current code length */
  int symbol;       /* symbol index in original or sorted table */
  int key;          /* reversed prefix code */
  int step;         /* step size to replicate values in current table */
  int low;          /* low bits for current root entry */
  int mask;         /* mask for low bits */
  int table_bits;   /* key length of current table */
  int table_size;   /* size of current table */
  int total_size;   /* sum of root table size and 2nd level table sizes */
  /* symbols sorted by code length */
  std::vector<int> sorted(code_lengths_size);
  /* offsets in sorted table for each length */
  uint16_t offset[kHuffmanMaxLength + 1];
  int max_length = 1;

  /* generate offsets into sorted symbol table by code length */
  {
    uint16_t sum = 0;
    for (len = 1; len <= kHuffmanMaxLength; len++) {
      offset[len] = sum;
      if (count[len]) {
        sum = static_cast<uint16_t>(sum + count[len]);
        max_length = len;
      }
    }
  }

  /* sort symbols by length, by symbol order within each length */
  for (symbol = 0; symbol < code_lengths_size; symbol++) {
    if (code_lengths[symbol] != 0) {
      sorted[offset[code_lengths[symbol]]++] = symbol;
    }
  }

  next = 0;
  table_bits = root_bits;
  table_size = 1 << table_bits;
  total_size = table_size;
  table->resize(total_size);

  /* special case code with only one value */
  if (offset[kHuffmanMaxLength] == 1) {
    code.bits = 0;
    code.value = static_cast<uint16_t>(sorted[0]);
    for (key = 0; key < total_size; ++key) {
      (*table)[key] = code;
    }
    return;
  }

  /* fill in root table */
  /* let's reduce the table size to a smaller size if possible, and */
  /* create the repetitions by memcpy if possible in the coming loop */
  if (table_bits > max_length) {
    table_bits = max_length;
    table_size = 1 << table_bits;
  }
  key = 0;
  symbol = 0;
  code.bits = 1;
  step = 2;
  do {
    for (; count[code.bits] != 0; --count[code.bits]) {
      code.value = static_cast<uint16_t>(sorted[symbol++]);
      ReplicateValue(&(*table)[key], step, table_size, code);
      key = GetNextKey(key, code.bits);
    }
    step <<= 1;
  } while (++code.bits <= table_bits);

  /* if root_bits != table_bits we only created one fraction of the */
  /* table, and we need to replicate it now. */
  while (total_size != table_size) {
    memcpy(&(*table)[table_size], &(*table)[0],
           table_size * sizeof((*table)[0]));
    table_size <<= 1;
  }

  /* fill in 2nd level tables and add pointers to root table */
  mask = total_size - 1;
  low = -1;
  for (len = root_bits + 1, step = 2; len <= max_length; ++len, step <<= 1) {
    for (; count[len] != 0; --count[len]) {
      if ((key & mask) != low) {
        next += table_size;
        table_bits = NextTableBitSize(count, len, root_bits);
        table_size = 1 << table_bits;
        total_size += table_size;
        table->resize(total_size);
        low = key & mask;
        (*table)[low].bits = static_cast<uint8_t>(table_bits + root_bits);
        (*table)[low].value = static_cast<uint16_t>(next - low);
      }
      code.bits = static_cast<uint8_t>(len - root_bits);
      code.value = static_cast<uint16_t>(sorted[symbol++]);
      ReplicateValue(&(*table)[next + (key >> root_bits)], step, table_size,
                     code);
      key = GetNextKey(key, len);
    }
  }
}

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

int ReadHuffmanCodeLengths(const uint8_t* code_length_code_lengths,
                           std::vector<uint8_t>* code_lengths,
                           BitReader* input) {
  uint8_t prev_code_len = kDefaultCodeLength;
  int repeat = 0;
  uint8_t repeat_code_len = 0;
  int space = 32768;
  std::vector<HuffmanCode> table;

  uint16_t counts[16] = {0};
  for (int i = 0; i < kCodeLengthCodes; ++i) {
    ++counts[code_length_code_lengths[i]];
  }
  BuildHuffmanTable(&table, 5, code_length_code_lengths, kCodeLengthCodes,
                    &counts[0]);

  const int max_num_symbols = 1 << 16;
  code_lengths->reserve(256);
  while (code_lengths->size() < max_num_symbols && space > 0) {
    const HuffmanCode* p = &table[0];
    uint8_t code_len;
    input->FillBitBuffer();
    p += input->PeekFixedBits<5>();
    input->Advance(p->bits);
    code_len = static_cast<uint8_t>(p->value);
    if (code_len < kCodeLengthRepeatCode) {
      repeat = 0;
      code_lengths->push_back(code_len);
      if (code_len != 0) {
        prev_code_len = code_len;
        space -= 32768 >> code_len;
      }
    } else {
      const int extra_bits = code_len - 14;
      int old_repeat;
      int repeat_delta;
      uint8_t new_len = 0;
      if (code_len == kCodeLengthRepeatCode) {
        new_len = prev_code_len;
      }
      if (repeat_code_len != new_len) {
        repeat = 0;
        repeat_code_len = new_len;
      }
      old_repeat = repeat;
      if (repeat > 0) {
        repeat -= 2;
        repeat <<= extra_bits;
      }
      int next_repeat = input->ReadBits(extra_bits) + 3;
      repeat += next_repeat;
      repeat_delta = repeat - old_repeat;
      if (code_lengths->size() + repeat_delta > max_num_symbols) {
        return 0;
      }
      for (int i = 0; i < repeat_delta; ++i) {
        code_lengths->push_back(repeat_code_len);
      }
      if (repeat_code_len != 0) {
        space -= repeat_delta << (15 - repeat_code_len);
      }
    }
  }
  if (space != 0) {
    return 0;
  }
  return 1;
}

bool HuffmanDecodingData::ReadFromBitStream(BitReader* input) {
  int ok = 1;
  int simple_code_or_skip;

  std::vector<uint8_t> code_lengths;
  /* simple_code_or_skip is used as follows:
     1 for simple code;
     0 for no skipping, 2 skips 2 code lengths, 3 skips 3 code lengths */
  simple_code_or_skip = input->ReadBits(2);
  if (simple_code_or_skip == 1) {
    /* Read symbols, codes & code lengths directly. */
    int i;
    int symbols[4] = {0};
    int max_symbol = 0;
    const int num_symbols = input->ReadBits(2) + 1;
    for (i = 0; i < num_symbols; ++i) {
      symbols[i] = DecodeVarLenUint16(input);
      if (symbols[i] > max_symbol) max_symbol = symbols[i];
    }
    code_lengths.resize(max_symbol + 1);
    code_lengths[symbols[0]] = 1;
    for (i = 1; i < num_symbols; ++i) {
      code_lengths[symbols[i]] = 2;
    }
    switch (num_symbols) {
      case 1:
        break;
      case 3:
        ok = ((symbols[0] != symbols[1]) && (symbols[0] != symbols[2]) &&
              (symbols[1] != symbols[2]));
        break;
      case 2:
        ok = (symbols[0] != symbols[1]);
        code_lengths[symbols[1]] = 1;
        break;
      case 4:
        ok = ((symbols[0] != symbols[1]) && (symbols[0] != symbols[2]) &&
              (symbols[0] != symbols[3]) && (symbols[1] != symbols[2]) &&
              (symbols[1] != symbols[3]) && (symbols[2] != symbols[3]));
        if (input->ReadBits(1)) {
          code_lengths[symbols[2]] = 3;
          code_lengths[symbols[3]] = 3;
        } else {
          code_lengths[symbols[0]] = 2;
        }
        break;
    }
  } else { /* Decode Huffman-coded code lengths. */
    int i;
    uint8_t code_length_code_lengths[kCodeLengthCodes] = {0};
    int space = 32;
    int num_codes = 0;
    /* Static Huffman code for the code length code lengths */
    static const HuffmanCode huff[16] = {
        {2, 0}, {2, 4}, {2, 3}, {3, 2}, {2, 0}, {2, 4}, {2, 3}, {4, 1},
        {2, 0}, {2, 4}, {2, 3}, {3, 2}, {2, 0}, {2, 4}, {2, 3}, {4, 5},
    };
    for (i = simple_code_or_skip; i < kCodeLengthCodes && space > 0; ++i) {
      const int code_len_idx = kCodeLengthCodeOrder[i];
      const HuffmanCode* p = huff;
      uint8_t v;
      input->FillBitBuffer();
      p += input->PeekFixedBits<4>();
      input->Advance(p->bits);
      v = static_cast<uint8_t>(p->value);
      code_length_code_lengths[code_len_idx] = v;
      if (v != 0) {
        space -= (32 >> v);
        ++num_codes;
      }
    }
    ok = (num_codes == 1 || space == 0) &&
         ReadHuffmanCodeLengths(code_length_code_lengths, &code_lengths, input);
  }
  if (!ok) {
    return PIK_FAILURE("Failed to read Huffman data");
  }
  uint16_t counts[16] = {0};
  for (int i = 0; i < code_lengths.size(); ++i) {
    ++counts[code_lengths[i]];
  }
  BuildHuffmanTable(&table_, kHuffmanTableBits, &code_lengths[0],
                    code_lengths.size(), &counts[0]);
  return true;
}

}  // namespace pik
