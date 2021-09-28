/*
 * (c) Copyright 2019-2021 Xilinx, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#ifndef _XFCOMPRESSION_HUFFMAN_TREEGEN_HPP_
#define _XFCOMPRESSION_HUFFMAN_TREEGEN_HPP_

/**
 * @file huffman_treegen.hpp
 * @brief Header for modules used in Tree Generation kernel.
 *
 * This file is part of XF Compression Library.
 */
#include "compress_utils.hpp"
#include "zlib_specs.hpp"
#include "zstd_specs.hpp"

namespace xf {
namespace compression {

const static uint8_t c_tgnSymbolBits = 10;
const static uint8_t c_tgnBitlengthBits = 5;

const static uint8_t c_lengthHistogram = 18;

const static uint16_t c_tgnSymbolSize = c_litCodeCount;
const static uint16_t c_tgnTreeDepth = c_litCodeCount;
const static uint16_t c_tgnMaxBits = c_maxCodeBits;

typedef ap_uint<12> Histogram;

template <int MAX_FREQ_DWIDTH>
using Frequency = ap_uint<MAX_FREQ_DWIDTH>;

template <int MAX_FREQ_DWIDTH>
struct Symbol {
    ap_uint<c_tgnSymbolBits> value;
    Frequency<MAX_FREQ_DWIDTH> frequency;
};

struct Codeword {
    ap_uint<c_maxBits> codeword;
    ap_uint<c_tgnBitlengthBits> bitlength;
};

namespace details {

// local types and constants
const static uint8_t RADIX = 16;
const static uint8_t BITS_PER_LOOP = 4;

const static ap_uint<c_tgnSymbolBits> INTERNAL_NODE = -1;
typedef ap_uint<BITS_PER_LOOP> Digit;

template <int MAX_FREQ_DWIDTH = 32, int WRITE_MXC = 1>
void filter(hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& inFreq,
            Symbol<MAX_FREQ_DWIDTH>* heap,
            uint16_t* heapLength,
            hls::stream<ap_uint<c_tgnSymbolBits> >& symLength,
            uint16_t i_symSize) {
    uint16_t hpLen = 0;
    ap_uint<c_tgnSymbolBits> smLen = 0;
    bool read_flag = false;
    auto curFreq = inFreq.read();
    if (curFreq.strobe == 0) {
        heapLength[0] = 0xFFFF;
        return;
    }
filter:
    for (uint16_t n = 0; n < i_symSize; ++n) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 286 min = 19
        if (read_flag) curFreq = inFreq.read();
        auto cf = curFreq.data[0];
        read_flag = true;
        if (n == 256) {
            heap[hpLen].value = smLen = n;
            heap[hpLen].frequency = 1;
            ++hpLen;
        } else if (cf != 0) {
            heap[hpLen].value = smLen = n;
            heap[hpLen].frequency = cf;
            ++hpLen;
        }
    }

    heapLength[0] = hpLen;
    if (WRITE_MXC) symLength << smLen;
}

template <int MAX_FREQ_DWIDTH = 32>
void filter(Frequency<MAX_FREQ_DWIDTH>* inFreq,
            Symbol<MAX_FREQ_DWIDTH>* heap,
            uint16_t* heapLength,
            uint16_t* symLength,
            uint16_t i_symSize) {
    uint16_t hpLen = 0;
    uint16_t smLen = 0;
filter:
    for (uint16_t n = 0; n < i_symSize; ++n) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 286 min = 19
        auto cf = inFreq[n];
        if (n == 256) {
            heap[hpLen].value = smLen = n;
            heap[hpLen].frequency = 1;
            ++hpLen;
        } else if (cf != 0) {
            heap[hpLen].value = smLen = n;
            heap[hpLen].frequency = cf;
            ++hpLen;
        }
    }

    heapLength[0] = hpLen;
    symLength[0] = smLen;
}

template <int SYMBOL_SIZE, int SYMBOL_BITS, int MAX_FREQ_DWIDTH = 32>
void radixSort_1(Symbol<MAX_FREQ_DWIDTH>* heap, uint16_t n) {
    //#pragma HLS INLINE
    Symbol<MAX_FREQ_DWIDTH> prev_sorting[SYMBOL_SIZE];
    Digit current_digit[SYMBOL_SIZE];
    bool not_sorted = true;

    ap_uint<SYMBOL_BITS> digit_histogram[RADIX], digit_location[RADIX];
#pragma HLS ARRAY_PARTITION variable = digit_location complete dim = 1
#pragma HLS ARRAY_PARTITION variable = digit_histogram complete dim = 1

radix_sort:
    for (uint8_t shift = 0; shift < MAX_FREQ_DWIDTH && not_sorted; shift += BITS_PER_LOOP) {
#pragma HLS LOOP_TRIPCOUNT min = 3 max = 5 avg = 4
    init_histogram:
        for (ap_uint<5> i = 0; i < RADIX; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 16 avg = 16
#pragma HLS PIPELINE II = 1
            digit_histogram[i] = 0;
        }

        auto prev_freq = heap[0].frequency;
        not_sorted = false;
    compute_histogram:
        for (uint16_t j = 0; j < n; ++j) {
#pragma HLS LOOP_TRIPCOUNT min = 19 max = 286
#pragma HLS PIPELINE II = 1
            Symbol<MAX_FREQ_DWIDTH> val = heap[j];
            Digit digit = (val.frequency >> shift) & (RADIX - 1);
            current_digit[j] = digit;
            ++digit_histogram[digit];
            prev_sorting[j] = val;
            // check if not already in sorted order
            if (prev_freq > val.frequency) not_sorted = true;
            prev_freq = val.frequency;
        }
        digit_location[0] = 0;

    find_digit_location:
        for (uint8_t i = 0; (i < RADIX - 1) && not_sorted; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 16 avg = 16
#pragma HLS PIPELINE II = 1
            digit_location[i + 1] = digit_location[i] + digit_histogram[i];
        }

    re_sort:
        for (uint16_t j = 0; j < n && not_sorted; ++j) {
#pragma HLS LOOP_TRIPCOUNT min = 19 max = 286
#pragma HLS PIPELINE II = 1
            Digit digit = current_digit[j];
            heap[digit_location[digit]] = prev_sorting[j];
            ++digit_location[digit];
        }
    }
}

template <int SYMBOL_SIZE, int SYMBOL_BITS, int MAX_FREQ_DWIDTH = 32>
void radixSort(Symbol<MAX_FREQ_DWIDTH>* heap, uint16_t n) {
    //#pragma HLS INLINE
    Symbol<MAX_FREQ_DWIDTH> prev_sorting[SYMBOL_SIZE];
    Digit current_digit[SYMBOL_SIZE];
    bool not_sorted = true;

    ap_uint<SYMBOL_BITS> digit_histogram[RADIX], digit_location[RADIX];
#pragma HLS ARRAY_PARTITION variable = digit_location complete dim = 1
#pragma HLS ARRAY_PARTITION variable = digit_histogram complete dim = 1

radix_sort:
    for (uint8_t shift = 0; shift < MAX_FREQ_DWIDTH && not_sorted; shift += BITS_PER_LOOP) {
#pragma HLS LOOP_TRIPCOUNT min = 3 max = 5 avg = 4
    init_histogram:
        for (uint8_t i = 0; i < RADIX; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 16 avg = 16
#pragma HLS PIPELINE II = 1
            digit_histogram[i] = 0;
        }

        auto prev_freq = heap[0].frequency;
        not_sorted = false;
    compute_histogram:
        for (uint16_t j = 0; j < n; ++j) {
#pragma HLS LOOP_TRIPCOUNT min = 19 max = 286
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL FACTOR = 2
            Symbol<MAX_FREQ_DWIDTH> val = heap[j];
            Digit digit = (val.frequency >> shift) & (RADIX - 1);
            current_digit[j] = digit;
            ++digit_histogram[digit];
            prev_sorting[j] = val;
            // check if not already in sorted order
            if (prev_freq > val.frequency) not_sorted = true;
            prev_freq = val.frequency;
        }
        digit_location[0] = 0;

    find_digit_location:
        for (uint8_t i = 0; (i < RADIX - 1) && not_sorted; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 16 avg = 16
#pragma HLS PIPELINE II = 1
            digit_location[i + 1] = digit_location[i] + digit_histogram[i];
        }

    re_sort:
        for (uint16_t j = 0; j < n && not_sorted; ++j) {
#pragma HLS LOOP_TRIPCOUNT min = 19 max = 286
#pragma HLS PIPELINE II = 1
            Digit digit = current_digit[j];
            heap[digit_location[digit]] = prev_sorting[j];
            ++digit_location[digit];
        }
    }
}

template <int SYMBOL_SIZE, int SYMBOL_BITS, int MAX_FREQ_DWIDTH = 32>
void createTree(Symbol<MAX_FREQ_DWIDTH>* heap,
                int num_symbols,
                ap_uint<SYMBOL_BITS>* parent,
                ap_uint<SYMBOL_SIZE>& left,
                ap_uint<SYMBOL_SIZE>& right,
                Frequency<MAX_FREQ_DWIDTH>* frequency) {
    ap_uint<SYMBOL_BITS> tree_count = 0; // Number of intermediate node assigned a parent
    ap_uint<SYMBOL_BITS> in_count = 0;   // Number of inputs consumed
    ap_uint<SYMBOL_SIZE> tmp;
    left = 0;
    right = 0;

    // for case with less number of symbols
    if (num_symbols < 3) num_symbols++;
// this loop needs to run at least twice
create_heap:
    for (uint16_t i = 0; i < num_symbols; ++i) {
#pragma HLS PIPELINE II = 3
#pragma HLS LOOP_TRIPCOUNT min = 19 avg = 286 max = 286
        Frequency<MAX_FREQ_DWIDTH> node_freq = 0;
        Frequency<MAX_FREQ_DWIDTH> intermediate_freq = frequency[tree_count];
        Symbol<MAX_FREQ_DWIDTH> s = heap[in_count];
        tmp = 1;
        tmp <<= i;

        if ((in_count < num_symbols && s.frequency <= intermediate_freq) || tree_count == i) {
            // Pick symbol from heap
            // left[i] = s.value;       // set input symbol value as left node
            node_freq = s.frequency; // Add symbol frequency to total node frequency
            // move to the next input symbol
            ++in_count;
        } else {
            // pick internal node without a parent
            // left[i] = INTERNAL_NODE;           // Set symbol to indicate an internal node
            left |= tmp;
            node_freq = intermediate_freq; // Add child node frequency
            parent[tree_count] = i;        // Set this node as child's parent
            // Go to next parent-less internal node
            ++tree_count;
        }

        intermediate_freq = frequency[tree_count];
        s = heap[in_count];
        if ((in_count < num_symbols && s.frequency <= intermediate_freq) || tree_count == i) {
            // Pick symbol from heap
            // right[i] = s.value;
            frequency[i] = node_freq + s.frequency;
            ++in_count;
        } else {
            // Pick internal node without a parent
            // right[i] = INTERNAL_NODE;
            right |= tmp;
            frequency[i] = node_freq + intermediate_freq;
            parent[tree_count] = i;
            ++tree_count;
        }
    }
}

template <int SYMBOL_SIZE, int SYMBOL_BITS, int MAX_FREQ_DWIDTH = 32>
void computeBitLength(ap_uint<SYMBOL_BITS>* parent,
                      ap_uint<SYMBOL_SIZE>& left,
                      ap_uint<SYMBOL_SIZE>& right,
                      int num_symbols,
                      Histogram* length_histogram,
                      Frequency<MAX_FREQ_DWIDTH>* child_depth) {
    ap_uint<SYMBOL_SIZE> tmp;
    // for case with less number of symbols
    if (num_symbols < 2) num_symbols++;
    // Depth of the root node is 0.
    child_depth[num_symbols - 1] = 0;
// this loop needs to run at least once
traverse_tree:
    for (int16_t i = num_symbols - 2; i >= 0; --i) {
#pragma HLS LOOP_TRIPCOUNT min = 19 max = 286
#pragma HLS pipeline II = 1
        tmp = 1;
        tmp <<= i;
        uint32_t length = child_depth[parent[i]] + 1;
        child_depth[i] = length;
        bool is_left_internal = ((left & tmp) == 0);
        bool is_right_internal = ((right & tmp) == 0);

        if ((is_left_internal || is_right_internal)) {
            uint32_t children = 1; // One child of the original node was a symbol
            if (is_left_internal && is_right_internal) {
                children = 2; // Both the children of the original node were symbols
            }
            length_histogram[length] += children;
        }
    }
}

void truncateTree(Histogram* length_histogram, uint16_t c_tree_depth, int max_bit_len) {
    int j = max_bit_len;
move_nodes:
    for (uint16_t i = c_tree_depth - 1; i > max_bit_len; --i) {
#pragma HLS LOOP_TRIPCOUNT min = 572 max = 572 avg = 572
#pragma HLS PIPELINE II = 1
        // Look to see if there are any nodes at lengths greater than target depth
        int cnt = 0;
    reorder:
        for (; length_histogram[i] != 0;) {
#pragma HLS LOOP_TRIPCOUNT min = 3 max = 3 avg = 3
            if (j == max_bit_len) {
                // find the deepest leaf with codeword length < target depth
                --j;
            trctr_mv:
                while (length_histogram[j] == 0) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1 avg = 1
                    --j;
                }
            }
            // Move leaf with depth i to depth j + 1
            length_histogram[j] -= 1;     // The node at level j is no longer a leaf
            length_histogram[j + 1] += 2; // Two new leaf nodes are attached at level j+1
            length_histogram[i - 1] += 1; // The leaf node at level i+1 gets attached here
            length_histogram[i] -= 2;     // Two leaf nodes have been lost from  level i

            // now deepest leaf with codeword length < target length
            // is at level (j+1) unless (j+1) == target length
            ++j;
        }
    }
    // for (int i = 0; i < c_tree_depth; ++i) printf("%d. length_hist: %u\n", i, (uint16_t)length_histogram[i]);
}

template <int SYMBOL_BITS, int MAX_FREQ_DWIDTH = 32>
void canonizeTree(Symbol<MAX_FREQ_DWIDTH>* sorted,
                  uint32_t num_symbols,
                  Histogram* length_histogram,
                  ap_uint<4>* symbol_bits,
                  uint16_t i_treeDepth) {
    int16_t length = i_treeDepth;
    ap_uint<SYMBOL_BITS> count = 0;
// Iterate across the symbols from lowest frequency to highest
// Assign them largest bit length to smallest
// printf("num_symbols: %u, treeDepth: %u\n", num_symbols, i_treeDepth);
process_symbols:
    for (uint32_t k = 0; k < num_symbols; ++k) {
#pragma HLS LOOP_TRIPCOUNT max = 286 min = 256 avg = 286
        if (count == 0) {
            // find the next non-zero bit length k
            count = length_histogram[--length];
        canonize_inner:
            while (count == 0 && length >= 0) {
#pragma HLS LOOP_TRIPCOUNT min = 1 avg = 1 max = 2
#pragma HLS PIPELINE II = 1
                // n  is the number of symbols with encoded length k
                count = length_histogram[--length];
            }
        }
        if (length < 0) break;
        symbol_bits[sorted[k].value] = length; // assign symbol k to have length bits
        // printf("%u. val: %u, bitlen: %d\n", k, (uint16_t)sorted[k].value, length);
        --count; // keep assigning i bits until we have counted off n symbols
    }
    // printf("last_length: %d\n", length);
}

template <int MAX_LEN>
void createCodeword(ap_uint<4>* symbol_bits,
                    Histogram* length_histogram,
                    Codeword* huffCodes,
                    uint16_t cur_symSize,
                    uint16_t cur_maxBits,
                    uint16_t symCnt) {
    //#pragma HLS inline
    typedef ap_uint<MAX_LEN> LCL_Code_t;
    LCL_Code_t first_codeword[MAX_LEN + 1];
    //#pragma HLS ARRAY_PARTITION variable = first_codeword complete dim = 1

    // Computes the initial codeword value for a symbol with bit length i
    first_codeword[0] = 0;
first_codewords:
    for (uint16_t i = 1; i <= cur_maxBits; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 7 max = 15
#pragma HLS PIPELINE II = 1
        first_codeword[i] = (first_codeword[i - 1] + length_histogram[i - 1]) << 1;
    }

assign_codewords_mm:
    for (uint16_t k = 0; k < cur_symSize; ++k) {
#pragma HLS LOOP_TRIPCOUNT max = 286 min = 286 avg = 286
#pragma HLS PIPELINE II = 1
        uint8_t length = (uint8_t)symbol_bits[k];
        // if symbol has 0 bits, it doesn't need to be encoded
        LCL_Code_t out_reversed = first_codeword[length];
        out_reversed.reverse();
        out_reversed = out_reversed >> (MAX_LEN - length);

        huffCodes[k].codeword = (length == 0) ? (uint16_t)0 : (uint16_t)out_reversed;
        huffCodes[k].bitlength = length;
        first_codeword[length]++;
    }
    if (symCnt == 0) {
        huffCodes[0].bitlength = 1;
    }
}

template <int MAX_LEN>
void createCodeword(ap_uint<4>* symbol_bits,
                    Histogram* length_histogram,
                    hls::stream<DSVectorStream_dt<Codeword, 1> >& huffCodes,
                    uint16_t cur_symSize,
                    uint16_t cur_maxBits,
                    uint16_t symCnt) {
    //#pragma HLS inline
    typedef ap_uint<MAX_LEN> LCL_Code_t;
    LCL_Code_t first_codeword[MAX_LEN + 1];
    //#pragma HLS ARRAY_PARTITION variable = first_codeword complete dim = 1

    DSVectorStream_dt<Codeword, 1> hfc;
    hfc.strobe = 1;

    // Computes the initial codeword value for a symbol with bit length i
    first_codeword[0] = 0;
first_codewords:
    for (uint16_t i = 1; i <= cur_maxBits; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 7 max = 15
#pragma HLS PIPELINE II = 1
        first_codeword[i] = (first_codeword[i - 1] + length_histogram[i - 1]) << 1;
    }
    Codeword code;
assign_codewords_sm:
    for (uint16_t k = 0; k < cur_symSize; ++k) {
#pragma HLS LOOP_TRIPCOUNT max = 286 min = 286 avg = 286
#pragma HLS PIPELINE II = 1
        uint8_t length = (uint8_t)symbol_bits[k];
        // if symbol has 0 bits, it doesn't need to be encoded
        LCL_Code_t out_reversed = first_codeword[length];
        out_reversed.reverse();
        out_reversed = out_reversed >> (MAX_LEN - length);

        hfc.data[0].codeword = (length == 0 || symCnt == 0) ? (uint16_t)0 : (uint16_t)out_reversed;
        length = (symCnt == 0) ? 0 : length;
        // hfc.data[0].bitlength = (symCnt == 0) ? 0 : length;
        hfc.data[0].bitlength = (symCnt == 0 && k == 0) ? 1 : length;
        first_codeword[length]++;
        huffCodes << hfc;
    }
}

template <int MAX_LEN>
void createZstdCodeword(ap_uint<4>* symbol_bits,
                        Histogram* length_histogram,
                        hls::stream<DSVectorStream_dt<Codeword, 1> >& huffCodes,
                        uint16_t cur_symSize,
                        uint16_t cur_maxBits,
                        uint16_t symCnt) {
    //#pragma HLS inline
    bool allSameBlen = true;
    typedef ap_uint<MAX_LEN> LCL_Code_t;
    LCL_Code_t first_codeword[MAX_LEN + 1];
    //#pragma HLS ARRAY_PARTITION variable = first_codeword complete dim = 1

    DSVectorStream_dt<Codeword, 1> hfc;
    hfc.strobe = 1;

    // Computes the initial codeword value for a symbol with bit length i
    first_codeword[0] = 0;
    uint8_t uniq_bl_idx = 0;
find_uniq_blen_count:
    for (uint8_t i = 0; i < cur_maxBits + 1; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 12
        if (length_histogram[i] == cur_symSize) uniq_bl_idx = i;
    }
    // If only 1 uniq_blc for all symbols divide into 3 bitlens
    // Means, if all the bitlens are same(mainly bitlen-8) then use an alternate tree
    // Fix the current bitlength_histogram and symbol_bits so that it generates codes-bitlens for alternate tree
    if (uniq_bl_idx > 0) {
        length_histogram[7] = 1;
        length_histogram[9] = 2;
        length_histogram[8] -= 3;

        symbol_bits[0] = 7;
        symbol_bits[1] = 9;
        symbol_bits[2] = 9;
    }

    uint16_t nextCode = 0;
hflkpt_initial_codegen:
    for (uint8_t i = cur_maxBits; i > 0; --i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 11
        uint16_t cur = nextCode;
        nextCode += (length_histogram[i]);
        nextCode >>= 1;
        first_codeword[i] = cur;
    }
    Codeword code;
assign_codewords_sm:
    for (uint16_t k = 0; k < cur_symSize; ++k) {
#pragma HLS LOOP_TRIPCOUNT max = 256 min = 256 avg = 256
#pragma HLS PIPELINE II = 1
        uint8_t length = (uint8_t)symbol_bits[k];
        // length = (uniq_bl_idx > 0 && k > 2 && length > 8) ? 8 : length;	// not needed if treegen is optimal
        length = (symCnt == 0) ? 0 : length;
        code.codeword = (uint16_t)first_codeword[length];
        // get bitlength for code
        length = (symCnt == 0 && k == 0) ? 1 : length;
        code.bitlength = length;
        // write out codes
        hfc.data[0] = code;
        first_codeword[length]++;
        huffCodes << hfc;
    }
}

template <int SYMBOL_SIZE, int SYMBOL_BITS, int MAX_FREQ_DWIDTH = 32>
void huffConstructTreeStream_1(hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& inFreq,
                               hls::stream<DSVectorStream_dt<Symbol<MAX_FREQ_DWIDTH>, 1> >& heapStream,
                               hls::stream<ap_uint<9> >& heapLenStream,
                               hls::stream<ap_uint<c_tgnSymbolBits> >& maxCodes,
                               hls::stream<bool>& isEOBlocks) {
    // internal buffers
    Symbol<MAX_FREQ_DWIDTH> heap[SYMBOL_SIZE];
    DSVectorStream_dt<Symbol<MAX_FREQ_DWIDTH>, 1> hpVal;
    const ap_uint<9> hctMeta[2] = {c_litCodeCount, c_dstCodeCount};
    bool last = false;
filter_sort_ldblock:
    while (!last) {
    // filter-sort for literals and distances
    filter_sort_litdist:
        for (uint8_t metaIdx = 0; metaIdx < 2; metaIdx++) {
            ap_uint<9> i_symbolSize = hctMeta[metaIdx]; // current symbol size
            uint16_t heapLength = 0;

            // filter the input, write 0 heapLength at end of block
            filter<MAX_FREQ_DWIDTH>(inFreq, heap, &heapLength, maxCodes, i_symbolSize);

            // check for end of block
            last = (heapLength == 0xFFFF);
            if (metaIdx == 0) isEOBlocks << last;
            if (last) break;

            // sort the input
            radixSort<SYMBOL_SIZE, SYMBOL_BITS, MAX_FREQ_DWIDTH>(heap, heapLength);

            // send sorted frequencies
            heapLenStream << heapLength;
            hpVal.strobe = 1;
            for (uint16_t i = 0; i < heapLength; i++) {
                hpVal.data[0] = heap[i];
                heapStream << hpVal;
            }
        }
    }
}

template <int SYMBOL_SIZE, int SYMBOL_BITS, int MAX_FREQ_DWIDTH = 18>
void zstdFreqFilterSort(hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& inFreqStream,
                        hls::stream<DSVectorStream_dt<Symbol<MAX_FREQ_DWIDTH>, 1> >& heapStream,
                        hls::stream<ap_uint<9> >& heapLenStream,
                        hls::stream<bool>& eobStream) {
    // internal buffers
    Symbol<MAX_FREQ_DWIDTH> heap[SYMBOL_SIZE];
    DSVectorStream_dt<Symbol<MAX_FREQ_DWIDTH>, 1> hpVal;
    bool last = false;

    hls::stream<ap_uint<SYMBOL_BITS> > maxCodes("maxCodes");
#pragma HLS STREAM variable = maxCodes depth = 2

filter_sort_ldblock:
    while (!last) {
        // filter-sort for literals
        ap_uint<9> i_symbolSize = SYMBOL_SIZE; // current symbol size
        uint16_t heapLength = 0;

        // filter the input, write 0 heapLength at end of block
        filter<MAX_FREQ_DWIDTH, 0>(inFreqStream, heap, &heapLength, maxCodes, i_symbolSize);
        // dump maxcode
        // maxCodes.read();
        // check for end of block
        last = (heapLength == 0xFFFF);
        eobStream << last;
        if (last) break;

        // sort the input
        radixSort<SYMBOL_SIZE, SYMBOL_BITS, MAX_FREQ_DWIDTH>(heap, heapLength);

        // send sorted frequencies
        heapLenStream << heapLength;
        hpVal.strobe = 1;
        for (uint16_t i = 0; i < heapLength; i++) {
            hpVal.data[0] = heap[i];
            heapStream << hpVal;
        }
    }
}

template <int SYMBOL_SIZE, int SYMBOL_BITS, int MAX_FREQ_DWIDTH = 32>
void huffConstructTreeStream_2(hls::stream<DSVectorStream_dt<Symbol<MAX_FREQ_DWIDTH>, 1> >& heapStream,
                               hls::stream<ap_uint<9> >& heapLenStream,
                               hls::stream<bool>& isEOBlocks,
                               hls::stream<DSVectorStream_dt<Codeword, 1> >& outCodes) {
    ap_uint<SYMBOL_SIZE> left = 0;
    ap_uint<SYMBOL_SIZE> right = 0;
    ap_uint<SYMBOL_BITS> parent[SYMBOL_SIZE];
    Histogram length_histogram[c_lengthHistogram];

    Frequency<MAX_FREQ_DWIDTH> temp_array[SYMBOL_SIZE];
    Symbol<MAX_FREQ_DWIDTH> heap[SYMBOL_SIZE];
#pragma HLS BIND_STORAGE variable = heap type = ram_t2p impl = bram
#pragma HLS AGGREGATE variable = heap
    ap_uint<4> symbol_bits[SYMBOL_SIZE];
    const ap_uint<9> hctMeta[3][3] = {{c_litCodeCount, c_litCodeCount, c_maxCodeBits},
                                      {c_dstCodeCount, c_dstCodeCount, c_maxCodeBits},
                                      {c_blnCodeCount, c_blnCodeCount, c_maxBLCodeBits}};
    DSVectorStream_dt<Symbol<MAX_FREQ_DWIDTH>, 1> hpVal;
construct_tree_ldblock:
    while (isEOBlocks.read() == false) {
    construct_tree_litdist:
        for (uint8_t metaIdx = 0; metaIdx < 2; metaIdx++) {
            uint16_t i_symbolSize = hctMeta[metaIdx][0]; // current symbol size
            uint16_t i_treeDepth = hctMeta[metaIdx][1];  // current tree depth
            uint16_t i_maxBits = hctMeta[metaIdx][2];    // current max bits

            uint16_t heapLength = heapLenStream.read();

        init_buffers:
            for (uint16_t i = 0; i < i_symbolSize; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 19 max = 286
#pragma HLS PIPELINE II = 1
                parent[i] = 0;
                if (i < c_lengthHistogram) length_histogram[i] = 0;
                temp_array[i] = 0;
                if (i < heapLength) {
                    hpVal = heapStream.read();
                    heap[i] = hpVal.data[0];
                }
                symbol_bits[i] = 0;
            }

            // create tree
            createTree<SYMBOL_SIZE, SYMBOL_BITS, MAX_FREQ_DWIDTH>(heap, heapLength, parent, left, right, temp_array);

            // get bit-lengths from tree
            computeBitLength<SYMBOL_SIZE, SYMBOL_BITS, MAX_FREQ_DWIDTH>(parent, left, right, heapLength,
                                                                        length_histogram, temp_array);

            // truncate tree for any bigger bit-lengths
            truncateTree(length_histogram, c_lengthHistogram, i_maxBits);

            // canonize the tree
            canonizeTree<SYMBOL_BITS, MAX_FREQ_DWIDTH>(heap, heapLength, length_histogram, symbol_bits,
                                                       c_lengthHistogram);

            // generate huffman codewords
            createCodeword<c_tgnMaxBits>(symbol_bits, length_histogram, outCodes, i_symbolSize, i_maxBits, heapLength);
        }
    }
}

template <int SYMBOL_SIZE, int SYMBOL_BITS, int MAX_FREQ_DWIDTH, int MAX_BITS>
void zstdGetHuffmanCodes(hls::stream<DSVectorStream_dt<Symbol<MAX_FREQ_DWIDTH>, 1> >& heapStream,
                         hls::stream<ap_uint<9> >& heapLenStream,
                         hls::stream<bool>& isEOBlocks,
                         hls::stream<DSVectorStream_dt<Codeword, 1> >& outCodes) {
    ap_uint<SYMBOL_SIZE> left = 0;
    ap_uint<SYMBOL_SIZE> right = 0;
    ap_uint<SYMBOL_BITS> parent[SYMBOL_SIZE];
    Histogram length_histogram[c_lengthHistogram];
#pragma HLS ARRAY_PARTITION variable = length_histogram complete
    Frequency<MAX_FREQ_DWIDTH> temp_array[SYMBOL_SIZE];

    Symbol<MAX_FREQ_DWIDTH> heap[SYMBOL_SIZE];
#pragma HLS BIND_STORAGE variable = heap type = ram_t2p impl = bram
#pragma HLS AGGREGATE variable = heap

    ap_uint<4> symbol_bits[SYMBOL_SIZE];
    DSVectorStream_dt<Symbol<MAX_FREQ_DWIDTH>, 1> hpVal;
construct_tree_ldblock:
    while (isEOBlocks.read() == false) {
        uint16_t heapLength = heapLenStream.read();
    init_buffers:
        for (uint16_t i = 0; i < SYMBOL_SIZE; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 256 max = 256
#pragma HLS PIPELINE II = 1
            parent[i] = 0;
            if (i < c_lengthHistogram) length_histogram[i] = 0;
            temp_array[i] = 0;
            if (i < heapLength) {
                hpVal = heapStream.read();
                heap[i] = hpVal.data[0];
            }
            symbol_bits[i] = 0;
        }

        // create tree
        createTree<SYMBOL_SIZE, SYMBOL_BITS, MAX_FREQ_DWIDTH>(heap, heapLength, parent, left, right, temp_array);

        // get bit-lengths from tree
        computeBitLength<SYMBOL_SIZE, SYMBOL_BITS, MAX_FREQ_DWIDTH>(parent, left, right, heapLength, length_histogram,
                                                                    temp_array);

        // truncate tree for any bigger bit-lengths
        truncateTree(length_histogram, c_lengthHistogram, MAX_BITS);

        // canonize the tree
        canonizeTree<SYMBOL_BITS, MAX_FREQ_DWIDTH>(heap, heapLength, length_histogram, symbol_bits, MAX_BITS + 1);

        // generate huffman codewords
        createZstdCodeword<MAX_BITS>(symbol_bits, length_histogram, outCodes, SYMBOL_SIZE, MAX_BITS, heapLength);
    }
}

template <int SYMBOL_SIZE, int MAX_FREQ_DWIDTH, int MAX_BITS, int BLEN_BITS>
void huffCodeWeightDistributor(hls::stream<DSVectorStream_dt<Codeword, 1> >& hufCodeStream,
                               hls::stream<bool>& isEOBlocks,
                               hls::stream<DSVectorStream_dt<HuffmanCode_dt<MAX_BITS>, 1> >& outCodeStream,
                               hls::stream<IntVectorStream_dt<BLEN_BITS, 1> >& outWeightsStream,
                               hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& weightFreqStream) {
    // distribute huffman codes to multiple output streams and one separate bitlen stream
    DSVectorStream_dt<HuffmanCode_dt<MAX_BITS>, 1> outCode;
    IntVectorStream_dt<BLEN_BITS, 1> outWeight;
    IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> outFreq;
    int blk_n = 0;
distribute_code_block:
    while (isEOBlocks.read() == false) {
        ++blk_n;
        ap_uint<MAX_FREQ_DWIDTH> weightFreq[MAX_BITS + 1];
#pragma HLS ARRAY_PARTITION variable = weightFreq complete
        ap_uint<BLEN_BITS> blenBuf[SYMBOL_SIZE];
        ap_uint<BLEN_BITS> curMaxBitlen = 0;
        uint8_t maxWeight = 0;
        uint16_t maxVal = 0;
    init_freq_bl:
        for (uint8_t i = 0; i < MAX_BITS + 1; ++i) {
#pragma HLS PIPELINE off
            weightFreq[i] = 0;
        }
        outCode.strobe = 1;
        outWeight.strobe = 1;
        outFreq.strobe = 1;
        int cnt = 0;
    // printf("Huffman Codes\n");
    distribute_code_loop:
        for (uint16_t i = 0; i < SYMBOL_SIZE; ++i) {
#pragma HLS PIPELINE II = 1
            auto inCode = hufCodeStream.read();
            uint8_t hfblen = inCode.data[0].bitlength;
            uint16_t hfcode = inCode.data[0].codeword;
            outCode.data[0].code = hfcode;
            outCode.data[0].bitlen = hfblen;
            blenBuf[i] = hfblen;
            if (hfblen > curMaxBitlen) curMaxBitlen = hfblen;
            if (hfblen > 0) {
                maxVal = (uint16_t)i;
                ++cnt;
            }
            outCodeStream << outCode;
        }
    send_weights:
        for (ap_uint<9> i = 0; i < SYMBOL_SIZE; ++i) {
#pragma HLS PIPELINE II = 1
            auto bitlen = blenBuf[i];
            auto blenWeight = (uint8_t)((bitlen > 0) ? (uint8_t)(curMaxBitlen + 1 - bitlen) : 0);
            outWeight.data[0] = blenWeight;
            if (i < maxVal + 1) weightFreq[blenWeight]++;
            outWeightsStream << outWeight;
        }
        // write maxVal as first value
        outFreq.data[0] = maxVal;
        weightFreqStream << outFreq;
    // send weight frequencies
    send_weight_freq:
        for (uint8_t i = 0; i < MAX_BITS + 1; ++i) {
#pragma HLS PIPELINE II = 1
            outFreq.data[0] = weightFreq[i];
            weightFreqStream << outFreq;
            if (outFreq.data[0] > 0) maxWeight = i; // to be deduced by module reading this stream
        }
        // end of block
        outCode.strobe = 0;
        outWeight.strobe = 0;
        outFreq.strobe = 0;
        outCodeStream << outCode;
        outWeightsStream << outWeight;
        weightFreqStream << outFreq;
    }
    // end of all data
    outCode.strobe = 0;
    outWeight.strobe = 0;
    outFreq.strobe = 0;
    outCodeStream << outCode;
    outWeightsStream << outWeight;
    weightFreqStream << outFreq;
}

template <int SYMBOL_SIZE, int SYMBOL_BITS, int LENGTH_SIZE, int MAX_FREQ_DWIDTH = 32>
void huffConstructTreeStream(hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& inFreq,
                             hls::stream<bool>& isEOBlocks,
                             hls::stream<DSVectorStream_dt<Codeword, 1> >& outCodes,
                             hls::stream<ap_uint<c_tgnSymbolBits> >& maxCodes,
                             uint8_t metaIdx) {
#pragma HLS inline
    // construct huffman tree and generate codes and bit-lengths
    const ap_uint<9> hctMeta[3][3] = {{c_litCodeCount, c_litCodeCount, c_maxCodeBits},
                                      {c_dstCodeCount, c_dstCodeCount, c_maxCodeBits},
                                      {c_blnCodeCount, c_blnCodeCount, c_maxBLCodeBits}};

    ap_uint<9> i_symbolSize = hctMeta[metaIdx][0]; // current symbol size
    ap_uint<9> i_treeDepth = hctMeta[metaIdx][1];  // current tree depth
    ap_uint<9> i_maxBits = hctMeta[metaIdx][2];    // current max bits

    while (isEOBlocks.read() == false) {
        // internal buffers
        Symbol<MAX_FREQ_DWIDTH> heap[SYMBOL_SIZE];

        ap_uint<SYMBOL_SIZE> left = 0;
        ap_uint<SYMBOL_SIZE> right = 0;
        ap_uint<SYMBOL_BITS> parent[SYMBOL_SIZE];
        Histogram length_histogram[LENGTH_SIZE];

        Frequency<MAX_FREQ_DWIDTH> temp_array[SYMBOL_SIZE];
        //#pragma HLS resource variable=temp_array core=RAM_2P_BRAM

        ap_uint<4> symbol_bits[SYMBOL_SIZE];
        IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> curFreq;
        uint16_t heapLength = 0;

    init_buffers:
        for (ap_uint<9> i = 0; i < i_symbolSize; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 19 max = 286
#pragma HLS PIPELINE II = 1
            parent[i] = 0;
            if (i < LENGTH_SIZE) length_histogram[i] = 0;
            temp_array[i] = 0;
            symbol_bits[i] = 0;
            heap[i].value = 0;
            heap[i].frequency = 0;
        }
        // filter the input, write 0 heapLength at end of block
        filter<MAX_FREQ_DWIDTH>(inFreq, heap, &heapLength, maxCodes, i_symbolSize);

        // sort the input
        radixSort_1<SYMBOL_SIZE, SYMBOL_BITS, MAX_FREQ_DWIDTH>(heap, heapLength);

        // create tree
        createTree<SYMBOL_SIZE, SYMBOL_BITS, MAX_FREQ_DWIDTH>(heap, heapLength, parent, left, right, temp_array);

        // get bit-lengths from tree
        computeBitLength<SYMBOL_SIZE, SYMBOL_BITS, MAX_FREQ_DWIDTH>(parent, left, right, heapLength, length_histogram,
                                                                    temp_array);

        // truncate tree for any bigger bit-lengths
        truncateTree(length_histogram, LENGTH_SIZE, i_maxBits);

        // canonize the tree
        canonizeTree<SYMBOL_BITS, MAX_FREQ_DWIDTH>(heap, heapLength, length_histogram, symbol_bits, LENGTH_SIZE);

        // generate huffman codewords
        createCodeword<c_tgnMaxBits>(symbol_bits, length_histogram, outCodes, i_symbolSize, i_maxBits, heapLength);
    }
}

template <int MAX_FREQ_DWIDTH = 32>
void genBitLenFreq(hls::stream<DSVectorStream_dt<Codeword, 1> >& outCodes,
                   hls::stream<bool>& isEOBlocks,
                   hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& freq,
                   hls::stream<ap_uint<c_tgnSymbolBits> >& maxCode) {
    // iterate over blocks
    while (isEOBlocks.read() == false) {
        ap_uint<MAX_FREQ_DWIDTH> blFreq[19];
        const ap_uint<9> hctMeta[2] = {c_litCodeCount, c_dstCodeCount};
    init_bitlen_freq:
        for (uint8_t i = 0; i < 19; i++) {
#pragma HLS PIPELINE off
            blFreq[i] = 0;
        }

        for (uint8_t itr = 0; itr < 2; itr++) {
            int16_t prevlen = -1;
            int16_t curlen = 0;
            int16_t count = 0;
            int16_t max_count = 7;
            int16_t min_count = 4;

            int16_t nextlen = outCodes.read().data[0].bitlength;
            if (nextlen == 0) {
                max_count = 138;
                min_count = 3;
            }

            ap_uint<c_tgnSymbolBits> maximumCodeLength = maxCode.read();
        parse_tdata:
            for (ap_uint<c_tgnSymbolBits> n = 0; n <= maximumCodeLength; ++n) {
#pragma HLS LOOP_TRIPCOUNT min = 30 max = 286
#pragma HLS PIPELINE II = 1
                curlen = nextlen;
                if (n == maximumCodeLength) {
                    nextlen = 0xF;
                } else {
                    nextlen = outCodes.read().data[0].bitlength;
                }

                if (++count < max_count && curlen == nextlen) {
                    continue;
                } else if (count < min_count) {
                    blFreq[curlen] += count;
                } else if (curlen != 0) {
                    if (curlen != prevlen) {
                        blFreq[curlen]++;
                    }
                    blFreq[c_reusePrevBlen]++;
                } else if (count <= 10) {
                    blFreq[c_reuseZeroBlen]++;
                } else {
                    blFreq[c_reuseZeroBlen7]++;
                }

                count = 0;
                prevlen = curlen;
                if (nextlen == 0) {
                    max_count = 138, min_count = 3;
                } else if (curlen == nextlen) {
                    max_count = 6, min_count = 3;
                } else {
                    max_count = 7, min_count = 4;
                }
            }
        read_spare_codes:
            for (auto i = maximumCodeLength + 1; i < hctMeta[itr]; i++) {
#pragma HLS PIPELINE II = 1
                auto tmp = outCodes.read();
            }
        }

        IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> blf;
        blf.strobe = 1;
    output_bitlen_frequencies:
        for (uint8_t i = 0; i < 19; i++) {
#pragma HLS PIPELINE II = 1
            blf.data[0] = blFreq[i];
            freq << blf;
        }
    }
}

template <int MAX_FREQ_DWIDTH = 32>
void genBitLenFreq(Codeword* outCodes, Frequency<MAX_FREQ_DWIDTH>* blFreq, uint16_t maxCode) {
    //#pragma HLS inline
    // generate bit-length frequencies using literal and distance bit-lengths
    ap_uint<4> tree_len[c_litCodeCount];

copy_blens:
    for (int i = 0; i <= maxCode; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 30 max = 286
        tree_len[i] = (uint8_t)outCodes[i].bitlength;
    }
clear_rem_blens:
    for (int i = maxCode + 2; i < c_litCodeCount; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 30 max = 286
        tree_len[i] = 0;
    }
    tree_len[maxCode + 1] = (uint8_t)0xff;

    int16_t prevlen = -1;
    int16_t curlen = 0;
    int16_t count = 0;
    int16_t max_count = 7;
    int16_t min_count = 4;
    int16_t nextlen = tree_len[0];

    if (nextlen == 0) {
        max_count = 138;
        min_count = 3;
    }

parse_tdata:
    for (uint32_t n = 0; n <= maxCode; ++n) {
#pragma HLS LOOP_TRIPCOUNT min = 30 max = 286
#pragma HLS PIPELINE II = 3
        curlen = nextlen;
        nextlen = tree_len[n + 1];

        if (++count < max_count && curlen == nextlen) {
            continue;
        } else if (count < min_count) {
            blFreq[curlen] += count;
        } else if (curlen != 0) {
            if (curlen != prevlen) {
                blFreq[curlen]++;
            }
            blFreq[c_reusePrevBlen]++;
        } else if (count <= 10) {
            blFreq[c_reuseZeroBlen]++;
        } else {
            blFreq[c_reuseZeroBlen7]++;
        }

        count = 0;
        prevlen = curlen;
        if (nextlen == 0) {
            max_count = 138, min_count = 3;
        } else if (curlen == nextlen) {
            max_count = 6, min_count = 3;
        } else {
            max_count = 7, min_count = 4;
        }
    }
}

template <uint8_t SEND_EOS = 1>
void sendTrees(hls::stream<ap_uint<c_tgnSymbolBits> >& maxLdCodes,
               hls::stream<ap_uint<c_tgnSymbolBits> >& maxBlCodes,
               hls::stream<DSVectorStream_dt<Codeword, 1> >& Ldcodes,
               hls::stream<DSVectorStream_dt<Codeword, 1> >& Blcodes,
               hls::stream<bool>& isEOBlocks,
               hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> >& hfcodeStream) {
    const uint8_t bitlen_vals[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
#pragma HLS ARRAY_PARTITION variable = bitlen_vals complete dim = 1

    DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> outHufCode;
send_tree_outer:
    while (isEOBlocks.read() == false) {
        Codeword zeroValue;
        zeroValue.bitlength = 0;
        zeroValue.codeword = 0;
        Codeword outCodes[c_litCodeCount + c_dstCodeCount + c_blnCodeCount + 2];
#pragma HLS AGGREGATE variable = outCodes
        ap_uint<c_tgnSymbolBits> maxCodesReg[3];

        const uint16_t offsets[3] = {0, c_litCodeCount + 1, (c_litCodeCount + c_dstCodeCount + 2)};
        // initialize all the memory
        maxCodesReg[0] = maxLdCodes.read();
    read_litcodes:
        for (uint16_t i = 0; i < c_litCodeCount; ++i) {
#pragma HLS PIPELINE II = 1
            auto ldc = Ldcodes.read();
            outCodes[i] = ldc.data[0];
        }
        // last value write
        outCodes[c_litCodeCount] = zeroValue;

        maxCodesReg[1] = maxLdCodes.read();

    read_dstcodes:
        for (uint16_t i = 0; i < c_dstCodeCount; i++) {
#pragma HLS PIPELINE II = 1
            auto ldc = Ldcodes.read();
            outCodes[offsets[1] + i] = ldc.data[0];
        }

        outCodes[c_litCodeCount + c_dstCodeCount + 1] = zeroValue;

        //********************************************//
        outHufCode.strobe = 1;
    send_ltrees:
        for (uint16_t i = 0; i < c_litCodeCount; ++i) {
#pragma HLS PIPELINE II = 1
            auto cw = outCodes[i];
            outHufCode.data[0].code = cw.codeword;
            outHufCode.data[0].bitlen = cw.bitlength;
            hfcodeStream << outHufCode;
        }

    send_dtrees:
        for (uint16_t i = 0; i < c_dstCodeCount; ++i) {
#pragma HLS PIPELINE II = 1
            auto cw = outCodes[offsets[1] + i];
            outHufCode.data[0].code = cw.codeword;
            outHufCode.data[0].bitlen = cw.bitlength;
            hfcodeStream << outHufCode;
        }

        maxCodesReg[2] = maxBlCodes.read();

    read_blcodes:
        for (uint16_t i = offsets[2]; i < offsets[2] + c_blnCodeCount; ++i) {
#pragma HLS PIPELINE II = 1
            auto ldc = Blcodes.read();
            outCodes[i] = ldc.data[0];
        }

        ap_uint<c_tgnSymbolBits> bl_mxc;
        bool mxb_continue = true;
    bltree_blen:
        for (bl_mxc = c_blnCodeCount - 1; (bl_mxc >= 3) && mxb_continue; --bl_mxc) {
#pragma HLS PIPELINE II = 1
            auto cIdx = offsets[2] + bitlen_vals[bl_mxc];
            mxb_continue = (outCodes[cIdx].bitlength == 0);
        }

        maxCodesReg[2] = bl_mxc + 1;

        // Code from Huffman Encoder
        //********************************************//
        // Start of block = 4 and len = 3
        // For dynamic tree
        outHufCode.data[0].code = 4;
        outHufCode.data[0].bitlen = 3;
        hfcodeStream << outHufCode;
        // lcodes
        outHufCode.data[0].code = ((maxCodesReg[0] + 1) - 257);
        outHufCode.data[0].bitlen = 5;
        hfcodeStream << outHufCode;

        // dcodes
        outHufCode.data[0].code = ((maxCodesReg[1] + 1) - 1);
        outHufCode.data[0].bitlen = 5;
        hfcodeStream << outHufCode;

        // blcodes
        outHufCode.data[0].code = ((maxCodesReg[2] + 1) - 4);
        outHufCode.data[0].bitlen = 4;
        hfcodeStream << outHufCode;

        ap_uint<c_tgnSymbolBits> bitIndex = offsets[2];
    // Send BL length data
    send_bltree:
        for (ap_uint<c_tgnSymbolBits> rank = 0; rank < maxCodesReg[2] + 1; rank++) {
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 64
#pragma HLS PIPELINE II = 1
            outHufCode.data[0].code = outCodes[bitIndex + bitlen_vals[rank]].bitlength;
            outHufCode.data[0].bitlen = 3;
            hfcodeStream << outHufCode;
        } // BL data copy loop

        // Send Bitlengths for Literal and Distance Tree
        for (int tree = 0; tree < 2; tree++) {
            uint8_t prevlen = 0; // Last emitted Length
            uint8_t curlen = 0;  // Length of Current Code
            uint8_t nextlen =
                (tree == 0) ? outCodes[0].bitlength : outCodes[offsets[1]].bitlength; // Length of next code
            uint8_t count = 0;
            int max_count = 7; // Max repeat count
            int min_count = 4; // Min repeat count

            if (nextlen == 0) {
                max_count = 138;
                min_count = 3;
            }

            uint16_t max_code = (tree == 0) ? maxCodesReg[0] : maxCodesReg[1];

            Codeword temp = outCodes[bitIndex + c_reusePrevBlen];
            uint16_t reuse_prev_code = temp.codeword;
            uint8_t reuse_prev_len = temp.bitlength;
            temp = outCodes[bitIndex + c_reuseZeroBlen];
            uint16_t reuse_zero_code = temp.codeword;
            uint8_t reuse_zero_len = temp.bitlength;
            temp = outCodes[bitIndex + c_reuseZeroBlen7];
            uint16_t reuse_zero7_code = temp.codeword;
            uint8_t reuse_zero7_len = temp.bitlength;

        send_ltree:
            for (uint16_t n = 0; n <= max_code; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 286 max = 286
                curlen = nextlen;
                // Length of next code
                nextlen = (tree == 0) ? outCodes[n + 1].bitlength : outCodes[offsets[1] + n + 1].bitlength;

                if (++count < max_count && curlen == nextlen) {
                    continue;
                } else if (count < min_count) {
                    temp = outCodes[bitIndex + curlen];
                lit_cnt:
                    for (uint8_t cnt = count; cnt != 0; --cnt) {
#pragma HLS LOOP_TRIPCOUNT min = 10 max = 10
#pragma HLS PIPELINE II = 1
                        outHufCode.data[0].code = temp.codeword;
                        outHufCode.data[0].bitlen = temp.bitlength;
                        hfcodeStream << outHufCode;
                    }
                    count = 0;

                } else if (curlen != 0) {
                    if (curlen != prevlen) {
                        temp = outCodes[bitIndex + curlen];
                        outHufCode.data[0].code = temp.codeword;
                        outHufCode.data[0].bitlen = temp.bitlength;
                        hfcodeStream << outHufCode;
                        count--;
                    }
                    outHufCode.data[0].code = reuse_prev_code;
                    outHufCode.data[0].bitlen = reuse_prev_len;
                    hfcodeStream << outHufCode;

                    outHufCode.data[0].code = count - 3;
                    outHufCode.data[0].bitlen = 2;
                    hfcodeStream << outHufCode;

                } else if (count <= 10) {
                    outHufCode.data[0].code = reuse_zero_code;
                    outHufCode.data[0].bitlen = reuse_zero_len;
                    hfcodeStream << outHufCode;

                    outHufCode.data[0].code = count - 3;
                    outHufCode.data[0].bitlen = 3;
                    hfcodeStream << outHufCode;

                } else {
                    outHufCode.data[0].code = reuse_zero7_code;
                    outHufCode.data[0].bitlen = reuse_zero7_len;
                    hfcodeStream << outHufCode;

                    outHufCode.data[0].code = count - 11;
                    outHufCode.data[0].bitlen = 7;
                    hfcodeStream << outHufCode;
                }

                count = 0;
                prevlen = curlen;
                if (nextlen == 0) {
                    max_count = 138, min_count = 3;
                } else if (curlen == nextlen) {
                    max_count = 6, min_count = 3;
                } else {
                    max_count = 7, min_count = 4;
                }
            }
        }
        // ends huffman stream for each block, strobe eos not needed from this module
        outHufCode.data[0].bitlen = 0;
        hfcodeStream << outHufCode;
    }
    if (SEND_EOS > 0) {
        // end of huffman tree data for all blocks
        outHufCode.strobe = 0;
        hfcodeStream << outHufCode;
    }
}

void codeWordDistributor(hls::stream<DSVectorStream_dt<Codeword, 1> >& inStreamCode,
                         hls::stream<DSVectorStream_dt<Codeword, 1> >& outStreamCode1,
                         hls::stream<DSVectorStream_dt<Codeword, 1> >& outStreamCode2,
                         hls::stream<ap_uint<c_tgnSymbolBits> >& inStreamMaxCode,
                         hls::stream<ap_uint<c_tgnSymbolBits> >& outStreamMaxCode1,
                         hls::stream<ap_uint<c_tgnSymbolBits> >& outStreamMaxCode2,
                         hls::stream<bool>& isEOBlocks) {
    const int hctMeta[2] = {c_litCodeCount, c_dstCodeCount};
    while (isEOBlocks.read() == false) {
    distribute_litdist_codes:
        for (uint8_t i = 0; i < 2; i++) {
            auto maxCode = inStreamMaxCode.read();
            outStreamMaxCode1 << maxCode;
            outStreamMaxCode2 << maxCode;
        distribute_hufcodes_main:
            for (uint16_t j = 0; j < hctMeta[i]; j++) {
#pragma HLS PIPELINE II = 1
                auto inVal = inStreamCode.read();
                outStreamCode1 << inVal;
                outStreamCode2 << inVal;
            }
        }
    }
}

template <int SLAVES>
void streamDistributor(hls::stream<bool>& inStream, hls::stream<bool> outStream[SLAVES]) {
    do {
        bool i = inStream.read();
        for (int n = 0; n < SLAVES; n++) outStream[n] << i;
        if (i == 1) break;
    } while (1);
}

template <int MAX_FREQ_DWIDTH = 32>
void processBitLength(hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& frequencies,
                      hls::stream<bool>& isEOBlocks,
                      hls::stream<DSVectorStream_dt<Codeword, 1> >& outCodes,
                      hls::stream<ap_uint<c_tgnSymbolBits> >& maxCodes) {
    // read freqStream and generate codes for it
    // construct the huffman tree and generate huffman codes
    details::huffConstructTreeStream<c_blnCodeCount, c_tgnBitlengthBits, 15, MAX_FREQ_DWIDTH>(frequencies, isEOBlocks,
                                                                                              outCodes, maxCodes, 2);
}

template <int MAX_FREQ_DWIDTH = 24, uint8_t SEND_EOS = 1>
void zlibTreegenStream(hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& lz77TreeStream,
                       hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> >& hufCodeStream) {
    hls::stream<DSVectorStream_dt<Codeword, 1> > ldCodes("ldCodes");
    hls::stream<DSVectorStream_dt<Codeword, 1> > ldCodes1("ldCodes1");
    hls::stream<DSVectorStream_dt<Codeword, 1> > ldCodes2("ldCodes2");
    hls::stream<DSVectorStream_dt<Codeword, 1> > blCodes("blCodes");
    hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> > ldFrequency("ldFrequency");

    hls::stream<ap_uint<c_tgnSymbolBits> > ldMaxCodes("ldMaxCodes");
    hls::stream<ap_uint<c_tgnSymbolBits> > ldMaxCodes1("ldMaxCodes1");
    hls::stream<ap_uint<c_tgnSymbolBits> > ldMaxCodes2("ldMaxCodes2");
    hls::stream<ap_uint<c_tgnSymbolBits> > blMaxCodes("blMaxCodes");

    hls::stream<DSVectorStream_dt<Symbol<MAX_FREQ_DWIDTH>, 1> > heapStream("heapStream");
    hls::stream<ap_uint<9> > heapLenStream("heapLenStream");
    hls::stream<bool> isEOBlocks("eob_source");
    hls::stream<bool> eoBlocks[5];

#pragma HLS STREAM variable = heapStream depth = 320
#pragma HLS STREAM variable = ldCodes2 depth = 320
#pragma HLS STREAM variable = ldMaxCodes2 depth = 4

// DATAFLOW
#pragma HLS DATAFLOW
    huffConstructTreeStream_1<c_litCodeCount, c_tgnSymbolBits, MAX_FREQ_DWIDTH>(lz77TreeStream, heapStream,
                                                                                heapLenStream, ldMaxCodes, isEOBlocks);
    streamDistributor<5>(isEOBlocks, eoBlocks);
    huffConstructTreeStream_2<c_litCodeCount, c_tgnSymbolBits, MAX_FREQ_DWIDTH>(heapStream, heapLenStream, eoBlocks[0],
                                                                                ldCodes);
    codeWordDistributor(ldCodes, ldCodes1, ldCodes2, ldMaxCodes, ldMaxCodes1, ldMaxCodes2, eoBlocks[1]);
    genBitLenFreq<MAX_FREQ_DWIDTH>(ldCodes1, eoBlocks[2], ldFrequency, ldMaxCodes1);
    processBitLength<MAX_FREQ_DWIDTH>(ldFrequency, eoBlocks[3], blCodes, blMaxCodes);
    sendTrees<SEND_EOS>(ldMaxCodes2, blMaxCodes, ldCodes2, blCodes, eoBlocks[4], hufCodeStream);
}

} // end namespace details

template <int MAX_FREQ_DWIDTH, int MAX_BITS>
void zstdTreegenStream(hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& inFreqStream,
                       hls::stream<DSVectorStream_dt<HuffmanCode_dt<MAX_BITS>, 1> >& outCodeStream,
                       hls::stream<IntVectorStream_dt<4, 1> >& outWeightStream,
                       hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& weightFreqStream) {
#pragma HLS DATAFLOW
    hls::stream<DSVectorStream_dt<Symbol<MAX_FREQ_DWIDTH>, 1> > heapStream("heapStream");
    hls::stream<ap_uint<9> > heapLenStream("heapLenStream");
    hls::stream<DSVectorStream_dt<Codeword, 1> > hufCodeStream("hufCodeStream");
    hls::stream<bool> eobStream("eobStream");
    hls::stream<bool> eoBlocks[2];

    details::zstdFreqFilterSort<details::c_maxLitV + 1, c_tgnSymbolBits, MAX_FREQ_DWIDTH>(inFreqStream, heapStream,
                                                                                          heapLenStream, eobStream);
    details::streamDistributor<2>(eobStream, eoBlocks);
    details::zstdGetHuffmanCodes<details::c_maxLitV + 1, c_tgnSymbolBits, MAX_FREQ_DWIDTH, MAX_BITS>(
        heapStream, heapLenStream, eoBlocks[0], hufCodeStream);
    details::huffCodeWeightDistributor<details::c_maxLitV + 1, MAX_FREQ_DWIDTH, MAX_BITS, 4>(
        hufCodeStream, eoBlocks[1], outCodeStream, outWeightStream, weightFreqStream);
}

} // End of compression
} // End of xf

#endif // _XFCOMPRESSION_DEFLATE_TREES_HPP_
