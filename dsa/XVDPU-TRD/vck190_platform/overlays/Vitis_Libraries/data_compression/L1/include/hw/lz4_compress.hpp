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
#ifndef _XFCOMPRESSION_LZ4_COMPRESS_HPP_
#define _XFCOMPRESSION_LZ4_COMPRESS_HPP_

/**
 * @file lz4_compress.hpp
 * @brief Header for modules used in LZ4 compression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "hls_stream.h"
#include <ap_int.h>
#include <assert.h>
#include <stdint.h>
#include "lz_compress.hpp"
#include "lz_optional.hpp"
#include "mm2s.hpp"
#include "s2mm.hpp"
#include "stream_downsizer.hpp"
#include "stream_upsizer.hpp"

const int c_gmemBurstSize = 32;

namespace xf {
namespace compression {
namespace details {

template <int MAX_LIT_COUNT, int PARALLEL_UNITS>
static void lz4CompressPart1(hls::stream<ap_uint<32> >& inStream,
                             hls::stream<uint8_t>& lit_outStream,
                             hls::stream<ap_uint<64> >& lenOffset_Stream,
                             uint32_t input_size,
                             uint32_t max_lit_limit[PARALLEL_UNITS],
                             uint32_t index) {
    if (input_size == 0) return;

    uint8_t match_len = 0;
    uint32_t lit_count = 0;
    uint32_t lit_count_flag = 0;

    ap_uint<32> nextEncodedValue = inStream.read();
lz4_divide:
    for (uint32_t i = 0; i < input_size;) {
#pragma HLS PIPELINE II = 1
        ap_uint<32> tmpEncodedValue = nextEncodedValue;
        if (i < (input_size - 1)) nextEncodedValue = inStream.read();
        uint8_t tCh = tmpEncodedValue.range(7, 0);
        uint8_t tLen = tmpEncodedValue.range(15, 8);
        uint16_t tOffset = tmpEncodedValue.range(31, 16);
        uint32_t match_offset = tOffset;

        if (lit_count >= MAX_LIT_COUNT) {
            lit_count_flag = 1;
        } else if (tLen) {
            uint8_t match_len = tLen - 4; // LZ4 standard
            ap_uint<64> tmpValue;
            tmpValue.range(63, 32) = lit_count;
            tmpValue.range(15, 0) = match_len;
            tmpValue.range(31, 16) = match_offset;
            lenOffset_Stream << tmpValue;
            match_len = tLen - 1;
            lit_count = 0;
        } else {
            lit_outStream << tCh;
            lit_count++;
        }
        if (tLen)
            i += tLen;
        else
            i += 1;
    }
    if (lit_count) {
        ap_uint<64> tmpValue;
        tmpValue.range(63, 32) = lit_count;
        if (lit_count == MAX_LIT_COUNT) {
            lit_count_flag = 1;
            tmpValue.range(15, 0) = 777;
            tmpValue.range(31, 16) = 777;
        } else {
            tmpValue.range(15, 0) = 0;
            tmpValue.range(31, 16) = 0;
        }
        lenOffset_Stream << tmpValue;
    }
    max_lit_limit[index] = lit_count_flag;
}

static void lz4CompressPart2(hls::stream<uint8_t>& in_lit_inStream,
                             hls::stream<ap_uint<64> >& in_lenOffset_Stream,
                             hls::stream<ap_uint<8> >& outStream,
                             hls::stream<bool>& endOfStream,
                             hls::stream<uint32_t>& compressdSizeStream,
                             uint32_t input_size) {
    // LZ4 Compress STATES
    enum lz4CompressStates { WRITE_TOKEN, WRITE_LIT_LEN, WRITE_MATCH_LEN, WRITE_LITERAL, WRITE_OFFSET0, WRITE_OFFSET1 };
    uint32_t lit_len = 0;
    uint16_t outCntr = 0;
    uint32_t compressedSize = 0;
    enum lz4CompressStates next_state = WRITE_TOKEN;
    uint16_t lit_length = 0;
    uint16_t match_length = 0;
    uint16_t write_lit_length = 0;
    ap_uint<16> match_offset = 0;
    bool lit_ending = false;
    bool extra_match_len = false;
    bool readOffsetFlag = true;

lz4_compress:
    for (uint32_t inIdx = 0; (inIdx < input_size) || (!readOffsetFlag);) {
#pragma HLS PIPELINE II = 1
        ap_uint<8> outValue;
        ap_uint<64> nextLenOffsetValue;

        if (readOffsetFlag) {
            nextLenOffsetValue = in_lenOffset_Stream.read();
            readOffsetFlag = false;
        }

        if (next_state == WRITE_TOKEN) {
            lit_length = nextLenOffsetValue.range(63, 32);
            match_length = nextLenOffsetValue.range(15, 0);
            match_offset = nextLenOffsetValue.range(31, 16);
            inIdx += match_length + lit_length + 4;

            if (match_length == 777 && match_offset == 777) {
                inIdx = input_size;
                lit_ending = true;
            }

            lit_len = lit_length;
            write_lit_length = lit_length;
            if (match_offset == 0 && match_length == 0) {
                lit_ending = true;
            }
            if (lit_length >= 15) {
                outValue.range(7, 4) = 15;
                lit_length -= 15;
                next_state = WRITE_LIT_LEN;
                readOffsetFlag = false;
            } else if (lit_length) {
                outValue.range(7, 4) = lit_length;
                lit_length = 0;
                next_state = WRITE_LITERAL;
                readOffsetFlag = false;
            } else {
                outValue.range(7, 4) = 0;
                next_state = WRITE_OFFSET0;
                readOffsetFlag = false;
            }
            if (match_length >= 15) {
                outValue.range(3, 0) = 15;
                match_length -= 15;
                extra_match_len = true;
            } else {
                outValue.range(3, 0) = match_length;
                match_length = 0;
                extra_match_len = false;
            }
        } else if (next_state == WRITE_LIT_LEN) {
            if (lit_length >= 255) {
                outValue = 255;
                lit_length -= 255;
            } else {
                outValue = lit_length;
                next_state = WRITE_LITERAL;
                readOffsetFlag = false;
            }
        } else if (next_state == WRITE_LITERAL) {
            outValue = in_lit_inStream.read();
            write_lit_length--;
            if (write_lit_length == 0) {
                if (lit_ending) {
                    next_state = WRITE_TOKEN;
                    readOffsetFlag = true;
                } else {
                    next_state = WRITE_OFFSET0;
                    readOffsetFlag = false;
                }
            }
        } else if (next_state == WRITE_OFFSET0) {
            match_offset++; // LZ4 standard
            outValue = match_offset.range(7, 0);
            next_state = WRITE_OFFSET1;
            readOffsetFlag = false;
        } else if (next_state == WRITE_OFFSET1) {
            outValue = match_offset.range(15, 8);
            if (extra_match_len) {
                next_state = WRITE_MATCH_LEN;
                readOffsetFlag = false;
            } else {
                next_state = WRITE_TOKEN;
                readOffsetFlag = true;
            }
        } else if (next_state == WRITE_MATCH_LEN) {
            if (match_length >= 255) {
                outValue = 255;
                match_length -= 255;
            } else {
                outValue = match_length;
                next_state = WRITE_TOKEN;
                readOffsetFlag = true;
            }
        }
        if (compressedSize < input_size) {
            // Limiting compression size not more than input size.
            // Host code should ignore such blocks
            outStream << outValue;
            endOfStream << 0;
            compressedSize++;
        }
    }

    compressdSizeStream << compressedSize;
    outStream << 0;
    endOfStream << 1;
}

} // namespace compression
} // namespace xf
} // namespace details

namespace xf {
namespace compression {

/**
 * @brief This is the core compression module which seperates the input stream into two
 * output streams, one literal stream and other offset stream, then lz4 encoding is done.
 *
 * @param inStream Input data stream
 * @param outStream Output data stream
 * @param max_lit_limit Size for compressed stream
 * @param input_size Size of input data
 * @param endOfStream Stream indicating that all data is processed or not
 * @param compressdSizeStream Gives the compressed size for each 64K block
 *
 */
template <int MAX_LIT_COUNT, int PARALLEL_UNITS>
static void lz4Compress(hls::stream<ap_uint<32> >& inStream,
                        hls::stream<ap_uint<8> >& outStream,
                        uint32_t max_lit_limit[PARALLEL_UNITS],
                        uint32_t input_size,
                        hls::stream<bool>& endOfStream,
                        hls::stream<uint32_t>& compressdSizeStream,
                        uint32_t index) {
    hls::stream<uint8_t> lit_outStream("lit_outStream");
    hls::stream<ap_uint<64> > lenOffset_Stream("lenOffset_Stream");

#pragma HLS STREAM variable = lit_outStream depth = MAX_LIT_COUNT
#pragma HLS STREAM variable = lenOffset_Stream depth = c_gmemBurstSize

#pragma HLS BIND_STORAGE variable = lenOffset_Stream type = FIFO impl = SRL

#pragma HLS dataflow
    details::lz4CompressPart1<MAX_LIT_COUNT, PARALLEL_UNITS>(inStream, lit_outStream, lenOffset_Stream, input_size,
                                                             max_lit_limit, index);
    details::lz4CompressPart2(lit_outStream, lenOffset_Stream, outStream, endOfStream, compressdSizeStream, input_size);
}

template <class data_t,
          int DATAWIDTH = 512,
          int BURST_SIZE = 16,
          int NUM_BLOCK = 8,
          int M_LEN = 6,
          int MIN_MAT = 4,
          int LZ_MAX_OFFSET_LIM = 65536,
          int OFFSET_WIN = 65536,
          int MAX_M_LEN = 255,
          int MAX_LIT_CNT = 4096,
          int MIN_B_SIZE = 128>
void hlsLz4Core(hls::stream<data_t>& inStream,
                hls::stream<data_t>& outStream,
                hls::stream<bool>& outStreamEos,
                hls::stream<uint32_t>& compressedSize,
                uint32_t max_lit_limit[NUM_BLOCK],
                uint32_t input_size,
                uint32_t core_idx) {
    hls::stream<ap_uint<32> > compressdStream("compressdStream");
    hls::stream<ap_uint<32> > bestMatchStream("bestMatchStream");
    hls::stream<ap_uint<32> > boosterStream("boosterStream");
#pragma HLS STREAM variable = compressdStream depth = 8
#pragma HLS STREAM variable = bestMatchStream depth = 8
#pragma HLS STREAM variable = boosterStream depth = 8

#pragma HLS BIND_STORAGE variable = compressdStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = boosterStream type = FIFO impl = SRL

#pragma HLS dataflow
    xf::compression::lzCompress<M_LEN, MIN_MAT, LZ_MAX_OFFSET_LIM>(inStream, compressdStream, input_size);
    xf::compression::lzBestMatchFilter<M_LEN, OFFSET_WIN>(compressdStream, bestMatchStream, input_size);
    xf::compression::lzBooster<MAX_M_LEN>(bestMatchStream, boosterStream, input_size);
    xf::compression::lz4Compress<MAX_LIT_CNT, NUM_BLOCK>(boosterStream, outStream, max_lit_limit, input_size,
                                                         outStreamEos, compressedSize, core_idx);
}

template <class data_t,
          int DATAWIDTH = 512,
          int BURST_SIZE = 16,
          int NUM_BLOCK = 8,
          int M_LEN = 6,
          int MIN_MAT = 4,
          int LZ_MAX_OFFSET_LIM = 65536,
          int OFFSET_WIN = 65536,
          int MAX_M_LEN = 255,
          int MAX_LIT_CNT = 4096,
          int MIN_B_SIZE = 128>
void hlsLz4(const data_t* in,
            data_t* out,
            const uint32_t input_idx[NUM_BLOCK],
            const uint32_t output_idx[NUM_BLOCK],
            const uint32_t input_size[NUM_BLOCK],
            uint32_t output_size[NUM_BLOCK],
            uint32_t max_lit_limit[NUM_BLOCK]) {
    hls::stream<ap_uint<8> > inStream[NUM_BLOCK];
    hls::stream<bool> outStreamEos[NUM_BLOCK];
    hls::stream<ap_uint<8> > outStream[NUM_BLOCK];
#pragma HLS STREAM variable = outStreamEos depth = 2
#pragma HLS STREAM variable = inStream depth = c_gmemBurstSize
#pragma HLS STREAM variable = outStream depth = c_gmemBurstSize

#pragma HLS BIND_STORAGE variable = outStreamEos type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = inStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = outStream type = FIFO impl = SRL

    hls::stream<uint32_t> compressedSize[NUM_BLOCK];

#pragma HLS dataflow
    xf::compression::details::mm2multStreamSize<8, NUM_BLOCK, DATAWIDTH, BURST_SIZE>(in, input_idx, inStream,
                                                                                     input_size);

    for (uint8_t i = 0; i < NUM_BLOCK; i++) {
#pragma HLS UNROLL
        // lz4Core is instantiated based on the NUM_BLOCK
        hlsLz4Core<ap_uint<8>, DATAWIDTH, BURST_SIZE, NUM_BLOCK>(inStream[i], outStream[i], outStreamEos[i],
                                                                 compressedSize[i], max_lit_limit, input_size[i], i);
    }

    xf::compression::details::multStream2MM<8, NUM_BLOCK, DATAWIDTH, BURST_SIZE>(
        outStream, outStreamEos, compressedSize, output_idx, out, output_size);
}

template <class data_t,
          int DATAWIDTH = 512,
          int BURST_SIZE = 16,
          int NUM_BLOCK = 8,
          int M_LEN = 6,
          int MIN_MAT = 4,
          int LZ_MAX_OFFSET_LIM = 65536,
          int OFFSET_WIN = 65536,
          int MAX_M_LEN = 255,
          int MAX_LIT_CNT = 4096,
          int MIN_B_SIZE = 128>
void lz4CompressMM(const data_t* in, data_t* out, uint32_t* compressd_size, const uint32_t input_size) {
    uint32_t block_idx = 0;
    uint32_t block_length = 64 * 1024;
    uint32_t no_blocks = (input_size - 1) / block_length + 1;
    uint32_t max_block_size = 64 * 1024;
    uint32_t readBlockSize = 0;

    bool small_block[NUM_BLOCK];
    uint32_t input_block_size[NUM_BLOCK];
    uint32_t input_idx[NUM_BLOCK];
    uint32_t output_idx[NUM_BLOCK];
    uint32_t output_block_size[NUM_BLOCK];
    uint32_t max_lit_limit[NUM_BLOCK];
    uint32_t small_block_inSize[NUM_BLOCK];
#pragma HLS ARRAY_PARTITION variable = input_block_size dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = input_idx dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = output_idx dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = output_block_size dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = max_lit_limit dim = 0 complete

    // Figure out total blocks & block sizes
    for (uint32_t i = 0; i < no_blocks; i += NUM_BLOCK) {
        uint32_t nblocks = NUM_BLOCK;
        if ((i + NUM_BLOCK) > no_blocks) {
            nblocks = no_blocks - i;
        }

        for (uint32_t j = 0; j < NUM_BLOCK; j++) {
            if (j < nblocks) {
                uint32_t inBlockSize = block_length;
                if (readBlockSize + block_length > input_size) inBlockSize = input_size - readBlockSize;
                if (inBlockSize < MIN_B_SIZE) {
                    small_block[j] = 1;
                    small_block_inSize[j] = inBlockSize;
                    input_block_size[j] = 0;
                    input_idx[j] = 0;
                } else {
                    small_block[j] = 0;
                    input_block_size[j] = inBlockSize;
                    readBlockSize += inBlockSize;
                    input_idx[j] = (i + j) * max_block_size;
                    output_idx[j] = (i + j) * max_block_size;
                }
            } else {
                input_block_size[j] = 0;
                input_idx[j] = 0;
            }
            output_block_size[j] = 0;
            max_lit_limit[j] = 0;
        }

        // Call for parallel compression
        hlsLz4<data_t, DATAWIDTH, BURST_SIZE, NUM_BLOCK>(in, out, input_idx, output_idx, input_block_size,
                                                         output_block_size, max_lit_limit);

        for (uint32_t k = 0; k < nblocks; k++) {
            if (max_lit_limit[k]) {
                compressd_size[block_idx] = input_block_size[k];
            } else {
                compressd_size[block_idx] = output_block_size[k];
            }

            if (small_block[k] == 1) {
                compressd_size[block_idx] = small_block_inSize[k];
            }
            block_idx++;
        }
    }
}

} // namespace compression
} // namespace xf
#endif // _XFCOMPRESSION_LZ4_COMPRESS_HPP_
