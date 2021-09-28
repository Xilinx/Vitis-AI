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
#ifndef _XFCOMPRESSION_MM2S_HPP_
#define _XFCOMPRESSION_MM2S_HPP_

/**
 * @file mm2s.hpp
 * @brief Header for modules used for memory mapped to streaming interface conversion.
 *
 * This file is part of Vitis Data Compression Library.
 */
#include "compress_utils.hpp"
#include "hls_stream.h"

#include <ap_int.h>
#include <assert.h>
#include <stdint.h>
#include "stream_downsizer.hpp"

#define GET_DIFF_IF_BIG(x, y) (x > y) ? (x - y) : 0

#define STREAM_UTILS_MM2S_IF_NOT_FULL(bIdx, outStream, is_full, read_idx, write_idx, local_buffer) \
    is_full.range(bIdx, bIdx) = outStream.full();                                                  \
    if (!is_full.range(bIdx, bIdx) && (read_idx[bIdx] != write_idx[bIdx])) {                       \
        outStream << local_buffer[bIdx][read_idx[bIdx]];                                           \
        read_idx[bIdx] += 1;                                                                       \
    }

namespace xf {
namespace compression {

const int kGMemDWidth = 512;
typedef ap_uint<kGMemDWidth> uintMemWidth_t;

const int c_lTreeSize = 1024;
const int c_dTreeSize = 64;
const int c_bLTreeSize = 64;
const int c_maxCodeSize = 16;

namespace details {

template <int DATAWIDTH, int BURST_SIZE, int NUM_BLOCKS>
void mm2sNb(const ap_uint<DATAWIDTH>* in,
            const uint32_t _input_idx[NUM_BLOCKS],
            hls::stream<ap_uint<DATAWIDTH> > outStream[NUM_BLOCKS],
            const uint32_t _input_size[NUM_BLOCKS]) {
    /**
     * @brief This module reads 512bit data from memory interface and
     * writes to the stream. Writing to the multiple data streams is
     * non-blocking call which is done using is_full() API
     *
     * @tparam DATAWIDTH width of data bus
     * @tparam BURST_SIZE burst size of the data transfers
     * @tparam NUM_BLOCKS number of blocks
     *
     * @param in input memory address
     * @param _input_idx input index
     * @param outStream output stream
     * @param _input_size input stream size
     */

    const int c_byteSize = 8;
    const int c_wordSize = DATAWIDTH / c_byteSize;
    ap_uint<DATAWIDTH> local_buffer[NUM_BLOCKS][BURST_SIZE];
#pragma HLS ARRAY_PARTITION variable = local_buffer dim = 1 complete
#pragma HLS BIND_STORAGE variable = local_buffer type = RAM_2P impl = LUTRAM
    uint32_t read_idx[NUM_BLOCKS];
    uint32_t write_idx[NUM_BLOCKS];
    uint32_t read_size[NUM_BLOCKS];
    uint32_t input_idx[NUM_BLOCKS];
    uint32_t input_size[NUM_BLOCKS];
#pragma HLS ARRAY_PARTITION variable = read_idx dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = write_idx dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = read_size dim = 0 complete
    ap_uint<NUM_BLOCKS> pending;
    ap_uint<NUM_BLOCKS> is_full;
    for (uint32_t bIdx = 0; bIdx < NUM_BLOCKS; bIdx++) {
#pragma HLS UNROLL
        read_idx[bIdx] = 0;
        write_idx[bIdx] = 0;
        read_size[bIdx] = 0;
        input_idx[bIdx] = _input_idx[bIdx];
        input_size[bIdx] = _input_size[bIdx];
        pending.range(bIdx, bIdx) = 1;
    }
    while (pending) {
        pending = 0;
        for (uint32_t bIdx = 0; bIdx < NUM_BLOCKS; bIdx++) {
            uint32_t pending_bytes = (input_size[bIdx] > read_size[bIdx]) ? (input_size[bIdx] - read_size[bIdx]) : 0;
            if ((pending_bytes) && (read_idx[bIdx] == write_idx[bIdx])) {
                uint32_t pending_words = (pending_bytes - 1) / c_wordSize + 1;
                uint32_t burst_size = (pending_words > BURST_SIZE) ? BURST_SIZE : pending_words;
                uint32_t mem_read_byte_idx = read_size[bIdx] + input_idx[bIdx];
                uint32_t mem_read_word_idx = (mem_read_byte_idx) ? ((mem_read_byte_idx - 1) / c_wordSize + 1) : 0;
            gmem_rd:
                for (uint32_t i = 0; i < burst_size; i++) {
#pragma HLS PIPELINE II = 1
                    local_buffer[bIdx][i] = in[mem_read_word_idx + i];
                }
                pending.range(bIdx, bIdx) = 1;
                read_idx[bIdx] = 0;
                write_idx[bIdx] = burst_size;
                read_size[bIdx] += burst_size * c_wordSize;
            }
        }
        ap_uint<NUM_BLOCKS> terminate_all;
        terminate_all = 1;
        bool terminate = 0;
    mm2s:
        for (int i = 0; (terminate == 0) && (terminate_all != 0); i++) {
#pragma HLS PIPELINE II = 1
            for (uint8_t pb = 0; pb < NUM_BLOCKS; pb++) {
#pragma HLS UNROLL
                is_full.range(pb, pb) = outStream[pb].full();
                if (!is_full.range(pb, pb) && (read_idx[pb] != write_idx[pb])) {
                    outStream[pb] << local_buffer[pb][read_idx[pb]];
                    read_idx[pb] += 1;
                }
            }
            terminate = 0;
            for (uint32_t bIdx = 0; bIdx < NUM_BLOCKS; bIdx++) {
#pragma HLS UNROLL
                if (read_idx[bIdx] == write_idx[bIdx]) {
                    terminate_all.range(bIdx, bIdx) = 0;
                    if (read_size[bIdx] < input_size[bIdx]) {
                        terminate = 1;
                    }
                } else {
                    terminate_all.range(bIdx, bIdx) = 1;
                    pending.range(bIdx, bIdx) = 1;
                }
            }
        }
    }
}

template <int NUM_BLOCKS, int DATAWIDTH, int BURST_SIZE>
void mm2multStreamSimple(const ap_uint<DATAWIDTH>* in,
                         hls::stream<ap_uint<DATAWIDTH> > outStream[NUM_BLOCKS],
                         hls::stream<uint16_t> outSizeStream[NUM_BLOCKS],
                         const uint32_t input_idx[NUM_BLOCKS],
                         const uint32_t _input_size[NUM_BLOCKS]) {
    /**
     * @brief This module reads 512-bit data from memory interface and
     * writes to the output streams and output size streams
     *
     * @tparam DATAWIDTH input width of data bus
     * @tparam BURST_SIZE burst size of the data transfers
     * @tparam NUM_BLOCKS number of parallel blocks
     *
     * @param in input memory address
     * @param input_idx input index
     * @param outStream output stream
     * @param outSizeStream output size stream
     * @param _input_size input stream size
     */

    const int c_byteSize = 8;
    const int c_wordSize = DATAWIDTH / c_byteSize;

    ap_uint<NUM_BLOCKS> is_pending;
    uint32_t read_idx[NUM_BLOCKS];
    uint32_t read_size[NUM_BLOCKS];
    uint32_t input_size[NUM_BLOCKS];
#pragma HLS ARRAY_PARTITION variable = read_idx dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = read_size dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = input_size dim = 0 complete

    for (uint8_t vid = 0; vid < NUM_BLOCKS; vid++) {
#pragma HLS UNROLL
        read_idx[vid] = input_idx[vid] / c_wordSize;
        input_size[vid] = _input_size[vid];
        read_size[vid] = 0;
        is_pending.range(vid, vid) = 1;
    }

    while (is_pending) {
    parallel_ops:
        for (uint32_t vid = 0; vid < NUM_BLOCKS; vid++) {
#pragma HLS PIPELINE off
            bool isFull = (outSizeStream[vid]).full();
            uint32_t pendingBytes = (input_size[vid] > read_size[vid]) ? (input_size[vid] - read_size[vid]) : 0;
            is_pending.range(vid, vid) = (pendingBytes > 0) ? 1 : 0;
            uint32_t sizeWrite = 0;
            if (pendingBytes && !isFull) {
                uint32_t pendingWords = (pendingBytes - 1) / c_wordSize + 1;
                uint32_t burstSize = (pendingWords > BURST_SIZE) ? BURST_SIZE : pendingWords;
                sizeWrite = burstSize * c_wordSize;
                uint32_t rIdx = read_idx[vid];
            gmem_read:
                for (uint32_t midx = 0; midx < burstSize; midx++) {
                    outStream[vid] << in[rIdx + midx];
                }
                read_idx[vid] += burstSize;
                if (read_size[vid] + sizeWrite < input_size[vid]) {
                    outSizeStream[vid] << sizeWrite;
                    read_size[vid] += sizeWrite;
                } else {
                    outSizeStream[vid] << (input_size[vid] - read_size[vid]);
                    read_size[vid] = input_size[vid];
                }
            }
        }
    }

size_init:
    for (uint8_t vid = 0; vid < NUM_BLOCKS; vid++) {
#pragma HLS UNROLL
        outSizeStream[vid] << 0;
    }
}

template <int IN_DATAWIDTH, int OUT_DATAWIDTH>
void mm2multStreamDownSizer(hls::stream<ap_uint<IN_DATAWIDTH> >& inStream,
                            hls::stream<uint16_t>& inSizeStream,
                            hls::stream<ap_uint<OUT_DATAWIDTH> >& outStream) {
    /**
     * @brief This module reads 512-bit data from stream interface and
     * writes to the output stream in 8-bit chunks using the size stream.
     *
     * @tparam IN_DATAWIDTH input width of data bus
     * @tparam OUT_DATAWIDTH output width of the data bus
     *
     * @param inStream input stream
     * @param inSizeStream input size stream
     * @param outStream output stream
     */

    const int c_byteWidth = 8;
    const int c_inputWord = IN_DATAWIDTH / c_byteWidth;
    const int c_outWord = OUT_DATAWIDTH / c_byteWidth;
    const int factor = c_inputWord / c_outWord;
    ap_uint<IN_DATAWIDTH> inBuffer = 0;

downsizer_top:
    for (uint16_t inSize = inSizeStream.read(); inSize != 0; inSize = inSizeStream.read()) {
        uint16_t outSizeV = (inSize - 1) / c_outWord + 1;
    downsizer_assign:
        for (uint16_t itr = 0; itr < outSizeV; itr++) {
#pragma HLS PIPELINE II = 1
            int idx = itr % factor;
            if (idx == 0) inBuffer = inStream.read();
            ap_uint<OUT_DATAWIDTH> tmpValue = inBuffer.range((idx + 1) * OUT_DATAWIDTH - 1, idx * OUT_DATAWIDTH);
            outStream << tmpValue;
        }
    }
}

template <int OUT_DATAWIDTH = 8, int NUM_BLOCKS = 8, int IN_DATAWIDTH = 512, int BURST_SIZE = 16>
void mm2multStreamSize(const ap_uint<IN_DATAWIDTH>* in,
                       const uint32_t input_idx[NUM_BLOCKS],
                       hls::stream<ap_uint<OUT_DATAWIDTH> > outStream[NUM_BLOCKS],
                       const uint32_t _input_size[NUM_BLOCKS]) {
    /**
     * @brief This module reads 512-bit data from memory interface and
     * writes to the output streams in 8-bit chunks. Writing to the multiple data streams is
     * non-blocking call which is done using full() API
     *
     * @tparam NUM_BLOCKS number of parallel blocks
     * @tparam IN_DATAWIDTH input width of data bus
     * @tparam OUT_DATAWIDTH output width of the data bus
     * @tparam BURST_SIZE burst size of the data transfers
     *
     *
     * @param in input memory address
     * @param input_idx input index
     * @param outStream output stream
     * @param _input_size input size
     */

    const uint32_t c_depthOutStreamV = 2 * BURST_SIZE;
    // Array of Streams used as internal buffer.
    hls::stream<ap_uint<IN_DATAWIDTH> > outStreamV[NUM_BLOCKS];
    hls::stream<uint16_t> outStreamVSize[NUM_BLOCKS];
#pragma HLS STREAM variable = outStreamV depth = c_depthOutStreamV
#pragma HLS STREAM variable = outStreamVSize depth = 3
#pragma HLS BIND_STORAGE variable = outStreamV type = FIFO impl = SRL

#pragma HLS DATAFLOW
    xf::compression::details::mm2multStreamSimple<NUM_BLOCKS, IN_DATAWIDTH, BURST_SIZE>(in, outStreamV, outStreamVSize,
                                                                                        input_idx, _input_size);
downsizer:
    for (uint8_t vid = 0; vid < NUM_BLOCKS; vid++) {
#pragma HLS UNROLL
        xf::compression::details::mm2multStreamDownSizer<IN_DATAWIDTH, OUT_DATAWIDTH>(
            outStreamV[vid], outStreamVSize[vid], outStream[vid]);
    }
}

template <int DATAWIDTH, int BURST_SIZE>
void mm2SingleStream(const ap_uint<DATAWIDTH>* in,
                     hls::stream<ap_uint<DATAWIDTH> >& outStream,
                     hls::stream<uint16_t>& outSizeStream,
                     const uint32_t _input_size) {
    /**
     * @brief This module reads 512-bit data from memory interface and
     * writes to the output streams and output size streams
     *
     * @tparam DATAWIDTH input width of data bus
     * @tparam BURST_SIZE burst size of the data transfers
     *
     * @param in input memory address
     * @param outStream output stream
     * @param outSizeStream output size stream
     * @param _input_size input stream size
     */

    const int c_byteSize = 8;
    const int c_wordSize = DATAWIDTH / c_byteSize;
    const uint32_t c_burstSize = BURST_SIZE * c_wordSize;

    uint32_t read_idx = 0;
    uint32_t read_size = 0;
    uint32_t input_size = _input_size;

mm2StreamSimple:
    for (uint32_t idx = 0; idx < input_size; idx += c_burstSize) {
        uint32_t pendingBytes = (input_size > read_size) ? (input_size - read_size) : 0;
        uint32_t sizeWrite = 0;
        uint32_t pendingWords = (pendingBytes - 1) / c_wordSize + 1;
        uint32_t burstSize = (pendingWords > BURST_SIZE) ? BURST_SIZE : pendingWords;
        sizeWrite = burstSize * c_wordSize;
        if (read_size + sizeWrite < input_size) {
            outSizeStream << sizeWrite;
            read_size += sizeWrite;
        } else {
            outSizeStream << (input_size - read_size);
            read_size = input_size;
        }
    gmem_read:
        for (uint32_t midx = 0; midx < burstSize; midx++) {
#pragma HLS PIPELINE II = 1
            outStream << in[read_idx + midx];
        }
        read_idx += burstSize;
    }
    outSizeStream << 0;
}

template <int IN_DATAWIDTH, int OUT_DATAWIDTH>
void mm2StreamDownSizer(hls::stream<ap_uint<IN_DATAWIDTH> >& inStream,
                        hls::stream<uint16_t>& inSizeStream,
                        hls::stream<ap_uint<OUT_DATAWIDTH> >& outStream) {
    /**
     * @brief This module reads 512-bit data from stream interface and
     * writes to the output stream in 8-bit chunks using the size stream.
     *
     * @tparam IN_DATAWIDTH input width of data bus
     * @tparam OUT_DATAWIDTH output width of the data bus
     *
     * @param inStream input stream
     * @param inSizeStream input size stream
     * @param outStream output stream
     */

    const int c_byteWidth = 8;
    const int c_inputWord = IN_DATAWIDTH / c_byteWidth;
    const int c_outWord = OUT_DATAWIDTH / c_byteWidth;
    const int factor = c_inputWord / c_outWord;
    ap_uint<IN_DATAWIDTH> inBuffer = 0;

downsizer_top:
    for (uint16_t inSize = inSizeStream.read(); inSize != 0; inSize = inSizeStream.read()) {
        uint16_t outSizeV = (inSize - 1) / c_outWord + 1;
    downsizer_assign:
        for (uint16_t itr = 0; itr < outSizeV; itr++) {
#pragma HLS PIPELINE II = 1
            int idx = itr % factor;
            if (idx == 0) {
                inBuffer = inStream.read();
            } else {
                inBuffer >>= OUT_DATAWIDTH;
            }
            outStream << inBuffer.range(OUT_DATAWIDTH - 1, 0);
        }
    }
}

template <int OUT_DATAWIDTH, int GMEM_DATAWIDTH = 512, int BURST_SIZE = 16>
void mm2Stream(const ap_uint<GMEM_DATAWIDTH>* in,
               hls::stream<ap_uint<OUT_DATAWIDTH> >& outStream,
               const uint32_t _input_size) {
    /**
     * @brief This module reads 512-bit data from memory interface and
     * writes to the output streams in 8-bit chunks. Writing to the multiple data streams is
     * non-blocking call which is done using is_full() API
     *
     * @tparam IN_DATAWIDTH input width of data bus
     * @tparam OUT_DATAWIDTH output width of the data bus
     * @tparam BURST_SIZE burst size of the data transfers
     *
     *
     * @param in input memory address
     * @param outStream output stream
     * @param _input_size input stream size
     */

    const uint32_t c_depthOutStreamV = 2 * BURST_SIZE;
    // Array of Streams used as internal buffer.
    hls::stream<ap_uint<GMEM_DATAWIDTH> > outStreamV;
    hls::stream<uint16_t> outStreamVSize;
#pragma HLS STREAM variable = outStreamV depth = c_depthOutStreamV
#pragma HLS STREAM variable = outStreamVSize depth = 2
#pragma HLS BIND_STORAGE variable = outStreamV type = FIFO impl = SRL

#pragma HLS DATAFLOW
    xf::compression::details::mm2SingleStream<GMEM_DATAWIDTH, BURST_SIZE>(in, outStreamV, outStreamVSize, _input_size);
    xf::compression::details::mm2StreamDownSizer<GMEM_DATAWIDTH, OUT_DATAWIDTH>(outStreamV, outStreamVSize, outStream);
}

template <int DATAWIDTH, int BURST_SIZE, int NUM_BLOCKS>
void mm2sNbRoundOff(const ap_uint<DATAWIDTH>* in,
                    const uint32_t _input_idx[NUM_BLOCKS],
                    hls::stream<ap_uint<DATAWIDTH> > outStream[NUM_BLOCKS],
                    const uint32_t _input_size[NUM_BLOCKS]) {
    /**
     * @brief This module is same as mm2sNb API but with an extra handling
     * rounding off the indexing to maximum buffer size for P2P decompression.
     *
     * @tparam DATAWIDTH width of data bus
     * @tparam BURST_SIZE burst size of the data transfers
     * @tparam NUM_BLOCKS number of blocks
     *
     * @param in input memory address
     * @param _input_idx input index
     * @param outStream output stream
     * @param _input_size input stream size
     * @param max_buffer_size_in_bytes Maximum buffer size for indexing
     */

    const int c_byteSize = 8;
    const int c_wordSize = DATAWIDTH / c_byteSize;
    ap_uint<DATAWIDTH> local_buffer[NUM_BLOCKS][BURST_SIZE];
#pragma HLS ARRAY_PARTITION variable = local_buffer dim = 1 complete
#pragma HLS BIND_STORAGE variable = local_buffer type = RAM_2P impl = LUTRAM
    uint32_t read_idx[NUM_BLOCKS];
    uint32_t write_idx[NUM_BLOCKS];
    uint32_t read_size[NUM_BLOCKS];
    uint32_t input_idx[NUM_BLOCKS];
    uint32_t input_size[NUM_BLOCKS];
#pragma HLS ARRAY_PARTITION variable = read_idx dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = write_idx dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = read_size dim = 0 complete
    ap_uint<NUM_BLOCKS> pending;
    ap_uint<NUM_BLOCKS> is_full;
    for (uint32_t bIdx = 0; bIdx < NUM_BLOCKS; bIdx++) {
#pragma HLS UNROLL
        read_idx[bIdx] = 0;
        write_idx[bIdx] = 0;
        read_size[bIdx] = 0;
        input_idx[bIdx] = _input_idx[bIdx];
        input_size[bIdx] = _input_size[bIdx] + (input_idx[bIdx] % c_wordSize);
        pending.range(bIdx, bIdx) = 1;
    }
    while (pending) {
        pending = 0;
        for (uint32_t bIdx = 0; bIdx < NUM_BLOCKS; bIdx++) {
            uint32_t pending_bytes = (input_size[bIdx] > read_size[bIdx]) ? (input_size[bIdx] - read_size[bIdx]) : 0;
            if ((pending_bytes) && (read_idx[bIdx] == write_idx[bIdx])) {
                uint32_t pending_words = (pending_bytes - 1) / c_wordSize + 1;
                uint32_t burst_size = (pending_words > BURST_SIZE) ? BURST_SIZE : pending_words;
                uint32_t mem_read_byte_idx = read_size[bIdx] + input_idx[bIdx];
                uint32_t mem_read_word_idx = 0;
                if (mem_read_byte_idx)
                    mem_read_word_idx = (mem_read_byte_idx % c_wordSize) ? (mem_read_byte_idx - 1) / c_wordSize
                                                                         : ((mem_read_byte_idx - 1) / c_wordSize + 1);
                else
                    mem_read_word_idx = 0;

            gmem_rd:
                for (uint32_t i = 0; i < burst_size; i++) {
#pragma HLS PIPELINE II = 1
                    local_buffer[bIdx][i] = in[mem_read_word_idx + i];
                }
                pending.range(bIdx, bIdx) = 1;
                read_idx[bIdx] = 0;
                write_idx[bIdx] = burst_size;
                read_size[bIdx] += burst_size * c_wordSize;
            }
        }
        ap_uint<NUM_BLOCKS> terminate_all;
        terminate_all = 1;
        bool terminate = 0;
    mm2s:
        for (int i = 0; (terminate == 0) && (terminate_all != 0); i++) {
#pragma HLS PIPELINE II = 1
            for (uint8_t pb = 0; pb < NUM_BLOCKS; pb++) {
#pragma HLS UNROLL
                is_full.range(pb, pb) = outStream[pb].full();
                if (!is_full.range(pb, pb) && (read_idx[pb] != write_idx[pb])) {
                    outStream[pb] << local_buffer[pb][read_idx[pb]];
                    read_idx[pb] += 1;
                }
            }
            terminate = 0;
            for (uint32_t bIdx = 0; bIdx < NUM_BLOCKS; bIdx++) {
#pragma HLS UNROLL
                if (read_idx[bIdx] == write_idx[bIdx]) {
                    terminate_all.range(bIdx, bIdx) = 0;
                    if (read_size[bIdx] < input_size[bIdx]) {
                        terminate = 1;
                    }
                } else {
                    terminate_all.range(bIdx, bIdx) = 1;
                    pending.range(bIdx, bIdx) = 1;
                }
            }
        }
    }
}

template <int DATAWIDTH>
void mm2sSimple(const ap_uint<DATAWIDTH>* in, hls::stream<ap_uint<DATAWIDTH> >& outstream, uint32_t inputSize) {
    /**
     * @brief Read data from 512-bit wide axi memory interface and
     *        write to stream.
     *
     * @tparam DATAWIDTH    width of data bus
     *
     * @param in            pointer to input memory
     * @param outstream     output stream
     * @param inputSize     size of the data
     *
     */
    const int c_byte_size = 8;
    const int c_word_size = DATAWIDTH / c_byte_size;
    const int inSize_gmemwidth = (inputSize - 1) / c_word_size + 1;

mm2s_simple:
    for (int i = 0; i < inSize_gmemwidth; i++) {
#pragma HLS PIPELINE II = 1
        outstream << in[i];
    }
}

template <int DATAWIDTH, int BURST_SIZE>
void mm2sSimple(const ap_uint<DATAWIDTH>* in,
                hls::stream<ap_uint<DATAWIDTH> >& outstream,
                uint32_t inputSize,
                uint32_t numItr) {
    /**
     * @brief Read data from DATAWIDTH wide axi memory interface and
     *        write to stream.
     *
     * @tparam DATAWIDTH    width of data bus
     *
     * @param in            pointer to input memory
     * @param outstream     output stream
     * @param inputSize     size of the data
     *
     */
    const int c_byte_size = 8;
    const int c_word_size = DATAWIDTH / c_byte_size;
    const int inSize_gmemwidth = (inputSize - 1) / c_word_size + 1;

    int allignedwidth = inSize_gmemwidth / BURST_SIZE;
    allignedwidth = ((inSize_gmemwidth - allignedwidth) > 0) ? allignedwidth + 1 : allignedwidth;

    for (auto z = 0; z < numItr; z++) {
        int i = 0;
        ap_uint<DATAWIDTH> temp;
    mm2s_simple:
        for (; i < allignedwidth * BURST_SIZE; i += BURST_SIZE) {
            for (uint32_t j = 0; j < BURST_SIZE; j++) {
#pragma HLS PIPELINE II = 1
                temp = in[i + j];
                if ((i + j) < inSize_gmemwidth) outstream << temp;
            }
        }
    }
}

template <int DATAWIDTH, int BURST_SIZE>
void mm2sSimple(const ap_uint<DATAWIDTH>* in, hls::stream<ap_uint<DATAWIDTH> >& outstream, uint32_t inputSize) {
    /**
     * @brief Read data from DATAWIDTH wide axi memory interface and
     *        write to stream.
     *
     * @tparam DATAWIDTH    width of data bus
     *
     * @param in            pointer to input memory
     * @param outstream     output stream
     * @param inputSize     size of the data
     *
     */
    const int c_byte_size = 8;
    const int c_word_size = DATAWIDTH / c_byte_size;
    const int inSize_gmemwidth = (inputSize - 1) / c_word_size + 1;

    int allignedwidth = inSize_gmemwidth / BURST_SIZE;
    allignedwidth = ((inSize_gmemwidth - allignedwidth) > 0) ? allignedwidth + 1 : allignedwidth;

    int i = 0;
    ap_uint<DATAWIDTH> temp;
mm2s_simple:
    for (; i < allignedwidth * BURST_SIZE; i += BURST_SIZE) {
        for (uint32_t j = 0; j < BURST_SIZE; j++) {
#pragma HLS PIPELINE II = 1
            temp = in[i + j];
            if ((i + j) < inSize_gmemwidth) outstream << temp;
        }
    }
}

template <int DATAWIDTH, int BURST_SIZE>
void mm2sSimple(const ap_uint<DATAWIDTH>* in,
                hls::stream<ap_uint<DATAWIDTH> >& outstream,
                hls::stream<uint32_t>& sizeStream,
                hls::stream<uint32_t>& inSize) {
    /**
     * @brief Read data from DATAWIDTH wide axi memory interface and
     *        write to stream.
     *
     * @tparam DATAWIDTH    width of data bus
     *
     * @param in            pointer to input memory
     * @param outstream     output stream
     * @param inSize        size of the data
     * @param sizeStream    o/p size of the data
     *
     */
    const int c_byte_size = 8;
    const int c_word_size = DATAWIDTH / c_byte_size;
    uint32_t inputSize = inSize.read();
    sizeStream << inputSize;
    uint32_t inSize_gmemwidth = (inputSize - 1) / c_word_size + 1;

    int allignedwidth = inSize_gmemwidth / BURST_SIZE;
    allignedwidth = ((inSize_gmemwidth - allignedwidth) > 0) ? allignedwidth + 1 : allignedwidth;

    int i = 0;
    ap_uint<DATAWIDTH> temp;
mm2s_simple:
    for (; i < allignedwidth * BURST_SIZE; i += BURST_SIZE) {
    burst_transfer:
        for (uint32_t j = 0; j < BURST_SIZE; j++) {
#pragma HLS PIPELINE II = 1
            temp = in[i + j];
            if ((i + j) < inSize_gmemwidth) outstream << temp;
        }
    }
}

template <int DATAWIDTH, int BURST_SIZE>
void mm2Stream(const ap_uint<DATAWIDTH>* in,
               hls::stream<ap_uint<DATAWIDTH> >& outstream,
               hls::stream<ap_uint<32> >& checksumStream,
               uint32_t* checksumData,
               uint32_t inputSize,
               hls::stream<uint32_t>& outSizeStream,
               bool checksumType,
               hls::stream<ap_uint<2> >& checksumTypeStream) {
    /**
     * @brief Read data from DATAWIDTH wide axi memory interface and
     *        write to stream.
     *
     * @tparam DATAWIDTH    width of data bus
     *
     * @param in            pointer to input memory
     * @param outstream     output stream
     * @param inputSize     size of the data
     * @param outSizeStream output size stream
     *
     */
    const int c_byte_size = 8;
    const int c_word_size = DATAWIDTH / c_byte_size;
    const int inSize_gmemwidth = (inputSize - 1) / c_word_size + 1;

    checksumTypeStream << checksumType;
    // exit condition for checksum kernel
    checksumTypeStream << 3;

    outSizeStream << inputSize;
    checksumStream << checksumData[0];

    uint32_t allignedwidth = inSize_gmemwidth / BURST_SIZE;
    allignedwidth = ((inSize_gmemwidth - allignedwidth) > 0) ? allignedwidth + 1 : allignedwidth;

    uint32_t i = 0;
    ap_uint<DATAWIDTH> temp;
mm2s:
    for (; i < allignedwidth * BURST_SIZE; i += BURST_SIZE) {
        for (uint16_t j = 0; j < BURST_SIZE; j++) {
#pragma HLS PIPELINE II = 1
            temp = in[i + j];
            if ((i + j) < inSize_gmemwidth) outstream << temp;
        }
    }
}

} // namespace details
} // namespace compression
} // namespace xf

#endif // _XFCOMPRESSION_MM2S_HPP_
