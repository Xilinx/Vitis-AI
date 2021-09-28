/*
 * (c) Copyright 2021 Xilinx, Inc. All rights reserved.
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
#ifndef _XFCOMPRESSION_ZLIB_COMPRESS_HPP_
#define _XFCOMPRESSION_ZLIB_COMPRESS_HPP_

/**
 * @file zlib_compress.hpp
 * @brief Header for modules used in ZLIB compression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include <ap_int.h>
#include <assert.h>
#include <stdint.h>

#include "compress_utils.hpp"
#include "zlib_compress_details.hpp"
#include "zlib_specs.hpp"
#include "lz_optional.hpp"
#include "lz_compress.hpp"
#include "huffman_treegen.hpp"
#include "huffman_encoder.hpp"
#include "mm2s.hpp"
#include "s2mm.hpp"
#include "stream_downsizer.hpp"
#include "stream_upsizer.hpp"
#include "axi_stream_utils.hpp"
#include "ap_axi_sdata.h"
#include "checksum_wrapper.hpp"

namespace xf {
namespace compression {

template <int NUM_BLOCKS = 8,
          int MAX_BLOCK_SIZE = 32 * 1024,
          int MAX_FREQ_DWIDTH = 24,
          int MATCH_LEN = 6,
          int MIN_MATCH = 3,
          int LZ_MAX_OFFSET_LIMIT = 32 * 1024,
          int MAX_MATCH_LEN = 255>
void lz77Compress(hls::stream<IntVectorStream_dt<8, 1> >& inStream,
                  hls::stream<IntVectorStream_dt<9, 1> >& lz77OutStream,
                  hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& outTreeStream,
                  uint8_t i) {
#pragma HLS dataflow
    hls::stream<IntVectorStream_dt<32, 1> > compressedStream("compressedStream");
    hls::stream<IntVectorStream_dt<32, 1> > boosterStream("boosterStream");
#pragma HLS STREAM variable = compressedStream depth = 4
#pragma HLS STREAM variable = boosterStream depth = 4

    xf::compression::lzCompress<MAX_BLOCK_SIZE, uint32_t, MATCH_LEN, MIN_MATCH, LZ_MAX_OFFSET_LIMIT, NUM_BLOCKS>(
        inStream, compressedStream, i);
    xf::compression::lzBooster<MAX_MATCH_LEN>(compressedStream, boosterStream);
    xf::compression::lz77DivideStream<MAX_FREQ_DWIDTH>(boosterStream, lz77OutStream, outTreeStream);
}

template <int NUM_BLOCKS = 8,
          int MAX_BLOCK_SIZE = 32 * 1024,
          int MATCH_LEN = 6,
          int MIN_MATCH = 3,
          int LZ_MAX_OFFSET_LIMIT = 32 * 1024,
          int MAX_MATCH_LEN = 255>
void lz77CompressStatic(hls::stream<IntVectorStream_dt<8, 1> >& inStream,
                        hls::stream<IntVectorStream_dt<9, 1> >& lz77Out,
                        uint8_t i) {
#pragma HLS dataflow
    hls::stream<IntVectorStream_dt<32, 1> > compressedStream("compressedStream");
    hls::stream<IntVectorStream_dt<32, 1> > boosterStream("boosterStream");
#pragma HLS STREAM variable = compressedStream depth = 4
#pragma HLS STREAM variable = boosterStream depth = 4

    xf::compression::lzCompress<MAX_BLOCK_SIZE, uint32_t, MATCH_LEN, MIN_MATCH, LZ_MAX_OFFSET_LIMIT, NUM_BLOCKS>(
        inStream, compressedStream, i);
    xf::compression::lzBooster<MAX_MATCH_LEN>(compressedStream, boosterStream);
    xf::compression::lz77DivideStatic(boosterStream, lz77Out);
}

void zlibHuffmanEncoder(hls::stream<IntVectorStream_dt<9, 1> >& inStream,
                        hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> >& hufCodeInStream,
                        hls::stream<IntVectorStream_dt<8, 2> >& huffOutStream) {
#pragma HLS dataflow
    hls::stream<IntVectorStream_dt<32, 1> > encodedOutStream("encodedOutStream");
    hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> > hufCodeStream("hufCodeStream");
#pragma HLS STREAM variable = encodedOutStream depth = 4
#pragma HLS STREAM variable = hufCodeStream depth = 4

    xf::compression::details::huffmanProcessingUnit(inStream, encodedOutStream);
    xf::compression::huffmanEncoderStream(encodedOutStream, hufCodeInStream, hufCodeStream);
    xf::compression::details::bitPackingStream(hufCodeStream, huffOutStream);
}

void zlibHuffmanEncoderStatic(hls::stream<IntVectorStream_dt<9, 1> >& inStream,
                              hls::stream<IntVectorStream_dt<8, 2> >& huffOut) {
#pragma HLS DATAFLOW
    hls::stream<IntVectorStream_dt<32, 1> > encodedOutStream("encodedOutStream");
    hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> > hufCodeStream("hufCodeStream");
#pragma HLS STREAM variable = encodedOutStream depth = 4
#pragma HLS STREAM variable = hufCodeStream depth = 4

    xf::compression::details::huffmanProcessingUnit(inStream, encodedOutStream);
    xf::compression::huffmanEncoderStatic(encodedOutStream, hufCodeStream);
    xf::compression::details::bitPackingStatic(hufCodeStream, huffOut);
}

template <int BLOCK_SIZE_IN_KB = 32, int NUM_BLOCKS = 8>
void gzipMulticoreCompression(hls::stream<ap_uint<64> >& inStream,
                              hls::stream<uint32_t>& inSizeStream,
                              hls::stream<ap_uint<32> >& checksumInitStream,
                              hls::stream<ap_uint<64> >& outStream,
                              hls::stream<bool>& outStreamEos,
                              hls::stream<uint32_t>& outSizeStream,
                              hls::stream<ap_uint<32> >& checksumOutStream,
                              hls::stream<ap_uint<2> >& checksumTypeStream) {
#pragma HLS dataflow
    constexpr int c_blockSizeInKb = BLOCK_SIZE_IN_KB;
    constexpr int c_blockSize = c_blockSizeInKb * 1024;
    constexpr int c_numBlocks = NUM_BLOCKS;
    constexpr int c_twiceNumBlocks = 2 * NUM_BLOCKS;
    constexpr int c_blckEosDepth = (NUM_BLOCKS * NUM_BLOCKS) + NUM_BLOCKS;
    constexpr int c_defaultDepth = 4;
    constexpr int c_dwidth = 64;
    constexpr int c_maxBlockSize = 32768;
    constexpr int c_minBlockSize = 64;
    constexpr int c_checksumParallelBytes = 8;
    constexpr int c_bufferFifoDepth = c_blockSize / 8;
    constexpr int c_freq_dwidth = getDataPortWidth(c_blockSize);
    constexpr int c_size_dwidth = getDataPortWidth(c_checksumParallelBytes);
    constexpr int c_strdBlockDepth = c_minBlockSize / c_checksumParallelBytes;

    // Assertion for Maximum Supported Parallel Cores
    assert(c_numBlocks <= 8);

    hls::stream<ap_uint<c_dwidth> > checksumStream("checksumStream");
    hls::stream<ap_uint<5> > checksumSizeStream("checksumSizeStream");
    hls::stream<ap_uint<c_dwidth> > coreStream("coreStream");
    hls::stream<uint32_t> coreSizeStream("coreSizeStream");
    hls::stream<ap_uint<c_dwidth> > distStream[c_numBlocks];
    hls::stream<ap_uint<c_freq_dwidth> > distSizeStream[c_numBlocks];
    hls::stream<ap_uint<c_dwidth> > strdStream;
    hls::stream<ap_uint<16> > strdSizeStream;
    hls::stream<ap_uint<17> > upsizedCntr[c_numBlocks];
    hls::stream<IntVectorStream_dt<8, 1> > downStream[c_numBlocks];
    hls::stream<IntVectorStream_dt<9, 1> > lz77Stream[c_numBlocks];
    hls::stream<IntVectorStream_dt<8, 2> > huffStream[c_numBlocks];
    hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> > hufCodeStream[c_numBlocks];
    hls::stream<ap_uint<72> > lz77UpsizedStream[c_numBlocks]; // 72 bits
    hls::stream<ap_uint<9> > lz77PassStream[c_numBlocks];     // 9 bits
    hls::stream<IntVectorStream_dt<9, 1> > lz77DownsizedStream[c_numBlocks];
    hls::stream<IntVectorStream_dt<c_freq_dwidth, 1> > lz77Tree[c_numBlocks];
    hls::stream<ap_uint<c_dwidth + c_size_dwidth> > mergeStream[c_numBlocks];

#pragma HLS STREAM variable = checksumStream depth = c_defaultDepth
#pragma HLS STREAM variable = checksumSizeStream depth = c_defaultDepth
#pragma HLS STREAM variable = coreStream depth = c_defaultDepth
#pragma HLS STREAM variable = coreSizeStream depth = c_defaultDepth

#pragma HLS STREAM variable = distSizeStream depth = c_numBlocks
#pragma HLS STREAM variable = downStream depth = c_numBlocks
#pragma HLS STREAM variable = huffStream depth = c_numBlocks
#pragma HLS STREAM variable = lz77Tree depth = c_numBlocks
#pragma HLS STREAM variable = hufCodeStream depth = c_numBlocks
#pragma HLS STREAM variable = upsizedCntr depth = c_numBlocks
#pragma HLS STREAM variable = lz77DownsizedStream depth = c_numBlocks
#pragma HLS STREAM variable = lz77Stream depth = c_numBlocks

#pragma HLS STREAM variable = strdSizeStream depth = c_twiceNumBlocks
#pragma HLS STREAM variable = strdStream depth = c_strdBlockDepth
#pragma HLS STREAM variable = mergeStream depth = c_bufferFifoDepth
#pragma HLS STREAM variable = distStream depth = c_bufferFifoDepth

#pragma HLS BIND_STORAGE variable = lz77Tree type = FIFO impl = SRL

    if (BLOCK_SIZE_IN_KB >= 32) {
#pragma HLS STREAM variable = lz77UpsizedStream depth = c_bufferFifoDepth
#pragma HLS BIND_STORAGE variable = lz77UpsizedStream type = FIFO impl = URAM
#pragma HLS BIND_STORAGE variable = distStream type = FIFO impl = URAM
#pragma HLS BIND_STORAGE variable = mergeStream type = FIFO impl = URAM
    } else {
#pragma HLS STREAM variable = lz77PassStream depth = c_blockSize
#pragma HLS BIND_STORAGE variable = lz77PassStream type = FIFO impl = BRAM
#pragma HLS BIND_STORAGE variable = distStream type = FIFO impl = BRAM
#pragma HLS BIND_STORAGE variable = mergeStream type = FIFO impl = BRAM
    }

    // send input data to both checksum and for compression
    xf::compression::details::dataDuplicator<c_dwidth, c_checksumParallelBytes>(
        inStream, inSizeStream, checksumStream, checksumSizeStream, coreStream, coreSizeStream);

    // checksum kernel
    xf::compression::checksum32<c_checksumParallelBytes>(checksumInitStream, checksumStream, checksumSizeStream,
                                                         checksumOutStream, checksumTypeStream);

    // distrubute block size data into each block in round-robin fashion
    details::multicoreDistributor<c_freq_dwidth, c_dwidth, c_numBlocks, c_blockSize>(
        coreStream, coreSizeStream, strdStream, strdSizeStream, distStream, distSizeStream);

    // Parallel Buffers
    for (uint8_t i = 0; i < c_numBlocks; i++) {
#pragma HLS UNROLL
        xf::compression::details::streamDownSizerSize<c_dwidth, 8, c_freq_dwidth>(distStream[i], distSizeStream[i],
                                                                                  downStream[i]);
        lz77Compress<c_numBlocks, c_blockSize, c_freq_dwidth>(downStream[i], lz77Stream[i], lz77Tree[i], i);

        if (BLOCK_SIZE_IN_KB >= 32) {
            xf::compression::details::bufferUpsizer<9, 72>(lz77Stream[i], lz77UpsizedStream[i], upsizedCntr[i]);
        } else {
            xf::compression::details::sendBuffer<9>(lz77Stream[i], lz77PassStream[i], upsizedCntr[i]);
        }
    }

    // Single Call Treegen
    details::zlibTreegenStreamWrapper<c_numBlocks>(lz77Tree, hufCodeStream);

    // Parallel Huffman
    for (uint8_t i = 0; i < c_numBlocks; i++) {
#pragma HLS UNROLL
        if (BLOCK_SIZE_IN_KB >= 32) {
            xf::compression::details::bufferDownsizer<72, 9>(lz77UpsizedStream[i], lz77DownsizedStream[i],
                                                             upsizedCntr[i]);
        } else {
            xf::compression::details::receiveBuffer<9>(lz77PassStream[i], lz77DownsizedStream[i], upsizedCntr[i]);
        }

        zlibHuffmanEncoder(lz77DownsizedStream[i], hufCodeStream[i], huffStream[i]);

        xf::compression::details::simpleStreamUpsizer<16, c_dwidth, c_size_dwidth>(huffStream[i], mergeStream[i]);
    }

    // read all num block data in round-robin fashion and write into single outstream
    details::multicoreMerger<c_dwidth, c_size_dwidth, c_numBlocks, c_blockSize>(mergeStream, strdStream, strdSizeStream,
                                                                                outStream, outStreamEos, outSizeStream);
}

template <int BLOCK_SIZE_IN_KB = 32, int NUM_BLOCKS = 8, int STRATEGY = 0> // STRATEGY -> 0: GZIP; 1: ZLIB
void gzipMulticoreStaticCompressStream(hls::stream<IntVectorStream_dt<8, 8> >& inStream,
                                       hls::stream<IntVectorStream_dt<8, 8> >& outStream) {
#pragma HLS dataflow
    // Constants
    constexpr int c_blockSizeInKb = BLOCK_SIZE_IN_KB;
    constexpr int c_blockSize = c_blockSizeInKb * 1024;
    constexpr int c_numBlocks = NUM_BLOCKS;
    constexpr int c_twiceNumBlocks = 2 * NUM_BLOCKS;
    constexpr int c_blckEosDepth = (NUM_BLOCKS * NUM_BLOCKS) + NUM_BLOCKS;
    constexpr int c_defaultDepth = 4;
    constexpr int c_dwidth = 64;
    constexpr int c_maxBlockSize = 32768;
    constexpr int c_minBlockSize = 64;
    constexpr int c_checksumParallelBytes = 8;
    constexpr int c_bufferFifoDepth = c_blockSize / 8;
    constexpr int c_freq_dwidth = getDataPortWidth(c_blockSize);
    constexpr int c_size_dwidth = getDataPortWidth(c_checksumParallelBytes);
    constexpr int c_strdBlockDepth = c_minBlockSize / c_checksumParallelBytes;

    // Assertion for Maximum Supported Parallel Cores
    assert(c_numBlocks <= 8);

    // Internal Streams
    hls::stream<ap_uint<c_dwidth + c_size_dwidth> > coreStream("coreStream");
    hls::stream<ap_uint<c_dwidth + c_size_dwidth> > packedStream("packedStream");
    hls::stream<ap_uint<c_dwidth + c_size_dwidth> > distStream[c_numBlocks];
    hls::stream<ap_uint<c_dwidth + c_size_dwidth> > mergeStream[c_numBlocks];
    hls::stream<ap_uint<c_dwidth> > checksumStream("checksumStream");
    hls::stream<ap_uint<c_dwidth> > strdStream("strdStream");

    hls::stream<IntVectorStream_dt<8, 1> > downStream[c_numBlocks];
    hls::stream<IntVectorStream_dt<9, 1> > lz77Stream[c_numBlocks];
    hls::stream<IntVectorStream_dt<8, 2> > huffStream[c_numBlocks];
    hls::stream<IntVectorStream_dt<c_freq_dwidth, 1> > lz77Tree[c_numBlocks];
    hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> > hufCodeStream[c_numBlocks];

    hls::stream<ap_uint<5> > checksumSizeStream("checksumSizeStream");
    hls::stream<ap_uint<16> > strdSizeStream("strdSizeStream");
    hls::stream<ap_uint<32> > checksumInitStream("checksumInitStream");
    hls::stream<ap_uint<32> > checksumOutStream("checksumOutStream");
    hls::stream<ap_uint<4> > coreIdxStream("coreIdxStream");

    hls::stream<uint32_t> fileSizeStream("fileSizeStream");

    hls::stream<bool> blckEosStream("blckEosStream");

#pragma HLS STREAM variable = checksumStream depth = c_defaultDepth
#pragma HLS STREAM variable = checksumSizeStream depth = c_defaultDepth
#pragma HLS STREAM variable = coreStream depth = c_defaultDepth
#pragma HLS STREAM variable = packedStream depth = c_defaultDepth
#pragma HLS STREAM variable = coreIdxStream depth = c_defaultDepth
#pragma HLS STREAM variable = checksumInitStream depth = c_defaultDepth

#pragma HLS STREAM variable = downStream depth = c_numBlocks
#pragma HLS STREAM variable = huffStream depth = c_numBlocks
#pragma HLS STREAM variable = lz77Tree depth = c_numBlocks
#pragma HLS STREAM variable = hufCodeStream depth = c_numBlocks
#pragma HLS STREAM variable = lz77Stream depth = c_numBlocks

#pragma HLS STREAM variable = strdSizeStream depth = c_twiceNumBlocks
#pragma HLS STREAM variable = fileSizeStream depth = c_twiceNumBlocks
#pragma HLS STREAM variable = checksumOutStream depth = c_twiceNumBlocks

#pragma HLS STREAM variable = mergeStream depth = c_bufferFifoDepth
#pragma HLS STREAM variable = distStream depth = c_bufferFifoDepth
#pragma HLS STREAM variable = strdStream depth = c_strdBlockDepth
#pragma HLS STREAM variable = blckEosStream depth = c_blckEosDepth

    if (BLOCK_SIZE_IN_KB >= 32) {
#pragma HLS BIND_STORAGE variable = distStream type = FIFO impl = URAM
#pragma HLS BIND_STORAGE variable = mergeStream type = FIFO impl = URAM
    } else {
#pragma HLS BIND_STORAGE variable = distStream type = FIFO impl = BRAM
#pragma HLS BIND_STORAGE variable = mergeStream type = FIFO impl = BRAM
    }
#pragma HLS BIND_STORAGE variable = lz77Tree type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = strdStream type = FIFO impl = SRL

    // send input data to both checksum and for compression
    xf::compression::details::dataDuplicator<c_dwidth, c_size_dwidth, c_checksumParallelBytes, STRATEGY>(
        inStream, checksumInitStream, checksumStream, checksumSizeStream, coreStream);

    // checksum size less kernel
    if (STRATEGY == 0) { // CRC-32
        xf::compression::details::crc32<c_checksumParallelBytes>(checksumInitStream, checksumStream, checksumSizeStream,
                                                                 checksumOutStream);
    } else { // Adler-32
        xf::compression::details::adler32<c_checksumParallelBytes>(checksumInitStream, checksumStream,
                                                                   checksumSizeStream, checksumOutStream);
    }

    // distribute block size data into each block in round-robin fashion
    details::multicoreDistributor<uint32_t, c_dwidth, c_size_dwidth, c_numBlocks, c_blockSize, c_minBlockSize,
                                  STRATEGY>(coreStream, fileSizeStream, strdStream, strdSizeStream, blckEosStream,
                                            coreIdxStream, distStream);

    // Parallel Buffers
    for (uint8_t i = 0; i < c_numBlocks; i++) {
#pragma HLS UNROLL
        xf::compression::details::bufferDownsizer<c_dwidth, 8, c_size_dwidth>(distStream[i], downStream[i]);

        lz77CompressStatic<c_numBlocks, c_blockSize>(downStream[i], lz77Stream[i], i);

        zlibHuffmanEncoderStatic(lz77Stream[i], huffStream[i]);

        xf::compression::details::simpleStreamUpsizer<16, c_dwidth, c_size_dwidth>(huffStream[i], mergeStream[i]);
    }

    // GZIP/ZLIB Data Packer
    xf::compression::details::gzipZlibPackerEngine<c_numBlocks, STRATEGY>(
        mergeStream, packedStream, strdStream, strdSizeStream, fileSizeStream, checksumOutStream, coreIdxStream,
        blckEosStream);

    // Byte Alignment and Packing into a Single Stream
    xf::compression::details::bytePacker<c_dwidth, c_size_dwidth>(packedStream, outStream);
}

template <int BLOCK_SIZE_IN_KB = 32, int NUM_BLOCKS = 8, int STRATEGY = 0> // STRATEGY -> 0: GZIP; 1: ZLIB
void gzipMulticoreCompressStream(hls::stream<IntVectorStream_dt<8, 8> >& inStream,
                                 hls::stream<IntVectorStream_dt<8, 8> >& outStream) {
#pragma HLS dataflow
    // Constants
    constexpr int c_blockSizeInKb = BLOCK_SIZE_IN_KB;
    constexpr int c_blockSize = c_blockSizeInKb * 1024;
    constexpr int c_numBlocks = NUM_BLOCKS;
    constexpr int c_twiceNumBlocks = 2 * NUM_BLOCKS;
    constexpr int c_blckEosDepth = (NUM_BLOCKS * NUM_BLOCKS) + NUM_BLOCKS;
    constexpr int c_defaultDepth = 4;
    constexpr int c_dwidth = 64;
    constexpr int c_maxBlockSize = 32768;
    constexpr int c_minBlockSize = 64;
    constexpr int c_checksumParallelBytes = 8;
    constexpr int c_bufferFifoDepth = c_blockSize / 8;
    constexpr int c_freq_dwidth = getDataPortWidth(c_blockSize);
    constexpr int c_size_dwidth = getDataPortWidth(c_checksumParallelBytes);
    constexpr int c_strdBlockDepth = c_minBlockSize / c_checksumParallelBytes;

    // Assertion for Maximum Supported Parallel Cores
    assert(c_numBlocks <= 8);

    // Internal Streams
    hls::stream<ap_uint<c_dwidth + c_size_dwidth> > coreStream("coreStream");
    hls::stream<ap_uint<c_dwidth + c_size_dwidth> > packedStream("packedStream");
    hls::stream<ap_uint<c_dwidth + c_size_dwidth> > distStream[c_numBlocks];
    hls::stream<ap_uint<c_dwidth + c_size_dwidth> > mergeStream[c_numBlocks];
    hls::stream<ap_uint<c_dwidth> > checksumStream("checksumStream");
    hls::stream<ap_uint<c_dwidth> > strdStream("strdStream");

    hls::stream<IntVectorStream_dt<8, 1> > downStream[c_numBlocks];
    hls::stream<IntVectorStream_dt<9, 1> > lz77Stream[c_numBlocks];
    hls::stream<IntVectorStream_dt<8, 2> > huffStream[c_numBlocks];
    hls::stream<IntVectorStream_dt<9, 1> > lz77DownsizedStream[c_numBlocks];
    hls::stream<IntVectorStream_dt<c_freq_dwidth, 1> > lz77Tree[c_numBlocks];
    hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> > hufCodeStream[c_numBlocks];

    hls::stream<ap_uint<5> > checksumSizeStream("checksumSizeStream");
    hls::stream<ap_uint<16> > strdSizeStream("strdSizeStream");
    hls::stream<ap_uint<17> > upsizedCntr[c_numBlocks];
    hls::stream<ap_uint<32> > checksumInitStream("checksumInitStream");
    hls::stream<ap_uint<32> > checksumOutStream("checksumOutStream");
    hls::stream<ap_uint<72> > lz77UpsizedStream[c_numBlocks]; // 72 bits
    hls::stream<ap_uint<9> > lz77PassStream[c_numBlocks];     // 9 bits
    hls::stream<ap_uint<4> > coreIdxStream("coreIdxStream");
    hls::stream<ap_uint<4> > coreIdxStreamArr[2];

    hls::stream<uint32_t> fileSizeStream("fileSizeStream");

    hls::stream<bool> blckEosStream("blckEosStream");

#pragma HLS STREAM variable = checksumStream depth = c_defaultDepth
#pragma HLS STREAM variable = checksumSizeStream depth = c_defaultDepth
#pragma HLS STREAM variable = coreStream depth = c_defaultDepth
#pragma HLS STREAM variable = packedStream depth = c_defaultDepth
#pragma HLS STREAM variable = coreIdxStream depth = c_defaultDepth
#pragma HLS STREAM variable = checksumInitStream depth = c_defaultDepth

#pragma HLS STREAM variable = downStream depth = c_numBlocks
#pragma HLS STREAM variable = huffStream depth = c_numBlocks
#pragma HLS STREAM variable = lz77Tree depth = c_numBlocks
#pragma HLS STREAM variable = hufCodeStream depth = c_numBlocks
#pragma HLS STREAM variable = upsizedCntr depth = c_numBlocks
#pragma HLS STREAM variable = lz77DownsizedStream depth = c_numBlocks
#pragma HLS STREAM variable = lz77Stream depth = c_numBlocks

#pragma HLS STREAM variable = strdSizeStream depth = c_twiceNumBlocks
#pragma HLS STREAM variable = fileSizeStream depth = c_twiceNumBlocks
#pragma HLS STREAM variable = coreIdxStreamArr depth = c_twiceNumBlocks
#pragma HLS STREAM variable = checksumOutStream depth = c_twiceNumBlocks

#pragma HLS STREAM variable = mergeStream depth = c_bufferFifoDepth
#pragma HLS STREAM variable = distStream depth = c_bufferFifoDepth
#pragma HLS STREAM variable = strdStream depth = c_strdBlockDepth
#pragma HLS STREAM variable = blckEosStream depth = c_blckEosDepth

#pragma HLS BIND_STORAGE variable = lz77Tree type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = strdStream type = FIFO impl = SRL

    // URAM buffering used only for Octa-core designs
    if (BLOCK_SIZE_IN_KB >= 32) {
#pragma HLS STREAM variable = lz77UpsizedStream depth = c_bufferFifoDepth
#pragma HLS BIND_STORAGE variable = distStream type = FIFO impl = URAM
#pragma HLS BIND_STORAGE variable = mergeStream type = FIFO impl = URAM
#pragma HLS BIND_STORAGE variable = lz77UpsizedStream type = FIFO impl = URAM
    } else {
#pragma HLS STREAM variable = lz77PassStream depth = c_blockSize
#pragma HLS BIND_STORAGE variable = lz77PassStream type = FIFO impl = BRAM
#pragma HLS BIND_STORAGE variable = distStream type = FIFO impl = BRAM
#pragma HLS BIND_STORAGE variable = mergeStream type = FIFO impl = BRAM
    }

    // send input data to both checksum and for compression
    xf::compression::details::dataDuplicator<c_dwidth, c_size_dwidth, c_checksumParallelBytes, STRATEGY>(
        inStream, checksumInitStream, checksumStream, checksumSizeStream, coreStream);

    // checksum size less kernel
    if (STRATEGY == 0) { // CRC-32
        xf::compression::details::crc32<c_checksumParallelBytes>(checksumInitStream, checksumStream, checksumSizeStream,
                                                                 checksumOutStream);
    } else { // Adler-32
        xf::compression::details::adler32<c_checksumParallelBytes>(checksumInitStream, checksumStream,
                                                                   checksumSizeStream, checksumOutStream);
    }

    // distribute block size data into each block in round-robin fashion
    details::multicoreDistributor<uint32_t, c_dwidth, c_size_dwidth, c_numBlocks, c_blockSize, c_minBlockSize,
                                  STRATEGY>(coreStream, fileSizeStream, strdStream, strdSizeStream, blckEosStream,
                                            coreIdxStream, distStream);

    // Stream Duplicator
    details::streamDuplicator<4, 2>(coreIdxStream, coreIdxStreamArr);

    // Parallel Buffers
    for (uint8_t i = 0; i < c_numBlocks; i++) {
#pragma HLS UNROLL
        xf::compression::details::bufferDownsizer<c_dwidth, 8, c_size_dwidth>(distStream[i], downStream[i]);

        lz77Compress<c_numBlocks, c_blockSize, c_freq_dwidth>(downStream[i], lz77Stream[i], lz77Tree[i], i);

        if (BLOCK_SIZE_IN_KB >= 32) {
            xf::compression::details::bufferUpsizer<9, 72>(lz77Stream[i], lz77UpsizedStream[i], upsizedCntr[i]);
        } else {
            xf::compression::details::sendBuffer<9>(lz77Stream[i], lz77PassStream[i], upsizedCntr[i]);
        }
    }

    // Single Call Treegen
    details::zlibTreegenStreamWrapper<c_numBlocks>(lz77Tree, hufCodeStream, coreIdxStreamArr[0]);

    // Parallel Huffman
    for (uint8_t i = 0; i < c_numBlocks; i++) {
#pragma HLS UNROLL
        if (BLOCK_SIZE_IN_KB >= 32) {
            xf::compression::details::bufferDownsizer<72, 9>(lz77UpsizedStream[i], lz77DownsizedStream[i],
                                                             upsizedCntr[i]);
        } else {
            xf::compression::details::receiveBuffer<9>(lz77PassStream[i], lz77DownsizedStream[i], upsizedCntr[i]);
        }

        zlibHuffmanEncoder(lz77DownsizedStream[i], hufCodeStream[i], huffStream[i]);

        xf::compression::details::simpleStreamUpsizer<16, c_dwidth, c_size_dwidth>(huffStream[i], mergeStream[i]);
    }

    // GZIP/ZLIB Data Packer
    xf::compression::details::gzipZlibPackerEngine<c_numBlocks, STRATEGY>(
        mergeStream, packedStream, strdStream, strdSizeStream, fileSizeStream, checksumOutStream, coreIdxStreamArr[1],
        blckEosStream);

    // Byte Alignment and Packing into a Single Stream
    xf::compression::details::bytePacker<c_dwidth, c_size_dwidth>(packedStream, outStream);
}

template <int BLOCK_SIZE_IN_KB = 32, int NUM_BLOCKS = 8, int STRATEGY = 0> // STRATEGY -> 0: GZIP; 1: ZLIB
void gzipMulticoreCompressAxiStream(hls::stream<ap_axiu<64, 0, 0, 0> >& inAxiStream,
                                    hls::stream<ap_axiu<64, 0, 0, 0> >& outAxiStream) {
    constexpr int c_dwidth = 64;
    constexpr int c_defaultDepth = 4;
    hls::stream<IntVectorStream_dt<8, 8> > inStream("inStream");
    hls::stream<IntVectorStream_dt<8, 8> > outStream("outStream");

#pragma HLS STREAM variable = inStream depth = c_defaultDepth
#pragma HLS STREAM variable = outStream depth = c_defaultDepth

#pragma HLS BIND_STORAGE variable = inStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = outStream type = FIFO impl = SRL

#pragma HLS dataflow
    xf::compression::details::axiu2hlsStream<c_dwidth>(inAxiStream, inStream);

#ifdef STATIC_MODE
    xf::compression::gzipMulticoreStaticCompressStream<BLOCK_SIZE_IN_KB, NUM_BLOCKS, STRATEGY>(inStream, outStream);
#else
    xf::compression::gzipMulticoreCompressStream<BLOCK_SIZE_IN_KB, NUM_BLOCKS, STRATEGY>(inStream, outStream);
#endif

    xf::compression::details::hlsStream2axiu<c_dwidth>(outStream, outAxiStream);
}

} // End namespace compression
} // End namespace xf

#endif // _XFCOMPRESSION_ZLIB_COMPRESS_HPP_
