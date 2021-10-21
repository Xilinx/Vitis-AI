/*
 * Copyright 2019-2021 Xilinx, Inc.
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
 */

/**
 * @file lz4_compress_stream.cpp
 * @brief Source for LZ4 compression streaming kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "lz4_compress_stream.hpp"

const int kGMemBurstSize = 16;
const int c_lz4MaxLiteralCount = MAX_LIT_COUNT;
typedef ap_uint<32> compressd_dt;
typedef ap_uint<64> lz4_compressd_dt;
typedef ap_uint<8> streamDt;

extern "C" {
void xilLz4CompressStream(hls::stream<ap_axiu<8, 0, 0, 0> >& inaxistream,
                          hls::stream<ap_axiu<8, 0, 0, 0> >& outaxistream,
                          uint32_t inputSize) {
#pragma HLS interface axis port = inaxistream
#pragma HLS interface axis port = outaxistream
#pragma HLS interface s_axilite port = inputSize bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    hls::stream<compressd_dt> compressdStream("compressdStream");
    hls::stream<compressd_dt> bestMatchStream("bestMatchStream");
    hls::stream<compressd_dt> boosterStream("boosterStream");
    hls::stream<uint32_t> compressedSize("compressedSize");
    hls::stream<streamDt> inStream("inStream");
    hls::stream<streamDt> outStream("outStream");
    hls::stream<bool> lz4OutEos("lz4OutEos");

#pragma HLS STREAM variable = inStream depth = 2
#pragma HLS STREAM variable = outStream depth = 2
#pragma HLS STREAM variable = compressedSize depth = 2
#pragma HLS STREAM variable = compressdStream depth = 8
#pragma HLS STREAM variable = bestMatchStream depth = 8
#pragma HLS STREAM variable = boosterStream depth = 8
#pragma HLS STREAM variable = lz4OutEos depth = 8

#pragma HLS BIND_STORAGE variable = inStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = outStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = compressdStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = boosterStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = lz4OutEos type = FIFO impl = SRL

#pragma HLS dataflow
    uint32_t litLimit[1];
    litLimit[0] = 0; // max_lit_limit;

    xf::compression::details::kStreamRead<8>(inaxistream, inStream, inputSize);

    xf::compression::lzCompress<MATCH_LEN, MIN_MATCH, LZ_MAX_OFFSET_LIMIT>(inStream, compressdStream, inputSize);
    xf::compression::lzBestMatchFilter<MATCH_LEN, OFFSET_WINDOW>(compressdStream, bestMatchStream, inputSize);
    xf::compression::lzBooster<MAX_MATCH_LEN>(bestMatchStream, boosterStream, inputSize);
    xf::compression::lz4Compress<MAX_LIT_COUNT, PARALLEL_BLOCK>(boosterStream, outStream, litLimit, inputSize,
                                                                lz4OutEos, compressedSize, 0);
    xf::compression::details::kStreamWrite<8>(outaxistream, outStream, lz4OutEos, compressedSize);
}
}
