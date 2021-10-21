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

#include "hls_stream.h"
#include <ap_int.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "lz4_compress.hpp"
#include "lz_compress.hpp"
#include "lz_optional.hpp"

#define MAX_LIT_COUNT 4096
#define PARALLEL_BLOCK 1
#define LZ_MAX_OFFSET_LIMIT 65536
#define OFFSET_WINDOW (64 * 1024)
#define MAX_MATCH_LEN 255
#define MATCH_LEN 6

typedef ap_uint<32> compressd_dt;
typedef ap_uint<64> lz4_compressd_dt;
typedef ap_uint<8> uintV_t;

int const c_minMatch = 4;
int const c_matchLevel = 6;
int const c_minOffset = 1;
int const c_lz4MaxLiteralCount = MAX_LIT_COUNT;

void lz4CompressEngineRun(hls::stream<uintV_t>& inStream,
                          hls::stream<uintV_t>& lz4Out,
                          hls::stream<bool>& lz4Out_eos,
                          hls::stream<uint32_t>& lz4OutSize,
                          uint32_t max_lit_limit[PARALLEL_BLOCK],
                          uint32_t input_size,
                          uint32_t core_idx) {
    hls::stream<compressd_dt> compressdStream("compressdStream");
    hls::stream<xf::compression::compressd_dt> bestMatchStream("bestMatchStream");
    hls::stream<compressd_dt> boosterStream("boosterStream");

#pragma HLS STREAM variable = compressdStream depth = 8
#pragma HLS STREAM variable = bestMatchStream depth = 8
#pragma HLS STREAM variable = boosterStream depth = 8

#pragma HLS BIND_STORAGE variable = compressdStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = boosterStream type = FIFO impl = SRL

#pragma HLS dataflow
    xf::compression::lzCompress<MATCH_LEN, c_minMatch, LZ_MAX_OFFSET_LIMIT, c_matchLevel, c_minOffset>(
        inStream, compressdStream, input_size);
    xf::compression::lzBestMatchFilter<MATCH_LEN, OFFSET_WINDOW>(compressdStream, bestMatchStream, input_size);
    xf::compression::lzBooster<MAX_MATCH_LEN>(bestMatchStream, boosterStream, input_size);
    xf::compression::lz4Compress<MAX_LIT_COUNT, PARALLEL_BLOCK>(boosterStream, lz4Out, max_lit_limit, input_size,
                                                                lz4Out_eos, lz4OutSize, core_idx);
}

int main(int argc, char* argv[]) {
    hls::stream<uintV_t> bytestr_in("compressIn");
    hls::stream<uintV_t> bytestr_out("compressOut");

    hls::stream<bool> lz4Out_eos;
    hls::stream<uint32_t> lz4OutSize;
    uint32_t max_lit_limit[PARALLEL_BLOCK];
    uint32_t input_size;
    uint32_t core_idx;

    std::ifstream inputFile;
    std::fstream outputFile;

    // Input file open for input_size
    inputFile.open(argv[1], std::ofstream::binary | std::ofstream::in);
    if (!inputFile.is_open()) {
        std::cout << "Cannot open the input file!!" << std::endl;
        exit(0);
    }
    inputFile.seekg(0, std::ios::end);
    uint32_t fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);
    input_size = fileSize;
    uint32_t p = fileSize;

    // Pushing input file into input stream for compression
    while (p--) {
        uint8_t x;
        inputFile.read((char*)&x, 1);
        bytestr_in << x;
    }
    inputFile.close();

    // COMPRESSION CALL
    lz4CompressEngineRun(bytestr_in, bytestr_out, lz4Out_eos, lz4OutSize, max_lit_limit, input_size, 0);

    uint32_t outsize;
    outsize = lz4OutSize.read();
    std::cout << "------- Compression Ratio: " << (float)fileSize / outsize << " -------" << std::endl;

    outputFile.open(argv[2], std::fstream::binary | std::fstream::out);
    if (!outputFile.is_open()) {
        std::cout << "Cannot open the output file!!" << std::endl;
        exit(0);
    }

    outputFile.write((char*)&input_size, 4);

    bool eos_flag = lz4Out_eos.read();
    while (outsize > 0) {
        while (!eos_flag) {
            uint8_t w = bytestr_out.read();
            eos_flag = lz4Out_eos.read();
            outputFile.write((char*)&w, 1);
            outsize--;
        }
        if (!eos_flag) outsize = lz4OutSize.read();
    }
    uint8_t w = bytestr_out.read();
    outputFile.close();
}
