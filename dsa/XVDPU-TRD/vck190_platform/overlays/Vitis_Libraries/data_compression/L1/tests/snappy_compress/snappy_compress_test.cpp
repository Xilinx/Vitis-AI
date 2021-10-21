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

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <ap_int.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "hls_stream.h"
#include "lz_compress.hpp"
#include "lz_optional.hpp"
#include "snappy_compress.hpp"

#define PARALLEL_BLOCK 1
#ifdef LARGE_LIT_RANGE
#define MAX_LIT_COUNT 4090
#define MAX_LIT_STREAM_SIZE 4096
#else
#define MAX_LIT_COUNT 60
#define MAX_LIT_STREAM_SIZE 64
#endif
int const c_minMatch = 4;
#define LZ_MAX_OFFSET_LIMIT 65536
#define OFFSET_WINDOW (64 * 1024)
#define MAX_MATCH_LEN 255
#define MATCH_LEN 6

const int c_snappyMaxLiteralStream = MAX_LIT_STREAM_SIZE;

typedef ap_uint<8> uintV_t;

void snappyCompressEngineRun(hls::stream<uintV_t>& inStream,
                             hls::stream<uintV_t>& snappyOut,
                             hls::stream<bool>& snappyOut_eos,
                             hls::stream<uint32_t>& snappyOutSize,
                             uint32_t max_lit_limit[PARALLEL_BLOCK],
                             uint32_t input_size,
                             uint32_t core_idx) {
    hls::stream<xf::compression::compressd_dt> compressdStream("compressdStream");
    hls::stream<xf::compression::compressd_dt> bestMatchStream("bestMatchStream");
    hls::stream<xf::compression::compressd_dt> boosterStream("boosterStream");

#pragma HLS STREAM variable = compressdStream depth = 8
#pragma HLS STREAM variable = bestMatchStream depth = 8
#pragma HLS STREAM variable = boosterStream depth = 8

#pragma HLS BIND_STORAGE variable = compressdStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = boosterStream type = FIFO impl = SRL
#pragma HLS dataflow

    xf::compression::lzCompress<MATCH_LEN, c_minMatch, LZ_MAX_OFFSET_LIMIT>(inStream, compressdStream, input_size);
    xf::compression::lzBestMatchFilter<MATCH_LEN, OFFSET_WINDOW>(compressdStream, bestMatchStream, input_size);
    xf::compression::lzBooster<MAX_MATCH_LEN>(bestMatchStream, boosterStream, input_size);
    xf::compression::snappyCompress<MAX_LIT_COUNT, MAX_LIT_STREAM_SIZE, PARALLEL_BLOCK>(
        boosterStream, snappyOut, max_lit_limit, input_size, snappyOut_eos, snappyOutSize, core_idx);
}

int main(int argc, char* argv[]) {
    hls::stream<uintV_t> bytestr_in("compressIn");
    hls::stream<uintV_t> bytestr_out("compressOut");

    hls::stream<bool> snappyOut_eos;
    hls::stream<uint32_t> snappyOutSize;
    uint32_t max_lit_limit[PARALLEL_BLOCK];
    uint32_t input_size;
    uint32_t core_idx;

    std::ifstream inputFile;
    std::fstream outputFile;

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

    while (p--) {
        uint8_t x;
        inputFile.read((char*)&x, 1);
        bytestr_in << x;
    }
    inputFile.close();

    // COMPRESSION CALL
    snappyCompressEngineRun(bytestr_in, bytestr_out, snappyOut_eos, snappyOutSize, max_lit_limit, input_size, 0);

    uint32_t outsize;
    outsize = snappyOutSize.read();
    std::cout << "------- Compression Ratio: " << (float)fileSize / outsize << " -------" << std::endl;

    outputFile.open(argv[2], std::fstream::binary | std::fstream::out);
    if (!outputFile.is_open()) {
        std::cout << "Cannot open the output file!!" << std::endl;
        exit(0);
    }

    outputFile << input_size;

    bool eos_flag = snappyOut_eos.read();
    while (outsize > 0) {
        while (!eos_flag) {
            uint8_t w = bytestr_out.read();
            eos_flag = snappyOut_eos.read();
            outputFile.write((char*)&w, 1);
            outsize--;
        }
        if (!eos_flag) outsize = snappyOutSize.read();
    }
    uint8_t w = bytestr_out.read();
    outputFile.close();
}
