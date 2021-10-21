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
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "cmdlineparser.h"

#define ZSTD

#define PARALLEL_BYTES_READ 8
#define ZSTD_BLOCK_SIZE_KB 32
#define WINDOW_SIZE (32 * 1024)

// Window size or LZ History size can be modified as required above 128KB
// Below 128KB it must be equal to block size

#include "zstd_decompress.hpp"

constexpr int getDataPortWidth(int maxVal) {
    return (31 - __builtin_clz(maxVal));
}

uint64_t getFileSize(std::ifstream& file) {
    file.seekg(0, file.end);
    uint64_t file_size = file.tellg();
    file.seekg(0, file.beg);
    return file_size;
}

void decompressFrame(hls::stream<ap_uint<(8 * PARALLEL_BYTES_READ)> >& inStream,
                     hls::stream<ap_uint<4> >& inStrobe,
                     hls::stream<ap_uint<(8 * PARALLEL_BYTES_READ) + PARALLEL_BYTES_READ> >& outStream) {
    const int c_lmoDWidth = 1 + getDataPortWidth(WINDOW_SIZE);
    xf::compression::zstdDecompressStream<PARALLEL_BYTES_READ, ZSTD_BLOCK_SIZE_KB, WINDOW_SIZE, c_lmoDWidth>(
        inStream, inStrobe, outStream);
}

void validateFile(std::string& fileName, std::string& cmpFileName, uint8_t max_cr) {
    uint64_t inputSize;
    uint64_t originalSize;
    uint64_t outputSize;
    // original file
    std::ifstream origFile(fileName.c_str(), std::ifstream::binary);
    if (!origFile) {
        std::cout << "Unable to open file " << fileName << std::endl;
        return;
    }
    // compressed file
    std::ifstream inCmpFile(cmpFileName.c_str(), std::ifstream::binary);
    if (!inCmpFile) {
        std::cout << "Unable to open file " << fileName << std::endl;
        return;
    }

    inputSize = getFileSize(inCmpFile);
    originalSize = getFileSize(origFile);
    // decompress the file in a loop frame by frame
    std::string out_file_name = fileName + ".raw";

    hls::stream<ap_uint<(8 * PARALLEL_BYTES_READ)> > inputStream("inputStream");
    hls::stream<ap_uint<4> > inStrobe("inStrobe");
    hls::stream<ap_uint<(8 * PARALLEL_BYTES_READ) + PARALLEL_BYTES_READ> > outputStream("outputStream");

    uint8_t lbWidth = inputSize % PARALLEL_BYTES_READ;
    ap_uint<4> strb = PARALLEL_BYTES_READ;
    // Fill Header to Stream
    for (int i = 0; i < inputSize; i += PARALLEL_BYTES_READ) {
        ap_uint<(8 * PARALLEL_BYTES_READ)> val;
        inCmpFile.read((char*)&val, PARALLEL_BYTES_READ);
        if (inputSize < (i + PARALLEL_BYTES_READ)) strb = lbWidth;
        inStrobe << strb;
        inputStream << val;
    }
    inStrobe << 0;
    inputStream << 0;
    inCmpFile.close();
    decompressFrame(inputStream, inStrobe, outputStream);

    uint64_t k = 0;
    bool pass = true;
    uint64_t outCnt = 0;

    for (ap_uint<(8 * PARALLEL_BYTES_READ) + PARALLEL_BYTES_READ> val = outputStream.read(); val != 0;
         val = outputStream.read()) {
        // reading value from output stream
        ap_uint<(8 * PARALLEL_BYTES_READ)> o =
            val.range((8 * PARALLEL_BYTES_READ) + PARALLEL_BYTES_READ - 1, PARALLEL_BYTES_READ);
        ap_uint<PARALLEL_BYTES_READ> strb = val.range(PARALLEL_BYTES_READ - 1, 0);
        size_t size = __builtin_popcount(strb.to_uint());
        // writing output file
        // outFile.write((char*)&o, size);
        outCnt += size;

        // Comparing with input file
        ap_uint<(8 * PARALLEL_BYTES_READ)> g = 0;
        origFile.read((char*)&g, PARALLEL_BYTES_READ);
        if (o != g) {
            for (uint8_t v = 0; v < size; v++) {
                uint8_t e = g.range((v + 1) * 8 - 1, v * 8);
                uint8_t r = o.range((v + 1) * 8 - 1, v * 8);
                if (e != r) {
                    pass = false;
                    std::cout << "Expected=" << std::hex << e << " got=" << r << std::endl;
                    std::cout << "-----TEST FAILED: The input file and the file after "
                              << "decompression are not similar!-----" << std::endl;
                }
                if (!pass) break;
            }
        }
    }

    origFile.close();

    printf("\nOriginal Size: %lld, OutputSize: %lld\n", originalSize, outCnt);

    if (pass)
        std::cout << "Test PASSED" << std::endl;
    else
        std::cout << "Test FAILED" << std::endl;
}

int main(int argc, char* argv[]) {
    // parse the decompression
    uint8_t max_cr = 20;
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--file", "-f", "File to decompress", "");
    parser.addSwitch("--original_file", "-o", "Original file path", "");
    parser.addSwitch("--max_cr", "-mcr", "Max compression ratio", "");

    parser.parse(argc, argv);

    std::string inCmpFileName = parser.value("file");
    std::string originalFileName = parser.value("original_file");
    std::string max_cr_s = parser.value("max_cr");

    if (inCmpFileName.empty()) {
        std::cerr << "Error: Input file name not specified !!" << std::endl;
        return 0;
    }
    if (!max_cr_s.empty()) {
        max_cr = atoi(max_cr_s.c_str());
    }
    validateFile(originalFileName, inCmpFileName, max_cr);

    return 0;
}
