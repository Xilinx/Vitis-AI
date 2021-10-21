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
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>

#include "lz4_decompress.hpp"

#define MAX_OFFSET (64 * 1024)
#define HISTORY_SIZE MAX_OFFSET
#define PARALLEL_BYTES 8
typedef ap_uint<PARALLEL_BYTES * 8> uintV_t;
typedef ap_uint<(PARALLEL_BYTES * 8) + 8> uintS_t;

void lz4DecompressEngineRun(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& inStream,
                            hls::stream<ap_uint<(PARALLEL_BYTES * 8) + 8> >& outStream,
                            hls::stream<uint32_t>& outSizeStream,
                            const uint32_t input_size)

{
    xf::compression::lz4DecompressEngine<PARALLEL_BYTES, HISTORY_SIZE>(inStream, outStream, outSizeStream, input_size);
}

int main(int argc, char* argv[]) {
    hls::stream<uintV_t> inStream("inStream");
    hls::stream<uintS_t> outStream("decompressOut");
    hls::stream<uint32_t> outStreamSize("decompressOut");
    uint32_t input_size;

    std::fstream outputFile;
    outputFile.open(argv[1], std::fstream::binary | std::fstream::in);
    if (!outputFile.is_open()) {
        std::cout << "Cannot open the compressed file!!" << std::endl;
        exit(0);
    }
    outputFile.seekg(0, std::ios::end); // reaching to end of file
    uint32_t comp_length = (uint32_t)outputFile.tellg();
    outputFile.seekg(0, std::ios::beg);
    for (int i = 0; i < comp_length; i += PARALLEL_BYTES) {
        uintV_t x;
        outputFile.read((char*)&x, PARALLEL_BYTES);
        inStream << x;
    }

    // DECOMPRESSION CALL
    lz4DecompressEngineRun(inStream, outStream, outStreamSize, comp_length);

    std::ofstream outFile;
    outFile.open(argv[2], std::fstream::binary | std::fstream::out);
    std::ifstream originalFile;
    originalFile.open(argv[3], std::ofstream::binary | std::ofstream::in);
    if (!originalFile.is_open()) {
        std::cout << "Cannot open the original file!!" << std::endl;
        exit(0);
    }
    bool pass = true;
    uint32_t outSize = outStreamSize.read();
    uint32_t outCnt = 0;
    uintS_t g;
    uintS_t o = outStream.read();
    bool eosFlag = o.range((PARALLEL_BYTES + 1) * 8 - 1, PARALLEL_BYTES * 8);
    while (!eosFlag) {
        // writing output file
        if ((outCnt + PARALLEL_BYTES) < outSize) {
            outFile.write((char*)&o, PARALLEL_BYTES);
            outCnt += PARALLEL_BYTES;
        } else {
            outFile.write((char*)&o, outSize - outCnt);
            outCnt = outSize;
        }

        // Comparing with input file
        g = 0;
        originalFile.read((char*)&g, PARALLEL_BYTES);
        if (o != g) {
            uint8_t range = ((outSize - outCnt) > PARALLEL_BYTES) ? PARALLEL_BYTES : (outSize - outCnt);
            for (uint8_t v = 0; v < range; v++) {
                uint8_t e = g.range((v + 1) * 8 - 1, v * 8);
                uint8_t r = o.range((v + 1) * 8 - 1, v * 8);
                if (e != r) {
                    pass = false;
                    std::cout << "Expected=" << std::hex << e << " got=" << r << std::endl;
                    std::cout << "-----TEST FAILED: The input file and the file after "
                              << "decompression are not similar!-----" << std::endl;
                    exit(0);
                }
            }
        }

        // reading value from output stream
        o = outStream.read();
        eosFlag = o.range((PARALLEL_BYTES + 1) * 8 - 1, PARALLEL_BYTES * 8);
    }
    outFile.close();
    if (pass) {
        std::cout << "TEST PASSED" << std::endl;
    } else {
        std::cout << "TEST FAILED" << std::endl;
    }
    originalFile.close();
    outputFile.close();
}
