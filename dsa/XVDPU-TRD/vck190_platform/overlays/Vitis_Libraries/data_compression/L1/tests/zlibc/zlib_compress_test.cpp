/*
 * Copyright 2021 Xilinx, Inc.
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

#include <ap_int.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include "cmdlineparser.h"
#include "kernel_stream_utils.hpp"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "zlib_compress.hpp"

#define GMEM_DWIDTH 64
#define NUM_BLOCKS 8
#define BLOCK_SIZE_IN_KB 32
#define STRATEGY 1
typedef ap_axiu<GMEM_DWIDTH, 0, 0, 0> in_dT;
typedef ap_axiu<GMEM_DWIDTH, 0, 0, 0> out_dT;
typedef ap_axiu<32, 0, 0, 0> size_dT;

const uint32_t c_size = (GMEM_DWIDTH / 8);

void zlibcMulticoreStreaming(hls::stream<in_dT>& inStream, hls::stream<out_dT>& outStream) {
#pragma HLS INTERFACE AXIS port = inStream
#pragma HLS INTERFACE AXIS port = outStream
#pragma HLS INTERFACE ap_ctrl_none port = return

#pragma HLS DATAFLOW
    xf::compression::gzipMulticoreCompressAxiStream<BLOCK_SIZE_IN_KB, NUM_BLOCKS, STRATEGY>(inStream, outStream);
}

int main(int argc, char* argv[]) {
    std::string inputFileName = argv[1];
    std::string outputFileName = argv[2];

    // File Handling
    std::fstream inFile;
    inFile.open(inputFileName.c_str(), std::fstream::binary | std::fstream::in);
    if (!inFile.is_open()) {
        std::cout << "Cannot open the input file!!" << inputFileName << std::endl;
        exit(0);
    }
    std::ofstream outFile;
    outFile.open(outputFileName.c_str(), std::fstream::binary | std::fstream::out);

    hls::stream<in_dT> inStream("inStream");
    hls::stream<out_dT> outStream("outStream");
    hls::stream<size_dT> outSizeStream("outSizeStream");

    size_dT inSize;
    inFile.seekg(0, std::ios::end); // reaching to end of file
    const uint32_t inFileSize = (uint32_t)inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    auto numItr = 1;

    in_dT inData;
    for (int z = 0; z < numItr; z++) {
        inFile.seekg(0, std::ios::beg);
        // Input File back to back
        for (uint32_t i = 0; i < inFileSize; i += c_size) {
            ap_uint<GMEM_DWIDTH> v;
            bool last = false;
            uint32_t rSize = c_size;
            if (i + c_size >= inFileSize) {
                rSize = inFileSize - i;
                last = true;
            }
            inFile.read((char*)&v, rSize);
            inData.data = v;
            inData.keep = -1;
            inData.last = false;
            if (last) {
                uint32_t num = 0;
                inData.last = true;
                for (int b = 0; b < rSize; b++) {
                    num |= 1UL << b;
                }
                inData.keep = num;
            }
            inStream << inData;
        }

        // Compression Call
        zlibcMulticoreStreaming(inStream, outStream);

        uint32_t byteCounter = 0;
        // 1st file
        out_dT val;
        do {
            val = outStream.read();
            ap_uint<GMEM_DWIDTH> o = val.data;
            auto w_size = c_size;
            if (val.keep != -1) w_size = __builtin_popcount(val.keep);
            byteCounter += w_size;
            outFile.write((char*)&o, w_size);
        } while (!val.last);
    }

    inFile.close();
    outFile.close();
}
