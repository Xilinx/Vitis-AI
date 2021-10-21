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
#include "cmdlineparser.h"

#include "inflate.hpp"

#define LOW_OFFSET 1
#define MAX_OFFSET (32 * 1024)
#define HISTORY_SIZE MAX_OFFSET

#define HUFFMAN_TYPE xf::compression::FULL

#define IN_BITWIDTH 16
#define OUT_BITWIDTH (MULTIPLE_BYTES * 8)
#define LL_MODEL true
const uint32_t sizeof_in = (IN_BITWIDTH / 8);
const uint32_t strbSize = (OUT_BITWIDTH / 8);

// typedef ap_uint<IN_BITWIDTH> in_t;
// typedef ap_uint<OUT_BITWIDTH> out_t;
typedef ap_axiu<16, 0, 0, 0> in_t;
typedef ap_axiu<64, 0, 0, 0> out_t;

void gzipMultiByteDecompressEngineRun(hls::stream<in_t>& inStream, hls::stream<out_t>& outStream)

{
#pragma HLS INTERFACE AXIS port = inStream
#pragma HLS INTERFACE AXIS port = outStream
#pragma HLS INTERFACE ap_ctrl_none port = return
    const int c_decoderType = (int)HUFFMAN_TYPE;
#pragma HLS DATAFLOW

    xf::compression::inflateMultiByte<c_decoderType, MULTIPLE_BYTES, LL_MODEL, HISTORY_SIZE>(inStream, outStream);
}

void validateFile(std::string& fileName, std::string& originalFileName) {
    hls::stream<in_t> inStream("inStream");
    hls::stream<out_t> outStream("decompressOut");

    std::string outputFileName = fileName + ".out";

    std::ifstream inFile(fileName.c_str(), std::ifstream::binary);
    if (!inFile.is_open()) {
        std::cout << "Cannot open the compressed file!!" << fileName << std::endl;
        return;
    }
    inFile.seekg(0, std::ios::end); // reaching to end of file
    uint32_t comp_length = (uint32_t)inFile.tellg();
    inFile.seekg(0, std::ios::beg);
    for (uint32_t i = 0; i < comp_length; i += sizeof_in) {
        ap_uint<16> x;
        in_t val;
        inFile.read((char*)&x, sizeof_in);
        val.data = x;
        if (i + sizeof_in >= comp_length) {
            val.last = 1;
        } else {
            val.last = 0;
        }
        inStream << val;
    }
    inFile.close();

    // DECOMPRESSION CALL
    gzipMultiByteDecompressEngineRun(inStream, outStream);

    std::ofstream outFile;
    outFile.open(outputFileName.c_str(), std::ofstream::binary);

    std::ifstream originalFile;
    originalFile.open(originalFileName.c_str(), std::ifstream::binary);
    if (!originalFile.is_open()) {
        std::cout << "Cannot open the original file " << originalFileName << std::endl;
        return;
    }

    bool pass = true;
    uint64_t outCnt = 0;
    ap_uint<64> g;

    out_t val;
    do {
        val = outStream.read();
        ap_uint<64> o = val.data;
        auto w_size = 8;
        if (val.keep != -1) w_size = __builtin_popcount(val.keep);
        outCnt += w_size;
        outFile.write((char*)&o, w_size);
        g = 0;
        originalFile.read((char*)&g, strbSize);

        if (o != g) {
            for (uint8_t v = 0; v < w_size; v++) {
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
        if (!pass) break;
    } while (!val.last);

    std::cout << "Decompressed size: " << std::dec << outCnt << " ";
    outFile.close();
    if (pass) {
        std::cout << "\nTEST PASSED\n" << std::endl;
    } else {
        std::cout << "TEST FAILED\n" << std::endl;
    }
    originalFile.close();
}

int main(int argc, char* argv[]) {
    sda::utils::CmdLineParser parser;

    parser.addSwitch("--file_list", "-l", "List of files", "");
    parser.addSwitch("--compressed_file", "-f", "Compressed zlib file path", "");
    parser.addSwitch("--original_file", "-o", "Original file path", "");
    parser.addSwitch("--current_path", "-p", "Current data path", "");
    parser.parse(argc, argv);

    std::string listFileName = parser.value("file_list");
    std::string singleInputFileName = parser.value("compressed_file");
    std::string singleGoldenFileName = parser.value("original_file");
    std::string currentPath = parser.value("current_path");

    // parse the arguments
    if (!listFileName.empty()) {
        // validate multiple files
        if (currentPath.empty()) {
            std::cout << "Path for data not specified.." << std::endl;
            std::cout << "Expecting absolute paths for files in list file." << std::endl;
        }

        std::ifstream infilelist(listFileName.c_str());
        std::string curFileName;
        // decompress and validate
        while (std::getline(infilelist, curFileName)) {
            auto origninalFileName = currentPath + "/" + curFileName;    // get original file path
            auto dcmpFileName = currentPath + "/" + curFileName + ".gz"; // compressed file path

            std::cout << "File: " << curFileName << std::endl;
            validateFile(dcmpFileName, origninalFileName);
        }
    } else {
        std::cout << "File: " << singleInputFileName << std::endl;
        // decompress and validate
        validateFile(singleInputFileName, singleGoldenFileName);
    }

    return 0;
}
