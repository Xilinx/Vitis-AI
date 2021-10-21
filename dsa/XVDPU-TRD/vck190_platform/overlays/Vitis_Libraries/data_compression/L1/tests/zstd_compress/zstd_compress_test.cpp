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

#define DISABLE_DEPENDENCE
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

#define PARALLEL_BYTES_READ 4
#define ZSTD_BLOCK_SIZE 32768
#define WINDOW_SIZE ZSTD_BLOCK_SIZE
#define MIN_BLOCK_SIZE 64
#define PARALLEL_HUFFMAN_UNITS 1
#define PARALLEL_LITERALS 4
#define MIN_MATCH_LEN 3

#include "zstd_compress.hpp"

uint64_t getFileSize(std::ifstream& file) {
    file.seekg(0, file.end);
    uint64_t file_size = file.tellg();
    file.seekg(0, file.beg);
    return file_size;
}

void compressFile(hls::stream<IntVectorStream_dt<8, 1> >& inStream, hls::stream<IntVectorStream_dt<8, 4> >& outStream) {
    xf::compression::zstdCompressCore<ZSTD_BLOCK_SIZE, WINDOW_SIZE, MIN_BLOCK_SIZE>(inStream, outStream);
}

void validateFile(std::string& fileName, std::string& cmpFileName) {
    uint64_t inputSize;
    uint64_t outputSize;
    // original file
    std::ifstream origFile(fileName.c_str(), std::ifstream::binary);
    if (!origFile) {
        std::cout << "Unable to open file " << fileName << std::endl;
        return;
    }
    // compressed file
    std::ofstream outCmpFile(cmpFileName.c_str(), std::ofstream::binary);
    if (!outCmpFile) {
        std::cout << "Unable to open file " << fileName << std::endl;
        return;
    }

    inputSize = getFileSize(origFile);
    // Decompress the file in a loop frame by frame
    std::string out_file_name = fileName + ".zst";

    hls::stream<IntVectorStream_dt<8, 1> > inStream("inStream");
    hls::stream<IntVectorStream_dt<8, 4> > outStream("outStream");
    // write input file block by block for 32KB block size
    IntVectorStream_dt<8, 1> inVal;
    inVal.strobe = 1;
    for (int i = 0; i < inputSize; ++i) {
        origFile.read((char*)inVal.data, 1);
        inStream << inVal;
    }
    // End of file/all data
    inVal.strobe = 0;
    inStream << inVal;
    origFile.close();
    compressFile(inStream, outStream);

    // write output to file
    outputSize = 0;
    for (auto outVal = outStream.read(); outVal.strobe > 0; outVal = outStream.read()) {
        ap_uint<32> outV = 0;
        outV.range(7, 0) = outVal.data[0];
        outV.range(15, 8) = outVal.data[1];
        outV.range(23, 16) = outVal.data[2];
        outV.range(31, 24) = outVal.data[3];
        outCmpFile.write((char*)&outV, (int)(outVal.strobe));
        outputSize += outVal.strobe;
    }
    printf("Output file size: %d\n", outputSize);
    outCmpFile.close();
}

int main(int argc, char* argv[]) {
    // parse the compression arguments
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--file", "-f", "File to compress", "");
    parser.addSwitch("--output", "-o", "Output compressed file", "");

    parser.parse(argc, argv);

    std::string inFileName = parser.value("file");
    std::string outFileName = parser.value("output");

    if (inFileName.empty()) {
        std::cerr << "Error: Input file name not specified !!" << std::endl;
        return 0;
    }
    if (outFileName.empty()) {
        outFileName.resize(inFileName.size() + 4);
        outFileName.assign(inFileName, 0, inFileName.size() + 4);
    }
    validateFile(inFileName, outFileName);
    return 0;
}
