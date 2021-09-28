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
#include "cmdlineparser.h"

#include "snappy_decompress_details.hpp"

#define BLOCK_SIZE 64
#define MULTIPLE_BYTES 8
#define NUM_BLOCKS 2

#define MAX_OFFSET (64 * 1024)
#define HISTORY_SIZE MAX_OFFSET

typedef ap_uint<MULTIPLE_BYTES * 8> uintS_t;
typedef ap_uint<(MULTIPLE_BYTES * 8) + 8> uintV_t;

void snappyMultiCoreDec(hls::stream<uintS_t>& inStream,
                        hls::stream<uint32_t>& inSizeStream,
                        hls::stream<uintV_t>& outStream,
                        hls::stream<uint32_t>& outSizeStream) {
    xf::compression::snappyMultiCoreDecompress<NUM_BLOCKS, MULTIPLE_BYTES, HISTORY_SIZE, BLOCK_SIZE>(
        inStream, inSizeStream, outStream, outSizeStream);
}

void processFile(std::string& compFile_name,
                 std::string& uncompFile_name,
                 std::string& origFile_name,
                 hls::stream<uint32_t>& inSizeStream) {
    hls::stream<uintS_t> inStream("inStream");
    hls::stream<uint32_t> inStreamSize("compLength");
    hls::stream<bool> inStreamEos("inStreamEos");
    hls::stream<uintV_t> outStream("decompressOut");
    hls::stream<uint32_t> outStreamSize("decompressOutSize");

    uint32_t comp_length = inSizeStream.read();
    inStreamSize << comp_length;
    inStreamSize << 0;

    std::fstream compFile;
    compFile.open(compFile_name, std::fstream::binary | std::fstream::in);
    if (!compFile.is_open()) {
        std::cout << "Cannot open the compressed file!!" << std::endl;
        exit(0);
    }

    // write data to stream
    for (int i = 0; i < comp_length; i += MULTIPLE_BYTES) {
        uintS_t x;
        compFile.read((char*)&x, MULTIPLE_BYTES);
        inStream << x;
    }

    // DECOMPRESSION CALL
    snappyMultiCoreDec(inStream, inStreamSize, outStream, outStreamSize);

    std::ofstream origFile;
    origFile.open(origFile_name, std::fstream::binary | std::fstream::out);
    std::ifstream uncompFile;
    uncompFile.open(uncompFile_name, std::ofstream::binary | std::ofstream::in);
    if (!uncompFile.is_open()) {
        std::cout << "Cannot open the original file!!" << std::endl;
        exit(0);
    }

    bool pass = true;
    uint32_t outSize = outStreamSize.read();

    uint32_t outCnt = 0;
    uintV_t g;
    uintV_t o = outStream.read();
    bool eosFlag = o.range((MULTIPLE_BYTES + 1) * 8 - 1, MULTIPLE_BYTES * 8);
    while (!eosFlag) {
        // writing output file
        if (outCnt + MULTIPLE_BYTES < outSize) {
            origFile.write((char*)&o, MULTIPLE_BYTES);
            outCnt += MULTIPLE_BYTES;
        } else {
            origFile.write((char*)&o, outSize - outCnt);
            outCnt = outSize;
        }

        // Comparing with input file
        g = 0;
        uncompFile.read((char*)&g, MULTIPLE_BYTES);
        if (o != g) {
            uint8_t range = ((outSize - outCnt) > MULTIPLE_BYTES) ? MULTIPLE_BYTES : (outSize - outCnt);
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
        eosFlag = o.range((MULTIPLE_BYTES + 1) * 8 - 1, MULTIPLE_BYTES * 8);
    }

    origFile.close();
    if (pass) {
        std::cout << "File: " << uncompFile_name << std::endl;
        std::cout << "TEST PASSED" << std::endl;
    } else {
        std::cout << "File: " << uncompFile_name << std::endl;
        std::cout << "TEST FAILED" << std::endl;
    }
    uncompFile.close();
    compFile.close();
}

int main(int argc, char* argv[]) {
    sda::utils::CmdLineParser parser;

    parser.addSwitch("--decompressFile", "-d", "File Decompress", "");
    parser.addSwitch("--file_list", "-l", "List of files", "");
    parser.addSwitch("--path", "-p", "path to files", "");
    parser.parse(argc, argv);

    std::string compressFileName = parser.value("decompressFile");
    std::string listFileName = parser.value("file_list");
    std::string filePath = parser.value("path");

    hls::stream<uint32_t> inSizeStream("compressedFileSize");

    std::string compFileName;
    std::string origFileName;
    std::string uncompFileName;

    // parse the arguments
    if (!listFileName.empty()) {
        std::ifstream infilelist(listFileName.c_str());

        while (std::getline(infilelist, compFileName)) {
            compFileName = filePath + "/" + compFileName;

            std::fstream compFile;
            compFile.open(compFileName, std::fstream::binary | std::fstream::in);
            if (!compFile.is_open()) {
                std::cout << "Cannot open the compressed file!!" << std::endl;
                exit(0);
            }
            compFile.seekg(0, std::ios::end); // reaching to end of file
            uint32_t comp_length = (uint32_t)compFile.tellg();
            compFile.seekg(0, std::ios::beg);

            // write insize to stream
            inSizeStream << comp_length;

            std::string line_in = compFileName;
            origFileName = line_in + ".orig";
            std::string delimiter = ".snappy";
            std::string token = line_in.substr(0, line_in.find(delimiter));
            token = token.substr(0, token.find(".xe2xd"));
            uncompFileName = token;

            // decompress and validate
            processFile(compFileName, uncompFileName, origFileName, inSizeStream);
        }
    } else {
        std::string line_in = compressFileName;
        std::fstream compFile;
        compFile.open(compressFileName, std::fstream::binary | std::fstream::in);
        if (!compFile.is_open()) {
            std::cout << "Cannot open the compressed file!!" << std::endl;
            exit(0);
        }
        compFile.seekg(0, std::ios::end); // reaching to end of file
        uint32_t comp_length = (uint32_t)compFile.tellg();
        compFile.seekg(0, std::ios::beg);

        // write insize to stream
        inSizeStream << comp_length;

        std::string origFileName = line_in + ".orig";
        std::string delimiter = ".snappy";
        std::string token = line_in.substr(0, line_in.find(delimiter));
        token = token.substr(0, token.find(".xe2xd"));
        std::string uncompFileName = token;

        // decompress and validate
        processFile(compressFileName, uncompFileName, origFileName, inSizeStream);
    }
}
