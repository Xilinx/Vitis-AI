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

#include "snappy_decompress.hpp"
#include "lz_decompress.hpp"

#define BLOCK_SIZE 65536
#define MAX_OFFSET 65536
#define HISTORY_SIZE MAX_OFFSET

typedef ap_uint<32> compressd_dt;
typedef ap_uint<8> uintV_t;

void snappyDecompressEngineRun(hls::stream<uintV_t>& inStream,
                               hls::stream<uintV_t>& outStream,
                               const uint32_t _input_size,
                               const uint32_t _output_size) {
    uint32_t input_size = _input_size;
    uint32_t output_size = _output_size;
    hls::stream<compressd_dt> decompressd_stream("decompressd_stream");
#pragma HLS STREAM variable = decompressd_stream depth = 8
#pragma HLS BIND_STORAGE variable = decompressd_stream type = FIFO impl = SRL

#pragma HLS dataflow
    xf::compression::snappyDecompress(inStream, decompressd_stream, input_size);
    xf::compression::lzDecompress<HISTORY_SIZE>(decompressd_stream, outStream, output_size);
}

int main(int argc, char* argv[]) {
    hls::stream<uintV_t> dec_bytestr_in("decompressIn");
    hls::stream<uintV_t> dec_bytestr_out("decompressOut");
    uint32_t input_size;

    std::ifstream originalFile;
    std::fstream inputFile;

    inputFile.open(argv[1], std::fstream::binary | std::fstream::in);
    if (!inputFile.is_open()) {
        std::cout << "Cannot open the compressed file!!" << std::endl;
        exit(0);
    }
    inputFile.seekg(0, inputFile.end);
    uint32_t compressSize = inputFile.tellg();
    inputFile.seekg(0, inputFile.beg);

    originalFile.open(argv[2], std::ofstream::binary | std::ofstream::in);
    if (!originalFile.is_open()) {
        std::cout << "Cannot open the original file!!" << std::endl;
        exit(0);
    }

    originalFile.seekg(0, originalFile.end);
    uint32_t output_size = originalFile.tellg();
    originalFile.seekg(0, originalFile.beg);

    bool pass = true;
    uint32_t blkSize = BLOCK_SIZE;
    uint32_t size = 0, incr = 0;
    uint32_t chunk_size = 0;
    char c, c1, c2, c3 = 0;

    inputFile.get(c);
    inputFile.get(c);
    inputFile.get(c);
    inputFile.get(c);
    inputFile.get(c);
    inputFile.get(c);
    inputFile.get(c);
    inputFile.get(c);
    inputFile.get(c);
    inputFile.get(c);

    for (int idxSize = 10; idxSize < compressSize; idxSize += 4) {
        if (incr + blkSize < output_size) {
            size = blkSize;
            incr += size;
        } else {
            size = output_size - incr;
        }

        inputFile.get(c);

        chunk_size = 0;
        // Chunk Compressed size
        inputFile.get(c1);
        inputFile.get(c2);
        inputFile.get(c3);
        uint8_t cbyte_1 = (uint8_t)c1;
        uint8_t cbyte_2 = (uint8_t)c2;
        uint8_t cbyte_3 = (uint8_t)c3;

        uint32_t temp = cbyte_3;
        temp <<= 16;
        chunk_size |= temp;
        temp = 0;
        temp = cbyte_2;
        temp <<= 8;
        chunk_size |= temp;
        temp = 0;
        chunk_size |= cbyte_1;

        uint32_t comp_length = chunk_size - 4;

        inputFile.clear();
        inputFile.seekg(idxSize + 8, std::ios::beg);
        uint32_t p = comp_length;
        for (int i = 0; i < p; i++) {
            uint8_t x;
            inputFile.read((char*)&x, 1);
            dec_bytestr_in << x;
        }
        idxSize += chunk_size;

        // DECOMPRESSION CALL
        snappyDecompressEngineRun(dec_bytestr_in, dec_bytestr_out, comp_length, size);

        uint32_t outputsize;
        outputsize = size;

        uint8_t s, t;
        for (uint32_t i = 0; i < outputsize; i++) {
            s = dec_bytestr_out.read();
            originalFile.read((char*)&t, 1);
            if (s == t)
                continue;
            else {
                pass = false;
                std::cout << "-----TEST FAILED: The input file and the file after "
                             "decompression are not similar!-----"
                          << std::endl;
                exit(0);
            }
        }
    }

    if (pass) {
        std::cout << "-----TEST PASSED: Original file and the file after decompression "
                     "are same.-------"
                  << std::endl;
    }
    originalFile.close();
    inputFile.close();
}
