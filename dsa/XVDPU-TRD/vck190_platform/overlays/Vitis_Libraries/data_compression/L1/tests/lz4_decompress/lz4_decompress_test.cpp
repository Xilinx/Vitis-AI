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
#include "lz_decompress.hpp"
#include "lz_optional.hpp"

#define MAX_OFFSET 65536
#define HISTORY_SIZE MAX_OFFSET

typedef ap_uint<32> compressd_dt;
typedef ap_uint<8> uintV_t;

void lz4DecompressEngineRun(hls::stream<uintV_t>& inStream,
                            hls::stream<uintV_t>& outStream,
                            const uint32_t _input_size,
                            const uint32_t _output_size) {
    uint32_t input_size = _input_size;
    uint32_t output_size = _output_size;
    hls::stream<compressd_dt> decompressd_stream("decompressd_stream");
#pragma HLS STREAM variable = decompressd_stream depth = 8
#pragma HLS BIND_STORAGE variable = decompressd_stream type = FIFO impl = SRL

#pragma HLS dataflow
    xf::compression::lz4Decompress(inStream, decompressd_stream, input_size);
    xf::compression::lzDecompress<HISTORY_SIZE>(decompressd_stream, outStream, output_size);
}

int main(int argc, char* argv[]) {
    hls::stream<uintV_t> dec_bytestr_in("decompressIn");
    hls::stream<uintV_t> dec_bytestr_out("decompressOut");
    uint32_t input_size;

    std::ifstream originalFile;
    std::fstream outputFile;

    outputFile.open(argv[1], std::fstream::binary | std::fstream::in);
    if (!outputFile.is_open()) {
        std::cout << "Cannot open the compressed file!!" << std::endl;
        exit(0);
    }
    uint32_t output_size;
    outputFile.read((char*)&output_size, 4);
    outputFile.seekg(0, std::ios::end);
    uint32_t comp_length = (uint32_t)outputFile.tellg() - 4;
    outputFile.seekg(4, std::ios::beg);
    uint32_t p = comp_length;
    for (int i = 0; i < p; i++) {
        uint8_t x;
        outputFile.read((char*)&x, 1);
        dec_bytestr_in << x;
    }

    // DECOMPRESSION CALL
    lz4DecompressEngineRun(dec_bytestr_in, dec_bytestr_out, comp_length, output_size);

    uint32_t outputsize;
    outputsize = output_size;

    originalFile.open(argv[2], std::ofstream::binary | std::ofstream::in);
    if (!originalFile.is_open()) {
        std::cout << "Cannot open the original file!!" << std::endl;
        exit(0);
    }
    uint8_t s, t;
    bool pass = true;
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
    if (pass) {
        std::cout << "-----TEST PASSED: Original file and the file after decompression "
                     "are same.-------"
                  << std::endl;
    }
    originalFile.close();
    outputFile.close();
}
