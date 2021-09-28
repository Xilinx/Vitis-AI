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
#include "xxhash.h"
#include <ap_int.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef ap_uint<512> data_t;

#include "lz4_compress.hpp"
#include "lz4_packer.hpp"

#define GMEM_DWIDTH 512
#define GMEM_BURST_SIZE 16
#define BLOCK_SIZE 64
#define BLOCK_LENGTH (BLOCK_SIZE * 1024)
#define PARALLEL_BLOCK 1
#define CONST_SIZE 2 * 1024

const uint32_t c_size = (GMEM_DWIDTH / 8);
const uint32_t c_csize = CONST_SIZE / c_size;

void hls_lz4CompressPacker(data_t* in,
                           data_t* out,
                           data_t* lz4Out,
                           uint32_t* inBlockSize,
                           uint32_t* compressedSize,
                           uint32_t* packerCompressedSize,
                           uint32_t xxhashVal,
                           uint32_t block_length,
                           uint32_t no_blocks,
                           uint32_t input_size) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem0 depth = c_csize
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem0 depth = c_csize
#pragma HLS INTERFACE m_axi port = lz4Out offset = slave bundle = gmem0 depth = c_csize
#pragma HLS INTERFACE m_axi port = compressedSize offset = slave bundle = gmem1 depth = c_csize
#pragma HLS INTERFACE m_axi port = packerCompressedSize offset = slave bundle = gmem1 depth = c_csize
#pragma HLS INTERFACE m_axi port = inBlockSize offset = slave bundle = gmem1 depth = c_csize
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = lz4Out bundle = control
#pragma HLS INTERFACE s_axilite port = compressedSize bundle = control
#pragma HLS INTERFACE s_axilite port = packerCompressedSize bundle = control
#pragma HLS INTERFACE s_axilite port = inBlockSize bundle = control
#pragma HLS INTERFACE s_axilite port = xxhashVal bundle = control
#pragma HLS INTERFACE s_axilite port = block_length bundle = control
#pragma HLS INTERFACE s_axilite port = no_blocks bundle = control
#pragma HLS INTERFACE s_axilite port = input_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::compression::lz4CompressMM<data_t, GMEM_DWIDTH, GMEM_BURST_SIZE, PARALLEL_BLOCK>(in, out, compressedSize,
                                                                                         input_size);
    xf::compression::lz4PackerMM<GMEM_DWIDTH, GMEM_BURST_SIZE>(in, out, lz4Out, inBlockSize, compressedSize,
                                                               packerCompressedSize, xxhashVal, block_length, no_blocks,
                                                               input_size);
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

    inFile.seekg(0, std::ios::end); // reaching to end of file
    uint32_t input_size = inFile.tellg();
    uint32_t inSizeV = 0;
    if (input_size > 0) inSizeV = (input_size - 1) / c_size + 1;
    std::cout << "DATA_SIZE: " << input_size << " PARALLEL_BLOCK: " << PARALLEL_BLOCK << std::endl;

    data_t source_in[CONST_SIZE];
    data_t source_out[CONST_SIZE];
    data_t dest_out[CONST_SIZE];
    uint32_t compressedSize[CONST_SIZE];
    uint32_t inBlockSize[CONST_SIZE];

    uint32_t packerCompressedSize = 0;
    uint32_t block_length = BLOCK_LENGTH;
    uint32_t block_size = BLOCK_SIZE;
    uint32_t no_blocks = 0;
    if (input_size > 0) no_blocks = (input_size - 1) / block_length + 1;
    uint32_t oIdx = 0, offset = 0;

    uint32_t temp_buff[10] = {xf::compression::FLG_BYTE,
                              BLOCK_SIZE,
                              input_size,
                              input_size >> 8,
                              input_size >> 16,
                              input_size >> 24,
                              0,
                              0,
                              0,
                              0};

    // xxhash is used to calculate hash value
    uint32_t xxh = XXH32(temp_buff, 10, 0);
    // This value is sent to Kernel 2
    uint32_t xxhash_val = (xxh >> 8);

    for (int i = 0; i < CONST_SIZE; i++) {
        source_in[i] = 0;
        source_out[i] = 0;
        dest_out[i] = 0;
        inBlockSize[i] = BLOCK_LENGTH;
    }

    inFile.seekg(0, std::ios::beg);
    int index = 0;

    for (uint32_t i = 0; i < input_size; i += c_size) {
        data_t x = 0;
        inFile.read((char*)&x, c_size);
        source_in[index++] = x;
    }

    // Call Compression and Packer Modules
    hls_lz4CompressPacker(source_in, source_out, dest_out, inBlockSize, compressedSize, &packerCompressedSize,
                          xxhash_val, block_size, no_blocks, input_size);

    uint32_t j = 0;
    for (uint32_t i = 0; i < packerCompressedSize; i += c_size) {
        data_t x = dest_out[j++];
        outFile.write((char*)&x, c_size);
    }

    outFile.close();
    inFile.close();
    return 0;
}
