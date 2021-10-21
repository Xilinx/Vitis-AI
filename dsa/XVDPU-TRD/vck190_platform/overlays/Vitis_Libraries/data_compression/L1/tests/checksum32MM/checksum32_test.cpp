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
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include <math.h>

#include "checksum_wrapper.hpp"
#include "zlib.h"
auto constexpr ADLER32 = 0;
auto constexpr CRC32 = 1;
auto constexpr CHECKSUM_MODE = ADLER32;
// auto constexpr CHECKSUM_MODE = CRC32;
auto constexpr PARALLEL_BYTE = 16;
auto constexpr CONST_SIZE = 0x8000;
auto constexpr HOST_BUFFER_SIZE = 10 * 1024;
auto constexpr c_size = CONST_SIZE;

// DUT
void hls_checksum32MM(const ap_uint<PARALLEL_BYTE * 8>* in,
                      ap_uint<32>* checksumData,
                      ap_uint<32> inputSize,
                      const bool checksumType) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem depth = c_size
#pragma HLS INTERFACE m_axi port = checksumData offset = slave bundle = gmem depth = 2
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = checksumData bundle = control
#pragma HLS INTERFACE s_axilite port = inputSize bundle = control
#pragma HLS INTERFACE s_axilite port = checksumType bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::compression::checksum32_mm<PARALLEL_BYTE>(in, checksumData, inputSize, checksumType);
}

// Testbench
int main(int argc, char* argv[]) {
    auto nerr = 0;
    bool constexpr c_mode = CHECKSUM_MODE;

    std::ifstream ifs;

    ifs.open(argv[1], std::ofstream::binary | std::ofstream::in);
    if (!ifs.is_open()) {
        std::cout << "Cannot open the input file!!" << std::endl;
        exit(0);
    }

    ifs.seekg(0, std::ios::end);
    uint32_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    ap_uint<PARALLEL_BYTE * 8> in[CONST_SIZE];
    ifs.read(reinterpret_cast<char*>(in), size);

    uint32_t golden = 0;
    if (c_mode == false) {
        unsigned long checksumTmp = 1;
        golden = adler32(checksumTmp, reinterpret_cast<const unsigned char*>(in), size);
    } else {
        unsigned long checksumTmp = 0;
        golden = crc32(checksumTmp, reinterpret_cast<const unsigned char*>(in), size);
    }

    // Calculating chunks of file
    auto no_blks = 0;
    if (size > 0) no_blks = (size - 1) / HOST_BUFFER_SIZE + 1;
    auto wordSize = PARALLEL_BYTE;
    auto readSize = 0;
    ap_uint<32> checksumData = 1;
    if (c_mode == true) checksumData = ~0;

    for (auto t = 0; t < no_blks; t++) {
        ap_uint<32> inSize = HOST_BUFFER_SIZE;
        ap_uint<32> offset = readSize / wordSize;
        if (readSize + inSize > size) inSize = size - readSize;
        readSize += inSize;

        hls_checksum32MM(&in[offset], &checksumData, inSize, c_mode);
    }

    if (c_mode == true) checksumData = ~checksumData;

    if (golden != checksumData) {
        std::cout << "checksum_out=" << checksumData << ", golden=" << golden << std::endl;
        nerr = 1;
    }

    std::cout << "TEST " << (nerr ? "FAILED" : "PASSED") << std::endl;
    return (nerr ? EXIT_FAILURE : EXIT_SUCCESS);
}
