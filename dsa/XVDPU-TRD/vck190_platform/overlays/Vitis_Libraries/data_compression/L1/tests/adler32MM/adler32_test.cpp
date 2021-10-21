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
auto constexpr PARALLEL_BYTES = 16;
auto constexpr CONST_SIZE = 0x8000;
auto constexpr HOST_BUFFER_SIZE = 10 * 1024;
auto constexpr c_size = CONST_SIZE;

// DUT
void hls_adler32MM(const ap_uint<PARALLEL_BYTES * 8>* in, ap_uint<32>* adlerData, ap_uint<32> inputSize) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem depth = c_size
#pragma HLS INTERFACE m_axi port = adlerData offset = slave bundle = gmem depth = 2
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = adlerData bundle = control
#pragma HLS INTERFACE s_axilite port = inputSize bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::compression::adler32_mm<PARALLEL_BYTES>(in, adlerData, inputSize);
}

// Testbench
int main(int argc, char* argv[]) {
    auto nerr = 0;

    std::ifstream ifs;

    ifs.open(argv[1], std::ofstream::binary | std::ofstream::in);
    if (!ifs.is_open()) {
        std::cout << "Cannot open the input file!!" << std::endl;
        exit(0);
    }

    ifs.seekg(0, std::ios::end);
    uint32_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    ap_uint<PARALLEL_BYTES * 8> in[CONST_SIZE];
    ifs.read(reinterpret_cast<char*>(in), size);

    unsigned long adlerTmp = 1;
    auto golden = adler32(adlerTmp, reinterpret_cast<const unsigned char*>(in), size);

    // Calculating chunks of file
    auto no_blks = 0;
    if (size > 0) no_blks = (size - 1) / HOST_BUFFER_SIZE + 1;
    auto wordSize = PARALLEL_BYTES;
    auto readSize = 0;
    ap_uint<32> adlerData = 1;

    for (auto t = 0; t < no_blks; t++) {
        ap_uint<32> inSize = HOST_BUFFER_SIZE;
        ap_uint<32> offset = readSize / wordSize;
        if (readSize + inSize > size) inSize = size - readSize;
        readSize += inSize;

        hls_adler32MM(&in[offset], &adlerData, inSize);
    }

    if (golden != adlerData) {
        std::cout << std::hex << "adler_out=" << adlerData << ", golden=" << golden << std::endl;
        nerr = 1;
    }

    std::cout << "TEST " << (nerr ? "FAILED" : "PASSED") << std::endl;
    return (nerr ? EXIT_FAILURE : EXIT_SUCCESS);
}
