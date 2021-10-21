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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include "hls_stream.h"
#include <ap_int.h>
#include <inttypes.h>
#include <vector>
#include "stream_upsizer.hpp"

#define dataSize 256
#define inWidth 8
#define outWidth 512

int const c_factor = outWidth / inWidth;

#define AOP uint32_t, inWidth, outWidth

void hls_streamUpsizer(hls::stream<ap_uint<inWidth> >& inStream,
                       hls::stream<ap_uint<outWidth> >& outStream,
                       uint32_t original_size) {
    xf::compression::details::streamUpsizer<AOP>(inStream, outStream, original_size);
}

int main(int argc, char* argv[]) {
    std::vector<uint8_t> source_in(dataSize);
    for (uint32_t i = 0; i < dataSize; i++) {
        source_in.push_back(rand() % (dataSize) + 1);
    }
    hls::stream<ap_uint<inWidth> > bytestr_in;
    hls::stream<ap_uint<outWidth> > bytestr_out;

    ap_uint<inWidth> input;
    uint32_t input_size = dataSize;
    uint32_t output_size = dataSize;
    uint32_t i = 0, index = 0, s_idx = 0;

    while (input_size--) {
        bytestr_in << source_in[index];
        index++;
    }

    input_size = dataSize;
    output_size = dataSize;

    hls_streamUpsizer(bytestr_in, bytestr_out, input_size);
    bool match = false;

    uint32_t output_size_byte = output_size / c_factor;
    while (output_size_byte--) {
        ap_uint<outWidth> outBuffer = bytestr_out.read();
        for (uint32_t i = 0; i < c_factor; i++) {
            uint8_t w = outBuffer.range((i + 1) * 8 - 1, i * 8);
            if (source_in[s_idx] == w) {
                s_idx++;
                match = true;
            } else {
                std::cout << "The input file and the output file are not same." << std::endl;
                std::cout << "Test Failed" << std::endl;
                exit(0);
            }
        }
    }
    if (match) std::cout << "TEST PASSED" << std::endl;
}
