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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <assert.h>
#include <iostream>
#include <fstream>
#include "hls_stream.h"
#include <ap_int.h>
#include "stream_downsizer.hpp"

#define testDataLen 256
#define inWidth 512
#define outWidth 8

#define AOP uint32_t, inWidth, outWidth

void streamDownsizerRun(hls::stream<ap_uint<inWidth> >& inStream,
                        hls::stream<ap_uint<outWidth> >& outStream,
                        uint32_t input_size) {
    xf::compression::details::streamDownsizer<AOP>(inStream, outStream, input_size);
}

int main(int argc, char* argv[]) {
    uint32_t i, j;
    std::vector<uint8_t> testdata(testDataLen);
    // Random test data
    for (i = 0; i < testDataLen; i++) {
        testdata.push_back(rand() % (testDataLen) + 1);
    }
    ap_uint<inWidth> inBuffer = 0;

    hls::stream<ap_uint<inWidth> > bytestr_in;
    hls::stream<ap_uint<outWidth> > bytestr_out;

    uint32_t input_size = testDataLen * sizeof(int8_t);
    uint32_t output_size = input_size;
    for (i = 0; i < (testDataLen * outWidth) / inWidth; i++) {
        for (j = 0; j < (inWidth / outWidth); j++) {
            inBuffer.range(((j + 1) * (outWidth)-1), (outWidth)*j) = testdata[(inWidth / outWidth) * i + j];
        }
        bytestr_in << inBuffer;
        inBuffer = 0;
    }
    streamDownsizerRun(bytestr_in, bytestr_out, input_size);
    i = 0;
    bool match = false;
    while (output_size--) {
        uint8_t w = bytestr_out.read();
        if (testdata[i] == w) {
            match = true;
            continue;
        } else {
            std::cout << "***TEST FAILED: The input file and the output file are not same.***" << std::endl;
            exit(0);
        }
        i++;
    }
    if (match) std::cout << "***TEST PASSED: The input file and the output file are same.***" << std::endl;
}
