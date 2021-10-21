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
#include "mm2s.hpp"
#include <ap_int.h>
#include <assert.h>
#include <fstream>
#include <inttypes.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define IN_DATAWIDTH 512
#define OUT_DATAWIDTH 16
#define BYTE_CNT (IN_DATAWIDTH / 8)
#define INPUT_SIZE_IN_MB 2
#define INPUT_SIZE (INPUT_SIZE_IN_MB * 1024 * 1024)
#define BLOCK_LENGTH (64 * 1024)
#define BURST_SIZE 16

const uint32_t c_inSizeV = INPUT_SIZE / BYTE_CNT;

// DUT
void hls_mm2sSimple(const ap_uint<IN_DATAWIDTH>* in,
                    hls::stream<ap_uint<OUT_DATAWIDTH> >& outStream,
                    const uint32_t inputSize) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem depth = c_inSizeV
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = inputSize bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::compression::details::mm2Stream<OUT_DATAWIDTH>(in, outStream, inputSize);
}

int main() {
    ap_uint<IN_DATAWIDTH>* source_in = new ap_uint<IN_DATAWIDTH>[ c_inSizeV ];
    for (uint32_t i = 0; i < c_inSizeV; i++) {
        source_in[i] = rand();
    }

    uint32_t inputSize = INPUT_SIZE;
    hls::stream<ap_uint<OUT_DATAWIDTH> > outStream;
    std::cout << "DATA_SIZE: " << c_inSizeV << std::endl;

    bool match = true;

    hls_mm2sSimple(source_in, outStream, inputSize);

    for (uint32_t i = 0; i < c_inSizeV; i++) {
        for (uint32_t j = 0; j < IN_DATAWIDTH / OUT_DATAWIDTH; j++) {
            ap_uint<OUT_DATAWIDTH> value = outStream.read();
            if (value != source_in[i].range(OUT_DATAWIDTH * (j + 1) - 1, j * OUT_DATAWIDTH)) {
                match = false;
#ifndef __SYNTHESIS__
                std::cout << "source[" << i
                          << "]: " << source_in[i].range(OUT_DATAWIDTH * (j + 1) - 1, j * OUT_DATAWIDTH)
                          << "\toutStream: " << value << std::endl;
#endif
                break;
            }
        }
    }

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
