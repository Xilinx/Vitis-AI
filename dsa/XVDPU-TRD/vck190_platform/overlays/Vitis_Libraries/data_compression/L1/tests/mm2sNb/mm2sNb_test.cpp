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
#define BYTE_CNT (IN_DATAWIDTH / 8)
#define INPUT_SIZE_IN_MB 2
#define INPUT_SIZE (INPUT_SIZE_IN_MB * 1024 * 1024)
#define BLOCK_LENGTH (64 * 1024)
#define PARALLEL_BLOCK 8
#define BURST_SIZE 16

const uint32_t c_no_blocks = (INPUT_SIZE - 1) / BLOCK_LENGTH + 1;
const uint32_t c_inSizeV = INPUT_SIZE / BYTE_CNT;
const uint32_t c_noElements = c_inSizeV / IN_DATAWIDTH;

// DUT
void hls_mm2sNb(const ap_uint<IN_DATAWIDTH>* in,
                const uint32_t input_idx[PARALLEL_BLOCK],
                hls::stream<ap_uint<IN_DATAWIDTH> > outStream[PARALLEL_BLOCK],
                const uint32_t input_size[PARALLEL_BLOCK]) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem depth = c_inSizeV
#pragma HLS INTERFACE m_axi port = input_idx bundle = gmem1 offset = slave
#pragma HLS INTERFACE m_axi port = input_size offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = input_idx bundle = control
#pragma HLS INTERFACE s_axilite port = input_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    xf::compression::details::mm2sNb<IN_DATAWIDTH, BURST_SIZE, PARALLEL_BLOCK>(in, input_idx, outStream, input_size);
}

int main() {
    ap_uint<IN_DATAWIDTH>* source_in = new ap_uint<IN_DATAWIDTH>[ c_inSizeV ];
    for (uint32_t i = 0; i < c_inSizeV; i++) {
        source_in[i] = rand();
    }

    uint32_t input_idx[PARALLEL_BLOCK];
    uint32_t input_size[PARALLEL_BLOCK];
    hls::stream<ap_uint<IN_DATAWIDTH> > outStream[PARALLEL_BLOCK];
    std::cout << "DATA_SIZE: " << c_inSizeV << " PARALLEL_BLOCK: " << PARALLEL_BLOCK << std::endl;

    uint32_t no_blocks = (INPUT_SIZE - 1) / BLOCK_LENGTH + 1;

    bool match = true;
    uint32_t index = 0;
    for (uint32_t i = 0; i < no_blocks; i += PARALLEL_BLOCK) {
        for (uint32_t j = 0; j < PARALLEL_BLOCK; j++) {
            input_idx[j] = (i + j) * BLOCK_LENGTH;
            input_size[j] = BLOCK_LENGTH;
        }

        hls_mm2sNb(source_in, input_idx, outStream, input_size);

        for (uint32_t bIdx = 0; bIdx < PARALLEL_BLOCK; bIdx++) {
            for (uint32_t k = 0; k < BURST_SIZE; k++) {
                for (uint32_t l = 0; l < c_noElements; l++) {
                    ap_uint<IN_DATAWIDTH> value = outStream[bIdx].read();
                    if (value != source_in[index]) {
                        match = false;
#ifndef __SYNTHESIS__
                        std::cout << "source[" << index << "]: " << source_in[index] << "\toutStream[" << bIdx
                                  << "]: " << value << std::endl;
#endif
                        break;
                    }
                    index++;
                }
            }
        }
    }

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
