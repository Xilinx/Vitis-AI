/*
 * Copyright 2019 Xilinx, Inc.
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

#include "kernel_sort.hpp"

void readM2S(int keyLength, KEY_TYPE inKey[LEN], hls::stream<KEY_TYPE>& keyStrm, hls::stream<bool>& endStrm) {
    const int len = LEN;
    for (int i = 0; i < keyLength; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = len min = len
        keyStrm.write(inKey[i]);
        endStrm.write(false);
    }
    endStrm.write(true);
}

void writeS2M(int keyLength, KEY_TYPE outKey[LEN], hls::stream<KEY_TYPE>& keyStrm, hls::stream<bool>& endStrm) {
    const int len = LEN;
    for (int i = 0; i < keyLength; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = len min = len
        outKey[i] = keyStrm.read();
        endStrm.read();
    }
    endStrm.read();
}

extern "C" void SortKernel(int order, int keyLength, KEY_TYPE inKey[LEN], KEY_TYPE outKey[LEN]) {
#pragma HLS dataflow
#ifndef HLS_TEST
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 64 max_read_burst_length = 64 bundle = gmem0 port = inKey
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 64 max_read_burst_length = 64 bundle = gmem1 port = outKey
#pragma HLS INTERFACE s_axilite port = order bundle = control
#pragma HLS INTERFACE s_axilite port = keyLength bundle = control
#pragma HLS INTERFACE s_axilite port = inKey bundle = control
#pragma HLS INTERFACE s_axilite port = outKey bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#endif
    hls::stream<KEY_TYPE> inKeyStrm;
    hls::stream<bool> inEndStrm;
    hls::stream<KEY_TYPE> outKeyStrm;
    hls::stream<bool> outEndStrm;
#pragma HLS stream variable = inKeyStrm depth = 4
#pragma HLS stream variable = inEndStrm depth = 4
#pragma HLS stream variable = outKeyStrm depth = 4
#pragma HLS stream variable = outEndStrm depth = 4
    readM2S(keyLength, inKey, inKeyStrm, inEndStrm);
    xf::database::compoundSort<KEY_TYPE, LEN, INSERT_LEN>(order, inKeyStrm, inEndStrm, outKeyStrm, outEndStrm);
    writeS2M(keyLength, outKey, outKeyStrm, outEndStrm);
}
