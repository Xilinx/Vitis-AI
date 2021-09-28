/*
 * Copyright 2021 Xilinx, Inc.
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

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

#define PTR_IN_WIDTH 16
#define PTR_OUT_WIDTH 16

extern "C" {

void s2mm(ap_int<PTR_OUT_WIDTH>* mem, hls::stream<qdma_axis<PTR_IN_WIDTH, 0, 0, 0> >& s, int size) {
#pragma HLS INTERFACE m_axi port = mem offset = slave bundle = gmem

#pragma HLS interface axis port = s

#pragma HLS INTERFACE s_axilite port = mem bundle = control
#pragma HLS INTERFACE s_axilite port = size bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II = 1
        qdma_axis<PTR_IN_WIDTH, 0, 0, 0> x = s.read();
        mem[i] = x.data;
    }
}
}
