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
#include "test.hpp"

#define N 1

void testEnc(ap_uint<64> in, ap_uint<64> key, ap_uint<64>& out) {
    ap_uint<64> outArr[N];
#pragma HLS ARRAY_PARTITION variable = outArr complete
    for (int i = 0; i < N; ++i) {
#pragma HLS pipeline
        xf::security::desEncrypt(in, key, outArr[i]);
    }
    out = outArr[0];
}

void testDec(ap_uint<64> in, ap_uint<64> key, ap_uint<64>& out) {
    ap_uint<64> outArr[N];
#pragma HLS ARRAY_PARTITION variable = outArr complete
    for (int i = 0; i < N; ++i) {
#pragma HLS pipeline
        xf::security::desDecrypt(in, key, outArr[i]);
    }

    out = outArr[0];
}
