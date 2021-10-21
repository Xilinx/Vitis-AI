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

/**
 * @file dut.cpp
 *
 * @brief This file contains top function of test case.
 */

#include <ap_fixed.h>
#include <ap_int.h>
#include "hls_stream.h"
#include "sobol_rsg.hpp"

#define NDIM 8

/**
 * @brief test function of sobol sequence generator for 1 dimension.
 *
 * @param num_of_rand is number of random number generated.
 * @param out_strm output data of stream.
 *
 */
void dut_1d(const int num_of_rand, hls::stream<ap_ufixed<32, 0> >& out_strm) {
    ap_ufixed<32, 0> result;

    xf::fintech::SobolRsg1D ssg;
    ssg.initialization();
Sobol_Loop:
    for (int i = 0; i < num_of_rand; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1024 max = 8192
#pragma HLS pipeline II = 1
        ssg.next(&result);
    }
    out_strm.write(result);
}

/**
 * @brief test function of sobol sequence generator for n dimension.
 *
 * @param num_of_rand is number of random number generated.
 * @param out_strm the stream of output result.
 *
 **/
void dut_nd(const int num_of_rand, hls::stream<ap_ufixed<32, 0> >& out_strm) {
    ap_ufixed<32, 0> b;
    ap_ufixed<32, 0> result[NDIM];
#pragma HLS ARRAY_PARTITION variable = result dim = 0

    xf::fintech::SobolRsg<NDIM> ssg_nd;
    ssg_nd.initialization();
Sobol_Loop_nd:
    for (int i = 0; i < num_of_rand; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1024 max = 8192
#pragma HLS pipeline II = 1
        ssg_nd.next(result);
    }
Copy_Loop_nd:
    for (int j = 0; j < NDIM; j++) {
#pragma HLS pipeline II = 1
        b = result[j];
        out_strm.write(b);
    }
}
