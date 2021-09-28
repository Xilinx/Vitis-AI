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

#include "covariance.hpp"
#define N 64
#define M 1536
#define DT double
#define DTLEN 64
#define TI 2
#define TO 2
void dut(int rows,
         int cols,
         hls::stream<ap_uint<DTLEN * TI> >& inMatStrm,
         hls::stream<ap_uint<DTLEN * TO> >& outCovStrm) {
    xf::fintech::covCoreStrm<DT, DTLEN, N, M, TI, TO>(rows, cols, inMatStrm, outCovStrm);
}
