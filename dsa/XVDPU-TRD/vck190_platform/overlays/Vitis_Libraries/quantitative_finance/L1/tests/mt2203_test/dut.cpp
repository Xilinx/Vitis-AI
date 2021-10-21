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

#include <ap_int.h>
#include "xf_fintech/rng.hpp"

/**
 * @brief test function for mT2203 rng
 *
 */
extern "C" void dut(const int num, ap_uint<32> st[4], ap_ufixed<32, 0> output[100]) {
    xf::fintech::MT2203 rngInst;

    rngInst.statusSetup(st[1], st[2], st[3]);

    rngInst.seedInitialization(st[0]);

    for (int i = 0; i < num; i++) {
#pragma HLS pipeline II = 1
        output[i] = rngInst.next();
    }
}
