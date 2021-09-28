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
#include "dut.hpp"

/**
 * @brief test function for Normal RNG
 *
 */
extern "C" void dut(const int num,
                    const int preRun,
                    ap_uint<32> st[4],
                    double outputMT19937ICN[SAMPLE_NUM],
                    double outputMT2203ICN[SAMPLE_NUM],
                    double outputMT19937BoxMuller[SAMPLE_NUM]) {
    xf::fintech::MT19937IcnRng<double> rngMT19937ICN;
    xf::fintech::MT2203IcnRng<double> rngMT2203ICN;
    xf::fintech::MT19937BoxMullerNormalRng rngMT19937BoxMuller;

    rngMT19937ICN.seedInitialization(st[0]);
    rngMT2203ICN.statusSetup(st[1], st[2], st[3]);
    rngMT2203ICN.seedInitialization(st[0]);
    rngMT19937BoxMuller.seedInitialization(st[0]);

    for (int i = 0; i < num; i++) {
#pragma HLS pipeline II = 1
        outputMT19937ICN[i] = rngMT19937ICN.next();
    }
    for (int i = 0; i < num; i++) {
#pragma HLS pipeline II = 1
        outputMT2203ICN[i] = rngMT2203ICN.next();
    }
    for (int i = 0; i < num; i++) {
#pragma HLS pipeline II = 1
        outputMT19937BoxMuller[i] = rngMT19937BoxMuller.next();
    }
}
