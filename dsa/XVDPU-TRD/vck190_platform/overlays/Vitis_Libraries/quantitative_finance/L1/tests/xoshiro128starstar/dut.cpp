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

#include <hls_stream.h>
#include "xoshiro128.hpp"

void dut(unsigned int seed[4], int n, hls::stream<ap_uint<32> >& rngStrm) {
    xf::fintech::XoShiRo128StarStar rng;
    rng.init(seed);
    rng.jump();

    for (int i = 0; i < n; i++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount max = 100 min = 100
        rngStrm.write(rng.next());
    }
}
