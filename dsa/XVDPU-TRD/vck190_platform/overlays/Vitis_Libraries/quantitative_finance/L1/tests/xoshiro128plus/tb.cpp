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

#include <iostream>
#include <hls_stream.h>
#include <ap_int.h>
#include "ext_xoshiro128plus.hpp"

void dut(unsigned int seed[4], int n, hls::stream<ap_uint<32> >& rngStrm);

int main() {
    int nerr = 0;
    unsigned n = 10000;
    unsigned int seed[4] = {1, 2, 3, 4};
    rng::s[0] = seed[0];
    rng::s[1] = seed[1];
    rng::s[2] = seed[2];
    rng::s[3] = seed[3];
    rng::jump();
    hls::stream<ap_uint<32> > rngStrm;
    dut(seed, n, rngStrm);
    for (int i = 0; i < n; i++) {
        unsigned int result = rngStrm.read();
        unsigned int golden = rng::next();
        if (result != golden) {
            std::cout << "rng[" << i << "]=" << result << ",golden[" << i << "]=" << golden << std::endl;
            nerr++;
        }
    }
    return nerr;
}
