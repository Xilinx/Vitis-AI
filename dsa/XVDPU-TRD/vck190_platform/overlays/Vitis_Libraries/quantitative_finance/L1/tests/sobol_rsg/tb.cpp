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

#include <ap_int.h>
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include "hls_stream.h"
#include "sobolrsg.hpp"

#define NUM_OF_RAND 8192
#define DIM 8

void dut_1d(const int num_of_rand, hls::stream<ap_ufixed<32, 0> >& out_strm);
void dut_nd(const int num_of_rand, hls::stream<ap_ufixed<32, 0> >& out_strm);

int main() {
    int nerror = 0;
    double out;
    hls::stream<ap_ufixed<32, 0> > out_strm;

    QuantLib::SobolRsg<DIM> rsg;
    rsg.initialization();
    double point[DIM];

    if (DIM == 1)
        dut_1d(NUM_OF_RAND, out_strm);
    else
        dut_nd(NUM_OF_RAND, out_strm);

    for (int i = 1; i < NUM_OF_RAND; i++) {
        rsg.nextSequence(point);
    }
    for (int i = 0; i < DIM; i++) {
        out = (double)out_strm.read();

        bool cmp = (double)fabs((point[i] - out)) < 0.0000000000000001 ? 1 : 0;
        if (!cmp) {
            nerror++;
            std::cout << "i=" << i << ",out=" << out << ",point=" << point[i] << std::endl;
        }
    }

    if (nerror != 0)
        std::cout << "\nFAIL: nerror = " << nerror << " errors found.\n";
    else
        std::cout << "\nPASS: no error found.\n";
    return nerror;
}
