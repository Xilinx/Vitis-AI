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

#include <vector> // std::vector
#include <iostream>
#include <stdlib.h>

#define AP_INT_MAX_W 4096

#include "xf_database/duplicate_col.hpp"
#include "hls_stream.h"

#define INT_RAND_MIN -100
#define INT_RAND_MAX 100

#define W 32

void xf_database_duplicate_col(hls::stream<ap_uint<W> >& d_in_strm,
                               hls::stream<bool>& e_in_strm,
                               hls::stream<ap_uint<W> >& d1_out_strm,
                               hls::stream<ap_uint<W> >& d2_out_strm,
                               hls::stream<bool>& e_out_strm) {
    xf::database::duplicateCol(d_in_strm, e_in_strm, d1_out_strm, d2_out_strm, e_out_strm);
}

#ifndef __SYNTHESIS__
// generate a random integer sequence between specified limits a and b (a<b);
uint rand_uint(uint a, uint b) {
    return rand() % (b - a + 1) + a;
}

//  generate test data
void generate_test_data(int len, std::vector<ap_uint<W> >& testvector) {
    for (int i = 0; i < len; i++) {
        uint randnum = rand_uint(1, 15);
        testvector.push_back((ap_uint<W>)randnum);
    }
    std::cout << " random test data generated! " << std::endl;
}

int test_function(int len) {
    std::vector<ap_uint<W> > testVector;
    hls::stream<ap_uint<W> > din_strm("din_strm");
    hls::stream<bool> e_in_strm("e_in_strm");
    hls::stream<ap_uint<W> > dout1_strm("dout1_strm");
    hls::stream<ap_uint<W> > dout2_strm("dout1_strm");
    hls::stream<bool> e_out_strm("e_dout_strm");
    // reference vector
    std::vector<ap_uint<W> > refvec;
    // generate test data
    generate_test_data(len, testVector);
    // prepare data to stream
    for (int i = 0; i < len; i++) {
        din_strm.write(testVector[i]);
        refvec.push_back(testVector[i]);
        e_in_strm.write(0);
    }
    e_in_strm.write(1);
    // run hls::func
    xf_database_duplicate_col(din_strm, e_in_strm, dout1_strm, dout2_strm, e_out_strm);
    // compare hls::func and refernece result
    int nerror = 0;
    int cnt = 0;
    // compare value
    while (!e_out_strm.read() && cnt < len) {
        ap_uint<W> out1_res = dout1_strm.read();
        ap_uint<W> out2_res = dout2_strm.read();
        bool comp_res = (refvec[cnt] == out1_res && refvec[cnt] == out2_res) ? 0 : 1;
        if (comp_res) {
            nerror++;
        }
        cnt++;
    }
    if (cnt != len) nerror++;
    return nerror;
}

int main() {
    int nerror = 0;
    nerror = test_function(4);

    if (nerror) {
        std::cout << "\nFAIL: nerror= " << nerror << " errors found.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return nerror;
}
#endif
