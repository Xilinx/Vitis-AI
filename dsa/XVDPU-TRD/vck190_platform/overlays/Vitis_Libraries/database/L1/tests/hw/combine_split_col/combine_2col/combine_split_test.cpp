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

#include "xf_database/combine_split_col.hpp"
#include "hls_stream.h"

#define GROUP_NUM 50
#define INT_RAND_MIN -100
#define INT_RAND_MAX 100

#define T1 17
#define T2 6
#define TO 23

template <int KEY_T1, int KEY_T2>
struct row_msg {
    ap_uint<KEY_T1> kin1;
    ap_uint<KEY_T2> kin2;
};

void xf_database_combine_col1(hls::stream<ap_uint<T1> >& kin1_strm,
                              hls::stream<ap_uint<T2> >& kin2_strm,
                              hls::stream<bool>& e_in_strm,
                              hls::stream<ap_uint<TO> >& kout_strm,
                              hls::stream<bool>& e_out_strm) {
    xf::database::combineCol(kin1_strm, kin2_strm, e_in_strm, kout_strm, e_out_strm);
}

#ifndef __SYNTHESIS__
// generate a random integer sequence between specified limits a and b (a<b);
uint rand_uint(uint a, uint b) {
    return rand() % (b - a + 1) + a;
}

// for each ap_uint<N> datatype, generate test data
template <int KEY_T1, int KEY_T2>
void generate_test_data(uint64_t len, std::vector<row_msg<KEY_T1, KEY_T2> >& testvector) {
    for (int i = 0; i < len; i++) {
        uint randnum = rand_uint(1, 15);
        uint randnum2 = rand_uint(1, 15);
        testvector.push_back({(ap_uint<KEY_T1>)randnum, (ap_uint<KEY_T2>)randnum2});
    }
    std::cout << " random test data generated! " << std::endl;
}

// len is the data number for each T1-T2-TO group
int test_function(int len) {
    std::vector<row_msg<T1, T2> > testVector;
    hls::stream<ap_uint<T1> > kin1_strm("kin1_strm");
    hls::stream<ap_uint<T2> > kin2_strm("kin2_strm");
    hls::stream<ap_uint<TO> > kout_strm("kout_strm");
    hls::stream<bool> e_in_strm("e_in_strm");
    hls::stream<bool> e_out_strm("e_dout_strm");
    // reference vector
    std::vector<ap_uint<TO> > refvec;
    // generate test data
    generate_test_data<T1, T2>(len, testVector);
    // prepare data to stream
    for (std::string::size_type i = 0; i < len; i++) {
        kin1_strm.write(testVector[i].kin1);
        kin2_strm.write(testVector[i].kin2);
        refvec.push_back((testVector[i].kin1, testVector[i].kin2));
        e_in_strm.write(0);
    }
    e_in_strm.write(1);
    // run hls::func
    xf_database_combine_col1(kin1_strm, kin2_strm, e_in_strm, kout_strm, e_out_strm);
    // compare hls::func and refernece result
    int nerror = 0;
    // compare value
    for (std::string::size_type i = 0; i < len; i++) {
        ap_uint<TO> out_res = kout_strm.read();
        bool comp_res = (refvec[i] == out_res) ? 1 : 0;
        if (!comp_res) {
            nerror++;
        }
    }
    // compare e flag
    for (std::string::size_type i = 0; i < len; i++) {
        bool estrm = e_out_strm.read();
        if (estrm) {
            nerror++;
        }
    }
    bool estrm = e_out_strm.read();
    if (!estrm) {
        nerror++;
    }
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
