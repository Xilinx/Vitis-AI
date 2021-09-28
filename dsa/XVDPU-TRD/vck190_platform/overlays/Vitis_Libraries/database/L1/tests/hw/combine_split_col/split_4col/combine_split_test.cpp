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

#define TIN ap_uint<42>
#define T1 ap_uint<6>
#define T2 ap_uint<10>
#define T3 ap_uint<17>
#define T4 ap_uint<9>

template <typename KEY_T1, typename KEY_T2, typename KEY_T3, typename KEY_T4>
struct row_msg {
    KEY_T1 kin1;
    KEY_T2 kin2;
    KEY_T3 kin3;
    KEY_T4 kin4;
};

void xf_database_split_col(hls::stream<TIN>& kin_strm,
                           hls::stream<bool>& e_in_strm,
                           hls::stream<T1>& kout1_strm,
                           hls::stream<T2>& kout2_strm,
                           hls::stream<T3>& kout3_strm,
                           hls::stream<T4>& kout4_strm,
                           hls::stream<bool>& e_out_strm) {
    xf::database::splitCol(kin_strm, e_in_strm, kout1_strm, kout2_strm, kout3_strm, kout4_strm, e_out_strm);
}

#ifndef __SYNTHESIS__
// generate a random integer sequence between specified limits a and b (a<b);
uint rand_uint(uint a, uint b) {
    return rand() % (b - a + 1) + a;
}

// for each ap_uint<N> datatype, generate test data
template <typename KEY_IN>
void generate_test_data(uint64_t len, std::vector<KEY_IN>& testvector) {
    for (int i = 0; i < len; i++) {
        uint randnum = rand_uint(1, 4294967295);
        testvector.push_back({(KEY_IN)randnum});
        std::cout << "testvector is " << (int)testvector[i] << std::endl;
    }
    std::cout << " random test data generated! " << std::endl;
}

// len is the data number for each T1-T2-O group
int test_function(int len) {
    std::vector<TIN> testVector;
    hls::stream<TIN> kin_strm("kin_strm");
    hls::stream<T1> kout1_strm("kout1_strm");
    hls::stream<T2> kout2_strm("kout2_strm");
    hls::stream<T3> kout3_strm("kout3_strm");
    hls::stream<T4> kout4_strm("kout4_strm");
    hls::stream<bool> e_in_strm("e_in_strm");
    hls::stream<bool> e_out_strm("e_dout_strm");
    // generate test data
    generate_test_data<TIN>(len, testVector);
    int nerror = 0;
    // ref vec
    std::vector<row_msg<T1, T2, T3, T4> > refvec;
    // prepare data to stream
    for (std::string::size_type i = 0; i < len; i++) {
        kin_strm.write(testVector[i]);
        TIN in;
        T1 out1_value;
        T2 out2_value;
        T3 out3_value;
        T4 out4_value;
        uint64_t width_t1 = out1_value.length();
        uint64_t width_t2 = out2_value.length();
        uint64_t width_t3 = out3_value.length();
        uint64_t width_t4 = out4_value.length();
        uint64_t width_in = in.length();
        out1_value = testVector[i].range(width_t1 - 1, 0);
        out2_value = testVector[i].range(width_t1 + width_t2 - 1, width_t1);
        out3_value = testVector[i].range(width_in - width_t4 - 1, width_t1 + width_t2);
        out4_value = testVector[i].range(width_in - 1, width_t1 + width_t2 + width_t3);

        refvec.push_back({out1_value, out2_value, out3_value, out4_value});
        std::cout << "refvec data is " << refvec[i].kin1 << ", " << refvec[i].kin2 << ", " << refvec[i].kin3
                  << std::endl;
        e_in_strm.write(0);
    }
    e_in_strm.write(1);
    // run hls::func
    xf_database_split_col(kin_strm, e_in_strm, kout1_strm, kout2_strm, kout3_strm, kout4_strm, e_out_strm);
    // compare hls::func and refernece result

    // compare value
    for (std::string::size_type i = 0; i < len; i++) {
        //        std::cout << "refvec[i] is " << refvec[i] << std::endl;
        T1 out_res1 = kout1_strm.read();
        T2 out_res2 = kout2_strm.read();
        T3 out_res3 = kout3_strm.read();
        T4 out_res4 = kout4_strm.read();
        bool comp_res1 = (refvec[i].kin1 == out_res1) ? 1 : 0;
        bool comp_res2 = (refvec[i].kin2 == out_res2) ? 1 : 0;
        bool comp_res3 = (refvec[i].kin3 == out_res3) ? 1 : 0;
        bool comp_res4 = (refvec[i].kin4 == out_res4) ? 1 : 0;
        if (!comp_res1 || !comp_res2 || !comp_res3 || !comp_res4) {
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
