/*
 * Copyright 2020 Xilinx, Inc.
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
#include "xf_database/dynamic_eval_v2_config.hpp"
// cout
#include <iostream>
#include <iomanip>
// rand
#include <cstdlib>
// uint32_t
#include <cstdint>
// assert
#include <cassert>

#include "ap_int.h"
#include "hls_stream.h"
#include "eval_dut.hpp"

int main(int argc, const char* argv[]) {
    using namespace xf::database;

    int nerror = 0;

    DynamicEvalV2Config ec("(-100 - b) * (10 - a) + (d + 3) * c");
    auto estr = ec.getConfigBits();
    std::cout << "test expr:(-100 - b) * (10 - a) + (d + 3) * c" << std::endl;
    hls::stream<ap_uint<32> > cfgs;
    for (int i = 0; i < ec.getConfigLen(); ++i) {
        cfgs.write(estr[i]);
    }

    hls::stream<int> col0_istrm, col1_istrm, col2_istrm, col3_istrm, ret_ostrm, refs;
    hls::stream<bool> e_istrm, e_ostrm;

    //-------------------------generate test data/golden data----------------------------
    const int ncase = 1000;

    for (int k = 0; k < ncase; ++k) {
        int a = rand() % 100;
        int b = rand() % 100;
        int c = rand() % 100;
        int d = rand() % 100;
        int rv = (-100 - b) * (10 - a) + (d + 3) * c;
        refs.write(rv);

        col0_istrm.write(a);
        col1_istrm.write(b);
        col2_istrm.write(c);
        col3_istrm.write(d);
        e_istrm.write(false);
    }
    e_istrm.write(true);

    eval2_dut(cfgs,                                                    //
              col0_istrm, col1_istrm, col2_istrm, col3_istrm, e_istrm, //
              ret_ostrm, e_ostrm);

    for (int k = 0; k < ncase; ++k) {
        int ref = refs.read();
        int result = ret_ostrm.read();
        e_ostrm.read();
        nerror += (result != ref);
    }
    e_ostrm.read();
    if (0 == nerror) {
        std::cout << "No Error Found,PASS\n";
    } else {
        std::cout << nerror << " cases unmatch!"
                  << "FAIL\n";
    }
    return nerror;
}
