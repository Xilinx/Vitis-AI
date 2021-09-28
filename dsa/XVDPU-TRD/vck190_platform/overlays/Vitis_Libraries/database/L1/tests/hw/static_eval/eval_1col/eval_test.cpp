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

#include "hls_stream.h"
#include "xf_database/static_eval.hpp"
#define LEN 10 // number of test data generated

int user_func(int a) {
    return a * a;
}

void dut(hls::stream<int>& in_strm,
         hls::stream<bool>& e_in_strm,
         hls::stream<int>& out_strm,
         hls::stream<bool>& e_out_strm) {
    xf::database::staticEval<int, int, user_func>(in_strm, e_in_strm, out_strm, e_out_strm);
}

// generate a random integer sequence between speified limits a and b (a<b);
int rand_int(int a, int b) {
    return rand() % (b - a + 1) + a;
}

void generate_test_data(uint64_t len, std::vector<int>& testvector) {
    for (int i = 0; i < len; i++) {
        int randnum = rand_int(-100, 100);
        testvector.push_back(randnum);
    }
}

int main() {
    hls::stream<int> in1_strm("in1_strm");
    hls::stream<bool> e_in_strm("e_in_strm");
    hls::stream<int> out_strm("out_strm");
    hls::stream<bool> e_out_strm("e_out_strm");

    // generate test dat
    std::vector<int> testvector;
    generate_test_data(LEN, testvector);
    // prepare data, write data into stream
    for (std::string::size_type i = 0; i < LEN; i++) {
        std::cout << "testvector data is " << testvector[i] << std::endl;
        in1_strm.write(testvector[i]);
        e_in_strm.write(0);
    }
    e_in_strm.write(1);
    std::cout << std::endl;

    // call ALU_block
    dut(in1_strm, e_in_strm, out_strm, e_out_strm);

    int nerror = 0;
    //===== check if the output flag e_out_strm is correct or not =====
    for (int i = 0; i < LEN; i++) {
        bool e = e_out_strm.read();
        if (e) {
            nerror++;
            std::cout << "the alu flag is incorrect" << std::endl;
        }
    }
    // read out the last flag that e should =1
    bool e = e_out_strm.read();
    if (!e) {
        nerror++;
        std::cout << "the alu data is incorrect" << std::endl;
    }
    // =======check the result with referenece
    for (int i = 0; i < LEN; i++) {
        bool cmp_dout = (user_func(testvector[i]) == out_strm.read()) ? 1 : 0;
        if (!cmp_dout) {
            nerror++;
            std::cout << "the alu data is incorrect" << std::endl;
        }
    }

    if (nerror) {
        std::cout << "\nFAIL: nerror= " << nerror << " errors found.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return nerror;
}
