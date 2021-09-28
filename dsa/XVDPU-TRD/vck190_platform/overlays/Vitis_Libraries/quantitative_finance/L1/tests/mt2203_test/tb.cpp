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
#include <iostream>
extern "C" {
#include "dc.h"
}

extern "C" void dut(const int num, ap_uint<32> st[4], ap_ufixed<32, 0> output[100]);

int main() {
    int num = 100;

    mt_struct* mts[2];
    init_dc(4172);
    for (int i = 0; i < 2; ++i) {
        printf("get first MT[%d]\n", i);
        mts[i] = get_mt_parameter_id(32, 2203, 0);
    }
    sgenrand_mt(1234, mts[0]);
    ap_ufixed<32, 0> dcmt_output[100];
    for (int i = 0; i < num; ++i) {
        ap_uint<32> dcmt_o = genrand_mt(mts[0]);
        dcmt_output[i].range(31, 0) = dcmt_o(31, 0);
    }
    ap_uint<32> st[4];
    st[0] = 1234;
    st[1] = mts[0]->aaa;
    st[2] = mts[0]->maskB;
    st[3] = mts[0]->maskC;

    ap_ufixed<32, 0> output[100];
    dut(num, st, output);

    for (int i = 0; i < num; ++i) {
        if (output[i] != dcmt_output[i]) {
            std::cout << "i:" << i << ", acut out :" << output[i] << ", ref out :" << dcmt_output[i] << std::endl;
            return -1;
        }
    }
    std::cout << "output correct." << std::endl;
    for (int i = 0; i < 2; ++i) free_mt_struct(mts[i]);
    return 0;
}
