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
#include "kernel.hpp"

#include <vector>
#include <string>
#include <iostream>

#include "xhostutils.hpp"

int main(int argc, const char* argv[]) {
    ap_uint<64>* buf0 = aligned_alloc<ap_uint<64> >(VEC_LEN * BUF_DEPTH);
    ap_uint<64>* buf1 = aligned_alloc<ap_uint<64> >(VEC_LEN * BUF_DEPTH);
    ap_uint<64>* bufo = aligned_alloc<ap_uint<64> >(2);
    int nrow = VEC_LEN * BUF_DEPTH - VEC_LEN;
    std::cout << "testing " << nrow << " rows\n";
    // first vec, only nrow
    buf0[0] = nrow;
    buf1[0] = 0;
    for (int i = 1; i < VEC_LEN; ++i) {
        buf0[i] = 0;
    }
    // following vec
    for (int i = 0; i < VEC_LEN * BUF_DEPTH - VEC_LEN; ++i) {
        if (i < nrow) {
            buf0[VEC_LEN + i] = i;
        } else {
            buf0[VEC_LEN + i] = 5;
        }
    }
    std::cout << "starting test..." << std::endl;
    Test((ap_uint<64 * VEC_LEN>*)buf0, (ap_uint<64 * VEC_LEN>*)buf1, bufo);
    std::cout << "done. result length: " << bufo[0] << ", result value " << bufo[1] << "\n";

    unsigned long ref = 0;
    for (int i = 0; i < nrow; ++i) {
        ref += i;
    }
    if (bufo[0] == 1 && bufo[1] == ref) {
        std::cout << "PASS" << std::endl;
        return 0;
    } else {
        std::cout << "FAIL" << std::endl;
        return 1;
    }
}
