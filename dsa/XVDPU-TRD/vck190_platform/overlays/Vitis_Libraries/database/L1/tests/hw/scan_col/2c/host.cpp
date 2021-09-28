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
    ap_uint<64>* bufo = aligned_alloc<ap_uint<64> >(1);
    int nrow = VEC_LEN * BUF_DEPTH;
    std::cout << "testing " << nrow << " rows\n";
    for (int i = 0; i < VEC_LEN * BUF_DEPTH; ++i) {
        if (i < nrow) {
            buf0[i] = i;
            buf1[i] = i * 2;
        } else {
            buf0[i] = 1;
            buf1[i] = 1;
        }
    }
    std::cout << "starting test..." << std::endl;
    Test((ap_uint<64 * VEC_LEN>*)buf0, (ap_uint<64 * VEC_LEN>*)buf1, nrow, bufo);
    std::cout << "done. result value " << *bufo << "\n";

    unsigned long ref = 0;
    for (int i = 0; i < nrow; ++i) {
        ref += i;
    }
    if (*bufo == ref) {
        std::cout << "PASS" << std::endl;
        return 0;
    } else {
        std::cout << "FAIL" << std::endl;
        return 1;
    }
}
