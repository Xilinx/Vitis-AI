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
#include <stdlib.h>

#include "dut.hpp"

int main(int argc, const char* argv[]) {
    int nerror = 0;

    hls::stream<ap_uint<WDATA> > ref_stream("reference");
    ap_uint<WDATA> ref_array[NDATA];
    for (int i = 0; i < NDATA; ++i) {
        ref_array[i] = 1;
    }
    for (int i = 0; i < NUM_SIZE; ++i) {
        if ((i & 1) == 0) {
            ref_array[i] = i;
        } else {
            ap_uint<WDATA> t = ref_array[i - 1];
            ref_stream.write(t);
        }
    }
    for (int i = 0; i < NUM_SIZE; i++) {
        ap_uint<WDATA> t = ref_array[i];
        ap_uint<WDATA> u = (t & 1) ? 1 : 0;
        ref_array[i] = u;
    }
    for (int i = 0; i < NDATA; ++i) {
        ap_uint<WDATA> t = ref_array[i];
        ref_stream.write(t);
    }

    ap_uint<WDATA> ii = 1;
    hls::stream<ap_uint<WDATA> > out_stream("output");

    dut(ii, out_stream);

    while (true) {
        ap_uint<WDATA> r = ref_stream.read();
        ap_uint<WDATA> o = out_stream.read();
        if (r != o) {
            if (!nerror) std::cout << "The data is incorrect, check implementation." << std::endl;
            nerror++;
        }
        if (ref_stream.size() == 0) {
            if (out_stream.size() == 0) {
                break;
            } else {
                std::cout << "The number of data is incorrect, check test case" << std::endl;
            }
        } else if (out_stream.size() == 0) {
            std::cout << "The number of data is incorrect, check test case" << std::endl;
        }
    }

    if (nerror) {
        std::cout << "\nFAIL: " << nerror << " elements are wrong.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return nerror;
}
