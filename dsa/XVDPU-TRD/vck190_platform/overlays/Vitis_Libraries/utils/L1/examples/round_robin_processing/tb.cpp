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

#include <vector>
#include <iostream>
#include <stdlib.h>

#include "code.hpp"

// double input as output
ap_uint<W_PU> compute(int d) {
    ap_uint<W_PU> nd = d;
    nd = nd << 1;
    return nd;
};

int test() {
    hls::stream<ap_uint<W_STRM> > istrm;
    hls::stream<bool> e_istrm;
    hls::stream<ap_uint<W_STRM> > ostrm;
    hls::stream<bool> e_ostrm;

    int tempa = W_STRM * NS / W_PU;
    int tempb = tempa * W_PU;
    int comp_count = tempb / W_STRM;

    std::cout << std::dec << "W_STRM  = " << W_STRM << std::endl;
    std::cout << std::dec << "W_PU    = " << W_PU << std::endl;
    std::cout << std::dec << "NPU     = " << NPU << std::endl;
    std::cout << std::dec << "NS      = " << NS << std::endl;
    for (int d = 0; d < NS; ++d) {
        istrm.write(d);
        e_istrm.write(false);
    }
    e_istrm.write(true);

    round_robin_mpu(istrm, e_istrm, ostrm, e_ostrm);

    int nerror = 0;
    int count = 0;
    bool first = true;

    while (!e_ostrm.read()) {
        ap_uint<W_STRM> d = ostrm.read();
        ap_uint<W_PU> gld = compute(count);
        if (count <= comp_count && d != gld) {
            nerror = 1;
            std::cout << "erro: "
                      << "c=" << count << ", gld=" << gld << ", "
                      << " data=" << d << std::endl;
        }
        count++;
    } // while
    std::cout << "\n total read: " << count << std::endl;
    if (count != comp_count) {
        nerror = 1;
        std::cout << "\n error:  total read = " << count << ", comp_count = " << comp_count << std::endl;
    }
    if (nerror) {
        std::cout << "\nFAIL: " << nerror << "the order is wrong.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return nerror;
}

int main() {
    return test();
}
