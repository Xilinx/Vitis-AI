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

int test() {
    hls::stream<ap_uint<W_STRM> > istrm;
    hls::stream<bool> e_istrm;
    hls::stream<ap_uint<W_TAG> > tg_strms[2];
    hls::stream<bool> e_tg_strms[2];
    hls::stream<ap_uint<W_STRM> > ostrm;
    hls::stream<bool> e_ostrm;

    std::cout << std::dec << "W_STRM  = " << W_STRM << std::endl;
    std::cout << std::dec << "W_TAG   = " << W_TAG << std::endl;
    std::cout << std::dec << "NTAG    = " << NTAG << std::endl;
    std::cout << std::dec << "NS      = " << NS << std::endl;
    for (int d = 0; d < NS; ++d) {
        istrm.write(d);
        e_istrm.write(false);
        ap_uint<W_TAG> tg = NTAG - d % NTAG;
        tg_strms[0].write(d);
        tg_strms[1].write(d);
        e_tg_strms[0].write(false);
        e_tg_strms[1].write(false);
    }
    e_istrm.write(true);
    e_tg_strms[0].write(true);
    e_tg_strms[1].write(true);

    // process
    test_core(istrm, e_istrm, tg_strms, e_tg_strms, ostrm, e_ostrm);

    // fetch back and check
    int nerror = 0;
    int count = 0;
    while (!e_ostrm.read()) {
        ap_uint<W_STRM> d = ostrm.read();
        ap_uint<W_STRM> gld = count + 1;
        if (count <= NS && d != gld) {
            nerror = 1;
            std::cout << "erro: "
                      << "c=" << count << ", gld=" << gld << ", "
                      << " data=" << d << std::endl;
        }
        count++;
    } // while
    std::cout << "\n total read: " << count << std::endl;
    if (count != NS) {
        nerror = 1;
        std::cout << "\n error:  total read = " << count << ", NS = " << NS << std::endl;
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
