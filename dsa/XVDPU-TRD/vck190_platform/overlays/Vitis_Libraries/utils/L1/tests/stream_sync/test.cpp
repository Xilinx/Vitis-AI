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

#include "xf_utils_hw/stream_sync.hpp"

#define W_STRM 16
#define NSTRM 16
#define NS (1024 * NSTRM)

typedef ap_uint<W_STRM> TYPE;

void test_core_sync(hls::stream<TYPE> istrms[NSTRM],
                    hls::stream<bool> e_istrms[NSTRM],
                    hls::stream<TYPE> ostrms[NSTRM],
                    hls::stream<bool>& e_ostrm) {
    xf::common::utils_hw::streamSync<TYPE, NSTRM>(istrms, e_istrms, ostrms, e_ostrm);
}

int test_sync() {
    hls::stream<TYPE> data_istrms[NSTRM];
    hls::stream<bool> e_data_istrms[NSTRM];
    hls::stream<TYPE> data_ostrms[NSTRM];
    hls::stream<bool> e_data_ostrm;

    std::cout << std::dec << "W_STRM  = " << W_STRM << std::endl;
    std::cout << std::dec << "NSTRM   = " << NSTRM << std::endl;
    std::cout << std::dec << "NS      = " << NS << std::endl;
    for (int d = 1; d <= NS; ++d) {
        int id = (d - 1) % NSTRM;
        data_istrms[id].write(d);
        e_data_istrms[id].write(false);
    }
    for (int i = 0; i < NSTRM; ++i) e_data_istrms[i].write(true);

    test_core_sync(data_istrms, e_data_istrms, data_ostrms, e_data_ostrm);
    int nerror = 0;
    int total = 0;
    bool last = e_data_ostrm.read();
    ap_uint<W_STRM> gld = 0;
    while (!last) {
        last = e_data_ostrm.read();
        for (int k = 0; k < NSTRM; ++k) {
            TYPE d = data_ostrms[k].read();
            total++;
            gld = gld + 1;
            if (d != gld) {
                nerror = 1;
                std::cout << "erro:"
                          << "  test data = " << d << "   "
                          << "gld data = " << gld << std::endl;
            } // if
        }     // for
    }         // while
    if (total != NS) nerror = 1;
    if (nerror) {
        std::cout << "\nFAIL: " << nerror << "the order is wrong.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return nerror;
}

int main() {
    return test_sync();
}
