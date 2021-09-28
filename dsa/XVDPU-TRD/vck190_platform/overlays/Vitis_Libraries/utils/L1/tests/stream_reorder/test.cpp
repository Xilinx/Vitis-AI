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

#include "xf_utils_hw/stream_reorder.hpp"

#define WS 4
#define W_STRM 16
#define NS (1024 * WS)

void test_core_reorder(hls::stream<int>& order_cfg,
                       hls::stream<ap_uint<W_STRM> >& istrm,
                       hls::stream<bool>& e_istrm,
                       hls::stream<ap_uint<W_STRM> >& ostrm,
                       hls::stream<bool>& e_ostrm) {
    xf::common::utils_hw::streamReorder<ap_uint<W_STRM>, WS>(order_cfg, istrm, e_istrm, ostrm, e_ostrm);
}

int test_reorder() {
    hls::stream<int> cfg;
    hls::stream<ap_uint<W_STRM> > data_istrm;
    hls::stream<bool> e_data_istrm;
    hls::stream<ap_uint<W_STRM> > data_ostrm;
    hls::stream<bool> e_data_ostrm;
    ap_uint<W_STRM> ba[WS];
    std::cout << std::dec << "W_STRM  = " << W_STRM << std::endl;
    std::cout << std::dec << "WS      = " << WS << std::endl;
    std::cout << std::dec << "NS      = " << NS << std::endl;
    for (int i = 0; i < WS; ++i) {
        cfg.write(WS - i - 1);
    }
    for (int d = 1; d <= NS; ++d) {
        ap_uint<W_STRM> data = d;
        data_istrm.write(data);
        e_data_istrm.write(false);
    }

    e_data_istrm.write(true);

    test_core_reorder(cfg, data_istrm, e_data_istrm, data_ostrm, e_data_ostrm);
    std::cout << "================================" << std::endl;
    int nerror = 0;
    bool last = e_data_ostrm.read();
    ap_uint<W_STRM> gld = 0;
    int c = 0;
    int total = 0;
    while (!last) {
        last = e_data_ostrm.read();
        ba[c] = data_ostrm.read();
        //    std::cout<<" t= "<< total << "   "<< ba[c] << std::endl;
        c++;
        total++;
        if (c == WS) {
            c = 0;
            for (int k = WS - 1; k >= 0; --k) {
                ap_uint<W_STRM> d = ba[k];
                gld = gld + 1;
                if (d != gld) {
                    nerror = 1;
                    std::cout << "erro:"
                              << "  test data = " << d << "   "
                              << "gld data = " << gld << std::endl;
                }
            } // for
        }     // if
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
    return test_reorder();
}
