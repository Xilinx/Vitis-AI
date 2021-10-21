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

#include "xf_utils_hw/stream_n_to_one/round_robin.hpp"

#define WIN_STRM 12
#define WOUT_STRM 64
#define NS 1024 * 8 - 3
#define NSTRM 8

void test_core_n_1(hls::stream<ap_uint<WIN_STRM> > istrms[NSTRM],
                   hls::stream<bool> e_istrms[NSTRM],
                   hls::stream<ap_uint<WOUT_STRM> >& ostrm,
                   hls::stream<bool>& e_ostrm) {
    xf::common::utils_hw::streamNToOne<WIN_STRM, WOUT_STRM, NSTRM>(istrms, e_istrms, ostrm, e_ostrm,
                                                                   xf::common::utils_hw::RoundRobinT());
}

int test_n_1() {
    hls::stream<ap_uint<WIN_STRM> > data_istrms[NSTRM];
    hls::stream<bool> e_data_istrms[NSTRM];
    hls::stream<ap_uint<WOUT_STRM> > data_ostrm;
    hls::stream<bool> e_data_ostrm;

    const int buf_width = xf::common::utils_hw::LCM<WIN_STRM * NSTRM, WOUT_STRM>::value;
    const int num_in = buf_width / WIN_STRM;
    const int num_out = buf_width / WOUT_STRM;

    ap_uint<buf_width> buff_test = 0;
    ap_uint<buf_width> buff_gld = 0;
    std::cout << std::dec << "WIN_STRM  = " << WIN_STRM << std::endl;
    std::cout << std::dec << "WOUT_STRM = " << WOUT_STRM << std::endl;
    std::cout << std::dec << "NSTRM     = " << NSTRM << std::endl;
    std::cout << std::dec << "NS        = " << NS << std::endl;
    for (int d = 0; d < NS; ++d) {
        ap_uint<WIN_STRM> id = d % NSTRM;
        data_istrms[id].write(d);
        e_data_istrms[id].write(false);
#if !defined(__SYNTHESIS__) && XF_UTIL_STRM_1NRR_DEBUG == 1
        std::cout << "id = " << id << "  "
                  << "d=" << d << std::endl;
#endif
    }

    for (int i = 0; i < NSTRM; ++i) e_data_istrms[i].write(true);

    test_core_n_1(data_istrms, e_data_istrms, data_ostrm, e_data_ostrm);
    int nerror = 0;
    int n = 0;
    int k = 0;
    bool last = e_data_ostrm.read();
    int kk = 0;
    while (!last) {
        // get num_out*WOUT_STRM bits from stream
        while (!last && k < num_out) {
            last = e_data_ostrm.read();
            ap_uint<WOUT_STRM> d = data_ostrm.read();
            buff_test.range((k + 1) * WOUT_STRM - 1, k * WOUT_STRM) = d;
            ++k;
            ++kk;
        }
        if (k == num_out) {
            k = 0;
            // get num_in*WIN_STRM bits from golden data;
            for (int i = 0; i < num_in; ++i) {
                buff_gld.range((i + 1) * WIN_STRM - 1, i * WIN_STRM) = n;
                ++n;
            }
            // compare (num_in*WIN_STRM = num_out*WOUT_STRM) bits
            if (buff_test != buff_gld) {
                nerror = 1;
                std::cout << "buff_test = " << buff_test << std::endl;
                std::cout << "buff_gld  = " << buff_gld << std::endl;
            }
        }                  // if (k == num_out)
    }                      // while(!last)
    int output_count = kk; // n + k ;
    if (k > 0) {
        for (int i = 0; i < num_in; ++i) {
            buff_gld.range((i + 1) * WIN_STRM - 1, i * WIN_STRM) = n;
            ++n;
        }
        int up = k * WOUT_STRM - 1;
        if (buff_test.range(up, 0) != buff_gld.range(up, 0)) {
            nerror = 1;
            std::cout << "buff_test = " << buff_test << std::endl;
            std::cout << "buff_gld  = " << buff_gld << std::endl;
        }
    }
    int comp_count = (NS) * (WIN_STRM) / (WOUT_STRM);
    if (comp_count != output_count) nerror = 1;
    int total_input = (NS)*WIN_STRM;
    int total_output = comp_count * WOUT_STRM;
    std::cout << "comp_count= " << comp_count << std::endl;
    std::cout << "output_count= " << output_count << std::endl;
    std::cout << "totally input NS*WIN_STRM = " << total_input << " bits through NStrm = " << NSTRM << " streams"
              << std::endl;
    std::cout << "totally output output_count*WOUT_STRM = " << total_output << " bits" << std::endl;
    std::cout << "discard " << total_input - total_output << " bits" << std::endl;
    std::cout << "compare comp_count*WOUT_STRM= " << comp_count * WOUT_STRM << " bits" << std::endl;

    if (nerror) {
        std::cout << "\nFAIL: " << nerror << "the order is wrong.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return nerror;
}

int main() {
    return test_n_1();
}
