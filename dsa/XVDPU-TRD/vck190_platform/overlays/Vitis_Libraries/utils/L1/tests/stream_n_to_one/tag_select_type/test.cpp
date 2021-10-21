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

#include "xf_utils_hw/stream_n_to_one/tag_select.hpp"

#define WIN_STRM 16
#define WTAG_STRM 3
#define NS (1024 * 8)
#define NSTRM (1 << (WTAG_STRM))

void test_core_n_1(hls::stream<float> data_istrms[NSTRM],
                   hls::stream<bool> e_data_istrms[NSTRM],
                   hls::stream<ap_uint<WTAG_STRM> >& tag_istrm,
                   hls::stream<bool>& e_tag_istrm,
                   hls::stream<float>& data_ostrm,
                   hls::stream<bool>& e_data_ostrm) {
    xf::common::utils_hw::streamNToOne<float, WTAG_STRM>(data_istrms, e_data_istrms, tag_istrm, e_tag_istrm, data_ostrm,
                                                         e_data_ostrm, xf::common::utils_hw::TagSelectT());
}

int test_n_1() {
    hls::stream<float> data_istrms[NSTRM];
    hls::stream<bool> e_data_istrms[NSTRM];
    hls::stream<ap_uint<WTAG_STRM> > tag_istrm;
    hls::stream<bool> e_tag_istrm;
    hls::stream<float> data_ostrm;
    hls::stream<bool> e_data_ostrm;

    for (int j = 0; j < NS; ++j) {
        int id = j % NSTRM;
        float d = j * j + 0.5;
        data_istrms[id].write(d);
        e_data_istrms[id].write(false);
        tag_istrm.write(id);
        e_tag_istrm.write(false);
#if !defined(__SYNTHESIS__) && XF_UTIL_STRM_1NRR_DEBUG == 1
        std::cout << "id=" << id << " "
                  << "data= " << d << std::endl;
#endif
    }

    for (int i = 0; i < NSTRM; ++i) e_data_istrms[i].write(true);

    e_tag_istrm.write(true);

    test_core_n_1(data_istrms, e_data_istrms, tag_istrm, e_tag_istrm, data_ostrm, e_data_ostrm);
    int nerror = 0;
    unsigned int n = 0;
    while (!e_data_ostrm.read()) {
        float data = data_ostrm.read();
        float gld = n * n + 0.5;
        if (data != gld) {
            nerror = 1;
            std::cout << " erro :  tag=" << n % NSTRM << " "
                      << "data= " << data << " gld= " << gld << std::endl;
        } else {
#if !defined(__SYNTHESIS__) && XF_UTIL_STRM_1NRR_DEBUG == 1
            std::cout << "tag=" << n % NSTRM << " "
                      << "data= " << data << std::endl;
#endif
        }
        n++;
    } // while
    std::cout << "totally input NS = " << NS << " float data through NSTRM = " << NSTRM << "streams" << std::endl;
    std::cout << "totally output n = " << n << " float data from the output stream" << std::endl;

    std::cout << "the length of the collected stream: " << n << std::endl;
    if (n != NS) nerror = 1;

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
