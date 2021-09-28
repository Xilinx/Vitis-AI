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

#include "xf_utils_hw/stream_one_to_n/round_robin.hpp"

#define WIN_STRM 512
#define WOUT_STRM 16
#define NS 1024 * 8
#define NSTRM 8

void test_core_1_n(hls::stream<ap_uint<WIN_STRM> >& data_istrm,
                   hls::stream<bool>& e_istrm,
                   hls::stream<ap_uint<WOUT_STRM> > data_ostrms[NSTRM],
                   hls::stream<bool> e_data_ostrms[NSTRM]) {
    xf::common::utils_hw::streamOneToN<WIN_STRM, WOUT_STRM, NSTRM>(data_istrm, e_istrm, data_ostrms, e_data_ostrms,
                                                                   xf::common::utils_hw::RoundRobinT());
}

int test_1_n() {
    hls::stream<ap_uint<WIN_STRM> > data_istrm;
    hls::stream<bool> e_istrm;
    hls::stream<ap_uint<WOUT_STRM> > data_ostrms[NSTRM];
    hls::stream<bool> e_data_ostrms[NSTRM];

    const int buf_width = xf::common::utils_hw::LCM<WIN_STRM, WOUT_STRM>::value;
    const int num_in = buf_width / WIN_STRM;
    const int num_out = buf_width / WOUT_STRM;

    ap_uint<buf_width> buff;
    std::cout << std::dec << "WIN_STRM  = " << WIN_STRM << std::endl;
    std::cout << std::dec << "WOUT_STRM = " << WOUT_STRM << std::endl;
    std::cout << std::dec << "NSTRM     = " << NSTRM << std::endl;
    std::cout << std::dec << "NS        = " << NS << std::endl;
    for (int d = 0; d < NS; ++d) {
        ap_uint<WIN_STRM> id = d % NSTRM;
        data_istrm.write(d);
        e_istrm.write(false);
#if !defined(__SYNTHESIS__) && XF_UTIL_STRM_1NRR_DEBUG == 1
        std::cout << "id = " << id << "  "
                  << "d=" << d << std::endl;
#endif
    }
    e_istrm.write(true);

    test_core_1_n(data_istrm, e_istrm, data_ostrms, e_data_ostrms);

    int nerror = 0;
    ap_uint<WIN_STRM> last_data = 0;
    ap_uint<NSTRM> ends = 0;
    bool first_data = true;
    int k = 0;
    bool exit_flag = false;
    int count = 0;
    int tempa = WIN_STRM * NS / WOUT_STRM;
    int tempb = tempa * WOUT_STRM;
    int comp_count = tempb / WIN_STRM; // comp_count<=NS, last data may not equal NS-1
    int len[NSTRM] = {0};
    int id = 0;
    std::cout << std::dec << "comp_count=" << comp_count << std::endl;
    while (exit_flag == false) {
        for (int kk = 0; kk < NSTRM; ++kk) {
            // read data from n streams
            if (ends[kk] == 0) {
                if (!e_data_ostrms[kk].read()) {
                    ap_uint<WOUT_STRM> d = data_ostrms[kk].read();
                    buff.range((k + 1) * WOUT_STRM - 1, k * WOUT_STRM) = d; // data_ostrms[kk].read();
                    len[kk]++;
                    k++;
                } else {
                    ends[kk] = 1;
                }
            }
        } // for k
        // check the data
        if (k == num_out) {
            k = 0;
#if !defined(__SYNTHESIS__) && XF_UTIL_STRM_1NRR_DEBUG == 1
            std::cout << std::hex << "buff =" << buff << std::endl;
#endif
            for (int c = 0; c < num_in; ++c) {
                ap_uint<WIN_STRM> d = buff.range((c + 1) * WIN_STRM - 1, c * WIN_STRM);
                count++;
                if (first_data) {
                    first_data = false;
                } else {
                    if (count <= comp_count && d != last_data + 1) {
                        nerror = 1;
                        std::cout << "erro: last_data=" << last_data << ", "
                                  << "current data=" << d << std::endl;
                    }
                    last_data = d;
                }
            } // for c
        }
        bool f = 1;
        for (int s = 0; s < NSTRM; ++s) f &= ends[s];
        exit_flag = f;
    } // while
    for (int c = 0; c < num_in; ++c) {
        if ((c + 1) * WIN_STRM < k * WOUT_STRM) {
            ap_uint<WIN_STRM> d = buff.range((c + 1) * WIN_STRM - 1, c * WIN_STRM);
            count++;
            if (first_data) {
                first_data = false;
                last_data = d;
            } else {
                if (count <= comp_count && d != last_data + 1) {
                    nerror = 1;
                    std::cout << "erro: last_data=" << last_data << ", "
                              << "current data=" << d << std::endl;
                }
                last_data = d;
            }
        } // if
    }     // for c
    for (int id = 0; id < NSTRM; id++)
        std::cout << std::dec << "the length of stream_" << id << ": " << len[id] << std::endl;
    if (nerror) {
        std::cout << "\nFAIL: " << nerror << "the order is wrong.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return nerror;
}

int main() {
    return test_1_n();
}
