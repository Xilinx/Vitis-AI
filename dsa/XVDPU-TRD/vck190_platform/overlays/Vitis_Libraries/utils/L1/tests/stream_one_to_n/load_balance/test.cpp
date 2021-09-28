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

#include "xf_utils_hw/stream_one_to_n/load_balance.hpp"

#define WIN_STRM 512
#define WOUT_STRM 16
#define NS (1024 * 2 * 2)
#define NSTRM 32

// no pause
void consume_one_s0(hls::stream<ap_uint<WOUT_STRM> >& c_istrm,
                    hls::stream<bool>& e_c_istrm,
                    hls::stream<ap_uint<WOUT_STRM> >& c_ostrm,
                    hls::stream<bool>& e_c_ostrm) {
    bool last = e_c_istrm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        bool em = c_istrm.empty();
        if (false == em) {
            ap_uint<WOUT_STRM> d = c_istrm.read();
            c_ostrm.write(d);
            e_c_ostrm.write(false);
            last = e_c_istrm.read();
        }
    } // while

    e_c_ostrm.write(true);
}
// read _SwL times and wait same circles
void consume_one_s2(bool f_sw,
                    int sw_l,
                    hls::stream<ap_uint<WOUT_STRM> >& c_istrm,
                    hls::stream<bool>& e_c_istrm,
                    hls::stream<ap_uint<WOUT_STRM> >& c_ostrm,
                    hls::stream<bool>& e_c_ostrm) {
    // const int sw_l= _SwL;
    int c = 0;
    bool sw = f_sw;
    bool last = e_c_istrm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        bool em = c_istrm.empty();
        if (false == sw && false == em) {
            ap_uint<WOUT_STRM> d = c_istrm.read();
            c_ostrm.write(d);
            e_c_ostrm.write(false);
            last = e_c_istrm.read();
        }
        if (++c == sw_l) {
            sw = !sw;
            c = 0;
        }
    } // while
    e_c_ostrm.write(true);
}
// case0
void consume_0(hls::stream<ap_uint<WOUT_STRM> > c_istrms[NSTRM],
               hls::stream<bool> e_c_istrms[NSTRM],
               hls::stream<ap_uint<WOUT_STRM> > c_ostrms[NSTRM],
               hls::stream<bool> e_c_ostrms[NSTRM]) {
#pragma HLS dataflow
    for (int i = 0; i < NSTRM; ++i) {
#pragma HLS unroll
        consume_one_s0(c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
    }
}

// case 1
void consume_1(hls::stream<ap_uint<WOUT_STRM> > c_istrms[NSTRM],
               hls::stream<bool> e_c_istrms[NSTRM],
               hls::stream<ap_uint<WOUT_STRM> > c_ostrms[NSTRM],
               hls::stream<bool> e_c_ostrms[NSTRM]) {
#pragma HLS dataflow
    for (int i = 0; i < NSTRM; ++i) {
#pragma HLS unroll
        if (i < 4)
            consume_one_s2(i % 4 == 0, 2, c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);

        else if (i < 8)
            consume_one_s2(i % 4 == 0, 4, c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
        else
            consume_one_s2(i % 4 == 0, 8, c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
    }
}
// case2
void consume_2(hls::stream<ap_uint<WOUT_STRM> > c_istrms[NSTRM],
               hls::stream<bool> e_c_istrms[NSTRM],
               hls::stream<ap_uint<WOUT_STRM> > c_ostrms[NSTRM],
               hls::stream<bool> e_c_ostrms[NSTRM]) {
#pragma HLS dataflow
    for (int i = 0; i < 2; ++i) {
#pragma HLS unroll
        consume_one_s0(c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
    }
    for (int i = 2; i < NSTRM; ++i) {
#pragma HLS unroll
        consume_one_s2(i % 4 == 0, i + 1, c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
    }
}

void test_core_1_n(hls::stream<ap_uint<WIN_STRM> >& data_istrm,
                   hls::stream<bool>& e_istrm,
                   hls::stream<ap_uint<WOUT_STRM> > data_ostrms[NSTRM],
                   hls::stream<bool> e_data_ostrms[NSTRM]) {
#pragma HLS dataflow
    // here the depth  is an influence factor of outputs' order
    hls::stream<ap_uint<WOUT_STRM> > c_strms[NSTRM];
#pragma HLS stream variable = c_strms depth = 4 // 1024
    hls::stream<bool> e_c_strms[NSTRM];
#pragma HLS stream variable = e_c_strms depth = 4 // 1024

    xf::common::utils_hw::streamOneToN<WIN_STRM, WOUT_STRM, NSTRM>(data_istrm, e_istrm, c_strms, e_c_strms,
                                                                   xf::common::utils_hw::LoadBalanceT());

    consume_1(c_strms, e_c_strms, data_ostrms, e_data_ostrms);
}

int test_1_n() {
    hls::stream<ap_uint<WIN_STRM> > data_istrm;
    hls::stream<bool> e_istrm;
    hls::stream<ap_uint<WOUT_STRM> > data_ostrms[NSTRM];
    hls::stream<bool> e_data_ostrms[NSTRM];
    int td[NS] = {0};
    const int buf_width = xf::common::utils_hw::LCM<WIN_STRM, WOUT_STRM>::value;
    const int num_in = buf_width / WIN_STRM;
    const int num_out = buf_width / WOUT_STRM;
    const int sw_l = 8;
    bool sw = false;

    ap_uint<buf_width> buff;
    std::cout << std::dec << "WIN_STRM  = " << WIN_STRM << std::endl;
    std::cout << std::dec << "WOUT_STRM = " << WOUT_STRM << std::endl;
    std::cout << std::dec << "NSTRM     = " << NSTRM << std::endl;
    std::cout << std::dec << "NS        = " << NS << std::endl;

    int t = WIN_STRM / WOUT_STRM;
    for (int d = 0; d < NS; d += t) {
        int i = d % (NSTRM);
        ap_uint<WIN_STRM> bd = 0;
        ap_uint<WOUT_STRM> sd = d;
        for (int j = 0; j < t; ++j) {
            bd.range((j + 1) * WOUT_STRM - 1, j * WOUT_STRM) = sd;
            sd++;
        }
        data_istrm.write(bd);
        e_istrm.write(false);
    }

    e_istrm.write(true);

    test_core_1_n(data_istrm, e_istrm, data_ostrms, e_data_ostrms);

    int nerror = 0;
    int count = 0;
    int tempa = WIN_STRM * NS / WOUT_STRM;
    int tempb = tempa * WOUT_STRM;
    int comp_count = tempb / WIN_STRM; // comp_count<=NS, last data may not equal NS-1
    int len[NSTRM] = {0};
    int id = 0;
    std::cout << std::dec << "comp_count=" << comp_count << std::endl;
    count = 0;
    for (int kk = 0; kk < NSTRM; ++kk) {
        std::cout << "stream " << kk << ":";
        while (!e_data_ostrms[kk].read()) {
            ap_uint<WOUT_STRM> d = data_ostrms[kk].read();
            td[d]++;
            len[kk]++;
            count++;
            std::cout << " " << d;
        }
        std::cout << std::endl;
    }
    for (int i = 0; i < NS; ++i)
        if (td[i] != 1) {
            nerror += 1;
            std::cout << "error: td[" << i << "] = " << td[i] << std::endl;
        }

    nerror += (count != NS);
    std::cout << "count = " << count << ", NS = " << NS << std::endl;
    for (int id = 0; id < NSTRM; id++) {
        std::cout << std::dec << "the length of stream_" << id << ": " << len[id] << std::endl;
    }
    if (nerror) {
        std::cout << "\nFAIL: " << nerror << " results are wrong.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return nerror;
}

int main() {
    return test_1_n();
}
