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
#include "xf_utils_hw/stream_n_to_one/load_balance.hpp"

#define NS (1024 * 2 * 4)
#define NSTRM 32

// no pause
void produce_one_s0(hls::stream<float>& c_istrm,
                    hls::stream<bool>& e_c_istrm,
                    hls::stream<float>& c_ostrm,
                    hls::stream<bool>& e_c_ostrm) {
    bool last = e_c_istrm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        // if ( false == em)
        {
            float d = c_istrm.read();
            c_ostrm.write(d);
            e_c_ostrm.write(false);
            last = e_c_istrm.read();
        }
    } // while

    e_c_ostrm.write(true);
}
// read sw_l cycles and wait same cycles
void produce_one_s1(bool f_sw,
                    int sw_l,
                    hls::stream<float>& c_istrm,
                    hls::stream<bool>& e_c_istrm,
                    hls::stream<float>& c_ostrm,
                    hls::stream<bool>& e_c_ostrm) {
    int c = 0;
    bool sw = f_sw;
    bool last = e_c_istrm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        // bool fl= e_c_ostrm.full();
        bool fl = c_ostrm.full();
        if (false == sw && fl == false) {
            float d = c_istrm.read();
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
void produce_0(hls::stream<float> c_istrms[NSTRM],
               hls::stream<bool> e_c_istrms[NSTRM],
               hls::stream<float> c_ostrms[NSTRM],
               hls::stream<bool> e_c_ostrms[NSTRM]) {
#pragma HLS dataflow
    for (int i = 0; i < NSTRM; ++i) {
#pragma HLS unroll
        produce_one_s0(c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
    }
}

// case 1
void produce_1(hls::stream<float> c_istrms[NSTRM],
               hls::stream<bool> e_c_istrms[NSTRM],
               hls::stream<float> c_ostrms[NSTRM],
               hls::stream<bool> e_c_ostrms[NSTRM]) {
#pragma HLS dataflow
    for (int i = 0; i < NSTRM; ++i) {
#pragma HLS unroll
        if (i < 2)
            produce_one_s1(false, 2 * 2, c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);

        else if (i < 4)
            produce_one_s1(false, 4 * 16, c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
        else
            produce_one_s1(false, 8 * 16, c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
    }
}
// case2
void produce_2(hls::stream<float> c_istrms[NSTRM],
               hls::stream<bool> e_c_istrms[NSTRM],
               hls::stream<float> c_ostrms[NSTRM],
               hls::stream<bool> e_c_ostrms[NSTRM]) {
#pragma HLS dataflow
    for (int i = 0; i < 2; ++i) {
#pragma HLS unroll
        produce_one_s0(c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
    }
    for (int i = 2; i < NSTRM; ++i) {
#pragma HLS unroll
        produce_one_s1(false, i + 1, c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
    }
}

void test_core_n_1(hls::stream<float> data_istrms[NSTRM],
                   hls::stream<bool> e_data_istrms[NSTRM],
                   hls::stream<float>& data_ostrm,
                   hls::stream<bool>& e_ostrm) {
/*
xf::util::level1::strm_one_to_n<WOUT_STRM, WIN_STRM,NSTRM>(
                       data_istrm,  e_istrm,
                       data_ostrms, e_data_ostrms,
                       xf::util::load_balance_t());
*/
#pragma HLS dataflow
    // here the depth  is an influence factor of outputs' order
    hls::stream<float> c_strms[NSTRM];
#pragma HLS stream variable = c_strms depth = 8
    hls::stream<bool> e_c_strms[NSTRM];
#pragma HLS stream variable = e_c_strms depth = 8

    produce_2(data_istrms, e_data_istrms, c_strms, e_c_strms);
    xf::common::utils_hw::streamNToOne<float, NSTRM>(c_strms, e_c_strms, data_ostrm, e_ostrm,
                                                     xf::common::utils_hw::LoadBalanceT());
}

int test_n_1() {
    hls::stream<float> data_istrms[NSTRM];
    hls::stream<bool> e_istrms[NSTRM];
    hls::stream<float> data_ostrm;
    hls::stream<bool> e_ostrm;
    int td[NS] = {0};

    std::cout << std::dec << "NSTRM     = " << NSTRM << std::endl;
    std::cout << std::dec << "NS        = " << NS << std::endl;

    for (int d = 0; d < NS; ++d) {
        int i = d % (NSTRM);
        float data = d + 0.5;
        data_istrms[i].write(data);
        e_istrms[i].write(false);
    }
    for (int i = 0; i < NSTRM; ++i) e_istrms[i].write(true);

    test_core_n_1(data_istrms, e_istrms, data_ostrm, e_ostrm);

    int nerror = 0;
    int count = 0;
    int comp_count = NS;
    std::cout << std::dec << "comp_count=" << comp_count << std::endl;
    count = 0;
    while (!e_ostrm.read()) {
        float data = data_ostrm.read();
        int d = (int)(data - 0.4);
        if (d >= 0 && d < NS)
            td[d]++;
        else {
            std::cout << std::dec << "erro: "
                      << "count=" << count << "  data=" << d << std::endl;
            nerror = 1;
        }
        count++;
    }
    std::cout << std::dec << "--------- " << std::endl;
    for (int i = 0; i < NS; ++i) {
        if (td[i] != 1) {
            std::cout << std::dec << "erro: "
                      << "i=" << i << "  td=" << td[i] << std::endl;
            nerror = 1;
        }
    }
    // nerror= (count == NS) ? nerror:1;
    std::cout << "count= " << count << "  "
              << "NS= " << NS << std::endl;
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
