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
#include "xf_utils_hw/stream_shuffle.hpp"
#include <iostream>
#include <stdlib.h>

#define NUM_INPUT 16
#define NUM_OUTPUT 20

#define STRM_LEN 10

#define DATA_TYPE int

int nerror;

void gld(ap_int<8> gld_cfg[NUM_OUTPUT],
         DATA_TYPE gld_input[NUM_INPUT][STRM_LEN],
         DATA_TYPE gld_output[NUM_OUTPUT][STRM_LEN]) {
    for (int i = 0; i < NUM_OUTPUT; i++) {
        for (int j = 0; j < STRM_LEN; j++) {
            if (gld_cfg[i] >= 0) gld_output[i][j] = gld_input[gld_cfg[i]][j];
        }
    }
}

void top(hls::stream<ap_uint<8 * NUM_OUTPUT> >& order_cfg,

         hls::stream<DATA_TYPE> istrms[NUM_INPUT],
         hls::stream<bool>& e_istrm,

         hls::stream<DATA_TYPE> ostrms[NUM_OUTPUT],
         hls::stream<bool>& e_ostrm) {
    xf::common::utils_hw::streamShuffle<NUM_INPUT, NUM_OUTPUT>(order_cfg, istrms, e_istrm, ostrms, e_ostrm);
}

int main() {
    nerror = 0;
    ap_uint<8 * NUM_OUTPUT> orders = 0;
    hls::stream<ap_uint<8 * NUM_OUTPUT> > order_cfg;

    hls::stream<DATA_TYPE> istrms[NUM_INPUT];
    hls::stream<bool> e_istrm;

    hls::stream<DATA_TYPE> ostrms[NUM_OUTPUT];
    hls::stream<bool> e_ostrm;

    ap_int<8> gld_cfg[NUM_OUTPUT];
    DATA_TYPE gld_input[NUM_INPUT][STRM_LEN];
    DATA_TYPE gld_output[NUM_OUTPUT][STRM_LEN];

    int i;

    for (i = 0; i < NUM_OUTPUT; i++) {
        for (int j = 0; j < STRM_LEN; j++) {
            gld_output[i][j] = 0;
        }
    }

    for (i = 0; i < NUM_INPUT / 2; i++) {
        orders.range(8 * i + 7, 8 * i) = ap_int<8>(i);
        gld_cfg[i] = i;
    }

    for (; i < NUM_OUTPUT; i++) {
        orders.range(8 * i + 7, 8 * i) = ap_int<8>(-10);
        gld_cfg[i] = -10;
    }

    order_cfg.write(orders);

    for (int j = 0; j < STRM_LEN; j++) {
        for (int i = 0; i < NUM_INPUT; i++) {
            istrms[i].write(i);
            gld_input[i][j] = i;
        }
        e_istrm.write(false);
    }
    e_istrm.write(true);

    gld(gld_cfg, gld_input, gld_output);

    top(order_cfg, istrms, e_istrm, ostrms, e_ostrm);

    DATA_TYPE test_data;
    bool rd_success = 0;
    bool e;

    e_ostrm.read();
    for (int i = 0; i < NUM_OUTPUT; i++) {
        for (int j = 0; j < STRM_LEN; j++) {
            rd_success = ostrms[i].read_nb(test_data);
            if (!rd_success) {
                nerror++;
                std::cout << "error: data loss" << std::endl;
            } else if (test_data != gld_output[i][j]) {
                nerror++;
                std::cout << "error: test data = " << test_data << " gold data = " << gld_output[i][j] << std::endl;
            }
        }
    }

    for (int j = 0; j < STRM_LEN; j++) {
        rd_success = e_ostrm.read_nb(e);
        if (!rd_success) {
            nerror++;
            std::cout << "error: end flag loss" << std::endl;
        }
    }

    if (nerror) {
        std::cout << "\nFAIL: " << nerror;
    } else {
        std::cout << "\nPASS: no error found.\n";
    }

    return nerror;
}
