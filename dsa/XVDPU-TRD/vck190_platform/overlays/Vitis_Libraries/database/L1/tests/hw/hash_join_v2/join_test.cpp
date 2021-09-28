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

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <vector>
#include <unordered_map>

#include <iostream>
#include <algorithm>

#include "hls_stream.h"
#include "join_dut.hpp"

#define PU_NM (1 << HJ_HW_P)

//-------------------------generate test data----------------------------

void generate_data(const int num,
                   hls::stream<ap_uint<WKEY> > key_strms[HJ_CH_NM],
                   hls::stream<ap_uint<WPAY> > pld_strms[HJ_CH_NM],
                   hls::stream<bool> e_strms[HJ_CH_NM]) {
    // generate s, twice
    for (int j = 0; j < 2; ++j) {
        // ensure same result
        srand(13);

        for (int i = 0; i < num; i++) {
            ap_uint<WKEY> key = i * 10 + (i % 10);
            ap_uint<WPAY> pld = rand() % 100;

            int ch = i % HJ_CH_NM;

            key_strms[ch].write(key);
            pld_strms[ch].write(pld);
            e_strms[ch].write(false);
        }
        for (int ch = 0; ch < HJ_CH_NM; ++ch) {
            e_strms[ch].write(true);
        }
    }

    // generate t
    {
        // ensure same result
        srand(17);

        for (int i = 0; i < num * 10; ++i) {
            ap_uint<WKEY> key = i;
            ap_uint<WPAY> pld = rand() % 100;

            int ch = i % HJ_CH_NM;

            key_strms[ch].write(key);
            pld_strms[ch].write(pld);
            e_strms[ch].write(false);
        }
        for (int ch = 0; ch < HJ_CH_NM; ++ch) {
            e_strms[ch].write(true);
        }
    }

    std::cout << "INFO: inner table " << num << " rows, outer table " << (num * 20) << " rows." << std::endl;
}

//-------------------------generate golden data-----------------------------

ap_uint<WPAY * 2> get_golden_sum(hls::stream<ap_uint<WKEY> > key_strms[HJ_CH_NM],
                                 hls::stream<ap_uint<WPAY> > pld_strms[HJ_CH_NM],
                                 hls::stream<bool> e_strms[HJ_CH_NM]) {
    ap_uint<WPAY* 2> sum = 0;
    int cnt = 0;

    std::unordered_multimap<uint32_t, uint32_t> ht1;

    // read s twice
    {
        // drop first round
        for (int ch = 0; ch < HJ_CH_NM; ++ch) {
            while (!e_strms[ch].read()) {
                key_strms[ch].read().to_uint();
                pld_strms[ch].read().to_uint();
            }
        }
        // use second round
        for (int ch = 0; ch < HJ_CH_NM; ++ch) {
            while (!e_strms[ch].read()) {
                uint32_t k = key_strms[ch].read().to_uint();
                uint32_t p = pld_strms[ch].read().to_uint();
                // insert into hash table
                ht1.insert(std::make_pair(k, p));
            }
        }
    }
    // read t once
    for (int ch = 0; ch < HJ_CH_NM; ++ch) {
        while (!e_strms[ch].read()) {
            uint32_t k = key_strms[ch].read().to_uint();
            uint32_t p = pld_strms[ch].read().to_uint();
            // check hash table
            auto its = ht1.equal_range(k);
            for (auto it = its.first; it != its.second; ++it) {
                uint32_t sp = it->second;
                sum += (p * sp);
                ++cnt;
            }
        }
    }

    std::cout << "INFO: CPU ref matched " << cnt << " rows, sum = " << sum << std::endl;
    return sum;
}

ap_uint<WPAY * 2> get_sum(hls::stream<ap_uint<WPAY * 2> >& pld_strm, hls::stream<bool>& e_strm) {
    ap_uint<WPAY* 2> sum = 0;
    int cnt = 0;

    while (!e_strm.read()) {
        ap_uint<WPAY* 2> p = pld_strm.read();
        ap_uint<WPAY> bp = p.range(WPAY - 1, 0);
        ap_uint<WPAY> sp = p.range(WPAY * 2 - 1, WPAY);
        sum += (bp * sp);
        ++cnt;
    }

    std::cout << "INFO: DUT generated " << cnt << " rows, sum = " << sum << std::endl;
    return sum;
}

template <int W>
int the_end(int nerror, ap_uint<W>* pu_ht[PU_NM]) {
    for (int i = 0; i < PU_NM; i++) {
        if (pu_ht[i]) {
            free(pu_ht[i]);
        }
    }

    if (nerror) {
        std::cout << "\nFAIL: " << nerror << " errors found.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }

    return nerror;
}

int main() {
    enum { TEST_SIZE = 256 };
    int nerror;

    // allocate internal buffer
    ap_uint<WKEY + WPAY>* pu_ht[PU_NM];

    for (int i = 0; i < PU_NM; i++) {
        pu_ht[i] = (ap_uint<WKEY + WPAY>*)malloc(((WKEY + WPAY) / 8) * BUFF_DEPTH);
        if (!pu_ht[i]) {
            std::cerr << "ERROR: cannot allocate buffer." << std::endl;
            return the_end(-1, pu_ht);
        }
    }

    ap_uint<WPAY * 2> result;
    {
        hls::stream<ap_uint<WKEY> > key_strms[HJ_CH_NM];
        hls::stream<ap_uint<WPAY> > pld_strms[HJ_CH_NM];
        hls::stream<bool> e_strms[HJ_CH_NM];
        generate_data(TEST_SIZE, key_strms, pld_strms, e_strms);

        hls::stream<ap_uint<WPAY * 2> > j_strm;
        hls::stream<bool> j_e_strm;
        join_dut(
            // input
            key_strms, pld_strms, e_strms,
            // output
            j_strm, j_e_strm,
            // tmps
            pu_ht[0], pu_ht[1], pu_ht[2], pu_ht[3], pu_ht[4], pu_ht[5], pu_ht[6], pu_ht[7]);

        result = get_sum(j_strm, j_e_strm);
    }

    ap_uint<WPAY * 2> gold;
    {
        hls::stream<ap_uint<WKEY> > key_strms[HJ_CH_NM];
        hls::stream<ap_uint<WPAY> > pld_strms[HJ_CH_NM];
        hls::stream<bool> e_strms[HJ_CH_NM];
        generate_data(TEST_SIZE, key_strms, pld_strms, e_strms);

        gold = get_golden_sum(key_strms, pld_strms, e_strms);
    }

    // check
    nerror = (result != gold);
    return the_end(nerror, pu_ht);
}
