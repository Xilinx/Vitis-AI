/*
 * Copyright 2020 Xilinx, Inc.
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

// XXX include header being test first, to ensure we detech missing dependency.
#include "xf_database/filter_config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <vector>

#include <iostream>
#include <algorithm>

#include "hls_stream.h"
#include "filter_dut.hpp"

//-------------------------generate test data----------------------------

void generate_data(const int num,
                   hls::stream<ap_uint<WKEY> > key_strms[4],
                   hls::stream<ap_uint<WPAY> >& pld_strm,
                   hls::stream<bool>& e_strm) {
    // ensure same result
    srand(17);

    for (int i = 0; i < num; i++) {
        key_strms[0].write((rand() % 30));
        key_strms[1].write((rand() % 30));
        key_strms[2].write((rand() % 10));
        key_strms[3].write((rand() % 10));

        pld_strm.write((rand() % 100));
        e_strm.write(false);
    }
    e_strm.write(true);
}

//-------------------------generate golden data-----------------------------

ap_uint<WPAY * 2> get_golden_sum(hls::stream<ap_uint<WKEY> > key_strms[4],
                                 hls::stream<ap_uint<WPAY> >& pld_strm,
                                 hls::stream<bool>& e_strm) {
    ap_uint<WPAY* 2> sum = 0;
    int cnt = 0;

    while (!e_strm.read()) {
        uint32_t k0 = key_strms[0].read().to_uint();
        uint32_t k1 = key_strms[1].read().to_uint();
        uint32_t k2 = key_strms[2].read().to_uint();
        uint32_t k3 = key_strms[3].read().to_uint();
        uint32_t p = pld_strm.read().to_uint();
        if ((k0 < 10 && k1 < 10) || (k2 < k3)) {
            sum += p;
            ++cnt;
        }
    }

    std::cout << "INFO: CPU ref matched " << cnt << " rows, sum = " << sum << std::endl;
    return sum;
}

//-------------------------sum up DUT data-----------------------------

ap_uint<WPAY * 2> get_sum(hls::stream<ap_uint<WPAY> >& pld_strm, hls::stream<bool>& e_strm) {
    ap_uint<WPAY* 2> sum = 0;
    int cnt = 0;

    while (!e_strm.read()) {
        ap_uint<WPAY> p = pld_strm.read();
        sum += p;
        ++cnt;
    }

    std::cout << "INFO: DUT generated " << cnt << " rows, sum = " << sum << std::endl;
    return sum;
}

// ----------------------------------------------------------------------------

int main() {
    enum { TEST_SIZE = (1 << 10) }; // 1k cases.
    int nerror;

    ap_uint<WPAY * 2> result;
    {
        using namespace xf::database;

        hls::stream<typename DynamicFilterInfo<4, WKEY>::cfg_type> filter_cfg_strm;

        // XXX this must match the get_golden_sum function.
        auto p = FilterConfig<WKEY>("(a < 10 && b < 10 ) || (c < d)").getConfigBits();
        for (unsigned i = 0; i < DynamicFilterInfo<4, WKEY>::dwords_num; ++i) {
            filter_cfg_strm.write(p[i]);
        }

        hls::stream<ap_uint<WKEY> > key_strms[4];
        hls::stream<ap_uint<WPAY> > pld_strm;
        hls::stream<bool> e_strm;
        generate_data(TEST_SIZE, key_strms, pld_strm, e_strm);

        hls::stream<ap_uint<WPAY> > f_strm;
        hls::stream<bool> e_f_strm;
        filter_dut(filter_cfg_strm,
                   // input
                   key_strms, pld_strm, e_strm,
                   // output
                   f_strm, e_f_strm);

        result = get_sum(f_strm, e_f_strm);
    }

    ap_uint<WPAY * 2> gold;
    {
        hls::stream<ap_uint<WKEY> > key_strms[4];
        hls::stream<ap_uint<WPAY> > pld_strm;
        hls::stream<bool> e_strm;
        generate_data(TEST_SIZE, key_strms, pld_strm, e_strm);

        gold = get_golden_sum(key_strms, pld_strm, e_strm);
    }

    // check
    nerror = (result != gold);

    if (nerror) {
        std::cout << "FAIL: " << nerror << " errors found.\n";
    } else {
        std::cout << "PASS: no error found.\n";
    }

    return nerror;
}
