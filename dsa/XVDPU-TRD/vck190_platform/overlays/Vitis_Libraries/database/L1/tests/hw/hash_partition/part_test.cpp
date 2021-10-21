
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

#include <iostream>
#include <algorithm>

#include "hls_stream.h"
#include "part_dut.hpp"
#include "xf_database/hash_lookup3.hpp"

#define ROW_NUM 10000
#define BIT_NUM 3
#define POWER_OF_PART_NUM (1 << BIT_NUM)

void generate_data(const int nrow,
                   hls::stream<ap_uint<64> > key_strms[CH_NM],
                   hls::stream<ap_uint<192> > pld_strms[CH_NM],
                   hls::stream<bool> e_strms[CH_NM]) {
    // ensure same result
    srand(13);

    for (int i = 0; i < nrow; i++) {
        ap_uint<64> key = i * 10 + (i % 10);
        ap_uint<192> pld = rand() % (1 << 30);

        int ch = i % CH_NM;

        key_strms[ch].write(key);
        pld_strms[ch].write(pld);
        e_strms[ch].write(false);
    }
    for (int ch = 0; ch < CH_NM; ++ch) {
        e_strms[ch].write(true);
    }

    std::cout << "INFO: input table " << nrow << " rows." << std::endl;
}

void generate_golden(hls::stream<ap_uint<64> > i_key_strms[CH_NM],
                     hls::stream<ap_uint<192> > i_pld_strms[CH_NM],
                     hls::stream<bool> i_e_strms[CH_NM],
                     ap_uint<256>* buf_o[POWER_OF_PART_NUM]) {
    hls::stream<ap_uint<64> > key_in;
    hls::stream<ap_uint<64> > key_out;
    int hash_cnt[POWER_OF_PART_NUM] = {0};
    for (int ch = 0; ch < CH_NM; ++ch) {
        while (!i_e_strms[ch].read()) {
            ap_uint<64> k = i_key_strms[ch].read();
            ap_uint<192> p = i_pld_strms[ch].read();
            key_in.write(k);
            xf::database::hashLookup3<64>(13, key_in, key_out);
            ap_uint<64> k_g = key_out.read();
            ap_uint<BIT_NUM> index = k_g(BIT_NUM - 1, 0);
            buf_o[index][hash_cnt[index]++] = (k, p);
        }
    }
}

int main(int argc, const char* argv[]) {
    int nerror = 0;
    const int nrow = ROW_NUM;
    const int bit_num = BIT_NUM;
    const int size_in_byte = ROW_NUM * 256 / 8;

    // allocate sw & hw result buffer
    ap_uint<256>* buf_sw[POWER_OF_PART_NUM];
    ap_uint<256>* buf_hw[POWER_OF_PART_NUM];
    for (int i = 0; i < POWER_OF_PART_NUM; i++) {
        buf_sw[i] = (ap_uint<256>*)malloc(size_in_byte);
        memset(buf_sw[i], 0, size_in_byte);
        if (!buf_sw[i]) {
            std::cerr << "ERROR: cannot allocate sw buffer." << std::endl;
            return 1;
        }
        buf_hw[i] = (ap_uint<256>*)malloc(size_in_byte);
        memset(buf_hw[i], 0, size_in_byte);
        if (!buf_hw[i]) {
            std::cerr << "ERROR: cannot allocate hw buffer." << std::endl;
            return 1;
        }
    }

    // define intermediate stream channel
    hls::stream<ap_uint<64> > k_strm_sw[CH_NM];
    hls::stream<ap_uint<192> > p_strm_sw[CH_NM];
    hls::stream<bool> e_strm_sw[CH_NM];
    hls::stream<ap_uint<64> > k_strm_hw[CH_NM];
    hls::stream<ap_uint<192> > p_strm_hw[CH_NM];
    hls::stream<bool> e_strm_hw[CH_NM];

    hls::stream<int> bit_num_strm;
    hls::stream<ap_uint<16> > bkpu_strm;
    hls::stream<ap_uint<10> > nm_strm;
    hls::stream<ap_uint<32> > kpld_strm[COL_NM];

    // golden result calculation
    generate_data(nrow, k_strm_sw, p_strm_sw, e_strm_sw);
    generate_golden(k_strm_sw, p_strm_sw, e_strm_sw, buf_sw);

    // hardware acclerate
    generate_data(nrow, k_strm_hw, p_strm_hw, e_strm_hw);

    bit_num_strm.write(bit_num);
    part_dut(bit_num_strm, k_strm_hw, p_strm_hw, e_strm_hw, bkpu_strm, nm_strm, kpld_strm);

    // collect hw result into buffer
    int hash_cnt[POWER_OF_PART_NUM] = {0};
    ap_uint<10> nm = nm_strm.read();
    while (nm) {
        ap_uint<16> bkpu = bkpu_strm.read();
        for (int n = 0; n < nm; n++) {
            for (int c = 0; c < COL_NM; c++) {
                ap_uint<32> d = kpld_strm[c].read();
                if (c == 0)
                    buf_hw[bkpu(9, 0)][hash_cnt[bkpu(9, 0)]].range(223, 192) = d;
                else if (c == 1)
                    buf_hw[bkpu(9, 0)][hash_cnt[bkpu(9, 0)]].range(31, 0) = d;
                else if (c == 2)
                    buf_hw[bkpu(9, 0)][hash_cnt[bkpu(9, 0)]].range(63, 32) = d;
                else if (c == 3)
                    buf_hw[bkpu(9, 0)][hash_cnt[bkpu(9, 0)]].range(95, 64) = d;
                else if (c == 4)
                    buf_hw[bkpu(9, 0)][hash_cnt[bkpu(9, 0)]].range(127, 96) = d;
                else if (c == 5)
                    buf_hw[bkpu(9, 0)][hash_cnt[bkpu(9, 0)]].range(159, 128) = d;
                else if (c == 6)
                    buf_hw[bkpu(9, 0)][hash_cnt[bkpu(9, 0)]].range(191, 160) = d;
                else
                    buf_hw[bkpu(9, 0)][hash_cnt[bkpu(9, 0)]].range(255, 224) = 0;
            }
            hash_cnt[bkpu(9, 0)]++;
        }
        nm = nm_strm.read();
    }

    // check sw VS hw results
    for (int i = 0; i < POWER_OF_PART_NUM; i++) {
        for (int j = 0; j < nrow; j++) {
            if (buf_hw[i][j] != buf_sw[i][j]) nerror++;
        }
    }

    // free allocated memory
    for (int i = 0; i < POWER_OF_PART_NUM; i++) {
        if (buf_sw[i]) free(buf_sw[i]);
        if (buf_hw[i]) free(buf_hw[i]);
    }

    return nerror;
}
