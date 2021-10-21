/*
 * (c) Copyright 2019-2021 Xilinx, Inc. All rights reserved.
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
 *
 */

/**
 * @file zlib_treegen_mm.cpp
 * @brief Source for treegen kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */
#include "zlib_treegen_mm.hpp"

typedef xf::compression::Frequency Frequency;
typedef xf::compression::Codeword Codeword;

void lcl_ddr2bram(Frequency* inFreq, uint32_t* lit_freq, uint32_t* dist_freq) {
    // copy input data from ddr to bram
    // copy literals
    int offset = 0;
    for (int i = 0; i < c_litCodeCount; ++i) {
        inFreq[i] = (Frequency)lit_freq[i];
    }
    offset += c_litCodeCount;
    // copy distances
    for (int i = 0; i < c_dstCodeCount; ++i) {
        inFreq[i + offset] = (Frequency)dist_freq[i];
    }
    offset += c_dstCodeCount;
    for (int i = 0; i < c_blnCodeCount; ++i) {
        inFreq[i + offset] = 0; // just initialize
    }
}

void lcl_bram2ddr(Codeword* outCodes,
                  uint16_t* maxCodes,
                  uint32_t* lit_code,
                  uint32_t* dist_code,
                  uint32_t* bl_code,
                  uint32_t* lit_blen,
                  uint32_t* dist_blen,
                  uint32_t* bl_blen,
                  uint32_t* max_codes) {
    // copy output data back to ddr from bram
    int offset = 0;
    // copy literal codes and blens
    for (int i = 0; i < c_litCodeCount; ++i) {
        lit_code[i] = (uint32_t)outCodes[i].codeword;
        lit_blen[i] = (uint32_t)outCodes[i].bitlength;
    }
    offset += c_litCodeCount;
    // copy distance codes and blens
    for (int i = 0; i < c_dstCodeCount; ++i) {
        dist_code[i] = (uint32_t)outCodes[offset + i].codeword;
        dist_blen[i] = (uint32_t)outCodes[offset + i].bitlength;
    }
    offset += c_dstCodeCount;
    // copy bit-length codes and blens
    for (int i = 0; i < c_blnCodeCount; ++i) {
        bl_code[i] = (uint32_t)outCodes[offset + i].codeword;
        bl_blen[i] = (uint32_t)outCodes[offset + i].bitlength;
    }
    // copy maxcodes
    for (int i = 0; i < 3; ++i) {
        max_codes[i] = (uint32_t)maxCodes[i];
    }
}

void treegenCore(uint32_t* lit_freq,
                 uint32_t* dist_freq,
                 uint32_t* lit_code,
                 uint32_t* dist_code,
                 uint32_t* bl_code,
                 uint32_t* lit_blen,
                 uint32_t* dist_blen,
                 uint32_t* bl_blen,
                 uint32_t* max_codes) {
    // internal buffers
    Frequency inFreq[c_litCodeCount + c_dstCodeCount + c_blnCodeCount];
    Codeword outCodes[c_litCodeCount + c_dstCodeCount + c_blnCodeCount];
    uint16_t maxCodes[3] = {0, 0, 0};

    // read literal and distance frequencies from DDR
    lcl_ddr2bram(inFreq, lit_freq, dist_freq);

    zlibTreegenCore(inFreq, outCodes, maxCodes);

    // write code and bit-length data to DDR
    lcl_bram2ddr(outCodes, maxCodes, lit_code, dist_code, bl_code, lit_blen, dist_blen, bl_blen, max_codes);
}

extern "C" {

void xilTreegenKernel(uint32_t* dyn_ltree_freq,
                      uint32_t* dyn_dtree_freq,
                      uint32_t* dyn_bltree_freq,
                      uint32_t* dyn_ltree_codes,
                      uint32_t* dyn_dtree_codes,
                      uint32_t* dyn_bltree_codes,
                      uint32_t* dyn_ltree_blen,
                      uint32_t* dyn_dtree_blen,
                      uint32_t* dyn_bltree_blen,
                      uint32_t* max_codes,
                      uint32_t block_size_in_kb,
                      uint32_t input_size,
                      uint32_t blocks_per_chunk) {
#pragma HLS INTERFACE m_axi port = dyn_ltree_freq offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = dyn_dtree_freq offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = dyn_bltree_freq offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = dyn_ltree_codes offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = dyn_dtree_codes offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = dyn_bltree_codes offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = dyn_ltree_blen offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = dyn_dtree_blen offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = dyn_bltree_blen offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = max_codes offset = slave bundle = gmem0

#pragma HLS INTERFACE s_axilite port = dyn_ltree_freq bundle = control
#pragma HLS INTERFACE s_axilite port = dyn_dtree_freq bundle = control
#pragma HLS INTERFACE s_axilite port = dyn_bltree_freq bundle = control
#pragma HLS INTERFACE s_axilite port = dyn_ltree_codes bundle = control
#pragma HLS INTERFACE s_axilite port = dyn_dtree_codes bundle = control
#pragma HLS INTERFACE s_axilite port = dyn_bltree_codes bundle = control
#pragma HLS INTERFACE s_axilite port = dyn_ltree_blen bundle = control
#pragma HLS INTERFACE s_axilite port = dyn_dtree_blen bundle = control
#pragma HLS INTERFACE s_axilite port = dyn_bltree_blen bundle = control
#pragma HLS INTERFACE s_axilite port = max_codes bundle = control
#pragma HLS INTERFACE s_axilite port = block_size_in_kb bundle = control
#pragma HLS INTERFACE s_axilite port = input_size bundle = control
#pragma HLS INTERFACE s_axilite port = blocks_per_chunk bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    const uint16_t c_ltree_size = 1024;
    const uint16_t c_dtree_size = 64;
    const uint16_t c_bltree_size = 64;

    for (uint8_t core_idx = 0; core_idx < blocks_per_chunk; core_idx++) {
        uint32_t l_cidx = core_idx * c_ltree_size;
        uint32_t d_cidx = core_idx * c_dtree_size;
        uint32_t bl_cidx = core_idx * c_bltree_size;
        uint32_t mxc_cidx = core_idx * 3;
        treegenCore(&(dyn_ltree_freq[l_cidx]), &(dyn_dtree_freq[d_cidx]), &(dyn_ltree_codes[l_cidx]),
                    &(dyn_dtree_codes[d_cidx]), &(dyn_bltree_codes[bl_cidx]), &(dyn_ltree_blen[l_cidx]),
                    &(dyn_dtree_blen[d_cidx]), &(dyn_bltree_blen[bl_cidx]), &(max_codes[mxc_cidx]));
    }
}
}
