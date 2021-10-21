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

/**
 * @file ac_tokenize.hpp
 */

#ifndef _XF_CODEC_AC_TOKENIZE_HPP_
#define _XF_CODEC_AC_TOKENIZE_HPP_

#include "kernel3/kernel3_common.hpp"

const int hls_kZeroDensityContextCount = 105;

// For DCT 8x8 there could be up to 63 non-zero AC coefficients (and one DC
// coefficient). To reduce the total number of contexts,
// the values are combined in pairs, i.e. 0..63 -> 0..31.
const uint32_t hls_kNonZeroBuckets = 32;

// TODO(user): find better clustering for PIK use case.
static const uint8_t hls_kCoeffFreqContext[64] = {
    0,  1,  2,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  8,  8,  9,  9,  9,  9,  10, 10,
    10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14,
    14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
};

// TODO(user): find better clustering for PIK use case.
static const uint8_t hls_kCoeffNumNonzeroContext[65] = { // 0xBAD=255,
    255, 0,  0,  16, 16, 16, 32, 32, 32, 32, 48, 48, 48, 48, 48, 48, 64, 64, 64, 64, 64, 64,
    64,  64, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 93, 93, 93, 93,
    93,  93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93};

const uint8_t hls_kSkipAndBitsSymbol[256] = {
    0,   1,   2,   3,   5,   10,  17,  32,  68,  83,  84,  85,  86,  87,  88,  89,  90,  4,   7,   12,  22,  31,
    43,  60,  91,  92,  93,  94,  95,  96,  97,  98,  99,  6,   14,  26,  36,  48,  66,  100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 8,   19,  34,  44,  57,  78,  110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 9,
    27,  39,  52,  61,  79,  120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 11,  28,  41,  53,  64,  80,  130,
    131, 132, 133, 134, 135, 136, 137, 138, 139, 13,  33,  46,  63,  72,  140, 141, 142, 143, 144, 145, 146, 147,
    148, 149, 150, 15,  35,  47,  65,  69,  151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 16,  37,  51,
    62,  74,  162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 18,  38,  50,  59,  75,  173, 174, 175, 176,
    177, 178, 179, 180, 181, 182, 183, 20,  40,  54,  76,  82,  184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
    194, 23,  42,  55,  77,  195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 24,  45,  56,  70,  207,
    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 25,  49,  58,  71,  219, 220, 221, 222, 223, 224, 225,
    226, 227, 228, 229, 230, 29,  67,  81,  231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 21,  30,
    73,  243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
};

//----------------------------------------------------------
void XAcc_TokenizeCoefficients5(const int32_t orders[3][64], // color dct_band
                                hls_Rect& rect,
                                hls::stream<dct_t> strm_coef_raster[8],
                                uint8_t ac_static_context_map[hls_kNumContexts],
                                hls::stream<ap_uint<13> >& strm_token_addr,
                                hls::stream<ap_uint<24> >& strm_token_symb,
                                hls::stream<ap_uint<24> >& strm_token_bits,
                                hls::stream<bool>& strm_e_addr,
                                hls::stream<bool>& strm_e_token);

void XAcc_TokenizeCoefficients6(const int32_t orders[3][64], // color dct_band
                                const group_rect rect,
                                hls::stream<dct_t>& strm_coef_raster,
                                uint8_t ac_static_context_map[hls_kNumContexts],
                                hls::stream<ap_uint<13> >& strm_token_addr,
                                hls::stream<hls_Token_symb>& strm_token_symb,
                                hls::stream<hls_Token_bits>& strm_token_bits,
                                hls::stream<bool>& strm_e_addr,
                                hls::stream<bool>& strm_e_token);

// for cosim
void hls_orderblk_tokennz(const int32_t orders[3][64], // color dct_band
                          const hls_blksize rect,
                          hls::stream<dct_t>& strm_coef_raster,
                          hls::stream<nzeros_t>& cnt_nz,
                          hls::stream<hls_Token>& strm_nz_token,
                          hls::stream<dct_t> strm_coef_orderd[64]);

void hls_order_blk(const hls_blksize rect,
                   const int32_t orders[3][64],
                   hls::stream<dct_t>& strm_coef_raster,
                   hls::stream<dct_t>& strm_coef_orderd);
#endif
