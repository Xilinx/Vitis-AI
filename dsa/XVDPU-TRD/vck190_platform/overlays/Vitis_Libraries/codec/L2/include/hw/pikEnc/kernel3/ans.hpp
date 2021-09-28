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
 * @file ans.hpp
 */

#ifndef _XF_CODEC_ANS_HPP_
#define _XF_CODEC_ANS_HPP_

#include "kernel3/kernel3_common.hpp"

void XAcc_WriteTokens_wapper( // input
    const int start,
    const int end,
    hls::stream<hls_Token_symb>& strm_ac_token_reverse,
    hls::stream<hls_Token_bits>& strm_token_bit,
    hls_ANSEncSymbolInfo codes[hls_kNumStaticContexts][hls_alphabet_size],
    uint8_t context_map[hls_kNumContexts], // table
    const bool is_dc,
    uint8_t dc_context_map[MAX_NUM_COLOR],

    // output
    uint32_t& pos,
    uint8_t& cnt_buffer,
    uint16_t& reg_buffer,
    hls::stream<uint16_t>& strm_pos_byte,
    hls::stream<bool>& strm_ac_e);

void hls_WriteTokensTop(

    // input
    const int total_token,
    hls::stream<hls_Token_symb>& strm_ac_token_reverse,
    hls::stream<hls_Token_bits>& strm_token_bit,
    hls_ANSEncSymbolInfo hls_codes[hls_kNumStaticContexts][hls_alphabet_size],
    uint8_t ac_static_context_map[hls_kNumContexts], // table
    const bool is_dc,
    uint8_t dc_context_map[MAX_NUM_COLOR],

    // output
    int& len_ac,
    hls::stream<uint16_t>& strm_ac_byte,
    hls::stream<bool>& strm_ac_e);

struct x_PosAndCount {
    uint32_t pos;
    uint32_t count;
};

#endif
