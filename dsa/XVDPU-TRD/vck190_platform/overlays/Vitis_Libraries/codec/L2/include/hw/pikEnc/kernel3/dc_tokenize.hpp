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
 * @file dc_tokenize.hpp
 */

#ifndef _XF_CODEC_DC_TOKENIZE_HPP_
#define _XF_CODEC_DC_TOKENIZE_HPP_

#include "kernel3/dc_shrink.hpp"
#include "kernel3/kernel3_common.hpp"

void hls_encode_dc_top(const bool rle,
                       const hls_Rect rect_dc,
                       hls::stream<dct_t>& strm_dc_y1,
                       hls::stream<dct_t>& strm_dc_y2,
                       hls::stream<dct_t>& strm_dc_y3,
                       hls::stream<dct_t>& strm_dc_x,
                       hls::stream<dct_t>& strm_dc_b,
                       hls::stream<addr_t>& strm_token_addr,
                       hls::stream<hls_Token_symb>& strm_token_symb,
                       hls::stream<hls_Token_bits>& strm_token_bits,
                       hls::stream<bool>& strm_e_addr,
                       hls::stream<bool>& strm_e_dc);

#endif
