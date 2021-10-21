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
 * @file ctrl_tokenize.hpp
 */

#ifndef _XF_CODEC_CTRL_TOKENIZE_HPP_
#define _XF_CODEC_CTRL_TOKENIZE_HPP_

#include "kernel3/kernel3_common.hpp"

void Xacc_TokenizeCtrlField_top(hls_Rect dc_rect,
                                hls::stream<uint8_t>& strm_strategy,
                                hls::stream<quant_t>& strm_quant_field,
                                hls::stream<arsigma_t>& strm_arsigma,
                                hls::stream<bool>& strm_strategy_block0,
                                hls::stream<bool>& strm_strategy_block1,
                                hls::stream<bool>& strm_strategy_block2,

                                hls::stream<addr_t>& strm_token_ct_addr,
                                hls::stream<hls_Token_symb>& strm_token_symb,
                                hls::stream<hls_Token_bits>& strm_token_bits,
                                hls::stream<bool>& strm_e_ct_addr,
                                hls::stream<bool>& strm_e_ctrl);
#endif
