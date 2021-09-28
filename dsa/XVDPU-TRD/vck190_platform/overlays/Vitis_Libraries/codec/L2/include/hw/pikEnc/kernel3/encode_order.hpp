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
 * @file encode_order.hpp
 */

#ifndef _XF_CODEC_ENCODE_ORDER_HPP_
#define _XF_CODEC_ENCODE_ORDER_HPP_

#include "kernel3/kernel3_common.hpp"

// Size of batch of Lehmer-transformed order of coefficients.
// If all codes in the batch are zero, then span is encoded with a single bit.
#define hls_kCoeffOrderCodeSpan (16)

void hls_EncodeCoeffOrder(hls::stream<int>& strm_order,
                          int& num_bits,
                          int& num,
                          hls::stream<nbits_t>& strm_nbits,
                          hls::stream<uint16_t>& strm_bits);

#endif
