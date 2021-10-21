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
#ifndef _XFCOMPRESSION_ZLIB_SPECS_HPP_
#define _XFCOMPRESSION_ZLIB_SPECS_HPP_

/**
 * @file zlib_specs.h
 * @brief Header for configuration for zlib compression and decompression kernels.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include <stdint.h>

const int gz_max_literal_count = 4096;

// Dynamic Huffman Related Content
const auto c_maxBits = 15;
// Literals
const auto c_literals = 256;

// Length codes
const auto c_lengthCodes = 29;

// Literal Codes
const auto c_literalCodes = (c_literals + 1 + c_lengthCodes);

// Distance Codes
const auto c_distanceCodes = 30;

// bit length codes
const auto c_blCodes = 19;

// Literal Tree size - 573
const auto c_heapSize = (2 * c_literalCodes + 1);

// Bit length codes must not exceed c_maxBlBits bits
const auto c_maxBlBits = 7;

const auto c_reusePrevBlen = 16;

const auto c_reuseZeroBlen = 17;

const auto c_reuseZeroBlen7 = 18;

const auto c_fixedDecoder = 0;
const auto c_dynamicDecoder = 1;
const auto c_fullDecoder = 2;

const uint16_t c_frequency_bits = 32;
const uint16_t c_codeword_bits = 20;
const uint16_t c_litCodeCount = 286;
const uint16_t c_dstCodeCount = 30;
const uint16_t c_blnCodeCount = 19;
const uint16_t c_maxCodeBits = 15;
const uint16_t c_maxBLCodeBits = 7;

#endif // _XFCOMPRESSION_ZLIB_SPECS_HPP_
