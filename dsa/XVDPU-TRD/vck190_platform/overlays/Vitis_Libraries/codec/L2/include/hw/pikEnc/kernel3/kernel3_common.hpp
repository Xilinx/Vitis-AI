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
 * @file kernel3_common.hpp
 * @brief JPEG XL codec common include struct.
 */

#ifndef __cplusplus
#error " pik_codec_common.hpp hls::stream<> interface, and thus requires C++"
#endif

#ifndef _XF_CODEC_KERNEL3_COMMON_HPP_
#define _XF_CODEC_KERNEL3_COMMON_HPP_

#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <stdint.h>

#include "pik_common.hpp"

#ifndef __SYNTHESIS__
// For debug
#include <bitset>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#endif

// ------------------------------------------------------------
// for dc Predictor
// ------------------------------------------------------------

#define hls_kNumPredictors (8)
// ------------------------------------------------------------
// for histogram
// ------------------------------------------------------------

typedef ap_uint<13> addr_t;

// for tokenize type
typedef int16_t dct_t;
typedef uint8_t nzeros_t;

struct hls_strategy {
    uint8_t strategy_;
    bool block_;
};
typedef int16_t quant_t;
typedef uint8_t arsigma_t;

struct hls_Token {
    uint16_t context; // 0~411
    uint8_t symbol;
    uint8_t nbits;
    uint16_t bits;
};

#define hls_kAcStrategyContexts (1)
#define hls_kQuantFieldContexts (1)
#define hls_kARParamsContexts (1)
#define hls_QuantContext (2)

static const int hls_kMaxNumSymbolsForSmallCode = 4;

#define hls_PackSigned(value) ((uint16_t)value << 1) ^ (((uint16_t)(~value) >> 15) - 1)

// to be remove
#define hls_Log2FloorNonZero_16b(n) 15 ^ __builtin_clz((uint16_t)n)
#define hls_Log2FloorNonZero_32b(n) 31 ^ __builtin_clz((uint32_t)n)
#define hls_Log2FloorNonZero_64b(n) 63 ^ __builtin_clz((uint64_t)n)
#define hls_Log2Floor_32b(n) n == 0 ? -1 : (31 ^ __builtin_clz((uint32_t)n))

#define hls_PackSigned_32b(value) ((uint32_t)value << 1) ^ (((uint32_t)(~value) >> 31) - 1)
#define hls_PackSigned_16b(value) ((uint16_t)value << 1) ^ (((uint16_t)(~value) >> 15) - 1)

typedef uint32_t hist_t;
typedef uint8_t nbits_t;

// for encode
const int32_t hls_kNaturalCoeffOrder8[8 * 8] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
                                                12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
                                                35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
                                                58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

const int32_t hls_kNaturalCoeffOrderLut8[8 * 8] = {0,  1,  5,  6,  14, 15, 27, 28, 2,  4,  7,  13, 16, 26, 29, 42,
                                                   3,  8,  12, 17, 25, 30, 41, 43, 9,  11, 18, 24, 31, 40, 44, 53,
                                                   10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60,
                                                   21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63};

// ------------------------------------------------------------
// for debug  population
// ------------------------------------------------------------
#define _XF_IMAGE_VOID_CAST static_cast<void>
#ifndef __SYNTHESIS__
#define _XF_IMAGE_PRINT(msg...) \
    do {                        \
        printf(msg);            \
    } while (0)
#else
#define _XF_IMAGE_PRINT(msg...) (_XF_IMAGE_VOID_CAST(0))
#endif

// ------------------------------------------------------------
// for noise
// ------------------------------------------------------------
const int hls_kMaxNoiseSize = 16;

// ------------------------------------------------------------
// for encode
// ------------------------------------------------------------

#define hls_kTileDimInBlocks (8)

struct hls_Rect { // to be removed
    int x0;
    int y0;
    uint16_t xsize;
    uint16_t ysize;
    uint8_t xsize_tiles;
    uint8_t ysize_tiles;
    uint8_t xsize_blocks; // use as const
    uint8_t ysize_blocks;
    uint8_t n_tiles;
};

struct group_rect {
    uint8_t xsize_tiles;
    uint8_t ysize_tiles;
    uint8_t xsize_blocks; // use as const
    uint8_t ysize_blocks;
};

struct hls_blksize {
    uint8_t xsize;
    uint8_t ysize;
};

struct hls_Token_symb {
    uint16_t context;
    uint8_t symbol;
};

struct hls_ANSEncSymbolInfo {
    uint16_t freq_;
    uint16_t start_;

    uint64_t ifreq_;
};

struct hls_TokenInfo {
    hls_ANSEncSymbolInfo info;
};

struct hls_Token_bits {
    uint16_t bits;
    uint8_t nbits;
};

typedef uint64_t hls_Runbit_t;
typedef uint32_t hls_Runbit_t2;

#define hls_RECIPROCAL_PRECISION 42
const int hls_kMaxClusters = 256;
const uint16_t hls_alphabet_size = 256;
#define MAX_ALPHABET_SIZE 256

void hls_WriteBits_strm(const nbits_t n_bits,
                        uint16_t bits,
                        int& num_bits,
                        int& num,
                        hls::stream<nbits_t>& strm_nbits,
                        hls::stream<uint16_t>& strm_bits);

void hls_WriteBits_strm_nodepend(const nbits_t n_bits,
                                 uint16_t bits,
                                 hls::stream<nbits_t>& strm_nbits,
                                 hls::stream<uint16_t>& strm_bits);

void hls_StoreVarLenUint16(
    uint32_t n, int& num_bits, int& num, hls::stream<nbits_t>& strm_nbits, hls::stream<uint16_t>& strm_bits);

float hls_FastLog2(int v);

void hls_WriteBitToStream(const int num_pair,
                          uint8_t& byte_tail,
                          hls::stream<nbits_t>& strm_nbits,
                          hls::stream<uint16_t>& strm_bits,
                          int& pos,
                          hls::stream<uint8_t>& strm_byte,
                          hls::stream<bool>& strm_histo_e);

void hls_WriteZeroesToByteBoundary(int* pos);

#endif
