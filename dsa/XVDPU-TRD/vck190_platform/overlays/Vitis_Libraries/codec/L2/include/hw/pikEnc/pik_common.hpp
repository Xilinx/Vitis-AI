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
 * @file pik_common.hpp
 */

#ifndef _XF_CODEC_PIK_COMMON_HPP_
#define _XF_CODEC_PIK_COMMON_HPP_

#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

#define MAX_EXPONENT_PIX (11)
#define MAX_NUM_COLOR 3
#define MAX_NUM_BLOCK88_W (1024)
#define MAX_NUM_BLOCK88_H (1024)
#define MAX_NUM_BLOCK88 (MAX_NUM_BLOCK88_W * MAX_NUM_BLOCK88_H)

#define MAX_NUM_PIXEL (8192 * 8192)
#define ALL_PIXEL (MAX_NUM_PIXEL * MAX_NUM_COLOR)

#define MAX_NUM_BLOCK88_W_TITLE (8)
#define MAX_NUM_BLOCK88_H_TITLE (8)

#define MAX_PIX_W (MAX_NUM_BLOCK88_W << 3)
#define MAX_PIX_H (MAX_NUM_BLOCK88_H << 3)
#define MAX_NUM_PIX (MAX_NUM_BLOCK88 * MAX_NUM_COLOR * 64)
#define MAX_NUM_COEF (MAX_NUM_BLOCK88 * MAX_NUM_COLOR * 64)
#define MAX_SIZE_COEF (MAX_NUM_COEF * 2)

#define BLKDIM (8)
#define BLOCK_SIZE (64)
#define hls_kDcGroupDimInBlocks (256)
#define hls_kGroupDim (512)
#define hls_kTileDim (64)
#define hls_kNumPredictors (8)

#define DCGROUP_SIZE (hls_kDcGroupDimInBlocks * hls_kDcGroupDimInBlocks)
#define ACGROUP_SIZE (hls_kGroupDim * hls_kGroupDim)
#define TILE_SIZE (hls_kTileDim * hls_kTileDim)

#define MAX_AC_GROUP (256)
#define MAX_DC_GROUP (16)

#define AXI_SZ (32)
#define AXI_WIDTH (AXI_SZ * 2)
#define BURST_LENTH (32)
#define DT_SZ (4)
typedef ap_int<8 * DT_SZ> DT;

#define BLKDIM (8)
#define MAX_EXPONENT_PIX (11)

#define DIVCEIL(a, b) ((a + b - 1) / b)
#define XBLOCKS_32X32 DIVCEIL(MAX_NUM_BLOCK88_W, 4)
#define YBLOCKS_32X32 DIVCEIL(MAX_NUM_BLOCK88_H, 4)

#define ELEM_SPACE (MAX_NUM_BLOCK88_W * MAX_NUM_BLOCK88_H * 8 * 8)
#define BUF_DEPTH ALL_PIXEL

#define AXI_OUT (MAX_NUM_COLOR * XBLOCKS_32X32 * YBLOCKS_32X32 * 32 * 32)
#define AXI_CMAP (MAX_NUM_BLOCK88 / 64 * 2 + 2)
#define AXI_QF (MAX_NUM_BLOCK88 + 2)

#define MAX_NUM_CONFIG (32)
#define MAX_NUM_DC (MAX_NUM_BLOCK88 * MAX_NUM_COLOR)
#define MAX_NUM_AC AXI_OUT

#define XGROUPS_512X512 DIVCEIL(MAX_NUM_BLOCK88_W, 64)
#define YGROUPS_512X512 DIVCEIL(MAX_NUM_BLOCK88_H, 64)
#define MAX_NUM_ORDER XGROUPS_512X512* YGROUPS_512X512* MAX_NUM_COLOR * 64
#define MAX_NUM_GROUP 256

#define hls_kHybridEncodingSplitToken (16)
#define hls_kHybridEncodingDirectSplitExponent (4)

#define hls_kRleSymStart (39)
#define hls_kEntropyCodingNumSymbols (78)

#define hls_kANSBufferSize (1 << 16)
#define hls_kMaxBufSize (3 << 16)
#define hls_kTokenMaxSize 24576
#define hls_kTotalSize 49152

#define hls_ANS_LOG_TAB_SIZE (10)
#define hls_ANS_TAB_SIZE (1 << hls_ANS_LOG_TAB_SIZE)
#define hls_ANS_TAB_MASK (hls_ANS_TAB_SIZE - 1)
#define hls_ANS_SIGNATURE (0x13) // Initial state, used as CRC.

#define hls_MAX_ALPHABET_SIZE 256
#define hls_kAlphabetSize (272)

#define hls_kOrderContexts 3

#define hls_kClustersLimit 64
#define hls_kNumStaticZdensContexts 7
#define hls_kNumStaticOrderFreeContexts 3
#define hls_kNumStaticContexts 24
#define hls_kMinClustersForHistogramRemap (24)
#define hls_NumHistograms 6144

#define hls_kNumContexts 411
#define MAX_DC_SIZE (4 * (2 * hls_kTotalSize) + 4096)
#define MAX_DC_HISTO_SIZE (1024 * (MAX_NUM_COLOR + 4))
#define MAX_AC_SIZE (4 * ((4 * hls_kTotalSize)) + 4096)
#define MAX_AC_HISTO_SIZE (hls_kNumStaticContexts * 1024)

template <typename I, typename F>
inline F bitsToF(I in) {
    union {
        I __I;
        F __F;
    } __T;
    __T.__I = in;
    return __T.__F;
}

template <typename F, typename I>
inline I fToBits(F in) {
    union {
        I __I;
        F __F;
    } __T;
    __T.__F = in;
    return __T.__I;
}

#endif
