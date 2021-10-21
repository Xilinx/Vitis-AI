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
#ifndef _XFCOMPRESSION_ZSTD_SPECS_HPP_
#define _XFCOMPRESSION_ZSTD_SPECS_HPP_

/**
 * @file zstd_specs.hpp
 * @brief Custom data types, constants and stored tables for use in ZSTD decompress kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */
namespace xf {
namespace compression {
namespace details {

typedef struct {
    uint8_t symbol;     // code to get length/offset basevalue
    uint8_t bitCount;   // bits to be read to generate next state
    uint16_t nextState; // basevalue for next state
} FseBSState;

typedef struct {
    ap_uint<8> symbol; // decoded symbol
    ap_uint<4> bitlen; // code bit-length
} HuffmanTable;

template <int OF_DWIDTH = 16, int ML_DWIDTH = 8, int LL_DWIDTH = OF_DWIDTH>
struct __attribute__((packed)) Sequence_dt {
    ap_uint<LL_DWIDTH> litlen;
    ap_uint<ML_DWIDTH> matlen;
    ap_uint<OF_DWIDTH> offset;
};

enum xfBlockType_t { RAW_BLOCK = 0, RLE_BLOCK, CMP_BLOCK, INVALID_BLOCK };
enum xfLitBlockType_t { RAW_LBLOCK = 0, RLE_LBLOCK, CMP_LBLOCK, TREELESS_LBLOCK };
enum xfSymbolCompMode_t { PREDEFINED_MODE = 0, RLE_MODE, FSE_COMPRESSED_MODE, REPEAT_MODE };

const ap_uint<32> c_magicNumber = 0xFD2FB528;
const uint32_t c_skipFrameMagicNumber = 0x184D2A50; // till +15 values
const uint32_t c_skippableFrameMask = 0xFFFFFFF0;

const uint8_t c_maxZstdHfBits = 11;
const uint16_t c_maxCharLit = 35;
const uint16_t c_maxCharDefOffset = 28;
const uint16_t c_maxCharOffset = 31;
const uint16_t c_maxCharMatchlen = 52;
const uint16_t c_maxCharHuffman = 255;

const uint16_t c_maxLitV = 255;
const uint8_t c_maxCodeLL = 35;
const uint8_t c_maxCodeML = 52;
const uint8_t c_maxCodeOF = 31;

const uint8_t c_fseMaxTableLogHF = 6;
const uint8_t c_fseMaxTableLogLL = 9;
const uint8_t c_fseMaxTableLogML = 9;
const uint8_t c_fseMaxTableLogOF = 8;
const uint8_t c_fseMinTableLog = 5;
const uint8_t c_fseMaxTableLog = 12;

const uint32_t c_baseLL[c_maxCharLit + 1] = {0,  1,  2,   3,   4,   5,    6,    7,    8,    9,     10,    11,
                                             12, 13, 14,  15,  16,  18,   20,   22,   24,   28,    32,    40,
                                             48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
const uint32_t c_baseML[c_maxCharMatchlen + 1] = {
    3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,   14,   15,   16,   17,    18,    19,   20,
    21, 22, 23, 24, 25, 26, 27, 28,  29,  30,  31,   32,   33,   34,   35,    37,    39,   41,
    43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539};

const uint8_t c_extraBitsLL[c_maxCharLit + 1] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  1,
                                                 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
const uint8_t c_extraBitsML[c_maxCharMatchlen + 1] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0,
                                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  1,  1, 1,
                                                      2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

// [litlen, offset, matlen]
const int16_t c_defaultDistribution[c_maxCharLit + c_maxCharDefOffset + c_maxCharMatchlen + 3] = { // litlen
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1,
    // offsets
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
    // matlen
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1};

const uint8_t c_litlenCode[64] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                  16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21,
                                  22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23,
                                  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24};

const uint8_t c_matlenCode[128] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38,
    38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42};

const uint32_t c_bitMask[32] = {0,         1,          3,          7,         0xF,       0x1F,      0x3F,
                                0x7F,      0xFF,       0x1FF,      0x3FF,     0x7FF,     0xFFF,     0x1FFF,
                                0x3FFF,    0x7FFF,     0xFFFF,     0x1FFFF,   0x3FFFF,   0x7FFFF,   0xFFFFF,
                                0x1FFFFF,  0x3FFFFF,   0x7FFFFF,   0xFFFFFF,  0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF,
                                0xFFFFFFF, 0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF};

const ap_uint<4> c_hufFixedBlen[256] = {
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8};
const uint8_t c_hufFixedBlenCnt[3] = {24, 120, 112};

const uint32_t c_litlenDeltaCode = 19;
const uint32_t c_matlenDeltaCode = 36;

const uint32_t c_rtbTable[8] = {0, 473195, 504333, 520860, 550000, 700000, 750000, 830000};

inline uint8_t bitsUsed31(uint32_t x) {
#pragma HLS INLINE
    return (uint8_t)(__builtin_clz(x) ^ 31);
}

template <int DWIDTH = 16>
inline uint8_t getLLCode(ap_uint<DWIDTH> litlen) {
#pragma HLS INLINE
    return (litlen > 63) ? bitsUsed31((uint32_t)litlen) + c_litlenDeltaCode : c_litlenCode[litlen];
}

template <int DWIDTH = 16>
inline uint8_t getMLCode(ap_uint<DWIDTH> matlen) {
#pragma HLS INLINE
    return (matlen > 127) ? bitsUsed31((uint32_t)matlen) + c_matlenDeltaCode : c_matlenCode[matlen];
}

/* Testing data */
uint16_t c_testSeqData[70][3] = { //{ll, ml, of}
    {59, 2, 61},  {22, 2, 47},  {4, 5, 35},   {26, 4, 68}, {3, 1, 140},  {36, 1, 139}, {22, 2, 130}, {0, 4, 84},
    {0, 1, 164},  {2, 1, 82},   {19, 9, 44},  {1, 3, 85},  {2, 4, 213},  {0, 1, 4},    {13, 1, 203}, {6, 3, 207},
    {13, 4, 56},  {13, 3, 84},  {31, 2, 202}, {7, 1, 206}, {17, 1, 274}, {9, 10, 356}, {0, 4, 229},  {3, 10, 36},
    {20, 1, 72},  {7, 5, 153},  {51, 1, 222}, {4, 1, 368}, {0, 3, 510},  {3, 1, 411},  {1, 3, 88},   {0, 5, 489},
    {3, 3, 518},  {34, 1, 527}, {5, 1, 165},  {7, 2, 22},  {0, 7, 587},  {0, 1, 288},  {3, 8, 467},  {5, 10, 693},
    {2, 11, 695}, {0, 3, 147},  {0, 12, 701}, {0, 2, 37},  {0, 1, 416},  {0, 1, 33},   {1, 31, 706}, {2, 2, 707},
    {2, 23, 1},   {0, 20, 701}, {0, 1, 696},  {0, 1, 132}, {0, 1, 65},   {0, 17, 704}, {0, 73, 698}, {0, 72, 692},
    {0, 27, 684}, {1, 3, 1},    {2, 37, 683}, {0, 1, 257}, {1, 1, 995},  {0, 5, 461},  {0, 51, 692}, {0, 1, 683},
    {0, 5, 336},  {0, 69, 691}, {0, 33, 685}, {0, 3, 194}, {0, 47, 691}, {0, 12, 684}};

const char c_testLitData[462] = {
    47,  42,  10,  32,  42,  32,  40,  99,  41,  32,  67,  111, 112, 121, 114, 105, 103, 104, 116, 32,  50,  48,
    49,  57,  32,  88,  105, 108, 105, 110, 120, 44,  32,  73,  110, 99,  46,  32,  65,  108, 108, 32,  115, 102,
    115, 100, 102, 32,  114, 101, 115, 101, 114, 118, 101, 100, 46,  10,  32,  76,  105, 99,  101, 110, 115, 101,
    100, 32,  117, 110, 100, 101, 114, 32,  116, 104, 101, 32,  65,  112, 100, 97,  99,  104, 101, 44,  32,  86,
    101, 114, 115, 105, 111, 110, 32,  50,  46,  48,  32,  40,  116, 100, 115, 102, 100, 115, 102, 104, 101, 32,
    34,  34,  41,  59,  121, 111, 117, 32,  109, 102, 102, 115, 102, 115, 97,  121, 32,  110, 111, 116, 32,  117,
    115, 101, 32,  116, 104, 105, 115, 32,  102, 105, 108, 101, 32,  101, 120, 99,  115, 100, 101, 112, 116, 32,
    105, 110, 32,  99,  111, 109, 112, 108, 105, 97,  110, 99,  101, 32,  119, 105, 116, 104, 32,  89,  97,  121,
    32,  111, 98,  116, 97,  105, 110, 32,  97,  32,  99,  111, 112, 121, 32,  111, 102, 32,  97,  116, 104, 116,
    116, 112, 58,  47,  47,  119, 119, 119, 46,  97,  112, 46,  111, 114, 103, 47,  108, 115, 47,  76,  73,  67,
    69,  78,  83,  69,  45,  50,  46,  48,  85,  110, 108, 101, 115, 115, 32,  114, 101, 113, 117, 105, 114, 115,
    100, 101, 100, 32,  98,  121, 32,  97,  112, 112, 108, 105, 99,  97,  98,  108, 101, 32,  108, 97,  119, 32,
    111, 114, 32,  97,  103, 114, 101, 101, 115, 100, 102, 100, 32,  116, 111, 119, 114, 105, 116, 105, 110, 103,
    44,  32,  115, 111, 102, 116, 119, 97,  114, 101, 100, 105, 115, 116, 114, 105, 98,  117, 116, 32,  105, 115,
    111, 110, 32,  97,  110, 32,  34,  65,  83,  32,  73,  83,  34,  32,  66,  65,  83,  73,  83,  44,  87,  73,
    84,  72,  79,  85,  84,  102, 32,  87,  65,  82,  82,  65,  78,  84,  73,  69,  83,  32,  79,  82,  32,  67,
    79,  78,  68,  73,  84,  73,  79,  78,  83,  32,  79,  70,  32,  65,  78,  89,  32,  75,  73,  78,  68,  44,
    32,  101, 105, 116, 104, 101, 114, 32,  101, 120, 112, 114, 111, 114, 32,  105, 32,  83,  101, 101, 32,  102,
    111, 115, 112, 101, 99,  105, 102, 105, 99,  32,  108, 97,  110, 103, 117, 97,  103, 101, 32,  103, 111, 118,
    101, 114, 110, 105, 110, 103, 32,  112, 101, 114, 109, 105, 115, 115, 32,  97,  110, 100, 108, 105, 109, 105,
    116, 97,  116, 115, 100, 102, 10,  32,  42,  47,  10,  102, 102, 102, 97,  99,  115, 100, 32,  115, 100, 100};

const uint8_t c_hfCodeTable[122] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 7,  31, 11, 32, 33, 34, 35, 36, 0,  1,
                                    12, 37, 13, 2,  12, 13, 14, 3,  15, 38, 39, 40, 41, 42, 43, 4,  5,  6,  44, 45, 46,
                                    47, 48, 14, 7,  16, 8,  17, 9,  49, 10, 15, 50, 11, 12, 51, 16, 18, 52, 53, 19, 17,
                                    20, 13, 14, 15, 16, 17, 54, 55, 56, 57, 58, 59, 60, 10, 21, 12, 13, 11, 14, 18, 19,
                                    12, 61, 62, 15, 22, 16, 17, 20, 18, 18, 13, 19, 21, 19, 22, 23, 23};

const uint8_t c_hfBitlenTable[122] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 3, 0, 7, 0, 0, 0, 0, 0, 8, 8, 7, 0, 7, 8, 6, 6, 7, 8,
                                      7, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 6, 8, 7, 8, 7, 8, 0, 8, 6, 0,
                                      8, 8, 0, 6, 7, 0, 0, 7, 6, 7, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 4, 7, 5,
                                      5, 4, 5, 6, 6, 4, 0, 0, 5, 7, 5, 5, 6, 8, 5, 4, 5, 6, 8, 6, 7, 6};

const int16_t c_litNormTable[64] = {16, 5, 4, 3, 2, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0,  0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0,  0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
const int16_t c_llNormTable[64] = {26, 5, 5, 5, 2, 2, -1, 3, 0, -1, 0, 0, 0, 3, 0, 0, -1, -1, -1, 2, -1, -1,
                                   2,  0, 2, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0,  0,  0, 0,  0,
                                   0,  0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0,  0,  0};
const int16_t c_mlNormTable[64] = {0, 44, 12, 16, 9, 9, 0, 2, 2, 2, 5, 2, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0,
                                   0, 2,  0,  0,  0, 2, 0, 0, 0, 2, 2, 0, 2, 0, 0, 2, 2, 0, 5, 0, 0, 0,
                                   0, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
const int16_t c_ofNormTable[64] = {-1, 0, -1, 0, -1, 4, 4, 7, 5, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0,  0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0,  0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

} // details
} // compression
} // xf
#endif
