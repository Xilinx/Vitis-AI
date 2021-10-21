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
 * @file des.hpp
 * @brief header file for Data Encryption Standard relate function.
 * This file part of Vitis Security Library.
 *
 * @detail Currently we have Aes256_Encryption for AES256 standard.
 */

#ifndef _XF_SECURITY_DES_HPP_
#define _XF_SECURITY_DES_HPP_

#include <ap_int.h>

#ifndef __SYNTHESIS__
#include <string>
#endif

namespace xf {
namespace security {
namespace internal {
#ifndef __SYNTHESIS__
template <int W>
void print(ap_uint<W> a) {
    for (int i = 0; i < W; ++i) {
        if (i % 8 == 0) {
            std::cout << " ";
        }

        std::cout << a[i];
    }

    std::cout << std::endl;
}

template <int W>
void print(std::string var, ap_uint<W> a) {
    std::cout << var << std::endl;
    for (int i = 0; i < W; ++i) {
        if (i % 8 == 0) {
            std::cout << " ";
        }

        std::cout << a[i];
    }

    std::cout << std::endl;
}

static void printArray(int arr[], int size, int interval) {
    std::cout << "{\n";
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i];

        if (i != size - 1) {
            std::cout << ", ";
        }

        if ((i + 1) % interval == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << "}" << std::endl;
}
#endif

const ap_uint<8> PermMap[64] = {58, 50, 42, 34, 26, 18, 10, 2,  60, 52, 44, 36, 28, 20, 12, 4,  62, 54, 46, 38, 30, 22,
                                14, 6,  64, 56, 48, 40, 32, 24, 16, 8,  57, 49, 41, 33, 25, 17, 9,  1,  59, 51, 43, 35,
                                27, 19, 11, 3,  61, 53, 45, 37, 29, 21, 13, 5,  63, 55, 47, 39, 31, 23, 15, 7};

// Reverse of PermMap, not necessary
const ap_uint<8> IPermMap[64] = {40, 8,  48, 16, 56, 24, 64, 32, 39, 7,  47, 15, 55, 23, 63, 31, 38, 6,  46, 14, 54, 22,
                                 62, 30, 37, 5,  45, 13, 53, 21, 61, 29, 36, 4,  44, 12, 52, 20, 60, 28, 35, 3,  43, 11,
                                 51, 19, 59, 27, 34, 2,  42, 10, 50, 18, 58, 26, 33, 1,  41, 9,  49, 17, 57, 25};

const ap_uint<8> ExtMap[48] = {32, 1,  2,  3,  4,  5,  4,  5,  6,  7,  8,  9,  8,  9,  10, 11,
                               12, 13, 12, 13, 14, 15, 16, 17, 16, 17, 18, 19, 20, 21, 20, 21,
                               22, 23, 24, 25, 24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32, 1};

const ap_uint<4> SP[8][64] = {
    {14, 4,  13, 1, 2,  15, 11, 8, 3, 10, 6, 12, 5,  9,  0,  7,  0,  15, 7,  4,  14, 2,
     13, 1,  10, 6, 12, 11, 9,  5, 3, 8,  4, 1,  14, 8,  13, 6,  2,  11, 15, 12, 9,  7,
     3,  10, 5,  0, 15, 12, 8,  2, 4, 9,  1, 7,  5,  11, 3,  14, 10, 0,  6,  13},
    {15, 1,  8, 14, 6,  11, 3,  4, 9, 7, 2,  13, 12, 0, 5, 10, 3,  13, 4,  7, 15, 2,  8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
     0,  14, 7, 11, 10, 4,  13, 1, 5, 8, 12, 6,  9,  3, 2, 15, 13, 8,  10, 1, 3,  15, 4, 2,  11, 6, 7, 12, 0, 5, 14, 9},
    {10, 0,  9,  14, 6, 3,  15, 5,  1,  13, 12, 7, 11, 4,  2,  8,  13, 7, 0,  9, 3, 4,
     6,  10, 2,  8,  5, 14, 12, 11, 15, 1,  13, 6, 4,  9,  8,  15, 3,  0, 11, 1, 2, 12,
     5,  10, 14, 7,  1, 10, 13, 0,  6,  9,  8,  7, 4,  15, 14, 3,  11, 5, 2,  12},
    {7, 13, 14, 3, 0, 6,  9, 10, 1,  2, 8,  5, 11, 12, 4,  15, 13, 8,  11, 5, 6, 15,
     0, 3,  4,  7, 2, 12, 1, 10, 14, 9, 10, 6, 9,  0,  12, 11, 7,  13, 15, 1, 3, 14,
     5, 2,  8,  4, 3, 15, 0, 6,  10, 1, 13, 8, 9,  4,  5,  11, 12, 7,  2,  14},
    {2,  12, 4, 1,  7,  10, 11, 6, 8, 5,  3, 15, 13, 0,  14, 9,  14, 11, 2,  12, 4,  7,
     13, 1,  5, 0,  15, 10, 3,  9, 8, 6,  4, 2,  1,  11, 10, 13, 7,  8,  15, 9,  12, 5,
     6,  3,  0, 14, 11, 8,  12, 7, 1, 14, 2, 13, 6,  15, 0,  9,  10, 4,  5,  3},
    {12, 1,  10, 15, 9,  2,  6, 8,  0, 13, 3,  4,  14, 7,  5, 11, 10, 15, 4, 2, 7, 12,
     9,  5,  6,  1,  13, 14, 0, 11, 3, 8,  9,  14, 15, 5,  2, 8,  12, 3,  7, 0, 4, 10,
     1,  13, 11, 6,  4,  3,  2, 12, 9, 5,  15, 10, 11, 14, 1, 7,  6,  0,  8, 13},
    {4, 11, 2,  14, 15, 0, 8, 13, 3,  12, 9, 7, 5, 10, 6, 1, 13, 0,  11, 7, 4, 9, 1,  10, 14, 3, 5, 12, 2,  15, 8, 6,
     1, 4,  11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5,  9, 2, 6,  11, 13, 8, 1, 4, 10, 7,  9,  5, 0, 15, 14, 2,  3, 12},
    {13, 2, 8, 4, 6,  15, 11, 1, 10, 9,  3,  14, 5, 0, 12, 7, 1, 15, 13, 8, 10, 3, 7,  4,  12, 5, 6, 11, 0, 14, 9, 2, 7,
     11, 4, 1, 9, 12, 14, 2,  0, 6,  10, 13, 15, 3, 5, 8,  2, 1, 14, 7,  4, 10, 8, 13, 15, 12, 9, 0, 3,  5, 6,  11}};

const ap_int<8> FPermMap[32] = {16, 7, 20, 21, 29, 12, 28, 17, 1,  15, 23, 26, 5,  18, 31, 10,
                                2,  8, 24, 14, 32, 27, 3,  9,  19, 13, 30, 6,  22, 11, 4,  25};

const ap_uint<8> PCMapC[28] = {57, 49, 41, 33, 25, 17, 9,  1,  58, 50, 42, 34, 26, 18,
                               10, 2,  59, 51, 43, 35, 27, 19, 11, 3,  60, 52, 44, 36};

const ap_uint<8> PCMapD[28] = {63, 55, 47, 39, 31, 23, 15, 7,  62, 54, 46, 38, 30, 22,
                               14, 6,  61, 53, 45, 37, 29, 21, 13, 5,  28, 20, 12, 4};

const ap_uint<8> PC2Map[48] = {14, 17, 11, 24, 1,  5,  3,  28, 15, 6,  21, 10, 23, 19, 12, 4,
                               26, 8,  16, 7,  27, 20, 13, 2,  41, 52, 31, 37, 47, 55, 30, 40,
                               51, 45, 33, 48, 44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32};

const ap_uint<8> subkeyIndex[16][48] = {
    {9,  50, 33, 59, 48, 16, 32, 56, 1, 8,  18, 41, 2,  34, 25, 24, 43, 57, 58, 0,  35, 26, 17, 40,
     21, 27, 38, 53, 36, 3,  46, 29, 4, 52, 22, 28, 60, 20, 37, 62, 14, 19, 44, 13, 12, 61, 54, 30},
    {1,  42, 25, 51, 40, 8,  24, 48, 58, 0,  10, 33, 59, 26, 17, 16, 35, 49, 50, 57, 56, 18, 9,  32,
     13, 19, 30, 45, 28, 62, 38, 21, 27, 44, 14, 20, 52, 12, 29, 54, 6,  11, 36, 5,  4,  53, 46, 22},
    {50, 26, 9,  35, 24, 57, 8,  32, 42, 49, 59, 17, 43, 10, 1,  0,  48, 33, 34, 41, 40, 2,  58, 16,
     60, 3,  14, 29, 12, 46, 22, 5,  11, 28, 61, 4,  36, 27, 13, 38, 53, 62, 20, 52, 19, 37, 30, 6},
    {34, 10, 58, 48, 8,  41, 57, 16, 26, 33, 43, 1,  56, 59, 50, 49, 32, 17, 18, 25, 24, 51, 42, 0,
     44, 54, 61, 13, 27, 30, 6,  52, 62, 12, 45, 19, 20, 11, 60, 22, 37, 46, 4,  36, 3,  21, 14, 53},
    {18, 59, 42, 32, 57, 25, 41, 0,  10, 17, 56, 50, 40, 43, 34, 33, 16, 1,  2,  9,  8,  35, 26, 49,
     28, 38, 45, 60, 11, 14, 53, 36, 46, 27, 29, 3,  4,  62, 44, 6,  21, 30, 19, 20, 54, 5,  61, 37},
    {2,  43, 26, 16, 41, 9,  25, 49, 59, 1,  40, 34, 24, 56, 18, 17, 0, 50, 51, 58, 57, 48, 10, 33,
     12, 22, 29, 44, 62, 61, 37, 20, 30, 11, 13, 54, 19, 46, 28, 53, 5, 14, 3,  4,  38, 52, 45, 21},
    {51, 56, 10, 0,  25, 58, 9,  33, 43, 50, 24, 18, 8, 40, 2,  1,  49, 34, 35, 42, 41, 32, 59, 17,
     27, 6,  13, 28, 46, 45, 21, 4,  14, 62, 60, 38, 3, 30, 12, 37, 52, 61, 54, 19, 22, 36, 29, 5},
    {35, 40, 59, 49, 9,  42, 58, 17, 56, 34, 8,  2,  57, 24, 51, 50, 33, 18, 48, 26, 25, 16, 43, 1,
     11, 53, 60, 12, 30, 29, 5,  19, 61, 46, 44, 22, 54, 14, 27, 21, 36, 45, 38, 3,  6,  20, 13, 52},
    {56, 32, 51, 41, 1,  34, 50, 9,  48, 26, 0,  59, 49, 16, 43, 42, 25, 10, 40, 18, 17, 8,  35, 58,
     3,  45, 52, 4,  22, 21, 60, 11, 53, 38, 36, 14, 46, 6,  19, 13, 28, 37, 30, 62, 61, 12, 5,  44},
    {40, 16, 35, 25, 50, 18, 34, 58, 32, 10, 49, 43, 33, 0,  56, 26, 9,  59, 24, 2,  1,  57, 48, 42,
     54, 29, 36, 19, 6,  5,  44, 62, 37, 22, 20, 61, 30, 53, 3,  60, 12, 21, 14, 46, 45, 27, 52, 28},
    {24, 0,  48, 9, 34, 2,  18, 42, 16, 59, 33, 56, 17, 49, 40, 10, 58, 43, 8,  51, 50, 41, 32, 26,
     38, 13, 20, 3, 53, 52, 28, 46, 21, 6,  4,  45, 14, 37, 54, 44, 27, 5,  61, 30, 29, 11, 36, 12},
    {8,  49, 32, 58, 18, 51, 2,  26, 0, 43, 17, 40, 1,  33, 24, 59, 42, 56, 57, 35, 34, 25, 16, 10,
     22, 60, 4,  54, 37, 36, 12, 30, 5, 53, 19, 29, 61, 21, 38, 28, 11, 52, 45, 14, 13, 62, 20, 27},
    {57, 33, 16, 42, 2,  35, 51, 10, 49, 56, 1, 24, 50, 17, 8,  43, 26, 40, 41, 48, 18, 9,  0, 59,
     6,  44, 19, 38, 21, 20, 27, 14, 52, 37, 3, 13, 45, 5,  22, 12, 62, 36, 29, 61, 60, 46, 4, 11},
    {41, 17, 0, 26, 51, 48, 35, 59, 33, 40, 50, 8,  34, 1,  57, 56, 10, 24, 25, 32, 2,  58, 49, 43,
     53, 28, 3, 22, 5,  4,  11, 61, 36, 21, 54, 60, 29, 52, 6,  27, 46, 20, 13, 45, 44, 30, 19, 62},
    {25, 1,  49, 10, 35, 32, 48, 43, 17, 24, 34, 57, 18, 50, 41, 40, 59, 8, 9,  16, 51, 42, 33, 56,
     37, 12, 54, 6,  52, 19, 62, 45, 20, 5,  38, 44, 13, 36, 53, 11, 30, 4, 60, 29, 28, 14, 3,  46},
    {17, 58, 41, 2,  56, 24, 40, 35, 9,  16, 26, 49, 10, 42, 33, 32, 51, 0,  1,  8,  43, 34, 25, 48,
     29, 4,  46, 61, 44, 11, 54, 37, 12, 60, 30, 36, 5,  28, 45, 3,  22, 27, 52, 21, 20, 6,  62, 38}};

template <int W>
static void convertEndian(ap_uint<W> in, ap_uint<W>& out) {
    // speed up
    // 8 * size and remaining
    // actually W == 64
    int size = W / 8;
    if (W % 8 != 0 /*(W >> 3) << 3 != W*/) {
        size++;
    }

    int start = 0;
    for (int i = 0; i < size; ++i) {
#pragma HLS pipeline
        int end = start + 8 > W ? W : start + 8;
        for (int j = start, k = end - 1; j < end; ++j, --k) {
            out[j] = in[k];
        }

        start += 8;
    }
}

// Convert number according to endian
static void convert(ap_uint<64> in, ap_uint<64>& out) {
    int start = 0;
    for (int i = 0; i < 8; ++i) {
#pragma HLS unroll
        int end = start + 8;
        for (int j = start; j < end; ++j) {
            int k = start + 7 - (j - start);
            out[j] = in[k];
        }

        start += 8;
    }
}

// Initialize subkeyC and subkeyD
static void initSubkeyCD(ap_uint<64> key, ap_uint<28>& subkeyC, ap_uint<28>& subkeyD) {
    for (int i = 0; i < 28; ++i) {
#pragma HLS unroll
        int id = PCMapC[i] - 1;
        subkeyC[i] = key[id];
        int idx = PCMapD[i] - 1;
        subkeyD[i] = key[idx];
    }
}

// Helper function for subkey index
static void initSubkeyIndex(int cIndexArray[28], int dIndexArray[28]) {
    for (int i = 0; i < 28; ++i) {
#pragma HLS unroll
        int id = PCMapC[i] - 1;
        cIndexArray[i] = id;
        int idx = PCMapD[i] - 1;
        dIndexArray[i] = idx;
    }
}

// Helper function for subkey index
template <int W>
static void leftRotateShiftArray(int in[W], int numShift) {
    int tmp[W];
    for (int i = 0; i < W; ++i) {
        int idx = (i + numShift) % W;
        tmp[i] = in[idx];
    }

    for (int i = 0; i < W; ++i) {
        in[i] = tmp[i];
    }
}

static void getSubkeyIndex(int iter, int cIndexArray[28], int dIndexArray[28], int subkeyIndex[48]) {
    int numShift = 2;
    int round = iter + 1;
    if (round == 1 || round == 2 || round == 9 || round == 16) {
        numShift = 1;
    }

    leftRotateShiftArray<28>(cIndexArray, numShift);
    leftRotateShiftArray<28>(dIndexArray, numShift);

    int cd[56];
    for (int i = 0; i < 28; ++i) {
        cd[i] = cIndexArray[i];
        cd[i + 28] = dIndexArray[i];
    }

    for (int i = 0; i < 48; ++i) {
        subkeyIndex[i] = cd[PC2Map[i] - 1];
    }
}

static void leftRotateShift(ap_uint<28>& in, int numShift) {
    ap_uint<28> tmp;
    for (int i = 0; i < 28; ++i) {
#pragma HLS unroll
        int idx = (i + numShift) % 28;
        tmp[i] = in[idx];
    }

    in = tmp;
}

// Get subkey for iteration iter, subkeyC and subkeyD are updated
static void getSubkey(int iter, ap_uint<28>& subkeyC, ap_uint<28>& subkeyD, ap_uint<48>& subkey) {
    int numShift = 2;
    int round = iter + 1;
    if (round == 1 || round == 2 || round == 9 || round == 16) {
        numShift = 1;
    }

    leftRotateShift(subkeyC, numShift);
    leftRotateShift(subkeyD, numShift);

    ap_uint<56> cd;
    cd(27, 0) = subkeyC;
    cd(55, 28) = subkeyD;

    for (int i = 0; i < 48; ++i) {
#pragma HLS unroll
        subkey[i] = cd[PC2Map[i] - 1];
    }
}

static void extend(ap_uint<32> in, ap_uint<48>& out) {
    for (int i = 0; i < 48; ++i) {
#pragma HLS unroll
        out[i] = in[ExtMap[i] - 1];
    }
};

static void substitute(ap_uint<48> in, ap_uint<32>& out) {
    int start = 0, outStart = 0;
    for (int i = 0; i < 8; ++i) {
#pragma HLS unroll
        ap_uint<6> block = in(start + 5, start);

        ap_uint<2> row;
        row[0] = block[5];
        row[1] = block[0];

        ap_uint<4> col;
        col[0] = block[4];
        col[1] = block[3];
        col[2] = block[2];
        col[3] = block[1];

        int rowId = (int)row;
        int idx = (rowId << 4) + (int)col;
        // std::cout << "idx " << idx << std::endl;
        ap_uint<4> mapped = SP[i][idx];
        // std::cout << "mapped " << (int)mapped << std::endl;

        out[outStart + 0] = mapped[3];
        out[outStart + 1] = mapped[2];
        out[outStart + 2] = mapped[1];
        out[outStart + 3] = mapped[0];

        start += 6;
        outStart += 4;
    }
}

static void perm(ap_uint<32> in, ap_uint<32>& out) {
    for (int i = 0; i < 32; ++i) {
#pragma HLS unroll
        out[i] = in[FPermMap[i] - 1];
    }
}

// Feistel function
static ap_uint<32> f(ap_uint<32> hb, ap_uint<48> subkey) {
    // half block expansion
    ap_uint<48> ehb;
    extend(hb, ehb);

    // key mixing
    ehb = ehb ^ subkey;

    // substitution
    ap_uint<32> prePerm;
    substitute(ehb, prePerm); // speed up

    // permutation
    ap_uint<32> result;
    perm(prePerm, result);

    return result;
}

// Initial permutation
static void initPerm(ap_uint<64> in, ap_uint<64>& out) {
    for (int i = 0; i < 64; ++i) {
#pragma HLS unroll
        out[i] = in[PermMap[i] - 1];
    }
}

// Final permutation
static void finalPerm(ap_uint<64> in, ap_uint<64>& out) {
    for (int i = 0; i < 64; ++i) {
#pragma HLS unroll
        int idx = PermMap[i] - 1; // PermMap range 1 - 64
        out[idx] = in[i];
    }
}

static void process(ap_uint<32>& l, ap_uint<32>& r, ap_uint<48> subkeys, bool enc) {
    const int IterNum = 16;
Loop_Round:
    for (int i = 0; i < IterNum; ++i) {
#pragma HLS pipeline
        int keyId = enc ? i : 15 - i;
        ap_uint<32> rr = l ^ f(r, subkeys[keyId]);

        // Update L and R, the order matters
        l = r;
        r = rr;
    }
}

static void keyIndexSchedule(ap_uint<64> key, int subkeyIndex[16][48]) {
    int cIndexArray[28];
    int dIndexArray[28];

    initSubkeyIndex(cIndexArray, dIndexArray);

    ap_uint<28> subkeyC, subkeyD;
    for (int i = 0; i < 28; ++i) {
        subkeyC[i] = key[cIndexArray[i]];
        subkeyD[i] = key[dIndexArray[i]];
    }

    for (int i = 0; i < 16; ++i) {
        getSubkeyIndex(i, cIndexArray, dIndexArray, subkeyIndex[i]);

#ifndef __SYNTHESIS__
        printArray(subkeyIndex[i], 48, 12);
        if (i != 15) {
            std::cout << ",\n";
        }
#endif
    }
}

static void keyScheduleOriginal(ap_uint<64> key, ap_uint<48> subkeys[16]) {
    ap_uint<28> subkeyC, subkeyD;
    initSubkeyCD(key, subkeyC, subkeyD);

    for (int i = 0; i < 16; ++i) {
#pragma HLS pipeline
        getSubkey(i, subkeyC, subkeyD, subkeys[i]);
    }
}

/**
 * @brief keySchedule is to schedule subkeys used in DES and 3DES
 *
 * @param key input original key, 64 bits.
 * @param subkeys output subkeys in encryption or decryption.
 *
 */
static void keySchedule(ap_uint<64> key, ap_uint<48> subkeys[16]) {
    for (int i = 0; i < 16; ++i) {
#pragma HLS unroll
        for (int j = 0; j < 48; ++j) {
#pragma HLS unroll
            int index = subkeyIndex[i][j];
            subkeys[i][j] = key[index];
        }
    }
}

} // namespace internal

/**
 * @brief desEncrypt is the basic function for ciphering one block with
 * one cipher key using DES
 *
 * @param in input one plain text to be encrypted, 64 bits.
 * @param key input cipher key, 64 bits.
 * @param out output encrypted text, 64 bits.
 *
 */
static void desEncrypt(ap_uint<64> in, ap_uint<64> key, ap_uint<64>& out) {
#pragma HLS pipeline
    using namespace internal;
    ap_uint<64> block;
    convert(in, block);

    ap_uint<64> cKey;
    convert(key, cKey);

    ap_uint<48> subkeys[16];
    keySchedule(cKey, subkeys);

    // ap_uint<28> subkeyC, subkeyD;
    // initSubkeyCD(cKey, subkeyC, subkeyD);

    // Initital permutation
    ap_uint<64> data;
    initPerm(block, data);

    // Stated in FIPS 46, the left most bit of a block is bit one
    ap_uint<32> l = data(31, 0);
    ap_uint<32> r = data(63, 32);

    // 16 rounds of processing
    const int IterNum = 16;
Loop_Round:
    for (int i = 0; i < IterNum; ++i) {
#pragma HLS pipeline
        // ap_uint<48> subkey;
        // getSubkey(i, subkeyC, subkeyD, subkey);

        ap_uint<32> rr = l ^ f(r, subkeys[i]);
        // ap_uint<32> rr = l ^ f(r, subkey);

        // Update L and R, the order matters
        l = r;
        r = rr;
    }

    ap_uint<64> prePerm;
    prePerm(31, 0) = r;
    prePerm(63, 32) = l;

    // Final permutation
    ap_uint<64> result;
    finalPerm(prePerm, result);

    convert(result, out);
}

/**
 * @brief desDecrypt is the basic function for decrypt one block with
 * one cipher key using DES
 *
 * @param in input one encrypted text to be decrypted, 64 bits.
 * @param cipherKey input cipher key, 64 bits.
 * @param out output decrypted text, 64 bits.
 *
 */
static void desDecrypt(ap_uint<64> in, ap_uint<64> cipherKey, ap_uint<64>& out) {
#pragma HLS pipeline
    using namespace internal;
    ap_uint<64> block;
    convert(in, block);

    ap_uint<64> cKey;
    convert(cipherKey, cKey);

    // ap_uint<48> subkeyArr[16];
    // keySchedule(cKey, subkeyArr);

    ap_uint<48> subkeys[16];
    // int subkeyIndex[16][48];
    // keyIndexSchedule(cKey, subkeyIndex);
    keySchedule(cKey, subkeys);

    // Initital permutation
    ap_uint<64> data;
    initPerm(block, data);

    // Stated in FIPS 46, the left most bit of a block is bit one
    ap_uint<32> l = data(31, 0);
    ap_uint<32> r = data(63, 32);

    // 16 rounds of processing
    const int IterNum = 16;
Loop_Round:
    for (int i = 0; i < IterNum; ++i) {
#pragma HLS pipeline
        int keyId = 15 - i;
        ap_uint<32> rr = l ^ f(r, subkeys[keyId]);

        // Update L and R, the order matters
        l = r;
        r = rr;
    }

    ap_uint<64> prePerm;
    prePerm(31, 0) = r;
    prePerm(63, 32) = l;

    // Final permutation
    ap_uint<64> result;
    finalPerm(prePerm, result);

    convert(result, out);
}

/**
 * @brief des3Encrypt is the basic function for ciphering one block with
 * three cipher keys using 3DES
 *
 * @param in input one plain text to be encrypted, 64 bits.
 * @param key1 input cipher key, 64 bits.
 * @param key2 input cipher key, 64 bits.
 * @param key3 input cipher key, 64 bits.
 * @param out output encrypted text, 64 bits.
 *
 */
static void des3Encrypt(ap_uint<64> in, ap_uint<64> key1, ap_uint<64> key2, ap_uint<64> key3, ap_uint<64>& out) {
#pragma HLS pipeline
    ap_uint<64> dat1, dat2;

    desEncrypt(in, key1, dat1);
    desDecrypt(dat1, key2, dat2);
    desEncrypt(dat2, key3, out);
}

/**
 * @brief des3Decrypt is the basic function for decrypt one block with
 * three cipher keys using 3DES
 *
 * @param in input one encrypted text to be decrypted, 64 bits.
 * @param key1 input cipher key, 64 bits.
 * @param key2 input cipher key, 64 bits.
 * @param key3 input cipher key, 64 bits.
 * @param out output decrypted text, 64 bits.
 *
 */
static void des3Decrypt(ap_uint<64> in, ap_uint<64> key1, ap_uint<64> key2, ap_uint<64> key3, ap_uint<64>& out) {
#pragma HLS pipeline
    ap_uint<64> dat1, dat2;

    desDecrypt(in, key3, dat1);
    desEncrypt(dat1, key2, dat2);
    desDecrypt(dat2, key1, out);
}

/*
static void desEncrypt2(ap_uint<64> in, ap_uint<64> key, ap_uint<64>& out, bool enc) {
#pragma HLS pipeline
    ap_uint<48> subkeys[16];
    keySchedule(key, subkeys);

    // Initital permutation
    ap_uint<64> data;
    initPerm(in, data);

    // Stated in FIPS 46, the left most bit of a block is bit one
    ap_uint<32> l = data(31, 0);
    ap_uint<32> r = data(63, 32);

    // 16 rounds of processing
    //process(l, r, subkeys, enc);
    const int IterNum = 16;
Loop_Round: for (int i = 0; i < IterNum; ++i) {
#pragma HLS pipeline
        int keyId = enc ? i : 15 - i;
        ap_uint<32> rr = l ^ f(r, subkeys[keyId]);

        // Update L and R, the order matters
        l = r;
        r = rr;
    }

    ap_uint<64> prePerm;
    prePerm(31, 0) = r;
    prePerm(63, 32) = l;

    // Final permutation
    ap_uint<64> result;
    finalPerm(prePerm, result);
}

static void des3EncryptBk(ap_uint<64> in, ap_uint<64> key1, ap_uint<64> key2, ap_uint<64> key3, ap_uint<64>& out) {
#pragma HLS pipeline
    ap_uint<64> block;
    convert(in, block);

    ap_uint<64> cKey1, cKey2, cKey3;
    convert(key1, cKey1);
    convert(key2, cKey2);
    convert(key3, cKey3);

    ap_uint<64> dat1, dat2, dat3;

    desEncrypt2(block, cKey1, dat1, true);
    desEncrypt2(dat1, cKey2, dat2, false);
    desEncrypt2(dat2, cKey3, dat3, true);

    convert(dat3, out);
}

static void des3DecryptBk(ap_uint<64> in, ap_uint<64> key1, ap_uint<64> key2, ap_uint<64> key3, ap_uint<64>& out) {
#pragma HLS pipeline
    ap_uint<64> block;
    convert(in, block);

    ap_uint<64> cKey1, cKey2, cKey3;
    convert(key1, cKey1);
    convert(key2, cKey2);
    convert(key3, cKey3);

    ap_uint<64> dat1, dat2, dat3;

    desEncrypt2(block, cKey3, dat1, false);
    desEncrypt2(dat1, cKey2, dat2, true);
    desEncrypt2(dat2, cKey1, dat3, false);

    convert(dat3, out);
}
*/

} // namespace security
} // namespace xf

#endif
