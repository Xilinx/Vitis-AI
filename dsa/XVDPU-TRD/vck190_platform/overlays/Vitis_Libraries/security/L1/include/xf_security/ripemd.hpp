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
 * @file ripmd160.hpp
 * @brief header file for RIPEMD160.
 * This file part of Vitis Security Library.
 *
 */

#ifndef _XF_SECURITY_RIPEMD160_HPP_
#define _XF_SECURITY_RIPEMD160_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include "xf_security/md4.hpp"
#if !defined(__SYNTHESIS__)
#include <iostream>
#endif

#define RIPEMD_ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

template <int N>
ap_uint<32> rotl(ap_uint<32> x) {
#pragma HLS inline
    return (x << N) | (x >> (32 - N));
}

namespace xf {
namespace security {
namespace internal {

template <int N>
ap_uint<32> f(ap_uint<32> x, ap_uint<32> y, ap_uint<32> z);

template <>
ap_uint<32> f<0>(ap_uint<32> x, ap_uint<32> y, ap_uint<32> z) {
#pragma HLS inline
    return x ^ y ^ z;
}

template <>
ap_uint<32> f<1>(ap_uint<32> x, ap_uint<32> y, ap_uint<32> z) {
#pragma HLS inline
    return (x & y) | ((~x) & z);
}

template <>
ap_uint<32> f<2>(ap_uint<32> x, ap_uint<32> y, ap_uint<32> z) {
#pragma HLS inline
    return (x | (~y)) ^ z;
}

template <>
ap_uint<32> f<3>(ap_uint<32> x, ap_uint<32> y, ap_uint<32> z) {
#pragma HLS inline
    return (x & z) | (y & (~z));
}

template <>
ap_uint<32> f<4>(ap_uint<32> x, ap_uint<32> y, ap_uint<32> z) {
#pragma HLS inline
    return x ^ (y | (~z));
}

template <int N>
void compress(ap_uint<32>& aLeft,
              ap_uint<32>& bLeft,
              ap_uint<32>& cLeft,
              ap_uint<32>& dLeft,
              ap_uint<32>& eLeft,
              ap_uint<32>& aRight,
              ap_uint<32>& bRight,
              ap_uint<32>& cRight,
              ap_uint<32>& dRight,
              ap_uint<32>& eRight,
              const blockType& block,
              const int pickLeft,
              const int pickRight,
              const int shiftLeft,
              const int shiftRight,
              const ap_uint<32> kLeft,
              const ap_uint<32> kRight) {
#pragma HLS inline
    ap_uint<32> tLeft = aLeft + f<N>(bLeft, cLeft, dLeft) + block.M[pickLeft] + kLeft;
    tLeft = RIPEMD_ROTL(tLeft, shiftLeft) + eLeft;
    aLeft = eLeft;
    eLeft = dLeft;
    dLeft = RIPEMD_ROTL(cLeft, 10);
    cLeft = bLeft;
    bLeft = tLeft;

    ap_uint<32> tRight = aRight + f<4 - N>(bRight, cRight, dRight) + block.M[pickRight] + kRight;
    tRight = RIPEMD_ROTL(tRight, shiftRight) + eRight;
    aRight = eRight;
    eRight = dRight;
    dRight = RIPEMD_ROTL(cRight, 10);
    cRight = bRight;
    bRight = tRight;
}

static void ripemd160Digest(hls::stream<blockType>& blockStrm,
                            hls::stream<ap_uint<64> >& nBlockStrm,
                            hls::stream<bool>& endNBlockStrm,
                            hls::stream<ap_uint<160> >& outStrm,
                            hls::stream<bool>& endOutStrm) {
    const ap_uint<32> kLeft[5] = {0x00000000UL, 0x5A827999UL, 0x6ED9EBA1UL, 0X8F1BBCDCUL, 0XA953FD4EUL};
#pragma HLS array_partition variable = kLeft dim = 0

    const ap_uint<32> kRight[5] = {0x50A28BE6UL, 0x5C4DD124UL, 0x6D703EF3UL, 0x7A6D76E9UL, 0x00000000UL};
#pragma HLS array_partition variable = kRight dim = 0

    const int rLeft[5][16] = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},  // Round 1
                              {7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8},  // Round 2
                              {3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12},  // Round 3
                              {1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2},  // Round 4
                              {4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13}}; // Round 5
#pragma HLS array_partition variable = rLeft dim = 0

    const int rRight[5][16] = {{5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12},  // Round 1
                               {6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2},  // Round 2
                               {15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13},  // Round 3
                               {8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14},  // Round 4
                               {12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11}}; // Round 5
#pragma HLS array_partition variable = rRight dim = 0

    const int sLeft[5][16] = {{11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8},  // Round 1
                              {7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12},  // Round 2
                              {11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5},  // Round 3
                              {11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12},  // Round 4
                              {9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6}}; // Round 5
#pragma HLS array_partition variable = sLeft dim = 0

    const int sRight[5][16] = {{8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6},  // Round 1
                               {9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11},  // Round 2
                               {9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5},  // Round 3
                               {15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8},  // Round 4
                               {8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11}}; // Round 5
#pragma HLS array_partition variable = sRight dim = 0

    while (!endNBlockStrm.read()) {
        ap_uint<32> h0 = 0x67452301UL;
        ap_uint<32> h1 = 0xEFCDAB89UL;
        ap_uint<32> h2 = 0x98BADCFEUL;
        ap_uint<32> h3 = 0x10325476UL;
        ap_uint<32> h4 = 0xC3D2E1F0UL;

        ap_uint<64> blockNum = nBlockStrm.read();

        for (int n = 0; n < blockNum; n++) {
#pragma HLS pipeline
            blockType block = blockStrm.read();

            ap_uint<32> aLeft = h0;
            ap_uint<32> bLeft = h1;
            ap_uint<32> cLeft = h2;
            ap_uint<32> dLeft = h3;
            ap_uint<32> eLeft = h4;

            ap_uint<32> aRight = h0;
            ap_uint<32> bRight = h1;
            ap_uint<32> cRight = h2;
            ap_uint<32> dRight = h3;
            ap_uint<32> eRight = h4;

            // Round 1
            for (int j = 0; j < 16; j++) { // sub iters
#pragma HLS unroll
                compress<0>(aLeft, bLeft, cLeft, dLeft, eLeft, aRight, bRight, cRight, dRight, eRight, block,
                            rLeft[0][j], rRight[0][j], sLeft[0][j], sRight[0][j], kLeft[0], kRight[0]);
            }
            // Round 1
            for (int j = 0; j < 16; j++) { // sub iters
#pragma HLS unroll
                compress<1>(aLeft, bLeft, cLeft, dLeft, eLeft, aRight, bRight, cRight, dRight, eRight, block,
                            rLeft[1][j], rRight[1][j], sLeft[1][j], sRight[1][j], kLeft[1], kRight[1]);
            }
            // Round 2
            for (int j = 0; j < 16; j++) { // sub iters
#pragma HLS unroll
                compress<2>(aLeft, bLeft, cLeft, dLeft, eLeft, aRight, bRight, cRight, dRight, eRight, block,
                            rLeft[2][j], rRight[2][j], sLeft[2][j], sRight[2][j], kLeft[2], kRight[2]);
            }
            // Round 3
            for (int j = 0; j < 16; j++) { // sub iters
#pragma HLS unroll
                compress<3>(aLeft, bLeft, cLeft, dLeft, eLeft, aRight, bRight, cRight, dRight, eRight, block,
                            rLeft[3][j], rRight[3][j], sLeft[3][j], sRight[3][j], kLeft[3], kRight[3]);
            }
            // Round 4
            for (int j = 0; j < 16; j++) { // sub iters
#pragma HLS unroll
                compress<4>(aLeft, bLeft, cLeft, dLeft, eLeft, aRight, bRight, cRight, dRight, eRight, block,
                            rLeft[4][j], rRight[4][j], sLeft[4][j], sRight[4][j], kLeft[4], kRight[4]);
            }

            ap_uint<32> tt = h1 + cLeft + dRight;
            h1 = h2 + dLeft + eRight;
            h2 = h3 + eLeft + aRight;
            h3 = h4 + aLeft + bRight;
            h4 = h0 + bLeft + cRight;
            h0 = tt;
        }

        ap_uint<160> res;
        res.range(31, 0) = h0;
        res.range(63, 32) = h1;
        res.range(95, 64) = h2;
        res.range(127, 96) = h3;
        res.range(159, 128) = h4;

        outStrm.write(res);
        endOutStrm.write(false);
    }
    endOutStrm.write(true);
}
} // namespace internal

static void ripemd160(hls::stream<ap_uint<32> >& inStrm,
                      hls::stream<ap_uint<64> >& inLenStrm,
                      hls::stream<bool>& endInLenStrm,
                      hls::stream<ap_uint<160> >& outStrm,
                      hls::stream<bool>& endOutStrm) {
#pragma HLS dataflow

    hls::stream<internal::blockType> blockStrm;
#pragma HLS stream variable = blockStrm depth = 2

    hls::stream<ap_uint<64> > nBlockStrm;
#pragma HLS stream variable = nBlockStrm depth = 2

    hls::stream<bool> endNBlockStrm;
#pragma HLS variable = endNBlockStrm depth = 2

    internal::preProcessing(inStrm, inLenStrm, endInLenStrm, blockStrm, nBlockStrm, endNBlockStrm);

    internal::ripemd160Digest(blockStrm, nBlockStrm, endNBlockStrm, outStrm, endOutStrm);
}

} // end of namespace security
} // end of namespace xf
#endif // _XF_SECURITY_RIPEMD160_HPP_
