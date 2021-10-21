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
 *
 * @file md5.hpp
 * @brief header file for MD5 related functions, including pre-processing and digest functions.
 * This file is part of Vitis Security Library.
 *
 * @detail The algorithm takes a message of arbitrary length as its input,
 * and produces a 128-bit "fingerprint/message digest".
 * Notice that the 16 operations which defined in round 1, 2, 3, and 4 respectively in the standard have dependencies.
 * Therefore, the MD5 digest process cannot achieve an II = 1.
 *
 */

#ifndef _XF_SECURITY_MD5_HPP_
#define _XF_SECURITY_MD5_HPP_

#include <ap_int.h>
#include <hls_stream.h>

// for debug
#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace security {
namespace internal {

// @brief Processing block
struct blockType {
    ap_uint<32> M[16];
};

/**
 * @brief Generate 512-bit processing blocks by padding and appending (pipeline).
 *
 * The algorithm reference is : "The MD5 Message-Digest Algorithm".
 * The optimization goal of this function is to yield a 512-bit block per cycle.
 *
 * @param msg_strm The message being hashed.
 * @param len_strm The message length in byte.
 * @param end_len_strm The flag to signal end of input message stream.
 * @param blk_strm The 512-bit hash block.
 * @param nblk_strm The number of hash block for this message.
 * @param end_nblk_strm End flag for number of hash block.
 *
 */

static void preProcessing(
    // inputs
    hls::stream<ap_uint<32> >& msg_strm,
    hls::stream<ap_uint<64> >& len_strm,
    hls::stream<bool>& end_len_strm,
    // outputs
    hls::stream<blockType>& blk_strm,
    hls::stream<ap_uint<64> >& nblk_strm,
    hls::stream<bool>& end_nblk_strm) {
    bool endFlag = end_len_strm.read();

LOOP_PREPROCESSING_MAIN:
    while (!endFlag) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        // read message length in byte
        ap_uint<64> len = len_strm.read();

        // prepare message length in bit which will be appended at the tail of the block according to the standard
        ap_uint<64> L = 8 * len;

        // total number blocks to digest in 512-bit
        ap_uint<64> blk_num = (len >> 6) + 1 + ((len & 0x3f) > 55);

        // inform digest function
        nblk_strm.write(blk_num);
        end_nblk_strm.write(false);

    LOOP_GEN_FULL_BLKS:
        for (ap_uint<64> j = 0; j < (ap_uint<64>)(len >> 6); ++j) {
#pragma HLS pipeline II = 16
#pragma HLS loop_tripcount min = 0 max = 1
            // message block
            blockType b0;
#pragma HLS array_partition variable = b0.M complete

        // this block will hold 16 words (32-bit for each) of message
        LOOP_GEN_ONE_FULL_BLK:
            for (ap_uint<5> i = 0; i < 16; ++i) {
#pragma HLS unroll
                // XXX algorithm assumes little-endian
                b0.M[i] = msg_strm.read();
            }

            // send the full block
            blk_strm.write(b0);
        }

        // number of bytes left which needs to be padded as a new full block
        ap_uint<6> left = (ap_uint<6>)(len & 0x3fUL);

        if (left == 0) {
            // end at block boundary, start with pad 1
            // last block
            blockType b;
#pragma HLS array_partition variable = b.M complete

            // pad 1
            b.M[0] = 0x00000080UL;

        // pad zero words
        LOOP_PAD_13_ZERO_WORDS:
            for (ap_uint<5> i = 1; i < 14; ++i) {
#pragma HLS unroll
                b.M[i] = 0;
            }

            // append L (low-order word first)
            b.M[14] = (ap_uint<32>)(0xffffffffUL & (L));
            b.M[15] = (ap_uint<32>)(0xffffffffUL & (L >> 32));

            // emit
            blk_strm.write(b);
        } else if (left < 56) {
            // can pad 1 and append L in current block
            // last block
            blockType b;
#pragma HLS array_partition variable = b.M complete

        LOOP_COPY_TAIL_AND_PAD_1:
            for (ap_uint<5> i = 0; i < 14; ++i) {
#pragma HLS pipeline
                if (i < (left >> 2)) {
                    // pad 1 byte not in this word
                    // XXX algorithm assumes little-endian
                    b.M[i] = msg_strm.read();
                } else if (i > (left >> 2)) {
                    // pad 1 byte not in this word, and no word to read
                    b.M[i] = 0UL;
                } else {
                    // pad 1 byte in this word
                    ap_uint<2> e = left & 0x3UL;
                    if (e == 0) {
                        // contains no message byte
                        b.M[i] = 0x00000080UL;
                    } else if (e == 1) {
                        // contains 1 message byte
                        ap_uint<32> l = msg_strm.read();
                        // XXX algorithm assumes little-endian
                        l = 0x000000ffUL & l;
                        b.M[i] = l | 0x00008000UL;
                    } else if (e == 2) {
                        // contains 2 message bytes
                        ap_uint<32> l = msg_strm.read();
                        // XXX algorithm assumes little-endian
                        l = 0x0000ffffUL & l;
                        b.M[i] = l | 0x00800000UL;
                    } else {
                        // contains 3 message bytes
                        ap_uint<32> l = msg_strm.read();
                        // XXX algorithm assumes little-endian
                        l = 0x00ffffffUL & l;
                        b.M[i] = l | 0x80000000UL;
                    }
                }
            }

            // append L (low-order word first)
            b.M[14] = (ap_uint<32>)(0xffffffffUL & (L));
            b.M[15] = (ap_uint<32>)(0xffffffffUL & (L >> 32));

            blk_strm.write(b);
        } else {
            // cannot append L
            // second-to-last block
            blockType b;
#pragma HLS array_partition variable = b.M complete

        // copy and pad 1
        LOOP_COPY_AND_PAD_1:
            for (ap_uint<5> i = 0; i < 16; ++i) {
#pragma HLS unroll
                if (i < (left >> 2)) { // pad 1 byte not in this word
                    // XXX algorithm assumes little-endian
                    b.M[i] = msg_strm.read();
                } else if (i > (left >> 2)) { // pad 1 byte not in this word, and no msg word to read
                    b.M[i] = 0UL;
                } else { // pad 1 byte in this word
                    ap_uint<2> e = left & 0x3UL;
                    if (e == 0) {
                        b.M[i] = 0x00000080UL;
                    } else if (e == 1) {
                        ap_uint<32> l = msg_strm.read();
                        // XXX algorithm assumes little-endian
                        l = 0x000000ffUL & l;
                        b.M[i] = l | 0x00008000UL;
                    } else if (e == 2) {
                        ap_uint<32> l = msg_strm.read();
                        // XXX algorithm assumes little-endian
                        l = 0x0000ffffUL & l;
                        b.M[i] = l | 0x00800000UL;
                    } else {
                        ap_uint<32> l = msg_strm.read();
                        // XXX algorithm assumes little-endian
                        l = 0x00ffffffUL & l;
                        b.M[i] = l | 0x80000000UL;
                    }
                }
            }

            // emit second-to-last block
            blk_strm.write(b);

            // last block
            blockType b1;
#pragma HLS array_partition variable = b1.M complete

        LOOP_PAD_14_ZERO_WORDS:
            for (ap_uint<5> i = 0; i < 14; ++i) {
#pragma HLS unroll
                b1.M[i] = 0;
            }

            // append L (low-order word first)
            b1.M[14] = (ap_uint<32>)(0xffffffffUL & (L));
            b1.M[15] = (ap_uint<32>)(0xffffffffUL & (L >> 32));

            // emit last block
            blk_strm.write(b1);
        }

        // still have message to handle
        endFlag = end_len_strm.read();
    }

    end_nblk_strm.write(true);

} // end preProcessing

/**
 * @brief Perform function F as defined in standard.
 *
 * The algorithm reference is : "The MD5 Message-Digest Algorithm".
 *
 * @param x The first 32-bit operand.
 * @param y The second 32-bit operand.
 * @param z The third 32-bit operand.
 *
 */

static ap_uint<32> F(
    // inputs
    ap_uint<32> x,
    ap_uint<32> y,
    ap_uint<32> z) {
#pragma HLS inline

    return ((x & y) | ((~x) & z));

} // end F

/**
 * @brief Perform function G as defined in standard.
 *
 * The algorithm reference is : "The MD5 Message-Digest Algorithm".
 *
 * @param x The first 32-bit operand.
 * @param y The second 32-bit operand.
 * @param z The third 32-bit operand.
 *
 */

static ap_uint<32> G(
    // inputs
    ap_uint<32> x,
    ap_uint<32> y,
    ap_uint<32> z) {
#pragma HLS inline

    return ((x & z) | (y & (~z)));

} // end G

/**
 * @brief Perform function H as defined in standard.
 *
 * The algorithm reference is : "The MD5 Message-Digest Algorithm".
 *
 * @param x The first 32-bit operand.
 * @param y The second 32-bit operand.
 * @param z The third 32-bit operand.
 *
 */

static ap_uint<32> H(
    // inputs
    ap_uint<32> x,
    ap_uint<32> y,
    ap_uint<32> z) {
#pragma HLS inline

    return (x ^ y ^ z);

} // end H

/**
 * @brief Perform function I as defined in standard.
 *
 * The algorithm reference is : "The MD5 Message-Digest Algorithm".
 *
 * @param x The first 32-bit operand.
 * @param y The second 32-bit operand.
 * @param z The third 32-bit operand.
 *
 */

static ap_uint<32> I(
    // inputs
    ap_uint<32> x,
    ap_uint<32> y,
    ap_uint<32> z) {
#pragma HLS inline

    return (y ^ (x | (~z)));

} // end I

/**
 *
 * @brief The implementation of rotate left (circular left shift) operation.
 *
 * The algorithm reference is : "The MD5 Message-Digest Algorithm".
 *
 * @tparam w The bit width of input x, default value is 32.
 *
 * @param n Number of bits for input x to be shifted.
 * @param x Word to be rotated.
 *
 */

template <unsigned int w>
ap_uint<w> ROTL(
    // inputs
    unsigned int n,
    ap_uint<w> x) {
#pragma HLS inline

    return ((x << n) | (x >> (w - n)));

} // end ROTL

/**
 *
 * @brief The implementation of the function defined in round 1.
 *
 * The algorithm reference is : "The MD5 Message-Digest Algorithm".
 * The operation is defined as : a = b + ((a + F(b, c, d) + X + T) <<< s).
 *
 * @param a The first 32-bit operand.
 * @param b The second 32-bit operand.
 * @param c The third 32-bit operand.
 * @param d The fourth 32-bit operand.
 * @param X The specific message word.
 * @param T the specific sine value.
 * @param s Number of bits to be shifted.
 *
 */

static void MD5Round1(
    // inputs
    ap_uint<32>& a,
    ap_uint<32> b,
    ap_uint<32> c,
    ap_uint<32> d,
    ap_uint<32> X,
    ap_uint<32> T,
    unsigned int s) {
    // perform the computation
    ap_uint<32> tmp = F(b, c, d);
    tmp = a + tmp + X + T;
    tmp = ROTL<32>(s, tmp);
    a = b + tmp;

} // end MD5Round1

/**
 *
 * @brief The implementation of the function defined in round 2.
 *
 * The algorithm reference is : "The MD5 Message-Digest Algorithm".
 * The operation is defined as : a = b + ((a + G(b, c, d) + X + T) <<< s).
 *
 * @param a The first 32-bit operand.
 * @param b The second 32-bit operand.
 * @param c The third 32-bit operand.
 * @param d The fourth 32-bit operand.
 * @param X The specific message word.
 * @param T the specific sine value.
 * @param s Number of bits to be shifted.
 *
 */

static void MD5Round2(
    // inputs
    ap_uint<32>& a,
    ap_uint<32> b,
    ap_uint<32> c,
    ap_uint<32> d,
    ap_uint<32> X,
    ap_uint<32> T,
    unsigned int s) {
    // perform the computation
    ap_uint<32> tmp = G(b, c, d);
    tmp = a + tmp + X + T;
    tmp = ROTL<32>(s, tmp);
    a = b + tmp;

} // end MD5Round2

/**
 *
 * @brief The implementation of the function defined in round 3.
 *
 * The algorithm reference is : "The MD5 Message-Digest Algorithm".
 * The operation is defined as : a = b + ((a + H(b, c, d) + X + T) <<< s).
 *
 * @param a The first 32-bit operand.
 * @param b The second 32-bit operand.
 * @param c The third 32-bit operand.
 * @param d The fourth 32-bit operand.
 * @param X The specific message word.
 * @param T the specific sine value.
 * @param s Number of bits to be shifted.
 *
 */

static void MD5Round3(
    // inputs
    ap_uint<32>& a,
    ap_uint<32> b,
    ap_uint<32> c,
    ap_uint<32> d,
    ap_uint<32> X,
    ap_uint<32> T,
    unsigned int s) {
    // perform the computation
    ap_uint<32> tmp = H(b, c, d);
    tmp = a + tmp + X + T;
    tmp = ROTL<32>(s, tmp);
    a = b + tmp;

} // end MD5Round3

/**
 *
 * @brief The implementation of the function defined in round 4.
 *
 * The algorithm reference is : "The MD5 Message-Digest Algorithm".
 * The operation is defined as : a = b + ((a + I(b, c, d) + X + T) <<< s).
 *
 * @param a The first 32-bit operand.
 * @param b The second 32-bit operand.
 * @param c The third 32-bit operand.
 * @param d The fourth 32-bit operand.
 * @param X The specific message word.
 * @param T the specific sine value.
 * @param s Number of bits to be shifted.
 *
 */

static void MD5Round4(
    // inputs
    ap_uint<32>& a,
    ap_uint<32> b,
    ap_uint<32> c,
    ap_uint<32> d,
    ap_uint<32> X,
    ap_uint<32> T,
    unsigned int s) {
    // perform the computation
    ap_uint<32> tmp = I(b, c, d);
    tmp = a + tmp + X + T;
    tmp = ROTL<32>(s, tmp);
    a = b + tmp;

} // end MD5Round4

/**
 * @brief The implementation of the digest part of MD5.
 *
 * The algorithm reference is : "The MD5 Message-Digest Algorithm".
 * The optimization goal of this function is for better performance.
 *
 * @param blk_strm The 512-bit hash block.
 * @param nblk_strm The number of hash block for this message.
 * @param end_nblk_strm End flag for number of hash block.
 * @param digest_strm The digest (fingerprint) stream.
 * @param end_digest_strm Flag to signal the end of the result.
 *
 */

static void MD5Digest(
    // inputs
    hls::stream<blockType>& blk_strm,
    hls::stream<ap_uint<64> >& nblk_strm,
    hls::stream<bool>& end_nblk_strm,
    // ouputs
    hls::stream<ap_uint<128> >& digest_strm,
    hls::stream<bool>& end_digest_strm) {
    // table of sine values
    ap_uint<32> T[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
#pragma HLS resource variable = T core = RAM_2P_LUTRAM

    bool endFlag = end_nblk_strm.read();

LOOP_MD5_MAIN:
    while (!endFlag) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        // Initialize the MD Buffer in low-order
        ap_uint<32> A = 0x67452301UL; // word A
        ap_uint<32> B = 0xefcdab89UL; // word B
        ap_uint<32> C = 0x98badcfeUL; // word C
        ap_uint<32> D = 0x10325476UL; // word D

        // total number blocks to digest
        ap_uint<64> blkNum = nblk_strm.read();

    LOOP_MD5_DIGEST_NBLK:
        for (ap_uint<64> n = 0; n < blkNum; n++) {
#pragma HLS loop_tripcount min = 10 max = 10 avg = 10
            // input block
            blockType blk = blk_strm.read();
#ifndef __SYNTHESIS__
            for (unsigned int i = 0; i < 16; i++) {
                std::cout << "M[" << i << "] = " << std::hex << blk.M[i] << std::endl;
            }
#endif

            // save original value of A, B, C, and D
            ap_uint<32> AA = A;
            ap_uint<32> BB = B;
            ap_uint<32> CC = C;
            ap_uint<32> DD = D;

        // round 1
        LOOP_MD5_DIGEST_ROUND1:
            for (ap_uint<3> i = 0; i < 4; i++) {
#pragma HLS pipeline off
                MD5Round1(A, B, C, D, blk.M[i * 4], T[i * 4], 7);
                MD5Round1(D, A, B, C, blk.M[i * 4 + 1], T[i * 4 + 1], 12);
                MD5Round1(C, D, A, B, blk.M[i * 4 + 2], T[i * 4 + 2], 17);
                MD5Round1(B, C, D, A, blk.M[i * 4 + 3], T[i * 4 + 3], 22);
            }
#ifndef __SYNTHESIS__
            std::cout << "Round 1 : a = " << std::hex << A << ", b = " << B << ", c = " << C << ", d = " << D
                      << std::endl;
#endif

        // round 2
        LOOP_MD5_DIGEST_ROUND2:
            for (ap_uint<3> i = 0; i < 4; i++) {
#pragma HLS pipeline off
                MD5Round2(A, B, C, D, blk.M[(i * 4 + 1) % 16], T[i * 4 + 16], 5);
                MD5Round2(D, A, B, C, blk.M[(i * 4 + 6) % 16], T[i * 4 + 17], 9);
                MD5Round2(C, D, A, B, blk.M[(i * 4 + 11) % 16], T[i * 4 + 18], 14);
                MD5Round2(B, C, D, A, blk.M[(i * 4) % 16], T[i * 4 + 19], 20);
            }
#ifndef __SYNTHESIS__
            std::cout << "Round 2 : a = " << std::hex << A << ", b = " << B << ", c = " << C << ", d = " << D
                      << std::endl;
#endif

        // round 3
        LOOP_MD5_DIGEST_ROUND3:
            for (ap_uint<3> i = 0; i < 4; i++) {
#pragma HLS pipeline off
                MD5Round3(A, B, C, D, blk.M[(i * 12 + 5) % 16], T[i * 4 + 32], 4);
                MD5Round3(D, A, B, C, blk.M[(i * 12 + 8) % 16], T[i * 4 + 33], 11);
                MD5Round3(C, D, A, B, blk.M[(i * 12 + 11) % 16], T[i * 4 + 34], 16);
                MD5Round3(B, C, D, A, blk.M[(i * 12 + 14) % 16], T[i * 4 + 35], 23);
            }
#ifndef __SYNTHESIS__
            std::cout << "Round 3 : a = " << std::hex << A << ", b = " << B << ", c = " << C << ", d = " << D
                      << std::endl;
#endif

        // round 4
        LOOP_MD5_DIGEST_ROUND4:
            for (ap_uint<3> i = 0; i < 4; i++) {
#pragma HLS pipeline off
                MD5Round4(A, B, C, D, blk.M[(i * 12 + 0) % 16], T[i * 4 + 48], 6);
                MD5Round4(D, A, B, C, blk.M[(i * 12 + 7) % 16], T[i * 4 + 49], 10);
                MD5Round4(C, D, A, B, blk.M[(i * 12 + 14) % 16], T[i * 4 + 50], 15);
                MD5Round4(B, C, D, A, blk.M[(i * 12 + 5) % 16], T[i * 4 + 51], 21);
            }
#ifndef __SYNTHESIS__
            std::cout << "Round 4 : a = " << std::hex << A << ", b = " << B << ", c = " << C << ", d = " << D
                      << std::endl;
#endif

            // increment the four working words
            A += AA;
            B += BB;
            C += CC;
            D += DD;
        }

        // emit digest
        ap_uint<128> digest;
        digest.range(127, 96) = D;
        digest.range(95, 64) = C;
        digest.range(63, 32) = B;
        digest.range(31, 0) = A;
        digest_strm.write(digest);
        end_digest_strm.write(false);

        endFlag = end_nblk_strm.read();
    }

    end_digest_strm.write(true);

} // end MD5Digest

} // namespace internal

/**
 * @brief Top function of MD5.
 *
 * The algorithm reference is : "The MD5 Message-Digest Algorithm".
 *
 * @param msg_strm The message being hashed.
 * @param len_strm The message length in byte.
 * @param end_len_strm The flag to signal end of input message stream.
 * @param digest_strm The digest (fingerprint) stream.
 * @param end_digest_strm Flag to signal the end of the result.
 *
 */

static void md5(
    // inputs
    hls::stream<ap_uint<32> >& msg_strm,
    hls::stream<ap_uint<64> >& len_strm,
    hls::stream<bool>& end_len_strm,
    // ouputs
    hls::stream<ap_uint<128> >& digest_strm,
    hls::stream<bool>& end_digest_strm) {
#pragma HLS dataflow

    // 512-bit processing block stream
    hls::stream<internal::blockType> blk_strm;
#pragma HLS stream variable = blk_strm depth = 2

    // number of blocks stream
    hls::stream<ap_uint<64> > nblk_strm;
#pragma HLS stream variable = nblk_strm depth = 2

    // end flag of number of blocks stream
    hls::stream<bool> end_nblk_strm;
#pragma HLS stream variable = end_nblk_strm depth = 2

    // padding and appending message words into blocks
    internal::preProcessing(msg_strm, len_strm, end_len_strm, blk_strm, nblk_strm, end_nblk_strm);

    // digest processing blocks into fingerprint by hash function
    internal::MD5Digest(blk_strm, nblk_strm, end_nblk_strm, digest_strm, end_digest_strm);

} // end md5

} // namespace security
} // namespace xf

#endif
