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
 * @file md4.hpp
 * @brief header file for MD4 related functions, including pre-processing and digest functions.
 * This file is part of Vitis Security Library.
 *
 * @detail The algorithm takes a message of arbitrary length as its input,
 * and produces a 128-bit "fingerprint/message digest".
 * Notice that the 16 operations which defined in round 1, 2, and 3 respectively in the standard have dependencies,
 * and each of them is optimized to an interval = 1. Therefore, considering the overhead,
 * the initiation interval (II) of MD4 digest process for a single processing block is 49 cycles is reasonable.
 *
 */

#ifndef _XF_SECURITY_MD4_HPP_
#define _XF_SECURITY_MD4_HPP_

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
 * The algorithm reference is : "The MD4 Message-Digest Algorithm".
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
 * The algorithm reference is : "The MD4 Message-Digest Algorithm".
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
 * The algorithm reference is : "The MD4 Message-Digest Algorithm".
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

    return ((x & y) | (x & z) | (y & z));

} // end G

/**
 * @brief Perform function H as defined in standard.
 *
 * The algorithm reference is : "The MD4 Message-Digest Algorithm".
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
 *
 * @brief The implementation of rotate left (circular left shift) operation.
 *
 * The algorithm reference is : "The MD4 Message-Digest Algorithm".
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
 * The algorithm reference is : "The MD4 Message-Digest Algorithm".
 * The operation is defined as : a = (a + F(b, c, d) + X[k]) <<< s.
 *
 * @param a The first 32-bit operand.
 * @param b The second 32-bit operand.
 * @param c The third 32-bit operand.
 * @param d The fourth 32-bit operand.
 * @param X The specific message word.
 * @param s Number of bits to be shifted.
 *
 */

static void MD4Round1(
    // inputs
    ap_uint<32>& a,
    ap_uint<32> b,
    ap_uint<32> c,
    ap_uint<32> d,
    ap_uint<32> X,
    unsigned int s) {
    // perform the computation
    ap_uint<32> tmp = F(b, c, d);
    tmp = a + tmp + X;
    a = ROTL<32>(s, tmp);

} // end MD4Round1

/**
 *
 * @brief The implementation of the function defined in round 2.
 *
 * The algorithm reference is : "The MD4 Message-Digest Algorithm".
 * The operation is defined as : a = (a + G(b, c, d) + X[k] + 0x5a827999) <<< s
 *
 * @param a The first 32-bit operand.
 * @param b The second 32-bit operand.
 * @param c The third 32-bit operand.
 * @param d The fourth 32-bit operand.
 * @param X The specific message word.
 * @param s Number of bits to be shifted.
 *
 */

static void MD4Round2(
    // inputs
    ap_uint<32>& a,
    ap_uint<32> b,
    ap_uint<32> c,
    ap_uint<32> d,
    ap_uint<32> X,
    unsigned int s) {
    // perform the computation
    ap_uint<32> tmp = G(b, c, d);
    tmp = a + tmp + X + 0x5a827999;
    a = ROTL<32>(s, tmp);

} // end MD4Round2

/**
 *
 * @brief The implementation of the function defined in round 3.
 *
 * The algorithm reference is : "The MD4 Message-Digest Algorithm".
 * The operation is defined as : a = (a + H(b, c, d) + X[k] + 0x6ed9eba1) <<< s
 *
 * @param a The first 32-bit operand.
 * @param b The second 32-bit operand.
 * @param c The third 32-bit operand.
 * @param d The fourth 32-bit operand.
 * @param X The specific message word.
 * @param s Number of bits to be shifted.
 *
 */

static void MD4Round3(
    // inputs
    ap_uint<32>& a,
    ap_uint<32> b,
    ap_uint<32> c,
    ap_uint<32> d,
    ap_uint<32> X,
    unsigned int s) {
    // perform the computation
    ap_uint<32> tmp = H(b, c, d);
    tmp = a + tmp + X + 0x6ed9eba1;
    a = ROTL<32>(s, tmp);

} // end MD4Round3

/**
 * @brief The implementation of the digest part of MD4.
 *
 * The algorithm reference is : "The MD4 Message-Digest Algorithm".
 * The optimization goal of this function is for better performance.
 *
 * @param blk_strm The 512-bit hash block.
 * @param nblk_strm The number of hash block for this message.
 * @param end_nblk_strm End flag for number of hash block.
 * @param digest_strm The digest (fingerprint) stream.
 * @param end_digest_strm Flag to signal the end of the result.
 *
 */

static void MD4Digest(
    // inputs
    hls::stream<blockType>& blk_strm,
    hls::stream<ap_uint<64> >& nblk_strm,
    hls::stream<bool>& end_nblk_strm,
    // ouputs
    hls::stream<ap_uint<128> >& digest_strm,
    hls::stream<bool>& end_digest_strm) {
    bool endFlag = end_nblk_strm.read();

LOOP_MD4_MAIN:
    while (!endFlag) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        // Initialize the MD Buffer in low-order
        ap_uint<32> A = 0x67452301UL; // word A
        ap_uint<32> B = 0xefcdab89UL; // word B
        ap_uint<32> C = 0x98badcfeUL; // word C
        ap_uint<32> D = 0x10325476UL; // word D

        // total number blocks to digest
        ap_uint<64> blkNum = nblk_strm.read();

    LOOP_MD4_DIGEST_NBLK:
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
            MD4Round1(A, B, C, D, blk.M[0], 3);
#ifndef __SYNTHESIS__
            std::cout << "Round 1 : a = " << std::hex << A << ", b = " << B << ", c = " << C << ", d = " << D
                      << std::endl;
#endif
            MD4Round1(D, A, B, C, blk.M[1], 7);
            MD4Round1(C, D, A, B, blk.M[2], 11);
            MD4Round1(B, C, D, A, blk.M[3], 19);
            MD4Round1(A, B, C, D, blk.M[4], 3);
            MD4Round1(D, A, B, C, blk.M[5], 7);
            MD4Round1(C, D, A, B, blk.M[6], 11);
            MD4Round1(B, C, D, A, blk.M[7], 19);
            MD4Round1(A, B, C, D, blk.M[8], 3);
            MD4Round1(D, A, B, C, blk.M[9], 7);
            MD4Round1(C, D, A, B, blk.M[10], 11);
            MD4Round1(B, C, D, A, blk.M[11], 19);
            MD4Round1(A, B, C, D, blk.M[12], 3);
            MD4Round1(D, A, B, C, blk.M[13], 7);
            MD4Round1(C, D, A, B, blk.M[14], 11);
            MD4Round1(B, C, D, A, blk.M[15], 19);

            // round 2
            MD4Round2(A, B, C, D, blk.M[0], 3);
#ifndef __SYNTHESIS__
            std::cout << "Round 2 : a = " << std::hex << A << ", b = " << B << ", c = " << C << ", d = " << D
                      << std::endl;
#endif
            MD4Round2(D, A, B, C, blk.M[4], 5);
            MD4Round2(C, D, A, B, blk.M[8], 9);
            MD4Round2(B, C, D, A, blk.M[12], 13);
            MD4Round2(A, B, C, D, blk.M[1], 3);
            MD4Round2(D, A, B, C, blk.M[5], 5);
            MD4Round2(C, D, A, B, blk.M[9], 9);
            MD4Round2(B, C, D, A, blk.M[13], 13);
            MD4Round2(A, B, C, D, blk.M[2], 3);
            MD4Round2(D, A, B, C, blk.M[6], 5);
            MD4Round2(C, D, A, B, blk.M[10], 9);
            MD4Round2(B, C, D, A, blk.M[14], 13);
            MD4Round2(A, B, C, D, blk.M[3], 3);
            MD4Round2(D, A, B, C, blk.M[7], 5);
            MD4Round2(C, D, A, B, blk.M[11], 9);
            MD4Round2(B, C, D, A, blk.M[15], 13);

            // round 3
            MD4Round3(A, B, C, D, blk.M[0], 3);
#ifndef __SYNTHESIS__
            std::cout << "Round 3 : a = " << std::hex << A << ", b = " << B << ", c = " << C << ", d = " << D
                      << std::endl;
#endif
            MD4Round3(D, A, B, C, blk.M[8], 9);
            MD4Round3(C, D, A, B, blk.M[4], 11);
            MD4Round3(B, C, D, A, blk.M[12], 15);
            MD4Round3(A, B, C, D, blk.M[2], 3);
            MD4Round3(D, A, B, C, blk.M[10], 9);
            MD4Round3(C, D, A, B, blk.M[6], 11);
            MD4Round3(B, C, D, A, blk.M[14], 15);
            MD4Round3(A, B, C, D, blk.M[1], 3);
            MD4Round3(D, A, B, C, blk.M[9], 9);
            MD4Round3(C, D, A, B, blk.M[5], 11);
            MD4Round3(B, C, D, A, blk.M[13], 15);
            MD4Round3(A, B, C, D, blk.M[3], 3);
            MD4Round3(D, A, B, C, blk.M[11], 9);
            MD4Round3(C, D, A, B, blk.M[7], 11);
            MD4Round3(B, C, D, A, blk.M[15], 15);

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

} // end MD4Digest

} // namespace internal

/**
 * @brief Top function of MD4.
 *
 * The algorithm reference is: "The MD4 Message-Digest Algorithm".
 *
 * @param msg_strm The message being hashed.
 * @param len_strm The message length in byte.
 * @param end_len_strm The flag to signal end of input message stream.
 * @param digest_strm The digest (fingerprint) stream.
 * @param end_digest_strm Flag to signal the end of the result.
 *
 */

static void md4(
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
    internal::MD4Digest(blk_strm, nblk_strm, end_nblk_strm, digest_strm, end_digest_strm);

} // end md4

} // namespace security
} // namespace xf

#endif
