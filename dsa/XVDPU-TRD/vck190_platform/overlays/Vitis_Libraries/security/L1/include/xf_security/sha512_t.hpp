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
 * @file sha512_t.hpp
 * @brief header file for SHA-384, SHA-512, and SHA-512/t related functions, including both pre-processing and digest
 * functions.
 * This file is part of Vitis Security Library.
 *
 * @detail The algorithm takes a message of arbitrary length as its input, and produces a 512-bit final hash value.
 * The message digest of SHA-384, SHA-512, and SHA-512/t is obtained by truncating the final hash value to its left-most
 * 384, 512, and t bits respectively.
 * The standard uses the big-endian convention.
 *
 */

#ifndef _XF_SECURITY_SHA512_T_HPP_
#define _XF_SECURITY_SHA512_T_HPP_

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
    ap_uint<64> M[16];
};

/**
 * @brief Generate 1024-bit processing blocks by padding and appending (pipeline).
 *
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 * The optimization goal of this function is to yield a 1024-bit block per cycle.
 *
 * @tparam w The bit width of each input message word, default value is 64.
 *
 * @param msg_strm The message being hashed.
 * @param len_strm The message length in byte.
 * @param end_len_strm The flag to signal end of input message stream.
 * @param blk_strm The 1024-bit hash block.
 * @param nblk_strm The number of hash block for this message.
 * @param end_nblk_strm End flag for number of hash block.
 *
 */

template <unsigned int w>
static void preProcessing(
    // inputs
    hls::stream<ap_uint<w> >& msg_strm,
    hls::stream<ap_uint<128> >& len_strm,
    hls::stream<bool>& end_len_strm,
    // outputs
    hls::stream<blockType>& blk_strm,
    hls::stream<ap_uint<128> >& nblk_strm,
    hls::stream<bool>& end_nblk_strm) {
    bool endFlag = end_len_strm.read();

LOOP_PREPROCESSING_MAIN:
    while (!endFlag) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        // read message length in byte
        ap_uint<128> len = len_strm.read();

        // prepare message length in bit which will be appended at the tail of the block according to the standard
        ap_uint<128> L = 8 * len;

        // total number blocks to digest in 1024-bit
        ap_uint<128> blk_num = (len >> 7) + 1 + ((len & 0x7f) > 111);

        // inform digest function
        nblk_strm.write(blk_num);
        end_nblk_strm.write(false);

    LOOP_GEN_FULL_BLKS:
        for (ap_uint<128> j = 0; j < (ap_uint<128>)(len >> 7); ++j) {
#pragma HLS pipeline II = 16
#pragma HLS loop_tripcount min = 0 max = 1
            // message block
            blockType b0;
#pragma HLS array_partition variable = b0.M complete

        // this block will hold 16 words (64-bit for each) of message
        LOOP_GEN_ONE_FULL_BLK:
            for (ap_uint<5> i = 0; i < 16; ++i) {
#pragma HLS unroll
                ap_uint<w> l = msg_strm.read();
                // XXX algorithm assumes big-endian
                l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40) |
                    ((0x0000000000ff0000UL & l) << 24) | ((0x00000000ff000000UL & l) << 8) |
                    ((0x000000ff00000000UL & l) >> 8) | ((0x0000ff0000000000UL & l) >> 24) |
                    ((0x00ff000000000000UL & l) >> 40) | ((0xff00000000000000UL & l) >> 56);
                b0.M[i] = l;
            }

            // send the full block
            blk_strm.write(b0);
        }

        // number of bytes left which needs to be padded as a new full block
        ap_uint<7> left = (ap_uint<7>)(len & 0x7fUL);

        if (left == 0) {
            // end at block boundary, start with pad 1
            // last block
            blockType b;
#pragma HLS array_partition variable = b.M complete

            // pad 1
            b.M[0] = 0x8000000000000000UL;

        // pad zero words
        LOOP_PAD_13_ZERO_WORDS:
            for (ap_uint<5> i = 1; i < 14; ++i) {
#pragma HLS unroll
                b.M[i] = 0;
            }

            // append L
            b.M[14] = (ap_uint<w>)(0xffffffffffffffffUL & (L >> w));
            b.M[15] = (ap_uint<w>)(0xffffffffffffffffUL & (L));

            // emit
            blk_strm.write(b);
        } else if (left < 112) {
            // can pad 1 and append L in current block
            // last block
            blockType b;
#pragma HLS array_partition variable = b.M complete

        LOOP_COPY_TAIL_AND_PAD_1:
            for (ap_uint<5> i = 0; i < 14; ++i) {
#pragma HLS pipeline
                if (i < (left >> 3)) {
                    // pad 1 byte not in this word
                    ap_uint<w> l = msg_strm.read();
                    // XXX algorithm assumes big-endian
                    l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40) |
                        ((0x0000000000ff0000UL & l) << 24) | ((0x00000000ff000000UL & l) << 8) |
                        ((0x000000ff00000000UL & l) >> 8) | ((0x0000ff0000000000UL & l) >> 24) |
                        ((0x00ff000000000000UL & l) >> 40) | ((0xff00000000000000UL & l) >> 56);
                    b.M[i] = l;
                } else if (i > (left >> 3)) {
                    // pad 1 byte not in this word, and no word to read
                    b.M[i] = 0UL;
                } else {
                    // pad 1 byte in this word
                    ap_uint<3> e = left & 0x7UL;
                    if (e == 0) {
                        // contains no message byte
                        b.M[i] = 0x8000000000000000UL;
                    } else if (e == 1) {
                        // contains 1 message byte
                        ap_uint<w> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x00000000000000ffUL & l) << 56);
                        b.M[i] = l | 0x0080000000000000UL;
                    } else if (e == 2) {
                        // contains 2 message bytes
                        ap_uint<w> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40);
                        b.M[i] = l | 0x0000800000000000UL;
                    } else if (e == 3) {
                        // contains 3 message bytes
                        ap_uint<w> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40) |
                            ((0x0000000000ff0000UL & l) << 24);
                        b.M[i] = l | 0x0000008000000000UL;
                    } else if (e == 4) {
                        // contains 4 message bytes
                        ap_uint<w> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40) |
                            ((0x0000000000ff0000UL & l) << 24) | ((0x00000000ff000000UL & l) << 8);
                        b.M[i] = l | 0x0000000080000000UL;
                    } else if (e == 5) {
                        // contains 5 message bytes
                        ap_uint<w> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40) |
                            ((0x0000000000ff0000UL & l) << 24) | ((0x00000000ff000000UL & l) << 8) |
                            ((0x000000ff00000000UL & l) >> 8);
                        b.M[i] = l | 0x0000000000800000UL;
                    } else if (e == 6) {
                        // contains 6 message bytes
                        ap_uint<w> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40) |
                            ((0x0000000000ff0000UL & l) << 24) | ((0x00000000ff000000UL & l) << 8) |
                            ((0x000000ff00000000UL & l) >> 8) | ((0x0000ff0000000000UL & l) >> 24);
                        b.M[i] = l | 0x0000000000008000UL;
                    } else {
                        // contains 7 message bytes
                        ap_uint<w> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40) |
                            ((0x0000000000ff0000UL & l) << 24) | ((0x00000000ff000000UL & l) << 8) |
                            ((0x000000ff00000000UL & l) >> 8) | ((0x0000ff0000000000UL & l) >> 24) |
                            ((0x00ff000000000000UL & l) >> 40);
                        b.M[i] = l | 0x0000000000000080UL;
                    }
                }
            }

            // append L
            b.M[14] = (ap_uint<w>)(0xffffffffffffffffUL & (L >> w));
            b.M[15] = (ap_uint<w>)(0xffffffffffffffffUL & (L));

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
                if (i < (left >> 3)) {
                    // pad 1 byte not in this word
                    ap_uint<w> l = msg_strm.read();
                    // XXX algorithm assumes big-endian
                    l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40) |
                        ((0x0000000000ff0000UL & l) << 24) | ((0x00000000ff000000UL & l) << 8) |
                        ((0x000000ff00000000UL & l) >> 8) | ((0x0000ff0000000000UL & l) >> 24) |
                        ((0x00ff000000000000UL & l) >> 40) | ((0xff00000000000000UL & l) >> 56);
                    b.M[i] = l;
                } else if (i > (left >> 3)) {
                    // pad 1 byte not in this word, and no msg word to read
                    b.M[i] = 0UL;
                } else {
                    // pad 1 byte in this word
                    ap_uint<3> e = left & 0x7UL;
                    if (e == 0) {
                        // contains no message byte
                        b.M[i] = 0x8000000000000000UL;
                    } else if (e == 1) {
                        // contains 1 message byte
                        ap_uint<w> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x00000000000000ffUL & l) << 56);
                        b.M[i] = l | 0x0080000000000000UL;
                    } else if (e == 2) {
                        // contains 2 message bytes
                        ap_uint<w> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40);
                        b.M[i] = l | 0x0000800000000000UL;
                    } else if (e == 3) {
                        // contains 3 message bytes
                        ap_uint<w> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40) |
                            ((0x0000000000ff0000UL & l) << 24);
                        b.M[i] = l | 0x0000008000000000UL;
                    } else if (e == 4) {
                        // contains 4 message bytes
                        ap_uint<w> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40) |
                            ((0x0000000000ff0000UL & l) << 24) | ((0x00000000ff000000UL & l) << 8);
                        b.M[i] = l | 0x0000000080000000UL;
                    } else if (e == 5) {
                        // contains 5 message bytes
                        ap_uint<w> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40) |
                            ((0x0000000000ff0000UL & l) << 24) | ((0x00000000ff000000UL & l) << 8) |
                            ((0x000000ff00000000UL & l) >> 8);
                        b.M[i] = l | 0x0000000000800000UL;
                    } else if (e == 6) {
                        // contains 6 message bytes
                        ap_uint<w> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40) |
                            ((0x0000000000ff0000UL & l) << 24) | ((0x00000000ff000000UL & l) << 8) |
                            ((0x000000ff00000000UL & l) >> 8) | ((0x0000ff0000000000UL & l) >> 24);
                        b.M[i] = l | 0x0000000000008000UL;
                    } else {
                        // contains 7 message bytes
                        ap_uint<w> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x00000000000000ffUL & l) << 56) | ((0x000000000000ff00UL & l) << 40) |
                            ((0x0000000000ff0000UL & l) << 24) | ((0x00000000ff000000UL & l) << 8) |
                            ((0x000000ff00000000UL & l) >> 8) | ((0x0000ff0000000000UL & l) >> 24) |
                            ((0x00ff000000000000UL & l) >> 40);
                        b.M[i] = l | 0x0000000000000080UL;
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

            // append L
            b1.M[14] = (ap_uint<w>)(0xffffffffffffffffUL & (L >> w));
            b1.M[15] = (ap_uint<w>)(0xffffffffffffffffUL & (L));

            // emit last block
            blk_strm.write(b1);
        }

        // still have message to handle
        endFlag = end_len_strm.read();
    }

    end_nblk_strm.write(true);

} // end preProcessing

/**
 *
 * @brief The implementation of right shift operation.
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 *
 * @tparam w The bit width of input x, default value is 64.
 *
 * @param n Number of bits for input x to be shifted.
 * @param x Word to be shifted.
 *
 */

template <unsigned int w>
ap_uint<w> SHR(
    // inputs
    unsigned int n,
    ap_uint<w> x) {
#pragma HLS inline

    return (x >> n);

} // end SHR

/**
 *
 * @brief The implementation of rotate right (circular right shift) operation.
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 *
 * @tparam w The bit width of input x, default value is 64.
 *
 * @param n Number of bits for input x to be shifted.
 * @param x Word to be rotated.
 *
 */

template <unsigned int w>
ap_uint<w> ROTR(
    // inputs
    unsigned int n,
    ap_uint<w> x) {
#pragma HLS inline

    return ((x >> n) | (x << (w - n)));

} // end ROTR

/**
 *
 * @brief The implementation of Ch(x,y,z).
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 *
 * @tparam w The bit width of input x, y, and z, default value is 64.
 *
 * @param x The first w-bit input word.
 * @param y The second w-bit input word.
 * @param z The third w-bit input word.
 *
 */

template <unsigned int w>
ap_uint<w> Ch(
    // inputs
    ap_uint<w> x,
    ap_uint<w> y,
    ap_uint<w> z) {
#pragma HLS inline

    return ((x & y) ^ ((~x) & z));

} // end Ch

/**
 *
 * @brief The implementation of Maj(x,y,z).
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 *
 * @tparam w The bit width of input x, y, and z, default value is 64.
 *
 * @param x The first w-bit input word.
 * @param y The second w-bit input word.
 * @param z The third w-bit input word.
 *
 */

template <unsigned int w>
ap_uint<w> Maj(
    // inputs
    ap_uint<w> x,
    ap_uint<w> y,
    ap_uint<w> z) {
#pragma HLS inline

    return ((x & y) ^ (x & z) ^ (y & z));

} // end Maj

/**
 *
 * @brief The implementation of upper-case letter sigma 0.
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 *
 * @tparam w The bit width of input x, default value is 64.
 *
 * @param x The operand.
 *
 */

template <unsigned int w>
ap_uint<w> BSIG0(
    // inputs
    ap_uint<w> x) {
#pragma HLS inline

    return (ROTR<w>(28, x) ^ ROTR<w>(34, x) ^ ROTR<w>(39, x));

} // end BSIG0

/**
 *
 * @brief The implementation of upper-case letter sigma 1.
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 *
 * @tparam w The bit width of input x, default value is 64.
 *
 * @param x The operand.
 *
 */

template <unsigned int w>
ap_uint<w> BSIG1(
    // inputs
    ap_uint<w> x) {
#pragma HLS inline

    return (ROTR<w>(14, x) ^ ROTR<w>(18, x) ^ ROTR<w>(41, x));

} // end BSIG1

/**
 *
 * @brief The implementation of lower-case letter sigma 0.
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 *
 * @tparam w The bit width of input x, default value is 64.
 *
 * @param x The operand.
 *
 */

template <unsigned int w>
ap_uint<w> SSIG0(
    // inputs
    ap_uint<w> x) {
#pragma HLS inline

    return (ROTR<w>(1, x) ^ ROTR<w>(8, x) ^ SHR<w>(7, x));

} // end SSIG0

/**
 *
 * @brief The implementation of lower-case letter sigma 1.
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 *
 * @tparam w The bit width of input x, default value is 64.
 *
 * @param x The operand.
 *
 */

template <unsigned int w>
ap_uint<w> SSIG1(
    // inputs
    ap_uint<w> x) {
#pragma HLS inline

    return (ROTR<w>(19, x) ^ ROTR<w>(61, x) ^ SHR<w>(6, x));

} // end SSIG1

/**
 *
 * @brief Duplicate 1 input stream to 2 output streams
 *
 * @tparam w The bit width of the streams.
 *
 * @param in_strm Input stream to be duplicated.
 * @param in_e_strm End flag of input stream.
 * @param out1_strm The first output stream.
 * @param out1_e_strm End flag of the first output stream.
 * @param out2_strm The second output stream.
 * @param out2_e_strm End flag of the second output stream.
 *
 */

template <unsigned int w>
void dup_strm(
    // stream in
    hls::stream<ap_uint<w> >& in_strm,
    hls::stream<bool>& in_e_strm,
    // stream out
    hls::stream<ap_uint<w> >& out1_strm,
    hls::stream<bool>& out1_e_strm,
    hls::stream<ap_uint<w> >& out2_strm,
    hls::stream<bool>& out2_e_strm) {
    bool e = in_e_strm.read();

LOOP_DUP_STREAM:
    while (!e) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
#pragma HLS pipeline II = 1
        ap_uint<w> in_r = in_strm.read();

        out1_strm.write(in_r);
        out1_e_strm.write(false);
        out2_strm.write(in_r);
        out2_e_strm.write(false);

        e = in_e_strm.read();
    }

    out1_e_strm.write(true);
    out2_e_strm.write(true);

} // end dup_strm

/**
 *
 * @brief Generate message schedule W (80 words) in stream.
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 *
 * @tparam w The bit width of message schedule W which defined in the standard, default value is 64.
 *
 * @param blk_strm Message block stream.
 * @param nblk_strm Number of message block stream.
 * @param end_nblk_strm End flag for number of message block stream.
 * @param w_strm The message schedule in stream.
 *
 */

template <unsigned int w>
void generateMsgSchedule(
    // inputs
    hls::stream<blockType>& blk_strm,
    hls::stream<ap_uint<128> >& nblk_strm,
    hls::stream<bool>& end_nblk_strm,
    // output
    hls::stream<ap_uint<w> >& w_strm) {
    bool end = end_nblk_strm.read();

LOOP_GEN_W_MAIN:
    while (!end) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        ap_uint<128> numBlk = nblk_strm.read();
    LOOP_GEN_W_NBLK:
        for (ap_uint<128> i = 0; i < numBlk; i++) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
            blockType blk = blk_strm.read();
#pragma HLS array_partition variable = blk.M complete

            // message schedule
            ap_uint<w> W[16];
#pragma HLS array_partition variable = W complete

        LOOP_SHA512_GEN_WT16:
            for (ap_uint<5> t = 0; t < 16; t++) {
#pragma HLS pipeline II = 1
                ap_uint<w> Wt = blk.M[t];
                W[t] = Wt;
                w_strm.write(Wt);
            }

        LOOP_SHA1_GEN_WT64:
            for (ap_uint<7> t = 16; t < 80; t++) {
#pragma HLS pipeline II = 1
                ap_uint<w> Wt = SSIG1<w>(W[14]) + W[9] + SSIG0<w>(W[1]) + W[0];
                for (ap_uint<5> i = 0; i < 15; i++) {
                    W[i] = W[i + 1];
                }
                W[15] = Wt;
                w_strm.write(Wt);
            }
        }
        end = end_nblk_strm.read();
    }

} // end generateMsgSchedule

/**
 *
 * @brief This function performs the computation of SHA-512.
 *
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 * The implementation is modified for better performance.
 *
 * @tparam w The bit width of each input message word, default value is 64.
 * @tparam hash_width The bit width of hash width, which depends on specific algorithm.
 *
 * @param w_strm Message schedule stream.
 * @param nblk_strm Number of message block stream.
 * @param end_nblk_strm End flag for number of message block stream.
 * @param digest_strm Output digest stream.
 * @param end_digest_strm End flag for output digest stream.
 *
 */

template <unsigned int w, unsigned int hash_width>
void SHA512Digest(
    // inputs
    hls::stream<ap_uint<w> >& w_strm,
    hls::stream<ap_uint<128> >& nblk_strm,
    hls::stream<bool>& end_nblk_strm,
    // outputs
    hls::stream<ap_uint<hash_width> >& digest_strm,
    hls::stream<bool>& end_digest_strm) {
    // the eighty constant 64-bit words of SHA-384, SHA-512, and SHA-512/t
    static const ap_uint<w> K[80] = {
        0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL, 0x3956c25bf348b538UL,
        0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL, 0xd807aa98a3030242UL, 0x12835b0145706fbeUL,
        0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL, 0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL,
        0xc19bf174cf692694UL, 0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
        0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL, 0x983e5152ee66dfabUL,
        0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL, 0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL,
        0x06ca6351e003826fUL, 0x142929670a0e6e70UL, 0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL,
        0x53380d139d95b3dfUL, 0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
        0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL, 0xd192e819d6ef5218UL,
        0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL, 0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL,
        0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL, 0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL,
        0x682e6ff3d6b2b8a3UL, 0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
        0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL, 0xca273eceea26619cUL,
        0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL, 0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL,
        0x113f9804bef90daeUL, 0x1b710b35131c471bUL, 0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL,
        0x431d67c49c100d4cUL, 0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL,
    };
#pragma HLS array_partition variable = K complete

    bool end = end_nblk_strm.read();

LOOP_SHA1_MAIN:
    while (!end) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        // the initial hash values, default value is SHA-512
        ap_uint<w> H[8];
        if (hash_width == 384) {
            // SHA-384
            H[0] = 0xcbbb9d5dc1059ed8UL;
            H[1] = 0x629a292a367cd507UL;
            H[2] = 0x9159015a3070dd17UL;
            H[3] = 0x152fecd8f70e5939UL;
            H[4] = 0x67332667ffc00b31UL;
            H[5] = 0x8eb44a8768581511UL;
            H[6] = 0xdb0c2e0d64f98fa7UL;
            H[7] = 0x47b5481dbefa4fa4UL;
        } else if (hash_width == 256) {
            // SHA-512/256
            H[0] = 0x22312194FC2BF72CUL;
            H[1] = 0x9F555FA3C84C64C2UL;
            H[2] = 0x2393B86B6F53B151UL;
            H[3] = 0x963877195940EABDUL;
            H[4] = 0x96283EE2A88EFFE3UL;
            H[5] = 0xBE5E1E2553863992UL;
            H[6] = 0x2B0199FC2C85B8AAUL;
            H[7] = 0x0EB72DDC81C52CA2UL;
        } else if (hash_width == 224) {
            // SHA-512/224
            H[0] = 0x8C3D37C819544DA2UL;
            H[1] = 0x73E1996689DCD4D6UL;
            H[2] = 0x1DFAB7AE32FF9C82UL;
            H[3] = 0x679DD514582F9FCFUL;
            H[4] = 0x0F6D2B697BD44DA8UL;
            H[5] = 0x77E36F7304C48942UL;
            H[6] = 0x3F9D85A86A1D36C8UL;
            H[7] = 0x1112E6AD91D692A1UL;
        } else {
            // SHA-512
            H[0] = 0x6a09e667f3bcc908UL;
            H[1] = 0xbb67ae8584caa73bUL;
            H[2] = 0x3c6ef372fe94f82bUL;
            H[3] = 0xa54ff53a5f1d36f1UL;
            H[4] = 0x510e527fade682d1UL;
            H[5] = 0x9b05688c2b3e6c1fUL;
            H[6] = 0x1f83d9abfb41bd6bUL;
            H[7] = 0x5be0cd19137e2179UL;
        }
#pragma HLS array_partition variable = H complete

        // total number blocks to digest
        ap_uint<128> blkNum = nblk_strm.read();

    LOOP_SHA1_DIGEST_NBLK:
        for (ap_uint<128> n = 0; n < blkNum; n++) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
            // load working variables
            ap_uint<w> a = H[0];
            ap_uint<w> b = H[1];
            ap_uint<w> c = H[2];
            ap_uint<w> d = H[3];
            ap_uint<w> e = H[4];
            ap_uint<w> f = H[5];
            ap_uint<w> g = H[6];
            ap_uint<w> h = H[7];

        // update working variables accordingly
        LOOP_SHA1_UPDATE_80_ROUNDS:
            for (ap_uint<7> t = 0; t < 80; t++) {
#pragma HLS pipeline II = 1
                ap_uint<w> Wt = w_strm.read();
#if !defined(__SYNTHESIS__) && __XF_SECURITY_SHA512_T_DEBUG__ == 1
                std::cout << "W[" << std::dec << t << "] = " << std::hex << Wt << std::endl;
#endif
                ap_uint<w> T1 = h + BSIG1<w>(e) + Ch<w>(e, f, g) + K[t] + Wt;
                ap_uint<w> T2 = BSIG0<w>(a) + Maj<w>(a, b, c);
                h = g;
                g = f;
                f = e;
                e = d + T1;
                d = c;
                c = b;
                b = a;
                a = T1 + T2;
#if !defined(__SYNTHESIS__) && __XF_SECURITY_SHA512_T_DEBUG__ == 1
                std::cout << "a = " << std::hex << a << std::endl;
                std::cout << "b = " << std::hex << b << std::endl;
                std::cout << "c = " << std::hex << c << std::endl;
                std::cout << "d = " << std::hex << d << std::endl;
                std::cout << "e = " << std::hex << e << std::endl;
                std::cout << "f = " << std::hex << f << std::endl;
                std::cout << "g = " << std::hex << g << std::endl;
                std::cout << "h = " << std::hex << h << std::endl;
#endif
            }

            // increment internal states with updated working variables
            H[0] = a + H[0];
            H[1] = b + H[1];
            H[2] = c + H[2];
            H[3] = d + H[3];
            H[4] = e + H[4];
            H[5] = f + H[5];
            H[6] = g + H[6];
            H[7] = h + H[7];
#if !defined(__SYNTHESIS__) && __XF_SECURITY_SHA512_T_DEBUG__ == 1
            std::cout << "H[0] = " << std::hex << H[0] << std::endl;
            std::cout << "H[1] = " << std::hex << H[1] << std::endl;
            std::cout << "H[2] = " << std::hex << H[2] << std::endl;
            std::cout << "H[3] = " << std::hex << H[3] << std::endl;
            std::cout << "H[4] = " << std::hex << H[4] << std::endl;
            std::cout << "H[5] = " << std::hex << H[5] << std::endl;
            std::cout << "H[6] = " << std::hex << H[6] << std::endl;
            std::cout << "H[7] = " << std::hex << H[7] << std::endl;
#endif
        }

        // emit digest
        ap_uint<8 * w> digest;
    LOOP_SHA512_EMIT:
        for (ap_uint<4> i = 0; i < 8; i++) {
#pragma HLS unroll
            ap_uint<w> l = H[i];
            // XXX shift algorithm's big-endian to HLS's little-endian
            ap_uint<8> byte0 = ((l >> 56) & 0xff);
            ap_uint<8> byte1 = ((l >> 48) & 0xff);
            ap_uint<8> byte2 = ((l >> 40) & 0xff);
            ap_uint<8> byte3 = ((l >> 32) & 0xff);
            ap_uint<8> byte4 = ((l >> 24) & 0xff);
            ap_uint<8> byte5 = ((l >> 16) & 0xff);
            ap_uint<8> byte6 = ((l >> 8) & 0xff);
            ap_uint<8> byte7 = (l & 0xff);
            digest.range(w * i + w - 1, w * i) = ((ap_uint<w>)byte0) | (((ap_uint<w>)byte1) << 8) |
                                                 (((ap_uint<w>)byte2) << 16) | (((ap_uint<w>)byte3) << 24) |
                                                 (((ap_uint<w>)byte4) << 32) | (((ap_uint<w>)byte5) << 40) |
                                                 (((ap_uint<w>)byte6) << 48) | (((ap_uint<w>)byte7) << 56);
        }
        // obtain the digest by trancating the left-most hash_width bits of 512-bit hash value
        digest_strm.write(digest.range(hash_width - 1, 0));
        end_digest_strm.write(false);

        end = end_nblk_strm.read();
    }

    end_digest_strm.write(true);

} // end SHA512Digest

/**
 *
 * @brief Top function of SHA-512.
 *
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 * The implementation dataflows the pre-processing part and message digest part.
 *
 * @tparam w The bit width of each input message word, default value is 64.
 * @tparam hash_width The bit width of hash width, which depends on specific algorithm.
 *
 * @param msg_strm The message being hashed.
 * @param len_strm The message length in byte.
 * @param end_len_strm The flag to signal end of input message stream.
 * @param digest_strm Output digest stream.
 * @param end_digest_strm End flag for output digest stream.
 *
 */

template <unsigned int w, unsigned int hash_width>
void sha512Top(
    // inputs
    hls::stream<ap_uint<w> >& msg_strm,
    hls::stream<ap_uint<128> >& len_strm,
    hls::stream<bool>& end_len_strm,
    // outputs
    hls::stream<ap_uint<hash_width> >& digest_strm,
    hls::stream<bool>& end_digest_strm) {
#pragma HLS dataflow

    // 1024-bit processing block stream
    hls::stream<blockType> blk_strm("blk_strm");
#pragma HLS stream variable = blk_strm depth = 32
#pragma HLS resource variable = blk_strm core = FIFO_LUTRAM

    // number of blocks stream
    hls::stream<ap_uint<128> > nblk_strm("nblk_strm");
#pragma HLS stream variable = nblk_strm depth = 32
#pragma HLS resource variable = nblk_strm core = FIFO_LUTRAM
    hls::stream<ap_uint<128> > nblk_strm1("nblk_strm1");
#pragma HLS stream variable = nblk_strm1 depth = 32
#pragma HLS resource variable = nblk_strm1 core = FIFO_LUTRAM
    hls::stream<ap_uint<128> > nblk_strm2("nblk_strm2");
#pragma HLS stream variable = nblk_strm2 depth = 32
#pragma HLS resource variable = nblk_strm2 core = FIFO_LUTRAM

    // end flag of number of blocks stream
    hls::stream<bool> end_nblk_strm("end_nblk_strm");
#pragma HLS stream variable = end_nblk_strm depth = 32
#pragma HLS resource variable = end_nblk_strm core = FIFO_LUTRAM
    hls::stream<bool> end_nblk_strm1("end_nblk_strm1");
#pragma HLS stream variable = end_nblk_strm1 depth = 32
#pragma HLS resource variable = end_nblk_strm1 core = FIFO_LUTRAM
    hls::stream<bool> end_nblk_strm2("end_nblk_strm2");
#pragma HLS stream variable = end_nblk_strm2 depth = 32
#pragma HLS resource variable = end_nblk_strm2 core = FIFO_LUTRAM

    // message schedule stream
    hls::stream<ap_uint<w> > w_strm("w_strm");
#pragma HLS stream variable = w_strm depth = 32
#pragma HLS resource variable = w_strm core = FIFO_LUTRAM

    // padding and appending message words into blocks
    preProcessing<w>(msg_strm, len_strm, end_len_strm, blk_strm, nblk_strm, end_nblk_strm);

    // duplicate number of block stream and its end flag stream
    dup_strm<128>(nblk_strm, end_nblk_strm, nblk_strm1, end_nblk_strm1, nblk_strm2, end_nblk_strm2);

    // generate the message schedule in stream
    generateMsgSchedule<w>(blk_strm, nblk_strm1, end_nblk_strm1, w_strm);

    // digest precessing blocks into hash value
    SHA512Digest<w, hash_width>(w_strm, nblk_strm2, end_nblk_strm2, digest_strm, end_digest_strm);

} // end sha512Top

} // end namespace internal

/**
 *
 * @brief SHA-384 algorithm with stream input and output.
 *
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 * The implementation dataflows the pre-processing part and message digest part.
 *
 * @tparam w The bit width of each input message word, default value is 64.
 *
 * @param msg_strm The message being hashed.
 * @param len_strm The message length in byte.
 * @param end_len_strm The flag to signal end of input message stream.
 * @param digest_strm Output digest stream.
 * @param end_digest_strm End flag for output digest stream.
 *
 */

template <unsigned int w>
void sha384(
    // inputs
    hls::stream<ap_uint<w> >& msg_strm,
    hls::stream<ap_uint<128> >& len_strm,
    hls::stream<bool>& end_len_strm,
    // outputs
    hls::stream<ap_uint<384> >& digest_strm,
    hls::stream<bool>& end_digest_strm) {
    internal::sha512Top<w, 384>(msg_strm, len_strm, end_len_strm, // input streams
                                digest_strm, end_digest_strm);    // output streams

} // end sha384

/**
 *
 * @brief SHA-512 algorithm with stream input and output.
 *
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 * The implementation dataflows the pre-processing part and message digest part.
 *
 * @tparam w The bit width of each input message word, default value is 64.
 *
 * @param msg_strm The message being hashed.
 * @param len_strm The message length in byte.
 * @param end_len_strm The flag to signal end of input message stream.
 * @param digest_strm Output digest stream.
 * @param end_digest_strm End flag for output digest stream.
 *
 */

template <unsigned int w>
void sha512(
    // inputs
    hls::stream<ap_uint<w> >& msg_strm,
    hls::stream<ap_uint<128> >& len_strm,
    hls::stream<bool>& end_len_strm,
    // outputs
    hls::stream<ap_uint<512> >& digest_strm,
    hls::stream<bool>& end_digest_strm) {
    internal::sha512Top<w, 512>(msg_strm, len_strm, end_len_strm, // input streams
                                digest_strm, end_digest_strm);    // output streams

} // end sha512

/**
 *
 * @brief SHA-512/t algorithm with stream input and output.
 *
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 * The implementation dataflows the pre-processing part and message digest part.
 *
 * @tparam w The bit width of each input message word, default value is 64.
 * @tparam t The bit width of the digest which depends on specific algorithm, typically is 224 or 256.
 *
 * @param msg_strm The message being hashed.
 * @param len_strm The message length in byte.
 * @param end_len_strm The flag to signal end of input message stream.
 * @param digest_strm Output digest stream.
 * @param end_digest_strm End flag for output digest stream.
 *
 */

template <unsigned int w, unsigned int t>
void sha512_t(
    // inputs
    hls::stream<ap_uint<w> >& msg_strm,
    hls::stream<ap_uint<128> >& len_strm,
    hls::stream<bool>& end_len_strm,
    // outputs
    hls::stream<ap_uint<t> >& digest_strm,
    hls::stream<bool>& end_digest_strm) {
    internal::sha512Top<w, t>(msg_strm, len_strm, end_len_strm, // input streams
                              digest_strm, end_digest_strm);    // output streams

} // end sha512_t

} // namespace security
} // namespace xf

#endif
