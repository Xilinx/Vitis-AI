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
 * @file sha1.hpp
 * @brief header file for SHA-1 related functions, including pre-processing and digest functions.
 * This file is part of Vitis Security Library.
 *
 * @detail The algorithm takes a message of arbitrary length as its input,
 * and produces a 160-bit message digest.
 * The standard of SHA-1 uses the big-endian convention, so that within each word, the most significant bit is stored in
 * the left-most bit.
 * For example, 291 is represented by 32-bit std::hex word 0x00000123
 *
 */

#ifndef _XF_SECURITY_SHA1_HPP_
#define _XF_SECURITY_SHA1_HPP_

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
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
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
                ap_uint<32> l = msg_strm.read();
                // XXX algorithm assumes big-endian
                l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                    ((0xff000000UL & l) >> 24);
                b0.M[i] = l;
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
            b.M[0] = 0x80000000UL;

        // pad zero words
        LOOP_PAD_13_ZERO_WORDS:
            for (ap_uint<5> i = 1; i < 14; ++i) {
#pragma HLS unroll
                b.M[i] = 0;
            }

            // append L
            b.M[14] = (ap_uint<32>)(0xffffffffUL & (L >> 32));
            b.M[15] = (ap_uint<32>)(0xffffffffUL & (L));

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
                    ap_uint<32> l = msg_strm.read();
                    // XXX algorithm assumes big-endian
                    l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                        ((0xff000000UL & l) >> 24);
                    b.M[i] = l;
                } else if (i > (left >> 2)) {
                    // pad 1 byte not in this word, and no word to read
                    b.M[i] = 0UL;
                } else {
                    // pad 1 byte in this word
                    ap_uint<2> e = left & 0x3UL;
                    if (e == 0) {
                        // contains no message byte
                        b.M[i] = 0x80000000UL;
                    } else if (e == 1) {
                        // contains 1 message byte
                        ap_uint<32> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x000000ffUL & l) << 24);
                        b.M[i] = l | 0x00800000UL;
                    } else if (e == 2) {
                        // contains 2 message bytes
                        ap_uint<32> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8);
                        b.M[i] = l | 0x00008000UL;
                    } else {
                        // contains 3 message bytes
                        ap_uint<32> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8);
                        b.M[i] = l | 0x00000080UL;
                    }
                }
            }

            // append L
            b.M[14] = (ap_uint<32>)(0xffffffffUL & (L >> 32));
            b.M[15] = (ap_uint<32>)(0xffffffffUL & (L));

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
                    ap_uint<32> l = msg_strm.read();
                    // XXX algorithm assumes big-endian
                    l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                        ((0xff000000UL & l) >> 24);
                    b.M[i] = l;
                } else if (i > (left >> 2)) { // pad 1 byte not in this word, and no msg word to read
                    b.M[i] = 0UL;
                } else { // pad 1 byte in this word
                    ap_uint<2> e = left & 0x3UL;
                    if (e == 0) {
                        b.M[i] = 0x80000000UL;
                    } else if (e == 1) {
                        ap_uint<32> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x000000ffUL & l) << 24);
                        b.M[i] = l | 0x00800000UL;
                    } else if (e == 2) {
                        ap_uint<32> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8);
                        b.M[i] = l | 0x00008000UL;
                    } else {
                        ap_uint<32> l = msg_strm.read();
                        // XXX algorithm assumes big-endian
                        l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8);
                        b.M[i] = l | 0x00000080UL;
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
            b1.M[14] = (ap_uint<32>)(0xffffffffUL & (L >> 32));
            b1.M[15] = (ap_uint<32>)(0xffffffffUL & (L));

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
 * @brief The implementation of rotate left (circular left shift) operation.
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
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
 * @brief The implementation of Ch(x,y,z), the sequence of logical functions of SHA-1 where 0 <= t <= 19.
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 *
 * @tparam w The bit width of input x, y, and z, default value is 32.
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
 * @brief The implementation of Parity(x,y,z), the sequence of logical functions of SHA-1 where 20 <= t <= 39, and 60 <=
 * t <= 79.
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 *
 * @tparam w The bit width of input x, y, and z, default value is 32.
 *
 * @param x The first w-bit input word.
 * @param y The second w-bit input word.
 * @param z The third w-bit input word.
 *
 */

template <unsigned int w>
ap_uint<w> Parity(
    // inputs
    ap_uint<w> x,
    ap_uint<w> y,
    ap_uint<w> z) {
#pragma HLS inline

    return (x ^ y ^ z);

} // end Parity

/**
 *
 * @brief The implementation of Maj(x,y,z), the sequence of logical functions of SHA-1 where 40 <= t <= 59.
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 *
 * @tparam w The bit width of input x, y, and z, default value is 32.
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
 * @brief Generate message schedule W (80 words) in stream.
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 *
 * @tparam w The bit width of message schedule W which defined in the standard.
 *
 * @param blk_strm Message block stream.
 * @param nblk_strm1 Number of message block stream.
 * @param end_nblk_strm1 End flag for number of message block stream.
 * @param w_strm The message schedule in stream.
 * @param nblk_strm2 Number of message block stream.
 * @param end_nblk_strm2 End flag for number of message block stream.
 */

template <unsigned int w>
void generateMsgSchedule(
    // inputs
    hls::stream<blockType>& blk_strm,
    hls::stream<ap_uint<64> >& nblk_strm1,
    hls::stream<bool>& end_nblk_strm1,
    // output
    hls::stream<ap_uint<w> >& w_strm,
    hls::stream<ap_uint<64> >& nblk_strm2,
    hls::stream<bool>& end_nblk_strm2) {
    //#pragma HLS inline

    bool end = end_nblk_strm1.read();
    end_nblk_strm2.write(end);

LOOP_GEN_W_MAIN:
    while (!end) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        ap_uint<64> numBlk = nblk_strm1.read();
        nblk_strm2.write(numBlk);
    LOOP_GEN_W_NBLK:
        for (ap_uint<64> i = 0; i < numBlk; i++) {
#pragma HLS loop_tripcount min = 10 max = 10 avg = 10
            blockType blk = blk_strm.read();
#pragma HLS array_partition variable = blk.M complete

            // message schedule
            ap_uint<w> W[16];
#pragma HLS array_partition variable = W complete

        LOOP_SHA1_GEN_WT16:
            for (ap_uint<5> t = 0; t < 16; t++) {
#pragma HLS pipeline II = 1
                ap_uint<w> Wt = blk.M[t];
                W[t] = Wt;
                w_strm.write(Wt);
            }

        LOOP_SHA1_GEN_WT64:
            for (ap_uint<7> t = 16; t < 80; t++) {
#pragma HLS pipeline II = 1
                ap_uint<w> Wt = ROTL<w>(1, W[13] ^ W[8] ^ W[2] ^ W[0]);
                for (unsigned int i = 0; i < 15; i++) {
                    W[i] = W[i + 1];
                }
                W[15] = Wt;
                w_strm.write(Wt);
            }
        }
        end = end_nblk_strm1.read();
        end_nblk_strm2.write(end);
    }

} // end generateMsgSchedule

/**
 *
 * @brief This function performs the computation of the secure hash algorithm.
 *
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 * The implementation is modified for better performance.
 *
 * @tparam w The bit width of each input message word, default value is 32.
 *
 * @param w_strm Message schedule stream.
 * @param nblk_strm Number of message block stream.
 * @param end_nblk_strm End flag for number of message block stream.
 * @param digest_strm Output digest stream.
 * @param end_digest_strm End flag for output digest stream.
 *
 */

template <unsigned int w>
void SHA1Digest(
    // inputs
    hls::stream<ap_uint<w> >& w_strm,
    hls::stream<ap_uint<64> >& nblk_strm,
    hls::stream<bool>& end_nblk_strm,
    // outputs
    hls::stream<ap_uint<5 * w> >& digest_strm,
    hls::stream<bool>& end_digest_strm) {
    // the eighty constant 32-bit words of SHA-1
    static const ap_uint<w> K[4] = {
        0x5a827999UL, //  0 <= t <= 19
        0x6ed9eba1UL, // 20 <= t <= 39
        0x8f1bbcdcUL, // 40 <= t <= 59
        0xca62c1d6UL  // 60 <= t <= 79
    };
#pragma HLS array_partition variable = K complete

    bool end = end_nblk_strm.read();

LOOP_SHA1_MAIN:
    while (!end) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        // the initial hash value of SHA-1
        ap_uint<w> H[5] = {0x67452301UL, 0xefcdab89UL, 0x98badcfeUL, 0x10325476UL, 0xc3d2e1f0UL};
#pragma HLS array_partition variable = H complete

        // total number blocks to digest
        ap_uint<64> blkNum = nblk_strm.read();

    LOOP_SHA1_DIGEST_NBLK:
        for (ap_uint<64> n = 0; n < blkNum; n++) {
#pragma HLS loop_tripcount min = 10 max = 10 avg = 10
            // load working variables
            ap_uint<w> a = H[0];
            ap_uint<w> b = H[1];
            ap_uint<w> c = H[2];
            ap_uint<w> d = H[3];
            ap_uint<w> e = H[4];

        // update working variables accordingly
        LOOP_SHA1_UPDATE_80_ROUNDS:
            for (ap_uint<7> t = 0; t < 80; t++) {
#pragma HLS pipeline II = 1
                ap_uint<w> Wt = w_strm.read();
#if !defined(__SYNTHESIS__) && DEBUG
                std::cout << "W[" << std::dec << t << "] = " << std::hex << Wt << std::endl;
#endif
                ap_uint<w> T;
                if (t < 20) {
                    T = ROTL<w>(5, a) + Ch<w>(b, c, d) + e + K[t / 20] + Wt;
                } else if ((40 <= t) && (t < 60)) {
                    T = ROTL<w>(5, a) + Maj<w>(b, c, d) + e + K[t / 20] + Wt;
                } else {
                    T = ROTL<w>(5, a) + Parity<w>(b, c, d) + e + K[t / 20] + Wt;
                }

                e = d;
                d = c;
                c = ROTL<w>(30, b);
                b = a;
                a = T;
#if !defined(__SYNTHESIS__) && DEBUG
                std::cout << "a = " << std::hex << a << std::endl;
                std::cout << "b = " << std::hex << b << std::endl;
                std::cout << "c = " << std::hex << c << std::endl;
                std::cout << "d = " << std::hex << d << std::endl;
                std::cout << "e = " << std::hex << e << std::endl;
#endif
            }

            // increment internal states with updated working variables
            H[0] = a + H[0];
            H[1] = b + H[1];
            H[2] = c + H[2];
            H[3] = d + H[3];
            H[4] = e + H[4];
#if !defined(__SYNTHESIS__) && DEBUG
            std::cout << "H[0] = " << std::hex << H[0] << std::endl;
            std::cout << "H[1] = " << std::hex << H[1] << std::endl;
            std::cout << "H[2] = " << std::hex << H[2] << std::endl;
            std::cout << "H[3] = " << std::hex << H[3] << std::endl;
            std::cout << "H[4] = " << std::hex << H[4] << std::endl;
#endif
        }

        // emit digest
        ap_uint<5 * w> digest;
    LOOP_SHA1_EMIT:
        for (ap_uint<3> i = 0; i < 5; i++) {
#pragma HLS unroll
            ap_uint<w> l = H[i];
            // XXX shift algorithm's big-endian to HLS's little-endian
            ap_uint<8> t0 = ((l >> 24) & 0xff);
            ap_uint<8> t1 = ((l >> 16) & 0xff);
            ap_uint<8> t2 = ((l >> 8) & 0xff);
            ap_uint<8> t3 = (l & 0xff);
            digest.range(w * i + w - 1, w * i) =
                ((ap_uint<w>)t0) | (((ap_uint<w>)t1) << 8) | (((ap_uint<w>)t2) << 16) | (((ap_uint<w>)t3) << 24);
        }
        digest_strm.write(digest);
        end_digest_strm.write(false);

        end = end_nblk_strm.read();
    }

    end_digest_strm.write(true);

} // end SHA1Digest

} // end namespace internal

/**
 *
 * @brief Top function of SHA-1.
 *
 * The algorithm reference is : "Secure Hash Standard", which published by NIST in February 2012.
 * The implementation dataflows the pre-processing part and message digest part.
 *
 * @tparam w The bit width of each input message word, default value is 32.
 *
 * @param msg_strm The message being hashed.
 * @param len_strm The message length in byte.
 * @param end_len_strm The flag to signal end of input message stream.
 * @param digest_strm Output digest stream.
 * @param end_digest_strm End flag for output digest stream.
 *
 */

template <unsigned int w>
void sha1(
    // inputs
    hls::stream<ap_uint<w> >& msg_strm,
    hls::stream<ap_uint<64> >& len_strm,
    hls::stream<bool>& end_len_strm,
    // outputs
    hls::stream<ap_uint<5 * w> >& digest_strm,
    hls::stream<bool>& end_digest_strm) {
#pragma HLS dataflow

    // 512-bit processing block stream
    hls::stream<internal::blockType> blk_strm("blk_strm");
#pragma HLS stream variable = blk_strm depth = 4
#pragma HLS resource variable = blk_strm core = FIFO_LUTRAM

    // number of blocks stream
    hls::stream<ap_uint<64> > nblk_strm1("nblk_strm1");
#pragma HLS stream variable = nblk_strm1 depth = 4
#pragma HLS resource variable = nblk_strm1 core = FIFO_LUTRAM
    hls::stream<ap_uint<64> > nblk_strm2("nblk_strm2");
#pragma HLS stream variable = nblk_strm2 depth = 4
#pragma HLS resource variable = nblk_strm2 core = FIFO_LUTRAM

    // end flag of number of blocks stream
    hls::stream<bool> end_nblk_strm1("end_nblk_strm1");
#pragma HLS stream variable = end_nblk_strm1 depth = 4
#pragma HLS resource variable = end_nblk_strm1 core = FIFO_LUTRAM
    hls::stream<bool> end_nblk_strm2("end_nblk_strm2");
#pragma HLS stream variable = end_nblk_strm2 depth = 4
#pragma HLS resource variable = end_nblk_strm2 core = FIFO_LUTRAM

    // message schedule stream
    hls::stream<ap_uint<w> > w_strm("w_strm");
#pragma HLS stream variable = w_strm depth = 320
#pragma HLS resource variable = w_strm core = FIFO_BRAM

    // padding and appending message words into blocks
    internal::preProcessing(msg_strm, len_strm, end_len_strm, blk_strm, nblk_strm1, end_nblk_strm1);

    // generate the message schedule in stream
    internal::generateMsgSchedule<w>(blk_strm, nblk_strm1, end_nblk_strm1, w_strm, nblk_strm2, end_nblk_strm2);

    // digest precessing blocks into hash value
    internal::SHA1Digest<w>(w_strm, nblk_strm2, end_nblk_strm2, digest_strm, end_digest_strm);

} // end sha1

} // namespace security
} // namespace xf

#endif
