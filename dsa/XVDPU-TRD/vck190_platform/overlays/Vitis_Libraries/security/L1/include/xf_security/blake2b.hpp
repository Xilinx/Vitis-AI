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
 * @file blake2b.hpp
 * @brief header file for BLAKE2B related functions, including mixing, compression, padding, and computing functions.
 * This file is part of Vitis Security Library.
 *
 * @detail The algorithm takes a message and a key of arbitrary length (0 <= message length <= 2^128 bytes, 0 <= key
 * length <= 64 bytes) as its input,
 * and produces a specified length (1 <= output length <= 64 bytes) digest".
 * Notice that the key is optional to be added to the hash process, you can get an unkeyed hashing by simply setting the
 * key length to zero.
 * A special case is that both key and message length are set to zero, which means an unkeyed hashing with an empty
 * message will be executed.
 *
 */

#ifndef _XF_SECURITY_BLAKE2B_HPP_
#define _XF_SECURITY_BLAKE2B_HPP_

#include <ap_int.h>
#include <hls_stream.h>

namespace xf {
namespace security {
namespace internal {

// @brief 1024-bit Processing block
struct blake2BlockType {
    ap_uint<64> M[16];
};

/**
 * @brief Generate 1024-bit processing blocks by padding (pipeline).
 *
 * The algorithm reference is : "The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)".
 * The optimization goal of this function is to yield a 1024-bit block per cycle.
 *
 * @tparam w Bit width of the message words in block, default value is 64.
 *
 * @param msg_strm The message being hashed.
 * @param msg_len_strm Message length in byte (0 <= msg_len <= 2^128).
 * @param key_strm The optional key.
 * @param key_len_strm Key length in byte (0 <= key_len <= 64).
 * @param end_len_strm The flag to signal end of input message stream.
 * @param blk_strm The 1024-bit hash block.
 * @param nblk_strm The number of hash block for this message.
 * @param end_nblk_strm End flag for number of hash block.
 * @param msg_len_out_strm Message length pass on to the digest process.
 * @param key_len_out_strm Key length pass on to the digest process.
 *
 */

void generateBlock(
    // inputs
    hls::stream<ap_uint<64> >& msg_strm,
    hls::stream<ap_uint<128> >& msg_len_strm,
    hls::stream<ap_uint<512> >& key_strm,
    hls::stream<ap_uint<8> >& key_len_strm,
    hls::stream<bool>& end_len_strm,
    // outputs
    hls::stream<blake2BlockType>& blk_strm,
    hls::stream<ap_uint<128> >& nblk_strm,
    hls::stream<bool>& end_nblk_strm,
    hls::stream<ap_uint<128> >& msg_len_out_strm,
    hls::stream<ap_uint<8> >& key_len_out_strm) {
    bool endFlag = end_len_strm.read();

LOOP_PREPROCESSING_MAIN:
    while (!endFlag) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        // read message length in byte
        ap_uint<128> msg_len = msg_len_strm.read();

        // read key length in byte
        ap_uint<8> key_len = key_len_strm.read();

        // total number blocks to digest in 1024-bit
        ap_uint<128> blk_num;
        // still need to send a zero block if both of the key length and message length is zero
        if ((key_len == 0) && (msg_len == 0)) {
            blk_num = 1;
        } else {
            blk_num = (key_len > 0) + (msg_len >> 7) + ((msg_len % 128) > 0);
        }

        // inform digest function
        msg_len_out_strm.write(msg_len);
        key_len_out_strm.write(key_len);
        nblk_strm.write(blk_num);
        end_nblk_strm.write(false);

        // generate key block
        blake2BlockType k;
#pragma HLS array_partition variable = k.M complete
        ap_uint<512> tmp_k = key_strm.read();
        for (int i = 0; i < 8; i++) {
#pragma HLS unroll
            k.M[i] = tmp_k.range(i * 64 + 63, i * 64);
            k.M[i + 8] = 0;
        }

    LOOP_GEN_KEY_BLK:

        // send key block
        if (key_len != 0) {
            blk_strm.write(k);
        }

    LOOP_GEN_FULL_MSG_BLKS:
        for (ap_uint<128> j = 0; j < (ap_uint<128>)(msg_len >> 7); ++j) {
#pragma HLS pipeline II = 16
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
            // message block
            blake2BlockType b0;

        // this block will hold 16 words (64-bit for each) of message
        LOOP_GEN_ONE_FULL_BLK:
            for (ap_uint<5> i = 0; i < 16; ++i) {
                // XXX algorithm assumes little-endian
                b0.M[i] = msg_strm.read();
            }

            // send the full block
            blk_strm.write(b0);
        }

        // number of bytes left which needs to be padded as a new full block
        ap_uint<7> left = (ap_uint<7>)(msg_len & 0x7fUL);

        // not end at block boundary, pad the remaining message bytes to a full block
        // or the special case of an unkeyed empty message, send a zero block
        if ((left > 0) | ((msg_len == 0) && (key_len == 0))) {
            // last message block
            blake2BlockType b;
#pragma HLS array_partition variable = b.M complete

        LOOP_PAD_ZEROS:
            for (ap_uint<5> i = 0; i < 16; i++) {
#pragma HLS unroll
                if (i < (left >> 3)) {
                    // still have full message words
                    // XXX algorithm assumes little-endian
                    b.M[i] = msg_strm.read();
                } else if (i > (left >> 3)) {
                    // no meesage word to read
                    b.M[i] = 0UL;
                } else {
                    // pad the 64-bit message word with specific zero bytes
                    ap_uint<3> e = left & 0x7UL;
                    if (e == 0) {
                        // contains no message byte
                        b.M[i] = 0x0UL;
                    } else if (e == 1) {
                        // contains 1 message byte
                        ap_uint<64> l = msg_strm.read();
                        // XXX algorithm assumes little-endian
                        b.M[i] = l & 0x00000000000000ffUL;
                    } else if (e == 2) {
                        // contains 2 message bytes
                        ap_uint<64> l = msg_strm.read();
                        // XXX algorithm assumes little-endian
                        b.M[i] = l & 0x000000000000ffffUL;
                    } else if (e == 3) {
                        // contains 3 message bytes
                        ap_uint<64> l = msg_strm.read();
                        // XXX algorithm assumes little-endian
                        b.M[i] = l & 0x0000000000ffffffUL;
                    } else if (e == 4) {
                        // contains 4 message bytes
                        ap_uint<64> l = msg_strm.read();
                        // XXX algorithm assumes little-endian
                        b.M[i] = l & 0x00000000ffffffffUL;
                    } else if (e == 5) {
                        // contains 5 message bytes
                        ap_uint<64> l = msg_strm.read();
                        // XXX algorithm assumes little-endian
                        b.M[i] = l & 0x000000ffffffffffUL;
                    } else if (e == 6) {
                        // contains 6 message bytes
                        ap_uint<64> l = msg_strm.read();
                        // XXX algorithm assumes little-endian
                        b.M[i] = l & 0x0000ffffffffffffUL;
                    } else {
                        // contains 7 message bytes
                        ap_uint<64> l = msg_strm.read();
                        // XXX algorithm assumes little-endian
                        b.M[i] = l & 0x00ffffffffffffffUL;
                    }
                }
            }

            // emit last block
            blk_strm.write(b);
        }

        // still have message to handle
        endFlag = end_len_strm.read();
    }

    end_nblk_strm.write(true);

} // end generateBlock

/**
 *
 * @brief The implementation of rotate right (circular right shift) operation.
 *
 * The algorithm reference is : "The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)".
 *
 * @tparam w The bit width of input x, default value is 64.
 * @tparam n Number of bits for input x to be shifted.
 *
 * @param x Word to be rotated.
 *
 */

template <unsigned int w = 64, unsigned int n = 0>
ap_uint<w> ROTR(
    // inputs
    ap_uint<w> x) {
#pragma HLS inline

    return ((x >> n) | (x << (w - n)));

} // end ROTR

/**
 * @brief Mixing function G as defined in standard.
 *
 * The algorithm reference is : "The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)".
 *
 * @tparam Bit width of the words, default value is 64.
 *
 * @param v Working vector.
 * @param a The first index.
 * @param b The second index.
 * @param c The third index.
 * @param d the fourth index.
 * @param x The first input working word.
 * @param y The second input working word.
 *
 */

void G(
    // in-out
    ap_uint<64> v[16],
    // inputs
    ap_uint<4> a,
    ap_uint<4> b,
    ap_uint<4> c,
    ap_uint<4> d,
    ap_uint<64> x,
    ap_uint<64> y) {
#pragma HLS inline

    v[a] = v[a] + v[b] + x;
    v[d] = ROTR<64, 32>(v[d] ^ v[a]);
    v[c] = v[c] + v[d];
    v[b] = ROTR<64, 24>(v[b] ^ v[c]);
    v[a] = v[a] + v[b] + y;
    v[d] = ROTR<64, 16>(v[d] ^ v[a]);
    v[c] = v[c] + v[d];
    v[b] = ROTR<64, 63>(v[b] ^ v[c]);

} // end G

void halfMixing2(
    // in-out
    ap_uint<64> v[16],
    // inputs
    ap_uint<64> m[16],
    ap_uint<4> mi0,
    ap_uint<4> mi1,
    ap_uint<4> mi2,
    ap_uint<4> mi3,
    ap_uint<4> mi4,
    ap_uint<4> mi5,
    ap_uint<4> mi6,
    ap_uint<4> mi7,
    ap_uint<4> mi8,
    ap_uint<4> mi9,
    ap_uint<4> mi10,
    ap_uint<4> mi11,
    ap_uint<4> mi12,
    ap_uint<4> mi13,
    ap_uint<4> mi14,
    ap_uint<4> mi15) {
#pragma HLS inline off

    G(v, 0, 4, 8, 12, m[mi0], m[mi1]);
    G(v, 1, 5, 9, 13, m[mi2], m[mi3]);
    G(v, 2, 6, 10, 14, m[mi4], m[mi5]);
    G(v, 3, 7, 11, 15, m[mi6], m[mi7]);

    G(v, 0, 5, 10, 15, m[mi8], m[mi9]);
    G(v, 1, 6, 11, 12, m[mi10], m[mi11]);
    G(v, 2, 7, 8, 13, m[mi12], m[mi13]);
    G(v, 3, 4, 9, 14, m[mi14], m[mi15]);

} // end halfMixing2

/**
 * @brief Compression function F as defined in standard.
 *
 * The optimization goal of this function is for better performance.
 * The algorithm reference is : "The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)".
 *
 * @tparam w Bit width of the words, default value is 64.
 * @tparam round Number of rounds, 12 for BLAKE2b and 10 for BLAKE2s.
 *
 * @param h State vector.
 * @param m Message block vector.
 * @param t Offset counter.
 * @param last Final block indicator.
 *
 */

void Compress(
    // in-out
    ap_uint<64> h[8],
    // inputs
    ap_uint<64> m[16],
    ap_uint<128> t,
    bool last) {
    ap_uint<64> blake2b_iv[8] = {0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
                                 0x510E527FADE682D1, 0x9B05688C2B3E6C1F, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179};

#pragma HLS array_partition variable = blake2b_iv complete

    // working variables
    ap_uint<64> v[16];
#pragma HLS array_partition variable = v complete
    for (ap_uint<4> i = 0; i < 8; i++) {
#pragma HLS unroll
        v[i] = h[i];
        v[i + 8] = blake2b_iv[i];
    }
    // xor with low word of the total number of bytes
    v[12] ^= t.range(63, 0);
    // high word
    v[13] ^= t.range(127, 64);
    // invert v[14] if its final block
    if (last) {
        v[14] = ~v[14];
    }

LOOP_CRYPTOGRAPHIC_MIXING:
    halfMixing2(v, m, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    halfMixing2(v, m, 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3);
    halfMixing2(v, m, 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4);
    halfMixing2(v, m, 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8);
    halfMixing2(v, m, 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13);
    halfMixing2(v, m, 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9);
    halfMixing2(v, m, 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11);
    halfMixing2(v, m, 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10);
    halfMixing2(v, m, 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5);
    halfMixing2(v, m, 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0);
    halfMixing2(v, m, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    halfMixing2(v, m, 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3);

    for (ap_uint<4> i = 0; i < 8; i++) {
#pragma HLS unroll
        h[i] ^= v[i] ^ v[i + 8];
    }

} // end Compress

/**
 * @brief The implementation of the digest prcoess of BLAKE2.
 *
 * The algorithm reference is : "The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)".
 * The optimization goal of this function is for better performance.
 *
 * @tparam w Bit width of the words, default value is 64.
 *
 * @param blk_strm The 512-bit hash block.
 * @param nblk_strm The number of hash block for this message.
 * @param end_nblk_strm End flag for number of hash block.
 * @param key_len_strm Key length in byte (0 <= key_len <= 64).
 * @param msg_len_strm Message length in byte (0 <= msg_len <= 2^128).
 * @param out_len_strm Result hash value length in byte (0 < out_len < 64).
 * @param digest_strm The full digest stream (result is stored in the lower out_len bytes).
 * @param end_digest_strm Flag to signal the end of the result.
 *
 */

void blake2bDigest(
    // inputs
    hls::stream<blake2BlockType>& blk_strm,
    hls::stream<ap_uint<128> >& nblk_strm,
    hls::stream<bool>& end_nblk_strm,
    hls::stream<ap_uint<8> >& key_len_strm,
    hls::stream<ap_uint<128> >& msg_len_strm,
    hls::stream<ap_uint<8> >& out_len_strm,
    // ouputs
    hls::stream<ap_uint<512> >& digest_strm,
    hls::stream<bool>& end_digest_strm) {
    // initialization vector
    ap_uint<64> blake2b_iv[8] = {0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
                                 0x510E527FADE682D1, 0x9B05688C2B3E6C1F, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179};
#pragma HLS array_partition variable = blake2b_iv complete

    bool endFlag = end_nblk_strm.read();

LOOP_BLAKE2B_MAIN:
    while (!endFlag) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        // state vector
        ap_uint<64> h[8];
#pragma HLS array_partition variable = h complete

        // read key length in byte
        ap_uint<64> key_len = key_len_strm.read();

        // read result hash value length in byte
        ap_uint<64> out_len = out_len_strm.read();

        // read message length in byte
        ap_uint<128> msg_len = msg_len_strm.read();

    // initialize state vector with initialization vector
    LOOP_INIT_STATE_VECTOR:
        for (ap_uint<4> i = 0; i < 8; i++) {
#pragma HLS unroll
            h[i] = blake2b_iv[i];
        }
        h[0] ^= ap_uint<64>(0x0000000001010000) ^ (key_len << 8) ^ out_len;

        // total number blocks to digest
        ap_uint<128> blkNum = nblk_strm.read();

        // total number of bytes
        ap_uint<128> t = 0;

        // last block flag
        bool last = false;

    LOOP_BLAKE2B_DIGEST_NBLK:
        for (ap_uint<128> n = 0; n < blkNum; n++) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
#pragma HLS pipeline
            // input block
            blake2BlockType blk = blk_strm.read();

            // unkeyed hashing
            if (key_len == 0) {
                // empty message
                if (msg_len == 0) {
                    last = true;
                    t = 0;
                    // the last block
                } else if (n == (blkNum - 1)) {
                    last = true;
                    t = msg_len;
                    // still have blocks to digest
                } else {
                    t += 128;
                }
                // optional key is selected
            } else {
                // empty message
                if (msg_len == 0) {
                    last = true;
                    t = 128;
                    // the last block
                } else if (n == (blkNum - 1)) {
                    last = true;
                    t = 128 + msg_len;
                    // still have blocks to digest
                } else {
                    t += 128;
                }
            }

            // hash core
            Compress(h, blk.M, t, last);
        }

        // emit digest
        ap_uint<512> digest;
    LOOP_EMIT_DIGEST:
        for (ap_uint<4> i = 0; i < 8; i++) {
#pragma HLS unroll
            digest.range(64 * i + 63, 64 * i) = h[i];
        }
        digest_strm.write(digest);
        end_digest_strm.write(false);

        endFlag = end_nblk_strm.read();
    }

    end_digest_strm.write(true);

} // end blake2bDigest

} // namespace internal

/**
 * @brief Top function of BLAKE2B.
 *
 * The algorithm reference is : "The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)".
 * The implementation dataflows the sub-modules.
 *
 * @param msg_strm The message being hashed.
 * @param msg_len_strm Message length in byte (0 <= msg_len <= 2^128).
 * @param key_strm The optional key.
 * @param key_len_strm Key length in byte (0 <= key_len <= 64).
 * @param out_len_strm Result hash value length in byte (0 < out_len < 64).
 * @param end_len_strm The flag to signal end of input message stream.
 * @param digest_strm The digest (hash value) stream.
 * @param end_digest_strm Flag to signal the end of the result.
 *
 */

void blake2b(
    // inputs
    hls::stream<ap_uint<64> >& msg_strm,
    hls::stream<ap_uint<128> >& msg_len_strm,
    hls::stream<ap_uint<512> >& key_strm,
    hls::stream<ap_uint<8> >& key_len_strm,
    hls::stream<ap_uint<8> >& out_len_strm,
    hls::stream<bool>& end_len_strm,
    // ouputs
    hls::stream<ap_uint<8 * 64> >& digest_strm,
    hls::stream<bool>& end_digest_strm) {
#pragma HLS dataflow

    // 1024-bit processing block stream
    hls::stream<internal::blake2BlockType> blk_strm("blk_strm");
#pragma HLS stream variable = blk_strm depth = 32
#pragma HLS resource variable = blk_strm core = FIFO_LUTRAM

    // number of blocks stream
    hls::stream<ap_uint<128> > nblk_strm("nblk_strm");
#pragma HLS stream variable = nblk_strm depth = 32
#pragma HLS resource variable = nblk_strm core = FIFO_LUTRAM

    // end flag of number of blocks stream
    hls::stream<bool> end_nblk_strm("end_nblk_strm");
#pragma HLS stream variable = end_nblk_strm depth = 32
#pragma HLS resource variable = end_nblk_strm core = FIFO_LUTRAM

    // key length stream from generateBlock to blake2bDigest
    hls::stream<ap_uint<8> > key_len_out_strm("key_len_out_strm");
#pragma HLS stream variable = key_len_out_strm depth = 32
#pragma HLS resource variable = key_len_out_strm core = FIFO_LUTRAM

    // message length stream from generateBlock to blake2bDigest
    hls::stream<ap_uint<128> > msg_len_out_strm("msg_len_out_strm");
#pragma HLS stream variable = msg_len_out_strm depth = 32
#pragma HLS resource variable = msg_len_out_strm core = FIFO_LUTRAM

    // padding key (optional) and message words into blocks
    internal::generateBlock(msg_strm, msg_len_strm, key_strm, key_len_strm, end_len_strm,            // in
                            blk_strm, nblk_strm, end_nblk_strm, msg_len_out_strm, key_len_out_strm); // out

    // digest processing blocks into hash value
    internal::blake2bDigest(blk_strm, nblk_strm, end_nblk_strm, key_len_out_strm, msg_len_out_strm,
                            out_len_strm,                  // in
                            digest_strm, end_digest_strm); // out

} // end blake2b

namespace internal {
void initBlock(blake2BlockType& block) {
#pragma HLS inline
    for (int i = 0; i < 16; i++) {
#pragma HLS unroll
        block.M[i] = 0;
    }
}

void genBlock(
    // input
    hls::stream<ap_uint<64> >& msg_strm,
    hls::stream<ap_uint<6> >& msg_pack_len_strm,
    hls::stream<ap_uint<512> >& key_strm,
    hls::stream<ap_uint<8> >& key_len_strm,
    hls::stream<ap_uint<8> >& hash_len_strm,
    // output
    hls::stream<blake2BlockType>& blk_strm,
    hls::stream<ap_uint<10> >& blk_len_strm,
    hls::stream<ap_uint<8> >& key_len_mid_strm,
    hls::stream<ap_uint<8> >& hash_len_mid_strm) {
    //
    ap_uint<6> pack_len = msg_pack_len_strm.read();
    while (pack_len[5] != 1) {
        ap_uint<8> key_len = key_len_strm.read();
        ap_uint<8> hash_len = hash_len_strm.read();
        key_len_mid_strm.write(key_len);
        hash_len_mid_strm.write(hash_len);

        blake2BlockType blk;
        ap_uint<10> blk_len;
        ap_uint<8> counter;

        // read key
        ap_uint<512> tmp_k = key_strm.read();
        for (int i = 0; i < 8; i++) {
#pragma HLS unroll
            blk.M[i] = tmp_k.range(i * 64 + 63, i * 64);
            blk.M[i + 8] = 0;
        }
        if (key_len != 0) {
            blk_len = 128;
            if ((pack_len[4] == 1) && (pack_len.range(3, 0) == 0)) {
                blk_len[8] = 1;
            } else {
                blk_len[8] = 0;
            }

            blk_strm.write(blk);
            blk_len_strm.write(blk_len);
        }

        // read msg
        initBlock(blk);
        blk_len = 0;
        counter = 0;
        while (pack_len[4] != 1) {
#pragma HLS pipeline II = 1
            blk.M[counter] = msg_strm.read();
            pack_len = msg_pack_len_strm.read();

            counter++;
            blk_len += 8;
            if (counter == 16) {
                blk_strm.write(blk);
                blk_len_strm.write(ap_uint<10>(128));
                counter = 0;
                blk_len = 0;
                initBlock(blk);
            }
        }

        if (pack_len.range(3, 0) == 0) {
            msg_strm.read();
            if (key_len == 0) {
                blk_strm.write(blk);

                blk_len_strm.write(ap_uint<10>(1 << 8));
            }
        } else {
            blk.M[counter] = msg_strm.read();
            blk_len += pack_len.range(3, 0);
            blk_len[8] = 1;
            blk_strm.write(blk);
            blk_len_strm.write(blk_len);
        }

        pack_len = msg_pack_len_strm.read();
    }
    blk_len_strm.write(ap_uint<10>(1 << 9));
}

void digestBlock(
    // input
    hls::stream<internal::blake2BlockType>& blk_strm,
    hls::stream<ap_uint<10> >& blk_len_strm,
    hls::stream<ap_uint<8> >& key_len_mid_strm,
    hls::stream<ap_uint<8> >& hash_len_mid_strm,
    // output
    hls::stream<ap_uint<512> >& digest_strm,
    hls::stream<bool>& end_digest_strm) {
    //
    ap_uint<64> blake2b_iv[8] = {0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
                                 0x510E527FADE682D1, 0x9B05688C2B3E6C1F, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179};

    ap_uint<10> blk_len = blk_len_strm.read();
    while (blk_len[9] == 0) {
        ap_uint<64> hash_len = hash_len_mid_strm.read();
        ap_uint<64> key_len = key_len_mid_strm.read();

        ap_uint<64> h[8];
#pragma HLS array_partition variable = h complete
        for (ap_uint<4> i = 0; i < 8; i++) {
#pragma HLS unroll
            h[i] = blake2b_iv[i];
        }
        h[0] ^= ap_uint<64>(0x0000000001010000) ^ (key_len << 8) ^ hash_len;

        ap_uint<128> t = 0;
        blake2BlockType blk;

        while (blk_len[8] == 0) {
            blk = blk_strm.read();
            t += 128;
            Compress(h, blk.M, t, false);

            blk_len = blk_len_strm.read();
        }

        blk = blk_strm.read();
        t += blk_len.range(7, 0);
        Compress(h, blk.M, t, true);

        ap_uint<512> digest;
        for (int i = 0; i < 8; i++) {
#pragma HLS unroll
            digest.range(i * 64 + 63, i * 64) = h[i];
        }
        digest_strm.write(digest);
        end_digest_strm.write(false);

        blk_len = blk_len_strm.read();
    }
    end_digest_strm.write(true);
}
}

void blake2b(
    // input
    hls::stream<ap_uint<64> >& msg_strm,
    hls::stream<ap_uint<6> >& msg_pack_len_strm,
    hls::stream<ap_uint<512> >& key_strm,
    hls::stream<ap_uint<8> >& key_len_strm,
    hls::stream<ap_uint<8> >& hash_len_strm,
    // output
    hls::stream<ap_uint<512> >& digest_strm,
    hls::stream<bool>& end_digest_strm) {
#pragma HLS dataflow
    hls::stream<internal::blake2BlockType> blk_strm;
#pragma HLS stream variable = blk_strm depth = 2
#pragma HLS resource variable = blk_strm core = FIFO_LUTRAM

    hls::stream<ap_uint<10> > blk_len_strm;
#pragma HLS stream variable = blk_len_strm depth = 2
#pragma HLS resource variable = blk_len_strm core = FIFO_LUTRAM

    hls::stream<ap_uint<8> > key_len_mid_strm;
#pragma HLS stream variable = key_len_mid_strm depth = 2
#pragma HLS resource variable = key_len_mid_strm core = FIFO_LUTRAM

    hls::stream<ap_uint<8> > hash_len_mid_strm;
#pragma HLS stream variable = hash_len_mid_strm depth = 2
#pragma HLS resource variable = hash_len_mid_strm core = FIFO_LUTRAM

    internal::genBlock(msg_strm, msg_pack_len_strm, key_strm, key_len_strm, hash_len_strm, // input
                       blk_strm, blk_len_strm, key_len_mid_strm, hash_len_mid_strm);       // output

    internal::digestBlock(blk_strm, blk_len_strm, key_len_mid_strm, hash_len_mid_strm, // input
                          digest_strm, end_digest_strm);                               // output
}

} // namespace security
} // namespace xf

#endif
