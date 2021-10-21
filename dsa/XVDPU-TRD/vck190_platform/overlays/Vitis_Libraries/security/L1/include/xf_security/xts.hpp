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
 * @file xts.hpp
 * @brief header file for XEX-based tweaked-codebook mode with ciphertext
 * stealing (XTS) block cipher mode of operation.
 * This file is part of Vitis Security Library.
 *
 * @detail Containing XTS mode with AES-128/256.
 * There is no loop-carried dependency in both encryption and decryption parts of XTS mode.
 *
 */

#ifndef _XF_SECURITY_XTS_HPP_
#define _XF_SECURITY_XTS_HPP_

#include <ap_int.h>
#include <hls_stream.h>

#include "aes.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace security {
namespace internal {

/**
 *
 * @brief aesXtsEncrypt is XTS encryption mode with AES single block cipher.
 *
 * The algorithm reference is : "IEEE Standard for Cryptographic Protection of
 * Data on Block-Oriented Storage Devices"
 * The implementation is optimized for better performance in FPGA.
 *
 * @tparam _keyWidth The bit-width of the cipher key, which is 128, 192, or 256.
 *
 * @param plaintext_strm Input block stream text to be encrypted, each block is 128 bits.
 * @param plaintext_e_strm End flag of block stream plaintext, 1 bit.
 * @param len_strm Total length of plaintext in bit, 64 bits.
 * @param cipherkey_strm Input two cipher key used in encryption, x bits for AES-x.
 * @param initialization_vector_strm Initialization vector for the fisrt
 * iteration of AES encrypition, 128 bits.
 * @param ciphertext_strm Output encrypted block stream text, 128 bits.
 * @param ciphertext_e_strm End flag of stream ciphertext, 1 bit.
 *
 */

template <unsigned int _keyWidth = 256>
void aesXtsEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintext_strm,
    hls::stream<bool>& plaintext_e_strm,
    hls::stream<ap_uint<64> >& len_strm,
    hls::stream<ap_uint<_keyWidth> >& cipherkey_strm,
    hls::stream<ap_uint<128> >& initialization_vector_strm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertext_strm,
    hls::stream<bool>& ciphertext_e_strm) {
    // register cipherkey and IV
    ap_uint<_keyWidth> key_1 = cipherkey_strm.read();
    ap_uint<_keyWidth> key_2 = cipherkey_strm.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "key1 = " << std::hex << key_1 << std::endl << "key2 = " << key_2 << std::endl;
#endif
    xf::security::aesEnc<_keyWidth> cipher;
    ap_uint<128> IV = initialization_vector_strm.read();
#ifndef __SYNTHESIS__
    std::cout << "initialization_vector = " << std::hex << IV << std::endl << std::endl;
#endif

    ap_uint<64> len_r = len_strm.read();
    const ap_uint<64> LEN_IN_BIT = len_r;
    const ap_uint<7> final_len = LEN_IN_BIT % 128; // length of final partial block
    const bool is_partial = (final_len != 0);      // if partial block exists

    //  bool next_plaintext = true;
    ap_uint<128> input_block;
    ap_uint<_keyWidth> input_key;
    ap_uint<128> output_block;
    ap_uint<128> t_vec, t_vec_final;
    ap_uint<128> ciphertext_r;
    ap_uint<128> ciphertext_r_sf;
    ap_uint<128> plaintext_reg;

    // counter for partial block
    ap_uint<64> block_cnt = 0;

    // calculate alpha
    input_block = IV;
    input_key = key_2;
    cipher.updateKey(key_2);
    cipher.process(input_block, input_key, t_vec);
    // xf::security::internal::aesEncrypt<_keyWidth>(input_block, input_key, t_vec);
    cipher.updateKey(key_1);
    bool e = plaintext_e_strm.read();
    while (!e) {
#pragma HLS PIPELINE II = 1
        // read a segment of plaintext, 128 bits
        e = plaintext_e_strm.read();
        ap_uint<128> plaintext_read = plaintext_strm.read();
        input_key = key_1;
#ifndef __SYNTHESIS__
        std::cout << "t_vec = " << std::hex << t_vec << std::endl;
#endif
        t_vec_final = t_vec;
        plaintext_reg = plaintext_read;
        input_block = plaintext_reg ^ t_vec;
#ifndef __SYNTHESIS__
        std::cout << "plaintext = " << std::hex << plaintext_reg << std::endl;
#endif

        // CIPH_k
        cipher.process(input_block, key_1, output_block);
// xf::security::internal::aesEncrypt<_keyWidth>(input_block, input_key, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // get the ciphertext for current interation by output_block
        ciphertext_r = output_block ^ t_vec;
        if (is_partial && (LEN_IN_BIT - block_cnt < 256)) {
            if (LEN_IN_BIT - block_cnt > 128) ciphertext_r_sf = ciphertext_r;
        } else {
            ciphertext_strm << ciphertext_r;
            ciphertext_e_strm << false;
#ifndef __SYNTHESIS__
            std::cout << "ciphertext = " << ciphertext_r << std::dec << ",block:" << block_cnt << std::endl;
#endif
        }

        if (is_partial) block_cnt += 128;

        // update T vector
        ap_uint<16> cin = 0;
        bool tt = t_vec[127];
        for (int j = 1; j < 16; j++) {
#pragma HLS unroll
            cin[j] = t_vec[8 * j - 1];
        }
        for (int j = 0; j < 16; j++) {
#pragma HLS unroll
            t_vec(8 * j + 7, 8 * j) = (t_vec(8 * j + 6, 8 * j), cin[j]);
        }
        if (tt) t_vec(7, 0) = t_vec(7, 0) ^ 0x87;
    }

    // write out final partial ciphertext
    if (is_partial) {
        ap_uint<128> final_plaintext = (ciphertext_r_sf(127, final_len), plaintext_reg(final_len - 1, 0));
        input_block = final_plaintext ^ t_vec_final;
        cipher.process(input_block, key_1, output_block);
        // xf::security::internal::aesEncrypt<_keyWidth>(input_block, input_key, output_block);
        ap_uint<128> t1 = output_block ^ t_vec_final;
        ciphertext_strm.write(t1);
        ciphertext_e_strm.write(false);

        ap_uint<128> tmp = 0;
        ap_uint<128> t = (tmp(127, final_len), ciphertext_r_sf(final_len - 1, 0));
        ciphertext_strm.write(t);
        ciphertext_e_strm.write(false);
    }

    // end of transfer
    ciphertext_e_strm.write(true);

} // end aesXtsEncrypt

/**
 *
 * @brief aesXtsDecrypt is XTS decryption mode with AES single block cipher.
 *
 * The algorithm reference is : "IEEE Standard for Cryptographic Protection of
 * Data on Block-Oriented Storage Devices"
 * The implementation is optimized for better performance in FPGA.
 *
 * @tparam _keyWidth The bit-width of the cipher key, which is 128, 192, or 256.
 *
 * @param ciphertext_strm Input block stream text to be decrypted, each block is 128 bits.
 * @param ciphertext_e_strm End flag of block stream ciphertext, 1 bit.
 * @param len_strm Total length of plaintext in bit, 64 bits.
 * @param cipherkey_strm Input two cipher key used in decryption, x bits for AES-x.
 * @param initialization_vector_strm Initialization vector for the fisrt
 * iteration of AES encrypition, 128 bits.
 * @param plaintext_strm Output decrypted block stream text, each block is 128 bits.
 * @param plaintext_e_strm End flag of block stream plaintext, 1 bit.
 *
 */

template <unsigned int _keyWidth = 256>
void aesXtsDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertext_strm,
    hls::stream<bool>& ciphertext_e_strm,
    hls::stream<ap_uint<64> >& len_strm,
    hls::stream<ap_uint<_keyWidth> >& cipherkey_strm,
    hls::stream<ap_uint<128> >& initialization_vector_strm,
    // stream out
    hls::stream<ap_uint<128> >& plaintext_strm,
    hls::stream<bool>& plaintext_e_strm) {
    // register cipherkey and IV
    ap_uint<_keyWidth> key_1 = cipherkey_strm.read();
    ap_uint<_keyWidth> key_2 = cipherkey_strm.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "key1 = " << std::hex << key_1 << std::endl << "key2 = " << key_2 << std::endl;
#endif
xf:;
    security::aesEnc<_keyWidth> cipher;
    security::aesDec<_keyWidth> decipher;
    ap_uint<128> IV = initialization_vector_strm.read();
#ifndef __SYNTHESIS__
    std::cout << "initialization_vector = " << std::hex << IV << std::endl << std::endl;
#endif

    ap_uint<64> len_r = len_strm.read();
    const ap_uint<64> LEN_IN_BIT = len_r;
    const ap_uint<7> final_len = LEN_IN_BIT % 128; // length of final partial block
    const bool is_partial = (final_len != 0);      // if partial block exists

    ap_uint<128> input_block;
    ap_uint<_keyWidth> input_key;
    ap_uint<128> output_block;
    ap_uint<128> t_vec, t_vec_final;
    ap_uint<128> plaintext_r;
    ap_uint<128> plaintext_r_sf;
    ap_uint<128> ciphertext_reg, ciphertext_last;

    // counter for partial block
    ap_uint<64> block_cnt = 0;

    // calculate alpha
    input_block = IV;
    input_key = key_2;
    cipher.updateKey(key_2);
    cipher.process(input_block, input_key, t_vec);
    decipher.updateKey(key_1);
    // xf::security::internal::aesEncrypt<_keyWidth>(input_block, input_key, t_vec);

    bool e = ciphertext_e_strm.read();
    while (!e) {
#pragma HLS PIPELINE II = 1
        // read a segment of ciphertext, 128 bits
        e = ciphertext_e_strm.read();
        ap_uint<128> ciphertext_read = ciphertext_strm.read();
        input_key = key_1;
#ifndef __SYNTHESIS__
        std::cout << "t_vec = " << std::hex << t_vec << std::endl;
#endif
        if (is_partial && (LEN_IN_BIT - block_cnt < 128)) {
            ciphertext_last = ciphertext_read;
        } else
            ciphertext_reg = ciphertext_read;
        input_block = ciphertext_reg ^ t_vec;
#ifndef __SYNTHESIS__
        std::cout << "ciphertext = " << std::hex << ciphertext_reg << std::endl;
#endif

        // CIPH_k
        decipher.process(input_block, input_key, output_block);
// xf::security::internal::aesDecrypt<_keyWidth>(input_block, input_key, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // get the ciphertext for current interation by output_block
        plaintext_r = output_block ^ t_vec;
        if (is_partial && (LEN_IN_BIT - block_cnt < 256)) {
            if (LEN_IN_BIT - block_cnt > 128) {
                t_vec_final = t_vec;
            } else {
                plaintext_r_sf = plaintext_r;
            }
        } else {
            plaintext_strm << plaintext_r;
            plaintext_e_strm << false;
#ifndef __SYNTHESIS__
            std::cout << "plaintext = " << plaintext_r << std::dec << ",block:" << block_cnt << std::endl;
#endif
        }

        if (is_partial) block_cnt += 128;

        // update T vector
        ap_uint<16> cin = 0;
        bool tt = t_vec[127];
        for (int j = 1; j < 16; j++) {
#pragma HLS unroll
            cin[j] = t_vec[8 * j - 1];
        }
        for (int j = 0; j < 16; j++) {
#pragma HLS unroll
            t_vec(8 * j + 7, 8 * j) = (t_vec(8 * j + 6, 8 * j), cin[j]);
        }
        if (tt) t_vec(7, 0) = t_vec(7, 0) ^ 0x87;
    }

    // write out final partial plaintext
    if (is_partial) {
        ap_uint<128> final_ciphertext = (plaintext_r_sf(127, final_len), ciphertext_last(final_len - 1, 0));
        input_block = final_ciphertext ^ t_vec_final;
        decipher.process(input_block, input_key, output_block);
        // xf::security::internal::aesDecrypt<_keyWidth>(input_block, input_key, output_block);
        ap_uint<128> t1 = output_block ^ t_vec_final;
        plaintext_strm.write(t1);
        plaintext_e_strm.write(false);

        ap_uint<128> tmp = 0;
        ap_uint<128> t = (tmp(127, final_len), plaintext_r_sf(final_len - 1, 0));
        plaintext_strm.write(t);
        plaintext_e_strm.write(false);
    }

    // end of transfer
    plaintext_e_strm.write(true);

} // end aesXtsDecrypt

} // namespace internal

/**
 *
 * @brief aes128XtsEncrypt is XTS encryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is : "IEEE Standard for Cryptographic Protection of
 * Data on Block-Oriented Storage Devices"
 * The implementation is optimized for better performance in FPGA.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param lenStrm Total length of plaintext in bit, 64 bits.
 * @param cipherkeyStrm Input two cipher key used in encryption, 128 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, 128 bits.
 * @param endCiphertextStrm End flag of stream ciphertext, 1 bit.
 *
 */

static void aes128XtsEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    hls::stream<ap_uint<64> >& lenStrm,
    hls::stream<ap_uint<128> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesXtsEncrypt<128>(plaintextStrm, endPlaintextStrm, lenStrm, cipherkeyStrm, IVStrm, ciphertextStrm,
                                 endCiphertextStrm);

} // end aesXts128Encrypt

/**
 *
 * @brief aes128XtsDecrypt is XTS decryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is : "IEEE Standard for Cryptographic Protection of
 * Data on Block-Oriented Storage Devices"
 * The implementation is optimized for better performance in FPGA.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param lenStrm Total length of plaintext in bit, 64 bits.
 * @param cipherkeyStrm Input two cipher key used in decryption, 128 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void aes128XtsDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    hls::stream<ap_uint<64> >& lenStrm,
    hls::stream<ap_uint<128> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesXtsDecrypt<128>(ciphertextStrm, endCiphertextStrm, lenStrm, cipherkeyStrm, IVStrm, plaintextStrm,
                                 endPlaintextStrm);

} // end aes128XtsDecrypt

/**
 *
 * @brief aes256XtsEncrypt is XTS encryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is : "IEEE Standard for Cryptographic Protection of
 * Data on Block-Oriented Storage Devices"
 * The implementation is optimized for better performance in FPGA.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param lenStrm Total length of plaintext in bit, 64 bits.
 * @param cipherkeyStrm Input two cipher key used in encryption, 256 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, 128 bits.
 * @param endCiphertextStrm End flag of stream ciphertext, 1 bit.
 *
 */

static void aes256XtsEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    hls::stream<ap_uint<64> >& lenStrm,
    hls::stream<ap_uint<256> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesXtsEncrypt<256>(plaintextStrm, endPlaintextStrm, lenStrm, cipherkeyStrm, IVStrm, ciphertextStrm,
                                 endCiphertextStrm);

} // end aesXts256Encrypt

/**
 *
 * @brief aes256XtsDecrypt is XTS decryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is : "IEEE Standard for Cryptographic Protection of
 * Data on Block-Oriented Storage Devices"
 * The implementation is optimized for better performance in FPGA.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param lenStrm Total length of plaintext in bit, 64 bits.
 * @param cipherkeyStrm Input two cipher key used in decryption, 256 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void aes256XtsDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    hls::stream<ap_uint<64> >& lenStrm,
    hls::stream<ap_uint<256> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesXtsDecrypt<256>(ciphertextStrm, endCiphertextStrm, lenStrm, cipherkeyStrm, IVStrm, plaintextStrm,
                                 endPlaintextStrm);

} // end aes256XtsDecrypt

} // namespace security
} // namespace xf

#endif
