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
 * @file cfb.hpp
 * @brief header file for Cipher Feedback (CFB) block cipher mode of operation.
 * This file is part of Vitis Security Library.
 *
 * @detail Containing CFB1/CFB8/CFB128 modes with AES-128/192/256 and DES.
 * Loop-carried dependency is enforced by the CFB encryption algorithm,
 * but no dependency in decryption part.
 *
 */

#ifndef _XF_SECURITY_CFB_HPP_
#define _XF_SECURITY_CFB_HPP_

#include <ap_int.h>
#include <hls_stream.h>

#include "xf_security/aes.hpp"
#include "xf_security/des.hpp"

// for debug
#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace security {
namespace internal {

/**
 *
 * @brief aesCfb1Encrypt is CFB1 encryption mode with AES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @tparam _keyWidth The bit-width of the cipher key, which is 128, 192, or 256.
 *
 * @param plaintext Input block stream text to be encrypted, each block is 128 bits.
 * @param plaintext_e End flag of block stream plaintext, 1 bit.
 * @param cipherkey Input cipher key used in encryption, x bits for AES-x.
 * @param initialization_vector Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertext Output encrypted block stream text, each block is 128 bits.
 * @param ciphertext_e End flag of stream ciphertext, 1 bit.
 *
 */

template <unsigned int _keyWidth = 256>
void aesCfb1Encrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintext,
    hls::stream<bool>& plaintext_e,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<_keyWidth> >& cipherkey,
    hls::stream<ap_uint<128> >& initialization_vector,
    // stream out
    hls::stream<ap_uint<128> >& ciphertext,
    hls::stream<bool>& ciphertext_e) {
    // register cipherkey
    ap_uint<_keyWidth> key_r = cipherkey.read();

    xf::security::aesEnc<_keyWidth> cipher;
    cipher.updateKey(key_r);
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif

    // register IV
    ap_uint<128> IV = initialization_vector.read();
#ifndef __SYNTHESIS__
    std::cout << "initialization_vector = " << std::hex << IV << std::endl << std::endl;
#endif

    // intermediate registers to perform the encryption chain
    bool next_plaintext = true;
    ap_uint<128> plaintext_r = 0;
    ap_uint<128> feedback_r = 0;
    ap_uint<128> input_block = 0;
    ap_uint<128> output_block = 0;
    ap_uint<128> ciphertext_r = 0;
    ap_uint<4> cfb_byte_cnt = 0;
    ap_uint<3> cfb_bit_cnt = 7;

    // set the initialization for ture
    bool initialization = true;

    bool e = plaintext_e.read();

encryption_cfb1_loop:
    while (!e) {
#pragma HLS PIPELINE
        // read a block of plaintext, 128 bits
        if (next_plaintext) { // mode CFB1/CFB8 needs multiple iteration to process one plaintext block
            plaintext_r = plaintext.read();
#ifndef __SYNTHESIS__
            std::cout << "plaintext    = " << std::hex << plaintext_r << std::endl;
#endif
        }

        // calculate input_block
        if (initialization) { // first iteration, input_block is IV
            input_block = IV;
            initialization = false;
        } else { // after first iteration, input_blcok is comprised by 127 bits of IV and 1 bit of ciphertext
            ap_uint<128> ibt;
            ibt.range(127, 121) = input_block(126, 120);
            ibt[120] = feedback_r[120];
            ibt.range(119, 113) = input_block(118, 112);
            ibt[112] = input_block[127];
            ibt.range(111, 105) = input_block(110, 104);
            ibt[104] = input_block[119];
            ibt.range(103, 97) = input_block(102, 96);
            ibt[96] = input_block[111];
            ibt.range(95, 89) = input_block(94, 88);
            ibt[88] = input_block[103];
            ibt.range(87, 81) = input_block(86, 80);
            ibt[80] = input_block[95];
            ibt.range(79, 73) = input_block(78, 72);
            ibt[72] = input_block[87];
            ibt.range(71, 65) = input_block(70, 64);
            ibt[64] = input_block[79];
            ibt.range(63, 57) = input_block(62, 56);
            ibt[56] = input_block[71];
            ibt.range(55, 49) = input_block(54, 48);
            ibt[48] = input_block[63];
            ibt.range(47, 41) = input_block(46, 40);
            ibt[40] = input_block[55];
            ibt.range(39, 33) = input_block(38, 32);
            ibt[32] = input_block[47];
            ibt.range(31, 25) = input_block(30, 24);
            ibt[24] = input_block[39];
            ibt.range(23, 17) = input_block(22, 16);
            ibt[16] = input_block[31];
            ibt.range(15, 9) = input_block(14, 8);
            ibt[8] = input_block[23];
            ibt.range(7, 1) = input_block(6, 0);
            ibt[0] = input_block[15];
            input_block = ibt;

            if ((15 == cfb_byte_cnt) && (0 == cfb_bit_cnt)) { // the last bit of the last byte
                cfb_byte_cnt = 0;
                cfb_bit_cnt = 7;
            } else if (0 < cfb_bit_cnt) { // in the middle of each byte
                --cfb_bit_cnt;
            } else if (0 == cfb_bit_cnt) { // the last bit of each byte
                cfb_bit_cnt = 7;
                ++cfb_byte_cnt;
            }
        }
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k
        cipher.process(input_block, key_r, output_block);
// xf::security::internal::aesEncrypt<_keyWidth>(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // feedback for the next iteration and get the ciphertext for current interation
        ciphertext_r[cfb_byte_cnt * 8 + cfb_bit_cnt] = plaintext_r[cfb_byte_cnt * 8 + cfb_bit_cnt] ^ output_block[7];
        feedback_r[120] = ciphertext_r[cfb_byte_cnt * 8 + cfb_bit_cnt];
#ifndef __SYNTHESIS__
        std::cout << "feedback     = " << std::hex << feedback_r << std::endl;
        std::cout << "ciphertext   = " << std::hex << ciphertext_r << std::endl;
#endif

        // write out ciphertext and decide whether to read a new plaintext block or not
        next_plaintext = false;
        if ((15 == cfb_byte_cnt) && (0 == cfb_bit_cnt)) {
            ciphertext.write(ciphertext_r);
            ciphertext_e.write(0);
            next_plaintext = true;
        }

        if (next_plaintext) {
            e = plaintext_e.read();
        }
    }

    ciphertext_e.write(1);

} // end aesCfb1Encrypt

/**
 *
 * @brief aesCfb1Decrypt is CFB1 decryption mode with AES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @tparam _keyWidth The bit-width of the cipher key, which is 128, 192, or 256.
 *
 * @param ciphertext Input block stream text to be decrypted, each block is 128 bits.
 * @param ciphertext_e End flag of block stream ciphertext, 1 bit.
 * @param cipherkey Input cipher key used in decryption, x bits for AES-x.
 * @param IV_strm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintext Output decrypted block stream text, each block is 128 bits.
 * @param plaintext_e End flag of stream plaintext, 1 bit.
 *
 */

template <unsigned int _keyWidth = 256>
void aesCfb1Decrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertext,
    hls::stream<bool>& ciphertext_e,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<_keyWidth> >& cipherkey,
    hls::stream<ap_uint<128> >& IV_strm,
    // stream out
    hls::stream<ap_uint<128> >& plaintext,
    hls::stream<bool>& plaintext_e) {
    // register cipherkey
    ap_uint<_keyWidth> key_r = cipherkey.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif

    xf::security::aesEnc<_keyWidth> cipher;
    cipher.updateKey(key_r);
    // register IV
    ap_uint<128> IV = IV_strm.read();
#ifndef __SYNTHESIS__
    std::cout << "initialization_vector = " << std::hex << IV << std::endl << std::endl;
#endif

    // intermediate registers to perform the decryption chain
    bool next_ciphertext = true;
    ap_uint<128> ciphertext_r = 0;
    ap_uint<128> plaintext_r = 0;
    ap_uint<128> feedback_r = 0;
    ap_uint<128> input_block = 0;
    ap_uint<128> output_block = 0;
    ap_uint<4> cfb_byte_cnt = 0;
    ap_uint<3> cfb_bit_cnt = 7;

    // set the initialization for ture
    bool initialization = true;

    bool e = ciphertext_e.read();

decryption_cfb1_loop:
    while (!e) {
#pragma HLS PIPELINE II = 1
        // read a block of ciphertext, 128 bits
        if (next_ciphertext) { // mode cfb1 needs 128 iterations to process one ciphertext block
            ciphertext_r = ciphertext.read();
#ifndef __SYNTHESIS__
            std::cout << "ciphertext    = " << std::hex << ciphertext_r << std::endl;
#endif
        }

        // calculate input_block
        if (initialization) { // first iteration, input_block is IV
            input_block = IV;
            initialization = false;
        } else { // after first iteration, input_block is calculated by ciphertext and input_block of last iteration
            ap_uint<128> input_block_r;
            input_block_r.range(127, 120) = input_block.range(7, 0);
            input_block_r.range(119, 112) = input_block.range(15, 8);
            input_block_r.range(111, 104) = input_block.range(23, 16);
            input_block_r.range(103, 96) = input_block.range(31, 24);
            input_block_r.range(95, 88) = input_block.range(39, 32);
            input_block_r.range(87, 80) = input_block.range(47, 40);
            input_block_r.range(79, 72) = input_block.range(55, 48);
            input_block_r.range(71, 64) = input_block.range(63, 56);
            input_block_r.range(63, 56) = input_block.range(71, 64);
            input_block_r.range(55, 48) = input_block.range(79, 72);
            input_block_r.range(47, 40) = input_block.range(87, 80);
            input_block_r.range(39, 32) = input_block.range(95, 88);
            input_block_r.range(31, 24) = input_block.range(103, 96);
            input_block_r.range(23, 16) = input_block.range(111, 104);
            input_block_r.range(15, 8) = input_block.range(119, 112);
            input_block_r.range(7, 0) = input_block.range(127, 120);
            input_block_r = (input_block_r << 1) + feedback_r[120];
            input_block.range(127, 120) = input_block_r.range(7, 0);
            input_block.range(119, 112) = input_block_r.range(15, 8);
            input_block.range(111, 104) = input_block_r.range(23, 16);
            input_block.range(103, 96) = input_block_r.range(31, 24);
            input_block.range(95, 88) = input_block_r.range(39, 32);
            input_block.range(87, 80) = input_block_r.range(47, 40);
            input_block.range(79, 72) = input_block_r.range(55, 48);
            input_block.range(71, 64) = input_block_r.range(63, 56);
            input_block.range(63, 56) = input_block_r.range(71, 64);
            input_block.range(55, 48) = input_block_r.range(79, 72);
            input_block.range(47, 40) = input_block_r.range(87, 80);
            input_block.range(39, 32) = input_block_r.range(95, 88);
            input_block.range(31, 24) = input_block_r.range(103, 96);
            input_block.range(23, 16) = input_block_r.range(111, 104);
            input_block.range(15, 8) = input_block_r.range(119, 112);
            input_block.range(7, 0) = input_block_r.range(127, 120);

            if ((15 == cfb_byte_cnt) && (0 == cfb_bit_cnt)) { // the last bit of the last byte
                cfb_byte_cnt = 0;
                cfb_bit_cnt = 7;
            } else if (0 < cfb_bit_cnt) { // in the middle of each byte
                --cfb_bit_cnt;
            } else if (0 == cfb_bit_cnt) { // the last bit of each byte
                cfb_bit_cnt = 7;
                ++cfb_byte_cnt;
            }
        }
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k
        cipher.process(input_block, key_r, output_block);
// xf::security::internal::aesEncrypt<_keyWidth>(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // feedback for the next iteration and get the plaintext for current interation
        feedback_r[120] = ciphertext_r[cfb_byte_cnt * 8 + cfb_bit_cnt];
        plaintext_r[cfb_byte_cnt * 8 + cfb_bit_cnt] = ciphertext_r[cfb_byte_cnt * 8 + cfb_bit_cnt] ^ output_block[7];
#ifndef __SYNTHESIS__
        std::cout << "plaintext   = " << std::hex << plaintext_r << std::endl;
        std::cout << "feedback     = " << std::hex << feedback_r << std::endl;
#endif

        // write out plaintext
        next_ciphertext = false;
        if ((15 == cfb_byte_cnt) && (0 == cfb_bit_cnt)) {
            plaintext.write(plaintext_r);
            plaintext_e.write(0);
            next_ciphertext = true;
        }

        if (next_ciphertext) {
            e = ciphertext_e.read();
        }
    }

    plaintext_e.write(1);

} // end aesCfb1Decrypt

/**
 *
 * @brief aesCfb8Encrypt is CFB8 encryption mode with AES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @tparam _keyWidth The bit-width of the cipher key, which is 128, 192, or 256.
 *
 * @param plaintext Input block stream text to be encrypted, each block is 128 bits.
 * @param plaintext_e End flag of block stream plaintext, 1 bit.
 * @param cipherkey Input cipher key used in encryption, x bits for AES-x.
 * @param initialization_vector Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertext Output encrypted block stream text, each block is 128 bits.
 * @param ciphertext_e End flag of stream ciphertext, 1 bit.
 *
 */

template <unsigned int _keyWidth = 256>
void aesCfb8Encrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintext,
    hls::stream<bool>& plaintext_e,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<_keyWidth> >& cipherkey,
    hls::stream<ap_uint<128> >& initialization_vector,
    // stream out
    hls::stream<ap_uint<128> >& ciphertext,
    hls::stream<bool>& ciphertext_e) {
    // register cipherkey
    ap_uint<_keyWidth> key_r = cipherkey.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif
    xf::security::aesEnc<_keyWidth> cipher;
    cipher.updateKey(key_r);
    // register IV
    ap_uint<128> IV = initialization_vector.read();
#ifndef __SYNTHESIS__
    std::cout << "initialization_vector = " << std::hex << IV << std::endl << std::endl;
#endif

    // intermediate registers to perform the encryption chain
    bool next_plaintext = true;
    ap_uint<128> plaintext_r = 0;
    ap_uint<128> feedback_r = 0;
    ap_uint<128> input_block = 0;
    ap_uint<128> output_block = 0;
    ap_uint<128> ciphertext_r = 0;
    ap_uint<4> cfb_byte_cnt = 0;

    // set the initialization for ture
    bool initialization = true;

    bool e = plaintext_e.read();

encryption_cfb8_loop:
    while (!e) {
#pragma HLS PIPELINE
        // read a block of plaintext, 128 bits
        if (next_plaintext) { // mode CFB1/CFB8 needs multiple iteration to process one plaintext block
            plaintext_r = plaintext.read();
#ifndef __SYNTHESIS__
            std::cout << "plaintext    = " << std::hex << plaintext_r << std::endl;
#endif
        }

        // calculate input_block
        if (initialization) { // first iteration, input_block is IV
            input_block = IV;
            initialization = false;
        } else { // after first iteration, input_blcok is comprised by 120 bits of IV and 8 bits of ciphertext
            input_block = (input_block >> 8) + (feedback_r(7, 0) << 120);
            if (15 == cfb_byte_cnt) {
                cfb_byte_cnt = 0;
            } else {
                ++cfb_byte_cnt;
            }
        }
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k
        cipher.process(input_block, key_r, output_block);
// xf::security::internal::aesEncrypt<_keyWidth>(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // feedback for the next iteration and get the ciphertext for current interation
        ciphertext_r(cfb_byte_cnt * 8 + 7, cfb_byte_cnt * 8) =
            plaintext_r(cfb_byte_cnt * 8 + 7, cfb_byte_cnt * 8) ^ output_block(7, 0);
        feedback_r(7, 0) = ciphertext_r(cfb_byte_cnt * 8 + 7, cfb_byte_cnt * 8);
#ifndef __SYNTHESIS__
        std::cout << "feedback     = " << std::hex << feedback_r << std::endl;
        std::cout << "ciphertext   = " << std::hex << ciphertext_r << std::endl;
#endif

        // write out ciphertext and decide whether to read a new plaintext block or not
        next_plaintext = false;
        if (15 == cfb_byte_cnt) {
            ciphertext.write(ciphertext_r);
            ciphertext_e.write(0);
            next_plaintext = true;
        }

        if (next_plaintext) {
            e = plaintext_e.read();
        }
    }

    ciphertext_e.write(1);

} // end aesCfb8Encrypt

/**
 *
 * @brief aesCfb8Decrypt is CFB8 decryption mode with AES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @tparam _keyWidth The bit-width of the cipher key, which is 128, 192, or 256.
 *
 * @param ciphertext Input block stream text to be decrypted, each block is 128 bits.
 * @param ciphertext_e End flag of block stream ciphertext, 1 bit.
 * @param cipherkey Input cipher key used in decryption, x bits for AES-x.
 * @param IV_strm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintext Output decrypted block stream text, each block is 128 bits.
 * @param plaintext_e End flag of stream plaintext, 1 bit.
 *
 */

template <unsigned int _keyWidth>
void aesCfb8Decrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertext,
    hls::stream<bool>& ciphertext_e,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<_keyWidth> >& cipherkey,
    hls::stream<ap_uint<128> >& IV_strm,
    // stream out
    hls::stream<ap_uint<128> >& plaintext,
    hls::stream<bool>& plaintext_e) {
    // register cipherkey
    ap_uint<_keyWidth> key_r = cipherkey.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif
    xf::security::aesEnc<_keyWidth> cipher;
    cipher.updateKey(key_r);
    // register IV
    ap_uint<128> IV = IV_strm.read();
#ifndef __SYNTHESIS__
    std::cout << "initialization_vector = " << std::hex << IV << std::endl << std::endl;
#endif

    // intermediate registers to perform the decryption chain
    bool next_ciphertext = true;
    ap_uint<128> ciphertext_r = 0;
    ap_uint<128> plaintext_r = 0;
    ap_uint<128> feedback_r = 0;
    ap_uint<128> input_block = 0;
    ap_uint<128> output_block = 0;
    ap_uint<4> cfb_byte_cnt = 0;

    // set the initialization for ture
    bool initialization = true;

    bool e = ciphertext_e.read();

decryption_cfb8_loop:
    while (!e) {
#pragma HLS PIPELINE II = 1
        // read a block of ciphertext, 128 bits
        if (next_ciphertext) { // mode cfb8 needs 16 iterations to process one ciphertext block
            ciphertext_r = ciphertext.read();
#ifndef __SYNTHESIS__
            std::cout << "ciphertext    = " << std::hex << ciphertext_r << std::endl;
#endif
        }

        // calculate input_block
        if (initialization) { // first iteration, input_block is IV
            input_block = IV;
            initialization = false;
        } else { // after first iteration, input_block is calculated by ciphertext and input_block of last iteration
            input_block.range(119, 0) = input_block.range(127, 8);
            input_block.range(127, 120) = feedback_r.range(7, 0);
            if (15 == cfb_byte_cnt) {
                cfb_byte_cnt = 0;
            } else {
                ++cfb_byte_cnt;
            }
        }
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k
        cipher.process(input_block, key_r, output_block);
// xf::security::internal::aesEncrypt<_keyWidth>(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // feedback for the next iteration and get the plaintext for current interation
        feedback_r(7, 0) = ciphertext_r(cfb_byte_cnt * 8 + 7, cfb_byte_cnt * 8);
        plaintext_r(cfb_byte_cnt * 8 + 7, cfb_byte_cnt * 8) =
            ciphertext_r.range(cfb_byte_cnt * 8 + 7, cfb_byte_cnt * 8) ^ output_block.range(7, 0);
#ifndef __SYNTHESIS__
        std::cout << "plaintext   = " << std::hex << plaintext_r << std::endl;
        std::cout << "feedback     = " << std::hex << feedback_r << std::endl;
#endif

        // write out plaintext
        next_ciphertext = false;
        if (15 == cfb_byte_cnt) {
            plaintext.write(plaintext_r);
            plaintext_e.write(0);
            next_ciphertext = true;
        }

        if (next_ciphertext) {
            e = ciphertext_e.read();
        }
    }

    plaintext_e.write(1);

} // end aesCfb8Decrypt

/**
 *
 * @brief aesCfb128Encrypt is CFB128 encryption mode with AES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @tparam _keyWidth The bit-width of the cipher key, which is 128, 192, or 256.
 *
 * @param plaintext Input block stream text to be encrypted, each block is 128 bits.
 * @param plaintext_e End flag of block stream plaintext, 1 bit.
 * @param cipherkey Input cipher key used in encryption, x bits for AES-x.
 * @param initialization_vector Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertext Output encrypted block stream text, each block is 128 bits.
 * @param ciphertext_e End flag of stream ciphertext, 1 bit.
 *
 */

template <unsigned int _keyWidth = 256>
void aesCfb128Encrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintext,
    hls::stream<bool>& plaintext_e,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<_keyWidth> >& cipherkey,
    hls::stream<ap_uint<128> >& initialization_vector,
    // stream out
    hls::stream<ap_uint<128> >& ciphertext,
    hls::stream<bool>& ciphertext_e) {
    // register cipherkey
    ap_uint<_keyWidth> key_r = cipherkey.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif
    xf::security::aesEnc<_keyWidth> cipher;
    cipher.updateKey(key_r);
    // register IV
    ap_uint<128> IV = initialization_vector.read();
#ifndef __SYNTHESIS__
    std::cout << "initialization_vector = " << std::hex << IV << std::endl << std::endl;
#endif

    // intermediate registers to perform the encryption chain
    ap_uint<128> plaintext_r = 0;
    ap_uint<128> feedback_r = 0;
    ap_uint<128> input_block = 0;
    ap_uint<128> output_block = 0;
    ap_uint<128> ciphertext_r = 0;

    // set the initialization for ture
    bool initialization = true;

    bool e = plaintext_e.read();

encryption_cfb128_loop:
    while (!e) {
#pragma HLS PIPELINE
        // read a block of plaintext, 128 bits
        plaintext_r = plaintext.read();
#ifndef __SYNTHESIS__
        std::cout << "plaintext    = " << std::hex << plaintext_r << std::endl;
#endif

        // calculate input_block
        if (initialization) { // first iteration, input_block is IV
            input_block = IV;
            initialization = false;
        } else { // after first iteration, input_blcok is ciphertext of last iteration
            input_block = feedback_r;
        }
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k
        cipher.process(input_block, key_r, output_block);
// xf::security::internal::aesEncrypt<_keyWidth>(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // feedback for the next iteration and get the ciphertext for current interation
        ciphertext_r = plaintext_r ^ output_block;
        feedback_r = ciphertext_r;
#ifndef __SYNTHESIS__
        std::cout << "feedback     = " << std::hex << feedback_r << std::endl;
        std::cout << "ciphertext   = " << std::hex << ciphertext_r << std::endl;
#endif

        // write out ciphertext
        ciphertext.write(ciphertext_r);
        ciphertext_e.write(0);

        e = plaintext_e.read();
    }

    ciphertext_e.write(1);

} // end aesCfb128Encrypt

/**
 *
 * @brief aesCfb128Decrypt is CFB128 decryption mode with AES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @tparam _keyWidth The bit-width of the cipher key, which is 128, 192, or 256.
 *
 * @param ciphertext Input block stream text to be decrypted, each block is 128 bits.
 * @param ciphertext_e End flag of block stream ciphertext, 1 bit.
 * @param cipherkey Input cipher key used in decryption, x bits for AES-x.
 * @param IV_strm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintext Output decrypted block stream text, each block is 128 bits.
 * @param plaintext_e End flag of stream plaintext, 1 bit.
 *
 */

template <unsigned int _keyWidth = 256>
void aesCfb128Decrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertext,
    hls::stream<bool>& ciphertext_e,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<_keyWidth> >& cipherkey,
    hls::stream<ap_uint<128> >& IV_strm,
    // stream out
    hls::stream<ap_uint<128> >& plaintext,
    hls::stream<bool>& plaintext_e) {
    // register cipherkey
    ap_uint<_keyWidth> key_r = cipherkey.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif
    xf::security::aesEnc<_keyWidth> cipher;
    cipher.updateKey(key_r);
    // register IV
    ap_uint<128> IV = IV_strm.read();
#ifndef __SYNTHESIS__
    std::cout << "initialization_vector = " << std::hex << IV << std::endl << std::endl;
#endif

    // intermediate registers to perform the decryption chain
    ap_uint<128> ciphertext_r = 0;
    ap_uint<128> plaintext_r = 0;
    ap_uint<128> feedback_r = 0;
    ap_uint<128> input_block = 0;
    ap_uint<128> output_block = 0;

    // set the initialization for ture
    bool initialization = true;

    bool e = ciphertext_e.read();

decryption_cfb128_loop:
    while (!e) {
#pragma HLS PIPELINE II = 1
        // read a block of ciphertext, 128 bits
        ciphertext_r = ciphertext.read();
#ifndef __SYNTHESIS__
        std::cout << "ciphertext    = " << std::hex << ciphertext_r << std::endl;
#endif

        // calculate input_block
        if (initialization) { // first iteration, input_block is IV
            input_block = IV;
            initialization = false;
        } else { // after first iteration, input_block is ciphertext of last iteration
            input_block = feedback_r;
        }
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k
        cipher.process(input_block, key_r, output_block);
// xf::security::internal::aesEncrypt<_keyWidth>(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // feedback for the next iteration and get the plaintext for current interation
        feedback_r = ciphertext_r;
        plaintext_r = ciphertext_r ^ output_block;
#ifndef __SYNTHESIS__
        std::cout << "plaintext   = " << std::hex << plaintext_r << std::endl;
        std::cout << "feedback     = " << std::hex << feedback_r << std::endl;
#endif

        // write out plaintext
        plaintext.write(plaintext_r);
        plaintext_e.write(0);

        e = ciphertext_e.read();
    }

    plaintext_e.write(1);

} // end aesCfb128Decrypt

} // namespace internal

/**
 *
 * @brief desCfb1Encrypt is CFB1 encryption mode with DES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each block is 64 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 64 bits for each key.
 * @param IVStrm Initialization vector for the fisrt iteration of DES encrypition, 64 bits.
 * @param ciphertextStrm Output encrypted block stream text, each block is 64 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void desCfb1Encrypt(
    // stream in
    hls::stream<ap_uint<64> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<64> >& cipherkeyStrm,
    hls::stream<ap_uint<64> >& IVStrm,
    // stream out
    hls::stream<ap_uint<64> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    // register cipherkey
    ap_uint<64> key_r = cipherkeyStrm.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif
    // register IV
    ap_uint<64> IV = IVStrm.read();
#ifndef __SYNTHESIS__
    std::cout << "initialization_vector = " << std::hex << IV << std::endl << std::endl;
#endif

    // intermediate registers to perform the encryption chain
    bool next_plaintext = true;
    ap_uint<64> plaintext_r = 0;
    ap_uint<64> feedback_r = 0;
    ap_uint<64> input_block = 0;
    ap_uint<64> output_block = 0;
    ap_uint<64> ciphertext_r = 0;
    ap_uint<4> cfb_byte_cnt = 0;
    ap_uint<3> cfb_bit_cnt = 7;

    // set the initialization for ture
    bool initialization = true;

    bool e = endPlaintextStrm.read();

encryption_cfb1_loop:
    while (!e) {
#pragma HLS PIPELINE
        // read a block of plaintext, 64 bits
        if (next_plaintext) { // mode CFB1/CFB8 needs multiple iterations to process one plaintext block
            plaintext_r = plaintextStrm.read();
#ifndef __SYNTHESIS__
            std::cout << "plaintext    = " << std::hex << plaintext_r << std::endl;
#endif
        }

        // calculate input_block
        if (initialization) { // first iteration, input_block is IV
            input_block = IV;
            initialization = false;
        } else { // after first iteration, input_blcok is comprised by 127 bits of IV and 1 bit of ciphertext
            ap_uint<64> ibt;
            ibt.range(63, 57) = input_block(62, 56);
            ibt[56] = feedback_r[56];
            ibt.range(55, 49) = input_block(54, 48);
            ibt[48] = input_block[63];
            ibt.range(47, 41) = input_block(46, 40);
            ibt[40] = input_block[55];
            ibt.range(39, 33) = input_block(38, 32);
            ibt[32] = input_block[47];
            ibt.range(31, 25) = input_block(30, 24);
            ibt[24] = input_block[39];
            ibt.range(23, 17) = input_block(22, 16);
            ibt[16] = input_block[31];
            ibt.range(15, 9) = input_block(14, 8);
            ibt[8] = input_block[23];
            ibt.range(7, 1) = input_block(6, 0);
            ibt[0] = input_block[15];
            input_block = ibt;

            if ((7 == cfb_byte_cnt) && (0 == cfb_bit_cnt)) { // the last bit of the last byte
                cfb_byte_cnt = 0;
                cfb_bit_cnt = 7;
            } else if (0 < cfb_bit_cnt) { // in the middle of each byte
                --cfb_bit_cnt;
            } else if (0 == cfb_bit_cnt) { // the last bit of each byte
                cfb_bit_cnt = 7;
                ++cfb_byte_cnt;
            }
        }
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k
        xf::security::desEncrypt(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // feedback for the next iteration and get the ciphertext for current interation
        ciphertext_r[cfb_byte_cnt * 8 + cfb_bit_cnt] = plaintext_r[cfb_byte_cnt * 8 + cfb_bit_cnt] ^ output_block[7];
        feedback_r[56] = ciphertext_r[cfb_byte_cnt * 8 + cfb_bit_cnt];
#ifndef __SYNTHESIS__
        std::cout << "feedback     = " << std::hex << feedback_r << std::endl;
        std::cout << "ciphertext   = " << std::hex << ciphertext_r << std::endl;
#endif

        // write out ciphertext and decide whether to read a new plaintext block or not
        next_plaintext = false;
        if ((7 == cfb_byte_cnt) && (0 == cfb_bit_cnt)) {
            ciphertextStrm.write(ciphertext_r);
            endCiphertextStrm.write(0);
            next_plaintext = true;
        }

        if (next_plaintext) {
            e = endPlaintextStrm.read();
        }
    }

    endCiphertextStrm.write(1);

} // end desCfb1Encrypt

/**
 *
 * @brief desCfb1Decrypt is CFB1 decryption mode with DES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 64 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 64 bits for each key.
 * @param IVStrm Initialization vector for the fisrt iteration of DES decrypition, 64 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 64 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void desCfb1Decrypt(
    // stream in
    hls::stream<ap_uint<64> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<64> >& cipherkeyStrm,
    hls::stream<ap_uint<64> >& IVStrm,
    // stream out
    hls::stream<ap_uint<64> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    // register cipherkey
    ap_uint<64> key_r = cipherkeyStrm.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif
    // register IV
    ap_uint<64> IV = IVStrm.read();
#ifndef __SYNTHESIS__
    std::cout << "initialization_vector = " << std::hex << IV << std::endl << std::endl;
#endif

    // intermediate registers to perform the decryption chain
    bool next_ciphertext = true;
    ap_uint<64> ciphertext_r = 0;
    ap_uint<64> plaintext_r = 0;
    ap_uint<64> feedback_r = 0;
    ap_uint<64> input_block = 0;
    ap_uint<64> output_block = 0;
    ap_uint<4> cfb_byte_cnt = 0;
    ap_uint<3> cfb_bit_cnt = 7;

    // set the initialization for ture
    bool initialization = true;

    bool e = endCiphertextStrm.read();

decryption_cfb1_loop:
    while (!e) {
#pragma HLS PIPELINE II = 1
        // read a block of ciphertext, 64 bits
        if (next_ciphertext) { // mode cfb1 needs 64 iterations to process one ciphertext block
            ciphertext_r = ciphertextStrm.read();
#ifndef __SYNTHESIS__
            std::cout << "ciphertext    = " << std::hex << ciphertext_r << std::endl;
#endif
        }

        // calculate input_block
        if (initialization) { // first iteration, input_block is IV
            input_block = IV;
            initialization = false;
        } else { // after first iteration, input_block is calculated by ciphertext and input_block of last iteration
            ap_uint<64> input_block_r;
            input_block_r.range(63, 56) = input_block.range(7, 0);
            input_block_r.range(55, 48) = input_block.range(15, 8);
            input_block_r.range(47, 40) = input_block.range(23, 16);
            input_block_r.range(39, 32) = input_block.range(31, 24);
            input_block_r.range(31, 24) = input_block.range(39, 32);
            input_block_r.range(23, 16) = input_block.range(47, 40);
            input_block_r.range(15, 8) = input_block.range(55, 48);
            input_block_r.range(7, 0) = input_block.range(63, 56);
            input_block_r = (input_block_r << 1) + feedback_r[56];
            input_block.range(63, 56) = input_block_r.range(7, 0);
            input_block.range(55, 48) = input_block_r.range(15, 8);
            input_block.range(47, 40) = input_block_r.range(23, 16);
            input_block.range(39, 32) = input_block_r.range(31, 24);
            input_block.range(31, 24) = input_block_r.range(39, 32);
            input_block.range(23, 16) = input_block_r.range(47, 40);
            input_block.range(15, 8) = input_block_r.range(55, 48);
            input_block.range(7, 0) = input_block_r.range(63, 56);

            if ((7 == cfb_byte_cnt) && (0 == cfb_bit_cnt)) { // the last bit of the last byte
                cfb_byte_cnt = 0;
                cfb_bit_cnt = 7;
            } else if (0 < cfb_bit_cnt) { // in the middle of each byte
                --cfb_bit_cnt;
            } else if (0 == cfb_bit_cnt) { // the last bit of each byte
                cfb_bit_cnt = 7;
                ++cfb_byte_cnt;
            }
        }
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k
        xf::security::desEncrypt(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // feedback for the next iteration and get the plaintext for current interation
        feedback_r[56] = ciphertext_r[cfb_byte_cnt * 8 + cfb_bit_cnt];
        plaintext_r[cfb_byte_cnt * 8 + cfb_bit_cnt] = ciphertext_r[cfb_byte_cnt * 8 + cfb_bit_cnt] ^ output_block[7];
#ifndef __SYNTHESIS__
        std::cout << "plaintext   = " << std::hex << plaintext_r << std::endl;
        std::cout << "feedback     = " << std::hex << feedback_r << std::endl;
#endif

        // write out plaintext
        next_ciphertext = false;
        if ((7 == cfb_byte_cnt) && (0 == cfb_bit_cnt)) {
            plaintextStrm.write(plaintext_r);
            endPlaintextStrm.write(0);
            next_ciphertext = true;
        }

        if (next_ciphertext) {
            e = endCiphertextStrm.read();
        }
    }

    endPlaintextStrm.write(1);

} // end desCfb1Decrypt

/**
 *
 * @brief aes128Cfb1Encrypt is CFB1 encryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each text block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 128 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, each cipher block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes128Cfb1Encrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<128> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesCfb1Encrypt<128>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, IVStrm, ciphertextStrm,
                                  endCiphertextStrm);

} // end aes128Cfb1Encrypt

/**
 *
 * @brief aes128Cfb1Decrypt is CFB1 decryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 128 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void aes128Cfb1Decrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<128> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesCfb1Decrypt<128>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, IVStrm, plaintextStrm,
                                  endPlaintextStrm);

} // end aes128Cfb1Decrypt

/**
 *
 * @brief aes192Cfb1Encrypt is CFB1 encryption mode with AES-192 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each text block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 192 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, each cipher block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes192Cfb1Encrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<192> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesCfb1Encrypt<192>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, IVStrm, ciphertextStrm,
                                  endCiphertextStrm);

} // end aes192Cfb1Encrypt

/**
 *
 * @brief aes192Cfb1Decrypt is CFB1 decryption mode with AES-192 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 192 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void aes192Cfb1Decrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<192> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesCfb1Decrypt<192>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, IVStrm, plaintextStrm,
                                  endPlaintextStrm);

} // end aes192Cfb1Decrypt

/**
 *
 * @brief aes256Cfb1Encrypt is CFB1 encryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each text block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 256 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, each cipher block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes256Cfb1Encrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<256> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesCfb1Encrypt<256>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, IVStrm, ciphertextStrm,
                                  endCiphertextStrm);

} // end aes256Cfb1Encrypt

/**
 *
 * @brief aes256Cfb1Decrypt is CFB1 decryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 256 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void aes256Cfb1Decrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<256> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesCfb1Decrypt<256>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, IVStrm, plaintextStrm,
                                  endPlaintextStrm);

} // end aes256Cfb1Decrypt

/**
 *
 * @brief desCfb8Encrypt is CFB8 encryption mode with DES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each block is 64 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 64 bits for each key.
 * @param IVStrm Initialization vector for the fisrt iteration of DES encrypition, 64 bits.
 * @param ciphertextStrm Output encrypted block stream text, each block is 64 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void desCfb8Encrypt(
    // stream in
    hls::stream<ap_uint<64> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<64> >& cipherkeyStrm,
    hls::stream<ap_uint<64> >& IVStrm,
    // stream out
    hls::stream<ap_uint<64> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    // register cipherkey
    ap_uint<64> key_r = cipherkeyStrm.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif
    // register IV
    ap_uint<64> IV = IVStrm.read();
#ifndef __SYNTHESIS__
    std::cout << "initialization_vector = " << std::hex << IV << std::endl << std::endl;
#endif

    // intermediate registers to perform the encryption chain
    bool next_plaintext = true;
    ap_uint<64> plaintext_r = 0;
    ap_uint<64> feedback_r = 0;
    ap_uint<64> input_block = 0;
    ap_uint<64> output_block = 0;
    ap_uint<64> ciphertext_r = 0;
    ap_uint<4> cfb_byte_cnt = 0;

    // set the initialization for ture
    bool initialization = true;

    bool e = endPlaintextStrm.read();

encryption_cfb8_loop:
    while (!e) {
#pragma HLS PIPELINE
        // read a block of plaintext, 64 bits
        if (next_plaintext) { // mode CFB1/CFB8 needs multiple iteration to process one plaintext block
            plaintext_r = plaintextStrm.read();
#ifndef __SYNTHESIS__
            std::cout << "plaintext    = " << std::hex << plaintext_r << std::endl;
#endif
        }

        // calculate input_block
        if (initialization) { // first iteration, input_block is IV
            input_block = IV;
            initialization = false;
        } else { // after first iteration, input_blcok is comprised by 120 bits of IV and 8 bits of ciphertext
            input_block = (input_block >> 8) + (feedback_r(7, 0) << 56);
            if (7 == cfb_byte_cnt) {
                cfb_byte_cnt = 0;
            } else {
                ++cfb_byte_cnt;
            }
        }
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k
        xf::security::desEncrypt(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // feedback for the next iteration and get the ciphertext for current interation
        ciphertext_r(cfb_byte_cnt * 8 + 7, cfb_byte_cnt * 8) =
            plaintext_r(cfb_byte_cnt * 8 + 7, cfb_byte_cnt * 8) ^ output_block(7, 0);
        feedback_r(7, 0) = ciphertext_r(cfb_byte_cnt * 8 + 7, cfb_byte_cnt * 8);
#ifndef __SYNTHESIS__
        std::cout << "feedback     = " << std::hex << feedback_r << std::endl;
        std::cout << "ciphertext   = " << std::hex << ciphertext_r << std::endl;
#endif

        // write out ciphertext and decide whether to read a new plaintext block or not
        next_plaintext = false;
        if (7 == cfb_byte_cnt) {
            ciphertextStrm.write(ciphertext_r);
            endCiphertextStrm.write(0);
            next_plaintext = true;
        }

        if (next_plaintext) {
            e = endPlaintextStrm.read();
        }
    }

    endCiphertextStrm.write(1);

} // end desCfb8Encrypt

/**
 *
 * @brief desCfb8Decrypt is CFB8 decryption mode with DES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 64 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 64 bits for each key.
 * @param IVStrm Initialization vector for the fisrt iteration of DES decrypition, 64 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 64 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void desCfb8Decrypt(
    // stream in
    hls::stream<ap_uint<64> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<64> >& cipherkeyStrm,
    hls::stream<ap_uint<64> >& IVStrm,
    // stream out
    hls::stream<ap_uint<64> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    // register cipherkey
    ap_uint<64> key_r = cipherkeyStrm.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif
    // register IV
    ap_uint<64> IV = IVStrm.read();
#ifndef __SYNTHESIS__
    std::cout << "initialization_vector = " << std::hex << IV << std::endl << std::endl;
#endif

    // intermediate registers to perform the decryption chain
    bool next_ciphertext = true;
    ap_uint<64> ciphertext_r = 0;
    ap_uint<64> plaintext_r = 0;
    ap_uint<64> feedback_r = 0;
    ap_uint<64> input_block = 0;
    ap_uint<64> output_block = 0;
    ap_uint<4> cfb_byte_cnt = 0;

    // set the initialization for ture
    bool initialization = true;

    bool e = endCiphertextStrm.read();

decryption_cfb8_loop:
    while (!e) {
#pragma HLS PIPELINE II = 1
        // read a block of ciphertext, 64 bits
        if (next_ciphertext) { // mode cfb8 needs 8 iterations to process one ciphertext block
            ciphertext_r = ciphertextStrm.read();
#ifndef __SYNTHESIS__
            std::cout << "ciphertext    = " << std::hex << ciphertext_r << std::endl;
#endif
        }

        // calculate input_block
        if (initialization) { // first iteration, input_block is IV
            input_block = IV;
            initialization = false;
        } else { // after first iteration, input_block is calculated by ciphertext and input_block of last iteration
            input_block.range(55, 0) = input_block.range(63, 8);
            input_block.range(63, 56) = feedback_r.range(7, 0);
            if (7 == cfb_byte_cnt) {
                cfb_byte_cnt = 0;
            } else {
                ++cfb_byte_cnt;
            }
        }
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k
        xf::security::desEncrypt(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // feedback for the next iteration and get the plaintext for current interation
        feedback_r(7, 0) = ciphertext_r(cfb_byte_cnt * 8 + 7, cfb_byte_cnt * 8);
        plaintext_r(cfb_byte_cnt * 8 + 7, cfb_byte_cnt * 8) =
            ciphertext_r.range(cfb_byte_cnt * 8 + 7, cfb_byte_cnt * 8) ^ output_block.range(7, 0);
#ifndef __SYNTHESIS__
        std::cout << "plaintext   = " << std::hex << plaintext_r << std::endl;
        std::cout << "feedback     = " << std::hex << feedback_r << std::endl;
#endif

        // write out plaintext
        next_ciphertext = false;
        if (7 == cfb_byte_cnt) {
            plaintextStrm.write(plaintext_r);
            endPlaintextStrm.write(0);
            next_ciphertext = true;
        }

        if (next_ciphertext) {
            e = endCiphertextStrm.read();
        }
    }

    endPlaintextStrm.write(1);

} // end desCfb8Decrypt

/**
 *
 * @brief aes128Cfb8Encrypt is CFB8 encryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each text block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 128 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, each cipher block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes128Cfb8Encrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<128> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesCfb8Encrypt<128>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, IVStrm, ciphertextStrm,
                                  endCiphertextStrm);

} // end aes128Cfb8Encrypt

/**
 *
 * @brief aes128Cfb8Decrypt is CFB8 decryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 128 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void aes128Cfb8Decrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<128> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesCfb8Decrypt<128>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, IVStrm, plaintextStrm,
                                  endPlaintextStrm);

} // end aes128Cfb8Decrypt

/**
 *
 * @brief aes192Cfb8Encrypt is CFB8 encryption mode with AES-192 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each text block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 192 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, each cipher block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes192Cfb8Encrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<192> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesCfb8Encrypt<192>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, IVStrm, ciphertextStrm,
                                  endCiphertextStrm);

} // end aes192Cfb8Encrypt

/**
 *
 * @brief aes192Cfb8Decrypt is CFB8 decryption mode with AES-192 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 192 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void aes192Cfb8Decrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<192> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesCfb8Decrypt<192>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, IVStrm, plaintextStrm,
                                  endPlaintextStrm);

} // end aes192Cfb8Decrypt

/**
 *
 * @brief aes256Cfb8Encrypt is CFB8 encryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each text block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 256 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, each cipher block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes256Cfb8Encrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<256> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesCfb8Encrypt<256>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, IVStrm, ciphertextStrm,
                                  endCiphertextStrm);

} // end aes256Cfb8Encrypt

/**
 *
 * @brief aes256Cfb8Decrypt is CFB8 decryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 256 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void aes256Cfb8Decrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<256> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesCfb8Decrypt<256>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, IVStrm, plaintextStrm,
                                  endPlaintextStrm);

} // end aes256Cfb8Decrypt

/**
 *
 * @brief desCfb128Encrypt is CFB128 encryption mode with DES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each block is 64 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 64 bits for each key.
 * @param IVStrm Initialization vector for the fisrt iteration of DES encrypition, 64 bits.
 * @param ciphertextStrm Output encrypted block stream text, each block is 64 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void desCfb128Encrypt(
    // stream in
    hls::stream<ap_uint<64> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<64> >& cipherkeyStrm,
    hls::stream<ap_uint<64> >& IVStrm,
    // stream out
    hls::stream<ap_uint<64> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    // register cipherkey
    ap_uint<64> key_r = cipherkeyStrm.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif
    // register IV
    ap_uint<64> IV = IVStrm.read();
#ifndef __SYNTHESIS__
    std::cout << "initialization_vector = " << std::hex << IV << std::endl << std::endl;
#endif

    // intermediate registers to perform the encryption chain
    ap_uint<64> plaintext_r = 0;
    ap_uint<64> feedback_r = 0;
    ap_uint<64> input_block = 0;
    ap_uint<64> output_block = 0;
    ap_uint<64> ciphertext_r = 0;

    // set the initialization for ture
    bool initialization = true;

    bool e = endPlaintextStrm.read();

encryption_cfb128_loop:
    while (!e) {
#pragma HLS PIPELINE
        // read a block of plaintext, 64 bits
        plaintext_r = plaintextStrm.read();
#ifndef __SYNTHESIS__
        std::cout << "plaintext    = " << std::hex << plaintext_r << std::endl;
#endif

        // calculate input_block
        if (initialization) { // first iteration, input_block is IV
            input_block = IV;
            initialization = false;
        } else { // after first iteration, input_blcok is ciphertext of last iteration
            input_block = feedback_r;
        }
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k
        xf::security::desEncrypt(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // feedback for the next iteration and get the ciphertext for current interation
        ciphertext_r = plaintext_r ^ output_block;
        feedback_r = ciphertext_r;
#ifndef __SYNTHESIS__
        std::cout << "feedback     = " << std::hex << feedback_r << std::endl;
        std::cout << "ciphertext   = " << std::hex << ciphertext_r << std::endl;
#endif

        // write out ciphertext
        ciphertextStrm.write(ciphertext_r);
        endCiphertextStrm.write(0);

        e = endPlaintextStrm.read();
    }

    endCiphertextStrm.write(1);

} // end desCfb128Encrypt

/**
 *
 * @brief desCfb128Decrypt is CFB128 decryption mode with DES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 64 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 64 bits for each key.
 * @param IVStrm Initialization vector for the fisrt iteration of DES decrypition, 64 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 64 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void desCfb128Decrypt(
    // stream in
    hls::stream<ap_uint<64> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<64> >& cipherkeyStrm,
    hls::stream<ap_uint<64> >& IVStrm,
    // stream out
    hls::stream<ap_uint<64> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    // register cipherkey
    ap_uint<64> key_r = cipherkeyStrm.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif

    // register IV
    ap_uint<64> IV = IVStrm.read();
#ifndef __SYNTHESIS__
    std::cout << "initialization_vector = " << std::hex << IV << std::endl << std::endl;
#endif
    // intermediate registers to perform the decryption chain
    ap_uint<64> ciphertext_r = 0;
    ap_uint<64> plaintext_r = 0;
    ap_uint<64> feedback_r = 0;
    ap_uint<64> input_block = 0;
    ap_uint<64> output_block = 0;

    // set the initialization for ture
    bool initialization = true;

    bool e = endCiphertextStrm.read();

decryption_cfb128_loop:
    while (!e) {
#pragma HLS PIPELINE II = 1
        // read a block of ciphertext, 64 bits
        ciphertext_r = ciphertextStrm.read();
#ifndef __SYNTHESIS__
        std::cout << "ciphertext    = " << std::hex << ciphertext_r << std::endl;
#endif

        // calculate input_block
        if (initialization) { // first iteration, input_block is IV
            input_block = IV;
            initialization = false;
        } else { // after first iteration, input_block is ciphertext of last iteration
            input_block = feedback_r;
        }
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k
        xf::security::desEncrypt(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // feedback for the next iteration and get the plaintext for current interation
        feedback_r = ciphertext_r;
        plaintext_r = ciphertext_r ^ output_block;
#ifndef __SYNTHESIS__
        std::cout << "plaintext   = " << std::hex << plaintext_r << std::endl;
        std::cout << "feedback     = " << std::hex << feedback_r << std::endl;
#endif

        // write out plaintext
        plaintextStrm.write(plaintext_r);
        endPlaintextStrm.write(0);

        e = endCiphertextStrm.read();
    }

    endPlaintextStrm.write(1);

} // end desCfb128Decrypt

/**
 *
 * @brief aes128Cfb128Encrypt is CFB128 encryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each text block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 128 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, each cipher block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes128Cfb128Encrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<128> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesCfb128Encrypt<128>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, IVStrm, ciphertextStrm,
                                    endCiphertextStrm);

} // end aes128Cfb128Encrypt

/**
 *
 * @brief aes128Cfb128Decrypt is CFB128 decryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 128 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void aes128Cfb128Decrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<128> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesCfb128Decrypt<128>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, IVStrm, plaintextStrm,
                                    endPlaintextStrm);

} // end aes128Cfb128Decrypt

/**
 *
 * @brief aes192Cfb128Encrypt is CFB128 encryption mode with AES-192 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each text block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 192 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, each cipher block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes192Cfb128Encrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<192> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesCfb128Encrypt<192>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, IVStrm, ciphertextStrm,
                                    endCiphertextStrm);

} // end aes192Cfb128Encrypt

/**
 *
 * @brief aes192Cfb128Decrypt is CFB128 decryption mode with AES-192 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 192 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void aes192Cfb128Decrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<192> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesCfb128Decrypt<192>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, IVStrm, plaintextStrm,
                                    endPlaintextStrm);

} // end aes192Cfb128Decrypt

/**
 *
 * @brief aes256Cfb128Encrypt is CFB128 encryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each text block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 256 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, each cipher block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes256Cfb128Encrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<256> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesCfb128Encrypt<256>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, IVStrm, ciphertextStrm,
                                    endCiphertextStrm);

} // end aes256Cfb128Encrypt

/**
 *
 * @brief aes256Cfb128Decrypt is CFB128 decryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 256 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void aes256Cfb128Decrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<256> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesCfb128Decrypt<256>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, IVStrm, plaintextStrm,
                                    endPlaintextStrm);

} // end aes256Cfb128Decrypt

} // namespace security
} // namespace xf

#endif
