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
 * @file ctr.hpp
 * @brief header file for Counter (CTR) block cipher mode of operation.
 * This file is part of Vitis Security Library.
 *
 * @detail Containing CTR mode with AES-128/192/256.
 * There is no loop-carried dependency in both encryption and decryption parts of CTR mode
 *
 */

#ifndef _XF_SECURITY_CTR_HPP_
#define _XF_SECURITY_CTR_HPP_

#include <ap_int.h>
#include <hls_stream.h>

#include "aes.hpp"

// for debug
#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace security {
namespace internal {

/**
 *
 * @brief aesCtrEncrypt is CTR encryption mode with AES single block cipher.
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
 * @param ciphertext Output encrypted block stream text, 128 bits.
 * @param ciphertext_e End flag of block stream ciphertext, 1 bit.
 *
 */

template <unsigned int _keyWidth = 256>
void aesCtrEncrypt(
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
    ap_uint<128> plaintext_r = 0;
    ap_uint<128> input_block = 0;
    ap_uint<128> input_block_r = 0;
    ap_uint<128> output_block = 0;
    ap_uint<128> ciphertext_r = 0;

    // set the initialization for ture
    bool initialization = true;

    bool e = plaintext_e.read();

encryption_ctr_loop:
    while (!e) {
#pragma HLS PIPELINE II = 1
        // read a block of plaintext, 128 bits
        plaintext_r = plaintext.read();
#ifndef __SYNTHESIS__
        std::cout << "plaintext    = " << std::hex << plaintext_r << std::endl;
#endif

        // calculate input_block
        if (initialization) {
            input_block = IV;
            initialization = false;
        } else {
            input_block_r.range(127, 120) = input_block(7, 0);
            input_block_r.range(119, 112) = input_block(15, 8);
            input_block_r.range(111, 104) = input_block(23, 16);
            input_block_r.range(103, 96) = input_block(31, 24);
            input_block_r.range(95, 88) = input_block(39, 32);
            input_block_r.range(87, 80) = input_block(47, 40);
            input_block_r.range(79, 72) = input_block(55, 48);
            input_block_r.range(71, 64) = input_block(63, 56);
            input_block_r.range(63, 56) = input_block(71, 64);
            input_block_r.range(55, 48) = input_block(79, 72);
            input_block_r.range(47, 40) = input_block(87, 80);
            input_block_r.range(39, 32) = input_block(95, 88);
            input_block_r.range(31, 24) = input_block(103, 96);
            input_block_r.range(23, 16) = input_block(111, 104);
            input_block_r.range(15, 8) = input_block(119, 112);
            input_block_r.range(7, 0) = input_block(127, 120);
            ++input_block_r;
            input_block.range(127, 120) = input_block_r(7, 0);
            input_block.range(119, 112) = input_block_r(15, 8);
            input_block.range(111, 104) = input_block_r(23, 16);
            input_block.range(103, 96) = input_block_r(31, 24);
            input_block.range(95, 88) = input_block_r(39, 32);
            input_block.range(87, 80) = input_block_r(47, 40);
            input_block.range(79, 72) = input_block_r(55, 48);
            input_block.range(71, 64) = input_block_r(63, 56);
            input_block.range(63, 56) = input_block_r(71, 64);
            input_block.range(55, 48) = input_block_r(79, 72);
            input_block.range(47, 40) = input_block_r(87, 80);
            input_block.range(39, 32) = input_block_r(95, 88);
            input_block.range(31, 24) = input_block_r(103, 96);
            input_block.range(23, 16) = input_block_r(111, 104);
            input_block.range(15, 8) = input_block_r(119, 112);
            input_block.range(7, 0) = input_block_r(127, 120);
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

        // get the ciphertext for current interation by output_block and plaintext
        ciphertext_r = plaintext_r ^ output_block;
#ifndef __SYNTHESIS__
        std::cout << "ciphertext   = " << std::hex << ciphertext_r << std::endl;
#endif

        // write out ciphertext
        ciphertext.write(ciphertext_r);
        ciphertext_e.write(0);

        e = plaintext_e.read();
    }

    ciphertext_e.write(1);

} // end aesCtrEncrypt

/**
 *
 * @brief aesCtrDecrypt is CTR decryption mode with AES single block cipher.
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
 * @param plaintext_e End flag of block stream plaintext, 1 bit.
 *
 */

template <unsigned int _keyWidth = 256>
void aesCtrDecrypt(
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
    ap_uint<128> input_block = 0;
    ap_uint<128> input_block_r = 0;
    ap_uint<128> output_block = 0;
    ap_uint<128> plaintext_r = 0;

    // set the initialization for ture
    bool initialization = true;

    bool e = ciphertext_e.read();

decryption_ctr_loop:
    while (!e) {
#pragma HLS PIPELINE II = 1
        // read a block of ciphertext, 128 bits
        ciphertext_r = ciphertext.read();
#ifndef __SYNTHESIS__
        std::cout << "ciphertext    = " << std::hex << ciphertext_r << std::endl;
#endif

        // calculate input_block
        if (initialization) {
            input_block = IV;
            initialization = false;
        } else {
            input_block_r.range(127, 120) = input_block(7, 0);
            input_block_r.range(119, 112) = input_block(15, 8);
            input_block_r.range(111, 104) = input_block(23, 16);
            input_block_r.range(103, 96) = input_block(31, 24);
            input_block_r.range(95, 88) = input_block(39, 32);
            input_block_r.range(87, 80) = input_block(47, 40);
            input_block_r.range(79, 72) = input_block(55, 48);
            input_block_r.range(71, 64) = input_block(63, 56);
            input_block_r.range(63, 56) = input_block(71, 64);
            input_block_r.range(55, 48) = input_block(79, 72);
            input_block_r.range(47, 40) = input_block(87, 80);
            input_block_r.range(39, 32) = input_block(95, 88);
            input_block_r.range(31, 24) = input_block(103, 96);
            input_block_r.range(23, 16) = input_block(111, 104);
            input_block_r.range(15, 8) = input_block(119, 112);
            input_block_r.range(7, 0) = input_block(127, 120);
            ++input_block_r;
            input_block.range(127, 120) = input_block_r(7, 0);
            input_block.range(119, 112) = input_block_r(15, 8);
            input_block.range(111, 104) = input_block_r(23, 16);
            input_block.range(103, 96) = input_block_r(31, 24);
            input_block.range(95, 88) = input_block_r(39, 32);
            input_block.range(87, 80) = input_block_r(47, 40);
            input_block.range(79, 72) = input_block_r(55, 48);
            input_block.range(71, 64) = input_block_r(63, 56);
            input_block.range(63, 56) = input_block_r(71, 64);
            input_block.range(55, 48) = input_block_r(79, 72);
            input_block.range(47, 40) = input_block_r(87, 80);
            input_block.range(39, 32) = input_block_r(95, 88);
            input_block.range(31, 24) = input_block_r(103, 96);
            input_block.range(23, 16) = input_block_r(111, 104);
            input_block.range(15, 8) = input_block_r(119, 112);
            input_block.range(7, 0) = input_block_r(127, 120);
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

        // get the plaintext for current interation by output_block and ciphertext
        plaintext_r = ciphertext_r ^ output_block;
#ifndef __SYNTHESIS__
        std::cout << "plaintext   = " << std::hex << plaintext_r << std::endl;
#endif

        // write out plaintext
        plaintext.write(plaintext_r);
        plaintext_e.write(0);

        e = ciphertext_e.read();
    }

    plaintext_e.write(1);

} // end aesCtrDecrypt

} // namespace internal

/**
 *
 * @brief aes128CtrEncrypt is CTR encryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 128 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes128CtrEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<128> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesCtrEncrypt<128>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, IVStrm, ciphertextStrm,
                                 endCiphertextStrm);

} // end aes128CtrEncrypt

/**
 *
 * @brief aes128CtrDecrypt is CTR decryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 128 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of stream plaintext, 1 bit.
 *
 */

static void aes128CtrDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<128> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesCtrDecrypt<128>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, IVStrm, plaintextStrm,
                                 endPlaintextStrm);

} // end aes128CtrDecrypt

/**
 *
 * @brief aes192CtrEncrypt is CTR encryption mode with AES-192 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 192 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes192CtrEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<192> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesCtrEncrypt<192>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, IVStrm, ciphertextStrm,
                                 endCiphertextStrm);

} // end aes192CtrEncrypt

/**
 *
 * @brief aes192CtrDecrypt is CTR decryption mode with AES-192 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 192 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of stream plaintext, 1 bit.
 *
 */

static void aes192CtrDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<192> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesCtrDecrypt<192>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, IVStrm, plaintextStrm,
                                 endPlaintextStrm);

} // end aes192CtrDecrypt

/**
 *
 * @brief aes256CtrEncrypt is CTR encryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 256 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES encrypition, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes256CtrEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey and initialization vector
    hls::stream<ap_uint<256> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesCtrEncrypt<256>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, IVStrm, ciphertextStrm,
                                 endCiphertextStrm);

} // end aes256CtrEncrypt

/**
 *
 * @brief aes256CtrDecrypt is CTR decryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream text to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 256 bits.
 * @param IVStrm Initialization vector for the fisrt iteration of AES decrypition, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of stream plaintext, 1 bit.
 *
 */

static void aes256CtrDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey & initialization vector
    hls::stream<ap_uint<256> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& IVStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesCtrDecrypt<256>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, IVStrm, plaintextStrm,
                                 endPlaintextStrm);

} // end aes256CtrDecrypt

} // namespace security
} // namespace xf
#endif
