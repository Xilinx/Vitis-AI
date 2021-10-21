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
 * @file ecb.hpp
 * @brief header file for Electronic Codebook Mode (ECB) block cipher mode of operation.
 * This file is part of Vitis Security Library.
 *
 * @detail Containing ECB mode with AES-128/192/256 and DES.
 * There is no loop-carried dependency in both encryption and decryption parts.
 *
 */

#ifndef _XF_SECURITY_ECB_HPP_
#define _XF_SECURITY_ECB_HPP_

#include <ap_int.h>
#include <hls_stream.h>

#include "aes.hpp"
#include "des.hpp"

// for debug
#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace security {
namespace internal {

/**
 *
 * @brief aesEcbEncrypt is ECB encryption mode with AES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @tparam _keyWidth The bit-width of the cipher key, which is 128, 192, or 256.
 *
 * @param plaintext Input block stream text to be encrypted, each block is 128 bits.
 * @param plaintext_e End flag of block stream plaintext, 1 bit.
 * @param cipherkey Input cipher key used in encryption, x bits for AES-x.
 * @param ciphertext Output encrypted block stream text, each block is 128 bits.
 * @param ciphertext_e End flag of block stream ciphertext, 1 bit.
 *
 */

template <unsigned int _keyWidth = 256>
void aesEcbEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintext,
    hls::stream<bool>& plaintext_e,
    // input cipherkey
    hls::stream<ap_uint<_keyWidth> >& cipherkey,
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
    // intermediate registers to perform the encryption chain
    ap_uint<128> plaintext_r = 0;
    ap_uint<128> input_block = 0;
    ap_uint<128> input_block_r = 0;
    ap_uint<128> output_block = 0;
    ap_uint<128> ciphertext_r = 0;

    // set the initialization for ture
    bool initialization = true;

    bool e = plaintext_e.read();

encryption_ecb_loop:
    while (!e) {
#pragma HLS PIPELINE II = 1
        // read a block of plaintext, 128 bits
        plaintext_r = plaintext.read();
#ifndef __SYNTHESIS__
        std::cout << "plaintext    = " << std::hex << plaintext_r << std::endl;
#endif

        // calculate input_block
        input_block = plaintext_r;
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
        ciphertext_r = output_block;
#ifndef __SYNTHESIS__
        std::cout << "ciphertext   = " << std::hex << ciphertext_r << std::endl;
#endif

        // write out ciphertext
        ciphertext.write(ciphertext_r);
        ciphertext_e.write(0);

        e = plaintext_e.read();
    }

    ciphertext_e.write(1);

} // end aesEcbEncrypt

/**
 *
 * @brief aesEcbDecrypt is ECB decryption mode with AES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @tparam _keyWidth The bit-width of the cipher key, which is 128, 192, or 256.
 *
 * @param ciphertext Input block stream to be decrypted, each block is 128 bits.
 * @param ciphertext_e End flag of block stream ciphertext, 1 bit.
 * @param cipherkey Input cipher key used in decryption, x bits for AES-x.
 * @param plaintext Output decrypted block stream text, each block is 128 bits.
 * @param plaintext_e End flag of block stream plaintext, 1 bit.
 *
 */

template <unsigned int _keyWidth = 256>
void aesEcbDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertext,
    hls::stream<bool>& ciphertext_e,
    // input cipherkey
    hls::stream<ap_uint<_keyWidth> >& cipherkey,
    // stream out
    hls::stream<ap_uint<128> >& plaintext,
    hls::stream<bool>& plaintext_e) {
    // register cipherkey
    ap_uint<_keyWidth> key_r = cipherkey.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif
    xf::security::aesDec<_keyWidth> decipher;
    decipher.updateKey(key_r);
    // intermediate registers to perform the decryption chain
    ap_uint<128> ciphertext_r = 0;
    ap_uint<128> input_block = 0;
    ap_uint<128> output_block = 0;
    ap_uint<128> plaintext_r = 0;

    bool e = ciphertext_e.read();

decryption_ecb_loop:
    while (!e) {
#pragma HLS PIPELINE II = 1
        // read a block of ciphertext, 128 bits
        ciphertext_r = ciphertext.read();
#ifndef __SYNTHESIS__
        std::cout << "ciphertext    = " << std::hex << ciphertext_r << std::endl;
#endif

        // calculate input block
        input_block = ciphertext_r;
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k^(-1)
        decipher.process(input_block, key_r, output_block);
// xf::security::internal::aesDecrypt<_keyWidth>(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // get the plaintext for current interation by output_block
        plaintext_r = output_block;
#ifndef __SYNTHESIS__
        std::cout << "plaintext   = " << std::hex << plaintext_r << std::endl;
#endif

        // write out plaintext
        plaintext.write(plaintext_r);
        plaintext_e.write(0);

        e = ciphertext_e.read();
    }

    plaintext_e.write(1);

} // end aesEcbDecrypt

} // namespace internal

/**
 *
 * @brief desEcbEncrypt is ECB encryption mode with DES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each block is 64 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 64 bits for each key.
 * @param ciphertextStrm Output encrypted block stream text, each block is 64 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void desEcbEncrypt(
    // stream in
    hls::stream<ap_uint<64> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey
    hls::stream<ap_uint<64> >& cipherkeyStrm,
    // stream out
    hls::stream<ap_uint<64> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    // register cipherkey
    ap_uint<64> key_r = cipherkeyStrm.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif

    // intermediate registers to perform the encryption chain
    ap_uint<64> plaintext_r = 0;
    ap_uint<64> input_block = 0;
    ap_uint<64> input_block_r = 0;
    ap_uint<64> output_block = 0;
    ap_uint<64> ciphertext_r = 0;

    // set the initialization for ture
    bool initialization = true;

    bool e = endPlaintextStrm.read();

encryption_ecb_loop:
    while (!e) {
#pragma HLS PIPELINE II = 1
        // read a block of plaintext, 64 bits
        plaintext_r = plaintextStrm.read();
#ifndef __SYNTHESIS__
        std::cout << "plaintext    = " << std::hex << plaintext_r << std::endl;
#endif

        // calculate input_block
        input_block = plaintext_r;
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k
        xf::security::desEncrypt(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // get the ciphertext for current interation by output_block and plaintext
        ciphertext_r = output_block;
#ifndef __SYNTHESIS__
        std::cout << "ciphertext   = " << std::hex << ciphertext_r << std::endl;
#endif

        // write out ciphertext
        ciphertextStrm.write(ciphertext_r);
        endCiphertextStrm.write(0);

        e = endPlaintextStrm.read();
    }

    endCiphertextStrm.write(1);

} // end desEcbEncrypt

/**
 *
 * @brief desEcbDecrypt is ECB decryption mode with DES single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream to be decrypted, each block is 64 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 64 bits for each key.
 * @param plaintextStrm Output decrypted block stream text, each block is 64 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void desEcbDecrypt(
    // stream in
    hls::stream<ap_uint<64> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey
    hls::stream<ap_uint<64> >& cipherkeyStrm,
    // stream out
    hls::stream<ap_uint<64> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    // register cipherkey
    ap_uint<64> key_r = cipherkeyStrm.read();
#ifndef __SYNTHESIS__
    std::cout << std::endl << "cipherkey = " << std::hex << key_r << std::endl;
#endif

    // intermediate registers to perform the decryption chain
    ap_uint<64> ciphertext_r = 0;
    ap_uint<64> input_block = 0;
    ap_uint<64> output_block = 0;
    ap_uint<64> plaintext_r = 0;

    bool e = endCiphertextStrm.read();

decryption_ecb_loop:
    while (!e) {
#pragma HLS PIPELINE II = 1
        // read a block of ciphertext, 64 bits
        ciphertext_r = ciphertextStrm.read();
#ifndef __SYNTHESIS__
        std::cout << "ciphertext    = " << std::hex << ciphertext_r << std::endl;
#endif

        // calculate input block
        input_block = ciphertext_r;
#ifndef __SYNTHESIS__
        std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

        // CIPH_k^(-1)
        xf::security::desDecrypt(input_block, key_r, output_block);
#ifndef __SYNTHESIS__
        std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

        // get the plaintext for current interation by output_block
        plaintext_r = output_block;
#ifndef __SYNTHESIS__
        std::cout << "plaintext   = " << std::hex << plaintext_r << std::endl;
#endif

        // write out plaintext
        plaintextStrm.write(plaintext_r);
        endPlaintextStrm.write(0);

        e = endCiphertextStrm.read();
    }

    endPlaintextStrm.write(1);

} // end desEcbDecrypt

/**
 *
 * @brief aes128EcbEncrypt is ECB encryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 128 bits.
 * @param ciphertextStrm Output encrypted block stream text, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes128EcbEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey
    hls::stream<ap_uint<128> >& cipherkeyStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesEcbEncrypt<128>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, ciphertextStrm, endCiphertextStrm);

} // end aes128EcbEncrypt

/**
 *
 * @brief aes128EcbDecrypt is ECB decryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 128 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void aes128EcbDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey
    hls::stream<ap_uint<128> >& cipherkeyStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesEcbDecrypt<128>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, plaintextStrm, endPlaintextStrm);

} // end aes128EcbDecrypt

/**
 *
 * @brief aes192EcbEncrypt is ECB encryption mode with AES-192 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 192 bits.
 * @param ciphertextStrm Output encrypted block stream text, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes192EcbEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey
    hls::stream<ap_uint<192> >& cipherkeyStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesEcbEncrypt<192>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, ciphertextStrm, endCiphertextStrm);

} // end aes192EcbEncrypt

/**
 *
 * @brief aes192EcbDecrypt is ECB decryption mode with AES-192 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 192 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void aes192EcbDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey
    hls::stream<ap_uint<192> >& cipherkeyStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesEcbDecrypt<192>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, plaintextStrm, endPlaintextStrm);

} // end aes192EcbDecrypt

/**
 *
 * @brief aes256EcbEncrypt is ECB encryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param plaintextStrm Input block stream text to be encrypted, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in encryption, 256 bits.
 * @param ciphertextStrm Output encrypted block stream text, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 *
 */

static void aes256EcbEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm,
    // input cipherkey
    hls::stream<ap_uint<256> >& cipherkeyStrm,
    // stream out
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm) {
    internal::aesEcbEncrypt<256>(plaintextStrm, endPlaintextStrm, cipherkeyStrm, ciphertextStrm, endCiphertextStrm);

} // end aes256EcbEncrypt

/**
 *
 * @brief aes256EcbDecrypt is ECB decryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is : "Recommendation for Block Cipher Modes of Operation - Methods and Techniques"
 * The implementation is modified for better performance.
 *
 * @param ciphertextStrm Input block stream to be decrypted, each block is 128 bits.
 * @param endCiphertextStrm End flag of block stream ciphertext, 1 bit.
 * @param cipherkeyStrm Input cipher key used in decryption, 256 bits.
 * @param plaintextStrm Output decrypted block stream text, each block is 128 bits.
 * @param endPlaintextStrm End flag of block stream plaintext, 1 bit.
 *
 */

static void aes256EcbDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertextStrm,
    hls::stream<bool>& endCiphertextStrm,
    // input cipherkey
    hls::stream<ap_uint<256> >& cipherkeyStrm,
    // stream out
    hls::stream<ap_uint<128> >& plaintextStrm,
    hls::stream<bool>& endPlaintextStrm) {
    internal::aesEcbDecrypt<256>(ciphertextStrm, endCiphertextStrm, cipherkeyStrm, plaintextStrm, endPlaintextStrm);

} // end aes256EcbDecrypt

} // namespace security
} // namespace xf

#endif
