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
 * @file gcm.hpp
 * @brief header file for Galois/Counter Mode (GCM) block cipher mode of operation.
 * This file is part of Vitis Security Library.
 *
 * @detail Containing GCTR encryption and decryption implementations.
 * GCM = GCTR + GMAC.
 *
 */

#ifndef _XF_SECURITY_GCM_HPP_
#define _XF_SECURITY_GCM_HPP_

#include <ap_int.h>
#include <hls_stream.h>

#include "aes.hpp"
#include "gmac.hpp"

// for debug
#if !defined(__SYNTHESIS__) && _XF_SECURITY_GCM_DEBUG_ == 1
#include <iostream>
#endif

namespace xf {
namespace security {
namespace internal {

/**
 *
 * @brief aesGctrEncrypt Encrypt plaintext to cihpertext.
 *
 * The algorithm reference is: "IEEE Standard for Authenticated Encryption with Length Expansion for Storage Devices"
 * The implementation is modified for better performance.
 *
 * @param plaintext The plaintext stream.
 * @param plaintext_length Length of plaintext in bits.
 * @param end_text_length Flag to signal the end of the text length stream.
 * @param cipherkey The cipherkey, x-bit for AES-x.
 * @param IV_Strm Initialization vector.
 * @param H_strm The hash subkey passed onto genGMAC.
 * @param E_K_Y0_strm E(K,Y0) as specified in standard passed onto genGMAC.
 * @param end_length End flag passed onto genGMAC.
 * @param ciphertext The ciphertext stream to output port.
 * @param ciphertext_length Length of ciphertext in bits to output port.
 * @param ciphertext1 The ciphertext stream to genGMAC.
 * @param ciphertext_length1 Length of ciphertext in bits to genGMAC.
 *
 */

template <unsigned int _keyWidth = 256>
void aesGctrEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintext,
    hls::stream<ap_uint<64> >& plaintext_length,
    hls::stream<bool>& end_text_length,
    // input cipherkey and initilization vector
    hls::stream<ap_uint<_keyWidth> >& cipherkey,
    hls::stream<ap_uint<96> >& IV_strm,
    // output hash key and E(K,Y0)
    hls::stream<ap_uint<128> >& H_strm,
    hls::stream<ap_uint<128> >& E_K_Y0_strm,
    hls::stream<bool>& end_length,
    // stream out
    hls::stream<ap_uint<128> >& ciphertext,
    hls::stream<ap_uint<64> >& ciphertext_length,
    hls::stream<ap_uint<128> >& ciphertext1,
    hls::stream<ap_uint<64> >& ciphertext_length1) {
    bool end = end_text_length.read();

    // inform genGMAC
    end_length.write(end);

    xf::security::aesEnc<_keyWidth> cipher;

    while (!end) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        // register cipherkey
        ap_uint<_keyWidth> K = cipherkey.read();
        cipher.updateKey(K);
#if !defined(__SYNTHESIS__) && _XF_SECURITY_GCM_DEBUG_ == 1
        std::cout << std::endl << "cipherkey = " << std::hex << K << std::endl;
#endif

        // generate initial counter block
        // XXX: The bit-width of IV is restricted to 96 in this implementation
        ap_uint<128> Y0;
        Y0.range(95, 0) = IV_strm.read();
        Y0.range(127, 96) = 0x01000000;
#if !defined(__SYNTHESIS__) && _XF_SECURITY_GCM_DEBUG_ == 1
        std::cout << "Y0 = " << std::hex << Y0 << std::endl;
#endif

        // intermediate registers to perform the CTR encryption chain
        ap_uint<128> plaintext_r = 0;
        ap_uint<128> input_block = 0;
        ap_uint<128> input_block_r = 0;
        ap_uint<128> output_block = 0;
        ap_uint<128> ciphertext_r = 0;

        // set the iteration controlling flag
        bool first = true;
        bool second = false;
        bool initialization = false;

        // total text length in bits
        ap_uint<64> plen = plaintext_length.read();

        // plaintext length is equal to ciphertext length
        ciphertext_length.write(plen);
        ciphertext_length1.write(plen);

        ap_uint<64> iEnd = plen / 128 + ((plen % 128) > 0) + 2;

    LOOP_GEN_CIPHER:
        for (ap_uint<64> i = 0; i < iEnd; i++) {
#pragma HLS loop_tripcount min = 10 max = 10 avg = 10
#pragma HLS PIPELINE II = 1
            // read a block of plaintext, 128 bits
            if (!first && !second) {
                plaintext_r = plaintext.read();
#if !defined(__SYNTHESIS__) && _XF_SECURITY_GCM_DEBUG_ == 1
                std::cout << "plaintext    = " << std::hex << plaintext_r << std::endl;
#endif
            }

            // calculate input_block
            if (first) {
                // the first iteration calculate the hash key
                input_block.range(127, 64) = 0x0000000000000000;
                input_block.range(63, 0) = 0x0000000000000000;
            } else if (second) {
                // the second iteration calculate the E(K,Y0)
                input_block = Y0;
            } else if (initialization) {
                // the third iteration uses Y0+1 as input_block
                input_block_r.range(127, 120) = Y0(7, 0);
                input_block_r.range(119, 112) = Y0(15, 8);
                input_block_r.range(111, 104) = Y0(23, 16);
                input_block_r.range(103, 96) = Y0(31, 24);
                input_block_r.range(95, 88) = Y0(39, 32);
                input_block_r.range(87, 80) = Y0(47, 40);
                input_block_r.range(79, 72) = Y0(55, 48);
                input_block_r.range(71, 64) = Y0(63, 56);
                input_block_r.range(63, 56) = Y0(71, 64);
                input_block_r.range(55, 48) = Y0(79, 72);
                input_block_r.range(47, 40) = Y0(87, 80);
                input_block_r.range(39, 32) = Y0(95, 88);
                input_block_r.range(31, 24) = Y0(103, 96);
                input_block_r.range(23, 16) = Y0(111, 104);
                input_block_r.range(15, 8) = Y0(119, 112);
                input_block_r.range(7, 0) = Y0(127, 120);
                input_block_r.range(31, 0) = input_block_r.range(31, 0) + 1;
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
                input_block_r.range(31, 0) = input_block_r.range(31, 0) + 1;
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
#if !defined(__SYNTHESIS__) && _XF_SECURITY_GCM_DEBUG_ == 1
            std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

            // CIPH_k
            cipher.process(input_block, K, output_block);
// xf::security::internal::aesEncrypt<_keyWidth>(input_block, K, output_block);
#if !defined(__SYNTHESIS__) && _XF_SECURITY_GCM_DEBUG_ == 1
            std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

            // get the ciphertext for current interation by output_block and plaintext
            ciphertext_r = plaintext_r ^ output_block;
#if !defined(__SYNTHESIS__) && _XF_SECURITY_GCM_DEBUG_ == 1
            std::cout << "ciphertext   = " << std::hex << ciphertext_r << std::endl;
#endif

            if (first) {
                // write out the hash key, and prepare for the second iteration
                H_strm.write(output_block);
                first = false;
                second = true;
            } else if (second) {
                // write out the E(K,Y0), and prepare for the third iteration
                E_K_Y0_strm.write(output_block);
                second = false;
                initialization = true;
            } else {
                // write out plaintext
                ciphertext.write(ciphertext_r);
                ciphertext1.write(ciphertext_r);
            }
        }

        // last message?
        end = end_text_length.read();

        // inform genGMAC
        end_length.write(end);
    }

} // end aesGctrEncrypt

/**
 *
 * @brief aesGctrDecrypt Decrypt ciphertext to plainrtext.
 *
 * The algorithm reference is: "IEEE Standard for Authenticated Encryption with Length Expansion for Storage Devices"
 * The implementation is modified for better performance.
 *
 * @param ciphertext The ciphertext stream.
 * @param ciphertext_length Length of ciphertext in bits.
 * @param end_text_length Flag to signal the end of the text length stream.
 * @param cipherkey The cipherkey, x-bit for AES-x.
 * @param IV_Strm Initialization vector.
 * @param H_strm The hash subkey passed onto genGMAC.
 * @param E_K_Y0_strm E(K,Y0) as specified in standard passed onto genGMAC.
 * @param end_length End flag passed onto genGMAC.
 * @param ciphertext_out The ciphertext stream passed onto genGMAC.
 * @param ciphertext_length_out Length of ciphertext in bits passed onto genGMAC.
 * @param plaintext The plaintext stream.
 * @param plaintext_length Length of plaintext in bits.
 *
 */

template <unsigned int _keyWidth = 256>
void aesGctrDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertext,
    hls::stream<ap_uint<64> >& ciphertext_length,
    hls::stream<bool>& end_text_length,
    // input cipherkey and initilization vector
    hls::stream<ap_uint<_keyWidth> >& cipherkey,
    hls::stream<ap_uint<96> >& IV_strm,
    // output hash key and E(K,Y0)
    hls::stream<ap_uint<128> >& H_strm,
    hls::stream<ap_uint<128> >& E_K_Y0_strm,
    hls::stream<bool>& end_length,
    // stream out
    hls::stream<ap_uint<128> >& ciphertext_out,
    hls::stream<ap_uint<64> >& ciphertext_length_out,
    hls::stream<ap_uint<128> >& plaintext,
    hls::stream<ap_uint<64> >& plaintext_length) {
    bool end = end_text_length.read();
    end_length.write(end);
    xf::security::aesEnc<_keyWidth> cipher;

    while (!end) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        // register cipherkey
        ap_uint<_keyWidth> K = cipherkey.read();
        cipher.updateKey(K);
#if !defined(__SYNTHESIS__) && _XF_SECURITY_GCM_DEBUG_ == 1
        std::cout << std::endl << "cipherkey = " << std::hex << K << std::endl;
#endif

        // generate initial counter block
        // XXX: The bit-width of IV is restricted to 96 in this implementation
        ap_uint<128> Y0;
        Y0.range(95, 0) = IV_strm.read();
        Y0.range(127, 96) = 0x01000000;
#if !defined(__SYNTHESIS__) && _XF_SECURITY_GCM_DEBUG_ == 1
        std::cout << "Y0 = " << std::hex << Y0 << std::endl;
#endif

        // intermediate registers to perform the CTR decryption chain
        ap_uint<128> ciphertext_r = 0;
        ap_uint<128> input_block = 0;
        ap_uint<128> input_block_r = 0;
        ap_uint<128> output_block = 0;
        ap_uint<128> plaintext_r = 0;

        // set the iteration controlling flag
        bool first = true;
        bool second = false;
        bool initialization = false;

        // total text length in bits
        ap_uint<64> clen = ciphertext_length.read();

        // plaintext length is equal to ciphertext length
        ciphertext_length_out.write(clen);
        plaintext_length.write(clen);

        ap_uint<64> iEnd = clen / 128 + ((clen % 128) > 0) + 2;

    LOOP_GEN_PLAIN:
        for (ap_uint<64> i = 0; i < iEnd; i++) {
#pragma HLS loop_tripcount min = 10 max = 10 avg = 10
#pragma HLS PIPELINE II = 1
            // read a block of ciphertext, 128 bits
            if (!first && !second) {
                ciphertext_r = ciphertext.read();
                ciphertext_out.write(ciphertext_r);
#if !defined(__SYNTHESIS__) && _XF_SECURITY_GCM_DEBUG_ == 1
                std::cout << "ciphertext    = " << std::hex << ciphertext_r << std::endl;
#endif
            }

            // calculate input_block
            if (first) { // the first iteration calculate the hash key
                input_block.range(127, 64) = 0x0000000000000000;
                input_block.range(63, 0) = 0x0000000000000000;
            } else if (second) { // the second iteration calculate the E(K,Y0)
                input_block = Y0;
            } else if (initialization) { // the third iteration uses Y0+1 as input_block
                input_block_r.range(127, 120) = Y0(7, 0);
                input_block_r.range(119, 112) = Y0(15, 8);
                input_block_r.range(111, 104) = Y0(23, 16);
                input_block_r.range(103, 96) = Y0(31, 24);
                input_block_r.range(95, 88) = Y0(39, 32);
                input_block_r.range(87, 80) = Y0(47, 40);
                input_block_r.range(79, 72) = Y0(55, 48);
                input_block_r.range(71, 64) = Y0(63, 56);
                input_block_r.range(63, 56) = Y0(71, 64);
                input_block_r.range(55, 48) = Y0(79, 72);
                input_block_r.range(47, 40) = Y0(87, 80);
                input_block_r.range(39, 32) = Y0(95, 88);
                input_block_r.range(31, 24) = Y0(103, 96);
                input_block_r.range(23, 16) = Y0(111, 104);
                input_block_r.range(15, 8) = Y0(119, 112);
                input_block_r.range(7, 0) = Y0(127, 120);
                input_block_r.range(31, 0) = input_block_r.range(31, 0) + 1;
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
                input_block_r.range(31, 0) = input_block_r.range(31, 0) + 1;
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
#if !defined(__SYNTHESIS__) && _XF_SECURITY_GCM_DEBUG_ == 1
            std::cout << "input_block  = " << std::hex << input_block << std::endl;
#endif

            // CIPH_k
            cipher.process(input_block, K, output_block);
// xf::security::internal::aesEncrypt<_keyWidth>(input_block, K, output_block);
#if !defined(__SYNTHESIS__) && _XF_SECURITY_GCM_DEBUG_ == 1
            std::cout << "output_block = " << std::hex << output_block << std::endl;
#endif

            // get the plaintext for current interation by output_block and ciphertext
            plaintext_r = ciphertext_r ^ output_block;
#if !defined(__SYNTHESIS__) && _XF_SECURITY_GCM_DEBUG_ == 1
            std::cout << "plaintext   = " << std::hex << plaintext_r << std::endl;
#endif

            if (first) { // write out the hash key, and prepare for the second iteration
                H_strm.write(output_block);
                first = false;
                second = true;
            } else if (second) { // write out the E(K,Y0), and prepare for the third iteration
                E_K_Y0_strm.write(output_block);
                second = false;
                initialization = true;
            } else { // write out plaintext
                plaintext.write(plaintext_r);
            }
        }

        // last message?
        end = end_text_length.read();
        end_length.write(end);
    }

} // end aesGctrDecrypt

/**
 *
 * @brief aesGcmEncrypt Top of GCM encryption mode with AES single block cipher.
 *
 * The algorithm reference is : "IEEE Standard for Authenticated Encryption with Length Expansion for Storage Devices"
 * The implementation is modified for better performance.
 *
 * @tparam _keyWidth The bit-width of the cipher key, which is 128, 192, or 256.
 *
 * @param plaintext Input block stream text to be encrypted, 128 bits.
 * @param plaintext_length Length of plaintext in bits.
 * @param cipherkey Input cipher key used in encryption, x bits for AES-x.
 * @param IV Initialization vector.
 * @param AAD Additional authenticated data for calculating the tag, 128 bits.
 * @param AAD_length Length of AAD in bits.
 * @param end__length Flag to signal the end of the text length stream.
 * @param ciphertext Output encrypted block stream text, 128 bits.
 * @param ciphertext_length Length of ciphertext in bits.
 * @param tag The MAC.
 * @param end_tag End flag for the MAC.
 *
 */

template <unsigned int _keyWidth = 256>
void aesGcmEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& plaintext,
    hls::stream<ap_uint<64> >& plaintext_length,
    // input cipherkey, initilization vector, and additional authenticated data
    hls::stream<ap_uint<_keyWidth> >& cipherkey,
    hls::stream<ap_uint<96> >& IV,
    hls::stream<ap_uint<128> >& AAD,
    hls::stream<ap_uint<64> >& AAD_length,
    hls::stream<bool>& end_length,
    // stream out
    hls::stream<ap_uint<128> >& ciphertext,
    hls::stream<ap_uint<64> >& ciphertext_length,
    // ouput tag
    hls::stream<ap_uint<128> >& tag,
    hls::stream<bool>& end_tag) {
#pragma HLS DATAFLOW

    hls::stream<ap_uint<128> > H_strm("H_strm");
#pragma HLS RESOURCE variable = H_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = H_strm depth = 32

    hls::stream<ap_uint<128> > E_K_Y0_strm("E_K_Y0_strm");
#pragma HLS RESOURCE variable = E_K_Y0_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = E_K_Y0_strm depth = 32

    hls::stream<bool> end_length_strm("end_length_strm");
#pragma HLS RESOURCE variable = end_length_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = end_length_strm depth = 32

    hls::stream<ap_uint<128> > ciphertext_strm("ciphertext_strm");
#pragma HLS RESOURCE variable = ciphertext_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = ciphertext_strm depth = 32

    hls::stream<ap_uint<64> > ciphertext_length_strm("ciphertext_length_strm");
#pragma HLS RESOURCE variable = ciphertext_length_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = ciphertext_length_strm depth = 32

    aesGctrEncrypt<_keyWidth>(plaintext, plaintext_length, end_length, cipherkey, IV, H_strm, E_K_Y0_strm,
                              end_length_strm, ciphertext_strm, ciphertext_length_strm, ciphertext, ciphertext_length);

    genGMAC(AAD, AAD_length, ciphertext_strm, ciphertext_length_strm, H_strm, E_K_Y0_strm, end_length_strm, tag,
            end_tag);

} // end aesGcmEncrypt

/**
 *
 * @brief aesGcmDecrypt Top of GCM decryption mode with AES single block cipher.
 *
 * The algorithm reference is : "IEEE Standard for Authenticated Encryption with Length Expansion for Storage Devices"
 * The implementation is modified for better performance.
 *
 * @param ciphertext Input block stream text to be decrypted, 128 bits.
 * @param ciphertext_length Length of ciphertext in bits.
 * @param cipherkey Input cipher key used in decryption, x bits for AES-x.
 * @param IV Initialization vector.
 * @param AAD Additional authenticated data for calculating the tag, 128 bits.
 * @param AAD_length Length of AAD in bits.
 * @param end__length Flag to signal the end of the text length stream.
 * @param plaintext Output decrypted block stream text, 128 bits.
 * @param plaintext_length Length of plaintext in bits.
 * @param tag The MAC.
 * @param end_tag End flag for the MAC.
 *
 */

template <unsigned int _keyWidth = 256>
void aesGcmDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& ciphertext,
    hls::stream<ap_uint<64> >& ciphertext_length,
    // input cipherkey, initilization vector, and additional authenticated data
    hls::stream<ap_uint<_keyWidth> >& cipherkey,
    hls::stream<ap_uint<96> >& IV,
    hls::stream<ap_uint<128> >& AAD,
    hls::stream<ap_uint<64> >& AAD_length,
    hls::stream<bool>& end_length,
    // stream out
    hls::stream<ap_uint<128> >& plaintext,
    hls::stream<ap_uint<64> >& plaintext_length,
    // ouput tag
    hls::stream<ap_uint<128> >& tag,
    hls::stream<bool>& end_tag) {
#pragma HLS DATAFLOW

    hls::stream<ap_uint<128> > H_strm("H_strm");
#pragma HLS RESOURCE variable = H_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = H_strm depth = 32

    hls::stream<ap_uint<128> > E_K_Y0_strm("E_K_Y0_strm");
#pragma HLS RESOURCE variable = E_K_Y0_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = E_K_Y0_strm depth = 32

    hls::stream<bool> end_length_strm("end_length_strm");
#pragma HLS RESOURCE variable = end_length_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = end_length_strm depth = 32

    hls::stream<ap_uint<128> > ciphertext_strm("ciphertext_strm");
#pragma HLS RESOURCE variable = ciphertext_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = ciphertext_strm depth = 32

    hls::stream<ap_uint<64> > ciphertext_length_strm("ciphertext_length_strm");
#pragma HLS RESOURCE variable = ciphertext_length_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = ciphertext_length_strm depth = 32

    aesGctrDecrypt<_keyWidth>(ciphertext, ciphertext_length, end_length, cipherkey, IV, H_strm, E_K_Y0_strm,
                              end_length_strm, ciphertext_strm, ciphertext_length_strm, plaintext, plaintext_length);

    genGMAC(AAD, AAD_length, ciphertext_strm, ciphertext_length_strm, H_strm, E_K_Y0_strm, end_length_strm, tag,
            end_tag);

} // end aesGcmDecrypt

} // namespace internal

/**
 *
 * @brief aes128GcmEncrypt is GCM encryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is : "IEEE Standard for Authenticated Encryption with Length Expansion for Storage Devices"
 * The implementation is modified for better performance.
 *
 * @param payloadStrm Input block stream text to be encrypted.
 * @param cipherkeyStrm Input cipher key used in encryption.
 * @param IVStrm Initialization vector stream.
 * @param AADStrm Additional authenticated data stream.
 * @param lenAADStrm Length of additional authenticated data in bits.
 * @param lenPldStrm Length of payload in bits.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param cipherStrm Output encrypted block stream text.
 * @param lenCphStrm Length of cipher in bits.
 * @param tagStrm The MAC stream.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

static void aes128GcmEncrypt(hls::stream<ap_uint<128> >& payloadStrm,
                             hls::stream<ap_uint<128> >& cipherkeyStrm,
                             hls::stream<ap_uint<96> >& IVStrm,
                             hls::stream<ap_uint<128> >& AADStrm,
                             hls::stream<ap_uint<64> >& lenAADStrm,
                             hls::stream<ap_uint<64> >& lenPldStrm,
                             hls::stream<bool>& endLenStrm,
                             hls::stream<ap_uint<128> >& cipherStrm,
                             hls::stream<ap_uint<64> >& lenCphStrm,
                             hls::stream<ap_uint<128> >& tagStrm,
                             hls::stream<bool>& endTagStrm) {
    internal::aesGcmEncrypt<128>(payloadStrm, lenPldStrm, cipherkeyStrm, IVStrm, AADStrm, lenAADStrm, endLenStrm,
                                 cipherStrm, lenCphStrm, tagStrm, endTagStrm);

} // end aes128GcmEncrypt

/**
 *
 * @brief aes128GcmDecrypt is GCM decryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is : "IEEE Standard for Authenticated Encryption with Length Expansion for Storage Devices"
 * The implementation is modified for better performance.
 *
 * @param payloadStrm Input block stream text to be decrypted.
 * @param cipherkeyStrm Input cipher key used in decryption.
 * @param IVStrm Initialization vector stream.
 * @param AADStrm Additional authenticated data stream.
 * @param lenAADStrm Length of additional authenticated data in bits.
 * @param lenPldStrm Length of payload in bits.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param cipherStrm Output decrypted block stream text.
 * @param lenCphStrm Length of cipher in bits.
 * @param tagStrm The MAC stream.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

static void aes128GcmDecrypt(hls::stream<ap_uint<128> >& payloadStrm,
                             hls::stream<ap_uint<128> >& cipherkeyStrm,
                             hls::stream<ap_uint<96> >& IVStrm,
                             hls::stream<ap_uint<128> >& AADStrm,
                             hls::stream<ap_uint<64> >& lenAADStrm,
                             hls::stream<ap_uint<64> >& lenPldStrm,
                             hls::stream<bool>& endLenStrm,
                             hls::stream<ap_uint<128> >& cipherStrm,
                             hls::stream<ap_uint<64> >& lenCphStrm,
                             hls::stream<ap_uint<128> >& tagStrm,
                             hls::stream<bool>& endTagStrm) {
    internal::aesGcmDecrypt<128>(payloadStrm, lenPldStrm, cipherkeyStrm, IVStrm, AADStrm, lenAADStrm, endLenStrm,
                                 cipherStrm, lenCphStrm, tagStrm, endTagStrm);

} // end aes128GcmDecrypt

/**
 *
 * @brief aes192GcmEncrypt is GCM encryption mode with AES-192 single block cipher.
 *
 * The algorithm reference is : "IEEE Standard for Authenticated Encryption with Length Expansion for Storage Devices"
 * The implementation is modified for better performance.
 *
 * @param payloadStrm Input block stream text to be encrypted.
 * @param cipherkeyStrm Input cipher key used in encryption.
 * @param IVStrm Initialization vector stream.
 * @param AADStrm Additional authenticated data stream.
 * @param lenAADStrm Length of additional authenticated data in bits.
 * @param lenPldStrm Length of payload in bits.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param cipherStrm Output encrypted block stream text.
 * @param lenCphStrm Length of cipher in bits.
 * @param tagStrm The MAC stream.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

static void aes192GcmEncrypt(hls::stream<ap_uint<128> >& payloadStrm,
                             hls::stream<ap_uint<192> >& cipherkeyStrm,
                             hls::stream<ap_uint<96> >& IVStrm,
                             hls::stream<ap_uint<128> >& AADStrm,
                             hls::stream<ap_uint<64> >& lenAADStrm,
                             hls::stream<ap_uint<64> >& lenPldStrm,
                             hls::stream<bool>& endLenStrm,
                             hls::stream<ap_uint<128> >& cipherStrm,
                             hls::stream<ap_uint<64> >& lenCphStrm,
                             hls::stream<ap_uint<128> >& tagStrm,
                             hls::stream<bool>& endTagStrm) {
    internal::aesGcmEncrypt<192>(payloadStrm, lenPldStrm, cipherkeyStrm, IVStrm, AADStrm, lenAADStrm, endLenStrm,
                                 cipherStrm, lenCphStrm, tagStrm, endTagStrm);

} // end aes192GcmEncrypt

/**
 *
 * @brief aes192GcmDecrypt is GCM decryption mode with AES-192 single block cipher.
 *
 * The algorithm reference is : "IEEE Standard for Authenticated Encryption with Length Expansion for Storage Devices"
 * The implementation is modified for better performance.
 *
 * @param payloadStrm Input block stream text to be decrypted.
 * @param cipherkeyStrm Input cipher key used in decryption.
 * @param IVStrm Initialization vector stream.
 * @param AADStrm Additional authenticated data stream.
 * @param lenAADStrm Length of additional authenticated data in bits.
 * @param lenPldStrm Length of payload in bits.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param cipherStrm Output decrypted block stream text.
 * @param lenCphStrm Length of cipher in bits.
 * @param tagStrm The MAC stream.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

static void aes192GcmDecrypt(hls::stream<ap_uint<128> >& payloadStrm,
                             hls::stream<ap_uint<192> >& cipherkeyStrm,
                             hls::stream<ap_uint<96> >& IVStrm,
                             hls::stream<ap_uint<128> >& AADStrm,
                             hls::stream<ap_uint<64> >& lenAADStrm,
                             hls::stream<ap_uint<64> >& lenPldStrm,
                             hls::stream<bool>& endLenStrm,
                             hls::stream<ap_uint<128> >& cipherStrm,
                             hls::stream<ap_uint<64> >& lenCphStrm,
                             hls::stream<ap_uint<128> >& tagStrm,
                             hls::stream<bool>& endTagStrm) {
    internal::aesGcmDecrypt<192>(payloadStrm, lenPldStrm, cipherkeyStrm, IVStrm, AADStrm, lenAADStrm, endLenStrm,
                                 cipherStrm, lenCphStrm, tagStrm, endTagStrm);

} // end aes192GcmDecrypt

/**
 *
 * @brief aes256GcmEncrypt is GCM encryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is : "IEEE Standard for Authenticated Encryption with Length Expansion for Storage Devices"
 * The implementation is modified for better performance.
 *
 * @param payloadStrm Input block stream text to be encrypted.
 * @param cipherkeyStrm Input cipher key used in encryption.
 * @param IVStrm Initialization vector stream.
 * @param AADStrm Additional authenticated data stream.
 * @param lenAADStrm Length of additional authenticated data in bits.
 * @param lenPldStrm Length of payload in bits.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param cipherStrm Output encrypted block stream text.
 * @param lenCphStrm Length of cipher in bits.
 * @param tagStrm The MAC stream.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

static void aes256GcmEncrypt(hls::stream<ap_uint<128> >& payloadStrm,
                             hls::stream<ap_uint<256> >& cipherkeyStrm,
                             hls::stream<ap_uint<96> >& IVStrm,
                             hls::stream<ap_uint<128> >& AADStrm,
                             hls::stream<ap_uint<64> >& lenAADStrm,
                             hls::stream<ap_uint<64> >& lenPldStrm,
                             hls::stream<bool>& endLenStrm,
                             hls::stream<ap_uint<128> >& cipherStrm,
                             hls::stream<ap_uint<64> >& lenCphStrm,
                             hls::stream<ap_uint<128> >& tagStrm,
                             hls::stream<bool>& endTagStrm) {
    internal::aesGcmEncrypt<256>(payloadStrm, lenPldStrm, cipherkeyStrm, IVStrm, AADStrm, lenAADStrm, endLenStrm,
                                 cipherStrm, lenCphStrm, tagStrm, endTagStrm);

} // end aes256GcmEncrypt

/**
 *
 * @brief aes256GcmDecrypt is GCM decryption mode with AES-2562 single block cipher.
 *
 * The algorithm reference is : "IEEE Standard for Authenticated Encryption with Length Expansion for Storage Devices"
 * The implementation is modified for better performance.
 *
 * @param payloadStrm Input block stream text to be decrypted.
 * @param cipherkeyStrm Input cipher key used in decryption.
 * @param IVStrm Initialization vector stream.
 * @param AADStrm Additional authenticated data stream.
 * @param lenAADStrm Length of additional authenticated data in bits.
 * @param lenPldStrm Length of payload in bits.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param cipherStrm Output decrypted block stream text.
 * @param lenCphStrm Length of cipher in bits.
 * @param tagStrm The MAC stream.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

static void aes256GcmDecrypt(hls::stream<ap_uint<128> >& payloadStrm,
                             hls::stream<ap_uint<256> >& cipherkeyStrm,
                             hls::stream<ap_uint<96> >& IVStrm,
                             hls::stream<ap_uint<128> >& AADStrm,
                             hls::stream<ap_uint<64> >& lenAADStrm,
                             hls::stream<ap_uint<64> >& lenPldStrm,
                             hls::stream<bool>& endLenStrm,
                             hls::stream<ap_uint<128> >& cipherStrm,
                             hls::stream<ap_uint<64> >& lenCphStrm,
                             hls::stream<ap_uint<128> >& tagStrm,
                             hls::stream<bool>& endTagStrm) {
    internal::aesGcmDecrypt<256>(payloadStrm, lenPldStrm, cipherkeyStrm, IVStrm, AADStrm, lenAADStrm, endLenStrm,
                                 cipherStrm, lenCphStrm, tagStrm, endTagStrm);

} // end aes256GcmDecrypt

} // namespace security
} // namespace xf

#endif
