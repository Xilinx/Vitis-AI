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
 * @file ccm.hpp
 * @brief header file for Counter with Cipher Block
 * Chaining-Message Authentication Code (CCM) block cipher mode of operation.
 * This file is part of Vitis Security Library.
 *
 * @detail Containing CCM mode with AES-128/192/256.
 * CCM = CTR + CBC-MAC.
 * Please be noticed that the counter block generation process of CCM is
 * quite different from the original CTR mode.
 *
 */

#ifndef _XF_SECURITY_CCM_HPP_
#define _XF_SECURITY_CCM_HPP_

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
 * @brief Impletmentation of formatting function as specified in standard.
 *
 * The algorithm reference is: "Recommendation for Block Cipher Modes of Operation: The CCM Mode for Authentication and
 * Confidentiality"
 * The implementation is modified for better performance.
 *
 * @tparam _t Length of the MAC in byte, t is an element of {4, 6, 8, 10, 12, 14, 16}.
 * @tparam _q Length in byte of the binary representation of the length of the payload in byte, q is an element of {2,
 * 3, 4, 5, 6, 7, 8}.
 *
 * @param ADStrm Associated data stream.
 * @param nonceStrm The nonce stream.
 * @param lenADStrm Length of associated data in byte.
 * @param lenPldStrm Length of payload in byte.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param blockStrm Block stream of the formatted input as specified in reference.
 * @param outLenADStrm Pass on the length of associated data to CBC-MAC.
 * @param outNonceStrm Pass on the nonce to CTR.
 *
 */

template <unsigned int _t = 16, unsigned int _q = 8>
void formatting(
    // stream in
    hls::stream<ap_uint<128> >& ADStrm,
    hls::stream<ap_uint<8 * (15 - _q)> >& nonceStrm,
    hls::stream<ap_uint<64> >& lenADStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<bool>& endLenStrm,
    // stream out
    hls::stream<ap_uint<128> >& blockStrm,
    hls::stream<ap_uint<64> >& outLenADStrm,
    hls::stream<ap_uint<8 * (15 - _q)> >& outNonceStrm) {
    bool end = endLenStrm.read();

    while (!end) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        // *** formatting of the control block ***
        // single block of the formatted input
        ap_uint<128> Blk = 0;

        // length of associated data
        ap_uint<64> lenAD = lenADStrm.read();
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
        std::cout << std::endl << "lenAD = " << std::dec << lenAD << std::endl;
#endif

        // pass on the AD length to CBC-MAC
        outLenADStrm.write(lenAD);

        // formatting of the flags octet in B0
        Blk.range(126, 126) = lenAD > 0;
        Blk.range(125, 123) = (_t - 2) >> 1;
        Blk.range(122, 120) = _q - 1;

        // formatting of the N in B0
        ap_uint<128> N = nonceStrm.read();
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
        std::cout << "nonce = " << std::hex << N << std::endl;
#endif
        ap_uint<128> nonce;
        nonce.range(7, 0) = N.range(127, 120);
        nonce.range(15, 8) = N.range(119, 112);
        nonce.range(23, 16) = N.range(111, 104);
        nonce.range(31, 24) = N.range(103, 96);
        nonce.range(39, 32) = N.range(95, 88);
        nonce.range(47, 40) = N.range(87, 80);
        nonce.range(55, 48) = N.range(79, 72);
        nonce.range(63, 56) = N.range(71, 64);
        nonce.range(71, 64) = N.range(63, 56);
        nonce.range(79, 72) = N.range(55, 48);
        nonce.range(87, 80) = N.range(47, 40);
        nonce.range(95, 88) = N.range(39, 32);
        nonce.range(103, 96) = N.range(31, 24);
        nonce.range(111, 104) = N.range(23, 16);
        nonce.range(119, 112) = N.range(15, 8);
        nonce.range(127, 120) = N.range(7, 0);

        // pass on the nonce to CTR
        outNonceStrm.write(N.range(8 * (15 - _q) - 1, 0));
        Blk.range(119, _q * 8) = nonce.range(127, 128 - (15 - _q) * 8);

        // formatting of the Q in B0
        ap_uint<64> lenPld = lenPldStrm.read();
        Blk.range(_q * 8 - 1, 0) = lenPld.range(_q * 8 - 1, 0);

        // emit B0
        blockStrm.write(Blk);

        // *** formatting of the associated data ***
        // header in byte of the encoded AD block
        unsigned char headerLen = 0;
        ap_uint<128> header = 0;
        if ((0 < lenAD) && (lenAD < (0x10000UL - 0x100UL))) {
            headerLen = 2;
            header = lenAD & 0xffffUL;
        } else if (((0x10000UL - 0x100UL) <= lenAD) && (lenAD < 0x100000000UL)) {
            headerLen = 6;
            header.range(47, 32) = 0xfffeUL;
            header.range(31, 0) = lenAD & 0xffffffffUL;
        } else if (0x100000000UL <= lenAD) {
            headerLen = 10;
            header.range(79, 64) = 0xffffUL;
            header.range(63, 0) = lenAD & 0xffffffffffffffffUL;
        }
        if (headerLen > 0) {
            Blk = header.range(headerLen * 8 - 1, 0) << (128 - headerLen * 8);
        }

    LOOP_AD_GEN:
        for (ap_uint<64> i = 0; i < (lenAD / 16 + ((lenAD % 16) > 0)); i++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 10 max = 10 avg = 10
            ap_uint<128> AD = ADStrm.read();
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
            std::cout << "AD = " << std::hex << AD << std::endl;
#endif
            ap_uint<128> ADReg;
            // change byte order
            ADReg.range(7, 0) = AD.range(127, 120);
            ADReg.range(15, 8) = AD.range(119, 112);
            ADReg.range(23, 16) = AD.range(111, 104);
            ADReg.range(31, 24) = AD.range(103, 96);
            ADReg.range(39, 32) = AD.range(95, 88);
            ADReg.range(47, 40) = AD.range(87, 80);
            ADReg.range(55, 48) = AD.range(79, 72);
            ADReg.range(63, 56) = AD.range(71, 64);
            ADReg.range(71, 64) = AD.range(63, 56);
            ADReg.range(79, 72) = AD.range(55, 48);
            ADReg.range(87, 80) = AD.range(47, 40);
            ADReg.range(95, 88) = AD.range(39, 32);
            ADReg.range(103, 96) = AD.range(31, 24);
            ADReg.range(111, 104) = AD.range(23, 16);
            ADReg.range(119, 112) = AD.range(15, 8);
            ADReg.range(127, 120) = AD.range(7, 0);
            Blk.range(127 - 8 * headerLen, 0) = ADReg.range(127, 8 * headerLen);
            blockStrm.write(Blk);
            Blk = ADReg << (128 - headerLen * 8);
        }
        // deal with the condition that we still have leftover AD
        if ((((lenAD + headerLen) % 16) > 0) && (((lenAD + headerLen) % 16) <= headerLen)) {
            blockStrm.write(Blk);
        }

        // last message?
        end = endLenStrm.read();
    }

} // end formatting

/**
 *
 * @brief Implementation of CTR encryption part in CCM.
 *
 * The algorithm reference is: "Recommendation for Block Cipher Modes of Operation: The CCM Mode for Authentication and
 * Confidentiality"
 * The implementation is modified for better performance.
 *
 * @tparam _q Length in byte of the binary representation of the length of the payload in byte, q is an element of {2,
 * 3, 4, 5, 6, 7, 8}.
 * @tparam _keyWidth Bit-width of the cipher key, typically 128, 192, or 256 for AES.
 *
 * @param payloadStrm Input block stream text to be encrypted.
 * @param cipherkeyStrm Input cipher key used in encryption.
 * @param nonceStrm The nonce stream.
 * @param lenPldStrm Length of payload in byte.
 * @param endLenPldStrm Flag to signal the end of the payload length stream.
 * @param outPayloadStrm Pass on the payload stream to CBC-MAC.
 * @param outCipherkeyStrm Pass on the cipherkey to CBC-MAC.
 * @param S0Strm First cipher used to generate the MAC.
 * @param cipherStrm Output encrypted block stream text.
 * @param lenCphStrm Length of cipher in byte.
 *
 */

template <unsigned int _q = 8, unsigned int _keyWidth = 256>
void aesCtrEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& payloadStrm,
    hls::stream<ap_uint<_keyWidth> >& cipherkeyStrm,
    hls::stream<ap_uint<8 * (15 - _q)> >& nonceStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<bool>& endLenPldStrm,
    // stream out
    hls::stream<ap_uint<128> >& outPayloadStrm,
    hls::stream<ap_uint<_keyWidth> >& outCipherkeyStrm,
    hls::stream<ap_uint<128> >& S0Strm,
    hls::stream<ap_uint<128> >& cipherStrm,
    hls::stream<ap_uint<64> >& lenCphStrm) {
    bool end = endLenPldStrm.read();
    xf::security::aesEnc<_keyWidth> cipher;

    while (!end) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        // register cipherkey
        ap_uint<_keyWidth> cipherkey = cipherkeyStrm.read();
        cipher.updateKey(cipherkey);
        // pass on the cipherkey to CBC-MAC
        outCipherkeyStrm.write(cipherkey);
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
        std::cout << std::endl << "cipherkey = " << std::hex << cipherkey << std::endl;
#endif

        // counter block
        ap_uint<128> Ctr = 0;

        // formatting of the flag byte in counter block
        Ctr.range(127, 123) = 0;
        Ctr.range(122, 120) = _q - 1;

        // formatting of the N in counter block
        ap_uint<128> N = nonceStrm.read();
        ap_uint<128> nonce;
        // change the byte order
        nonce.range(7, 0) = N.range(127, 120);
        nonce.range(15, 8) = N.range(119, 112);
        nonce.range(23, 16) = N.range(111, 104);
        nonce.range(31, 24) = N.range(103, 96);
        nonce.range(39, 32) = N.range(95, 88);
        nonce.range(47, 40) = N.range(87, 80);
        nonce.range(55, 48) = N.range(79, 72);
        nonce.range(63, 56) = N.range(71, 64);
        nonce.range(71, 64) = N.range(63, 56);
        nonce.range(79, 72) = N.range(55, 48);
        nonce.range(87, 80) = N.range(47, 40);
        nonce.range(95, 88) = N.range(39, 32);
        nonce.range(103, 96) = N.range(31, 24);
        nonce.range(111, 104) = N.range(23, 16);
        nonce.range(119, 112) = N.range(15, 8);
        nonce.range(127, 120) = N.range(7, 0);
        Ctr.range(119, _q * 8) = nonce.range(127, 128 - (15 - _q) * 8);

        // formatting of the index in counter block
        Ctr.range(_q * 8 - 1, 0) = 0;

        // intermediate registers to perform the CTR encryption chain
        ap_uint<128> payloadReg = 0;
        ap_uint<128> inBlock = 0;
        ap_uint<128> inBlockReg = 0;
        ap_uint<128> outBlock = 0;
        ap_uint<128> cipherReg = 0;

        // is initialized?
        bool isInit = false;

        // total payload length in byte
        ap_uint<64> lenPld = lenPldStrm.read();
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
        std::cout << "payload_len    = " << std::dec << lenPld << std::endl;
#endif

        // length of cipher is equal to length of payload
        lenCphStrm.write(lenPld);

    LOOP_CIPHER_GEN:
        for (ap_uint<64> i = 0; i < (lenPld / 16 + ((lenPld % 16) > 0) + 1); i++) {
#pragma HLS pipeline II = 1
            // read a block of payload, 128 bits
            if (isInit) {
                payloadReg = payloadStrm.read();
                // pass on the payload to CBC-MAC
                outPayloadStrm.write(payloadReg);
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
                std::cout << "payload  = " << std::hex << payloadReg << std::endl;
#endif
            }

            // calculate S0 in the first iteration
            if (!isInit) {
                inBlock = Ctr;
                // calculate ciphers in the rest iterations
            } else {
                inBlock.range(8 * _q - 1, 0) = inBlock.range(8 * _q - 1, 0) + 1;
            }

            // change the byte order
            inBlockReg.range(7, 0) = inBlock.range(127, 120);
            inBlockReg.range(15, 8) = inBlock.range(119, 112);
            inBlockReg.range(23, 16) = inBlock.range(111, 104);
            inBlockReg.range(31, 24) = inBlock.range(103, 96);
            inBlockReg.range(39, 32) = inBlock.range(95, 88);
            inBlockReg.range(47, 40) = inBlock.range(87, 80);
            inBlockReg.range(55, 48) = inBlock.range(79, 72);
            inBlockReg.range(63, 56) = inBlock.range(71, 64);
            inBlockReg.range(71, 64) = inBlock.range(63, 56);
            inBlockReg.range(79, 72) = inBlock.range(55, 48);
            inBlockReg.range(87, 80) = inBlock.range(47, 40);
            inBlockReg.range(95, 88) = inBlock.range(39, 32);
            inBlockReg.range(103, 96) = inBlock.range(31, 24);
            inBlockReg.range(111, 104) = inBlock.range(23, 16);
            inBlockReg.range(119, 112) = inBlock.range(15, 8);
            inBlockReg.range(127, 120) = inBlock.range(7, 0);
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
            std::cout << "inBlock  = " << std::hex << inBlock << std::endl;
#endif

            // CIPH_k
            cipher.process(inBlockReg, cipherkey, outBlock);
// xf::security::internal::aesEncrypt<_keyWidth>(inBlockReg, cipherkey, outBlock);
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
            std::cout << "outBlock = " << std::hex << outBlock << std::endl;
#endif

            // get the cipher for current interation by outBlock and payload
            cipherReg = payloadReg ^ outBlock;
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
            std::cout << "cipher   = " << std::hex << cipherReg << std::endl;
#endif

            // write out S0, and prepare the rest iterations
            if (!isInit) {
                S0Strm.write(outBlock);
                isInit = true;
                // write out ciphers
            } else {
                cipherStrm.write(cipherReg);
            }
        }

        // last message?
        end = endLenPldStrm.read();
    }

} // end aesCtrEncrypt

/**
 *
 * @brief Implementation of CTR decryption part in CCM.
 *
 * The algorithm reference is: "Recommendation for Block Cipher Modes of Operation: The CCM Mode for Authentication and
 * Confidentiality"
 * The implementation is modified for better performance.
 *
 * @tparam _q Length in byte of the binary representation of the length of the payload in byte, q is an element of {2,
 * 3, 4, 5, 6, 7, 8}.
 * @tparam _keyWidth Bit-width of the cipher key, typically 128, 192, or 256 for AES.
 *
 * @param payloadStrm Input block stream text to be decrypted.
 * @param cipherkeyStrm Input cipher key used in decryption.
 * @param nonceStrm The nonce stream.
 * @param lenPldStrm Length of payload in byte.
 * @param endLenPldStrm Flag to signal the end of the payload length stream.
 * @param outPayloadStrm Pass on the payload stream to CBC-MAC.
 * @param outCipherkeyStrm Pass on the cipherkey to CBC-MAC.
 * @param S0Strm First cipher used to generate the MAC.
 * @param cipherStrm Output decrypted block stream text.
 * @param lenCphStrm Length of cipher in byte.
 *
 */

template <unsigned int _q = 8, unsigned int _keyWidth = 256>
void aesCtrDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& payloadStrm,
    hls::stream<ap_uint<_keyWidth> >& cipherkeyStrm,
    hls::stream<ap_uint<8 * (15 - _q)> >& nonceStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<bool>& endLenPldStrm,
    // stream out
    hls::stream<ap_uint<128> >& outPayloadStrm,
    hls::stream<ap_uint<_keyWidth> >& outCipherkeyStrm,
    hls::stream<ap_uint<128> >& S0Strm,
    hls::stream<ap_uint<128> >& cipherStrm,
    hls::stream<ap_uint<64> >& lenCphStrm) {
    bool end = endLenPldStrm.read();
    xf::security::aesEnc<_keyWidth> cipher;
    while (!end) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        // register cipherkey
        ap_uint<_keyWidth> cipherkey = cipherkeyStrm.read();
        cipher.updateKey(cipherkey);
        // pass on the cipherkey to CBC-MAC
        outCipherkeyStrm.write(cipherkey);
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
        std::cout << std::endl << "cipherkey = " << std::hex << cipherkey << std::endl;
#endif

        // counter block
        ap_uint<128> Ctr = 0;

        // formatting of the flag in counter block
        Ctr.range(127, 123) = 0;
        Ctr.range(122, 120) = _q - 1;

        // formatting of the N in counter block
        ap_uint<128> N = nonceStrm.read();
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
        std::cout << "nonce = " << std::hex << N << std::endl;
#endif
        ap_uint<128> nonce;

        // change the byte order
        nonce.range(7, 0) = N.range(127, 120);
        nonce.range(15, 8) = N.range(119, 112);
        nonce.range(23, 16) = N.range(111, 104);
        nonce.range(31, 24) = N.range(103, 96);
        nonce.range(39, 32) = N.range(95, 88);
        nonce.range(47, 40) = N.range(87, 80);
        nonce.range(55, 48) = N.range(79, 72);
        nonce.range(63, 56) = N.range(71, 64);
        nonce.range(71, 64) = N.range(63, 56);
        nonce.range(79, 72) = N.range(55, 48);
        nonce.range(87, 80) = N.range(47, 40);
        nonce.range(95, 88) = N.range(39, 32);
        nonce.range(103, 96) = N.range(31, 24);
        nonce.range(111, 104) = N.range(23, 16);
        nonce.range(119, 112) = N.range(15, 8);
        nonce.range(127, 120) = N.range(7, 0);
        Ctr.range(119, _q * 8) = nonce.range(127, 128 - (15 - _q) * 8);

        // formatting of the index in counter block
        Ctr.range(_q * 8 - 1, 0) = 0;
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
        std::cout << "Ctr0 = " << std::hex << Ctr << std::endl;
#endif

        // intermediate registers to perform the CTR encryption chain
        ap_uint<128> payloadReg = 0;
        ap_uint<128> inBlock = 0;
        ap_uint<128> inBlockReg = 0;
        ap_uint<128> outBlock = 0;
        ap_uint<128> cipherReg = 0;

        // is initialized?
        bool isInit = false;

        // total payload length in byte
        ap_uint<64> lenPld = lenPldStrm.read();
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
        std::cout << "payload_len    = " << std::dec << lenPld << std::endl;
#endif

        // length of cipher is equal to length of payload
        lenCphStrm.write(lenPld);

    LOOP_CIPHER_GEN:
        for (ap_uint<64> i = 0; i < (lenPld / 16 + ((lenPld % 16) > 0) + 1); i++) {
#pragma HLS loop_tripcount min = 10 max = 10 avg = 10
#pragma HLS pipeline II = 1
            // read a block of payload, 128 bits
            if (isInit) {
                payloadReg = payloadStrm.read();
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
                std::cout << "payload    = " << std::hex << payloadReg << std::endl;
#endif
            }

            // calculate S0 in the first iteration
            if (!isInit) {
                inBlock = Ctr;
                // calculate ciphers in the rest iterations
            } else {
                inBlock.range(8 * _q - 1, 0) = inBlock.range(8 * _q - 1, 0) + 1;
            }

            // change the byte order
            inBlockReg.range(7, 0) = inBlock.range(127, 120);
            inBlockReg.range(15, 8) = inBlock.range(119, 112);
            inBlockReg.range(23, 16) = inBlock.range(111, 104);
            inBlockReg.range(31, 24) = inBlock.range(103, 96);
            inBlockReg.range(39, 32) = inBlock.range(95, 88);
            inBlockReg.range(47, 40) = inBlock.range(87, 80);
            inBlockReg.range(55, 48) = inBlock.range(79, 72);
            inBlockReg.range(63, 56) = inBlock.range(71, 64);
            inBlockReg.range(71, 64) = inBlock.range(63, 56);
            inBlockReg.range(79, 72) = inBlock.range(55, 48);
            inBlockReg.range(87, 80) = inBlock.range(47, 40);
            inBlockReg.range(95, 88) = inBlock.range(39, 32);
            inBlockReg.range(103, 96) = inBlock.range(31, 24);
            inBlockReg.range(111, 104) = inBlock.range(23, 16);
            inBlockReg.range(119, 112) = inBlock.range(15, 8);
            inBlockReg.range(127, 120) = inBlock.range(7, 0);
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
            std::cout << "inBlock  = " << std::hex << inBlock << std::endl;
#endif

            // CIPH_k
            cipher.process(inBlockReg, cipherkey, outBlock);
// xf::security::internal::aesEncrypt<_keyWidth>(inBlockReg, cipherkey, outBlock);
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
            std::cout << "outBlock = " << std::hex << outBlock << std::endl;
#endif

            // get the cipher for current interation by outBlock and payload
            cipherReg = payloadReg ^ outBlock;
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
            std::cout << "cipher   = " << std::hex << cipherReg << std::endl;
#endif

            // write out S0, and prepare the rest iterations
            if (!isInit) {
                S0Strm.write(outBlock);
                isInit = true;
                // write out ciphers
            } else {
                cipherStrm.write(cipherReg);
                // pass on the payload to CBC-MAC
                outPayloadStrm.write(cipherReg);
            }
        }

        // last message?
        end = endLenPldStrm.read();
    }

} // end aesCtrDecrypt

/**
 *
 * @brief Impletmentation of CBC-MAC in CCM.
 *
 * The algorithm reference is: "Recommendation for Block Cipher Modes of Operation: The CCM Mode for Authentication and
 * Confidentiality"
 * The implementation is modified for better performance.
 *
 * @tparam _t Length of the MAC in byte, t is an element of {4, 6, 8, 10, 12, 14, 16}.
 * @tparam _keyWidth Bit-width of the cipher key, typically 128, 192, or 256 for AES.
 *
 * @param payloadStrm Input block stream text.
 * @param lenADStrm Length of associated data in byte.
 * @param lenPldStrm Length of payload in byte.
 * @param blockStrm Block stream of the formatted input as specified in reference.
 * @param cipherkeyStrm Input cipher key, typically 128, 192, or 256 for AES.
 * @param S0Strm First cipher used to generate the MAC.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param tagStrm The MAC.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

template <unsigned int _t = 16, unsigned int _keyWidth = 256>
void CBC_MAC(
    // stream in
    hls::stream<ap_uint<128> >& payloadStrm,
    hls::stream<ap_uint<64> >& lenADStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<ap_uint<128> >& blockStrm,
    hls::stream<ap_uint<_keyWidth> >& cipherkeyStrm,
    hls::stream<ap_uint<128> >& S0Strm,
    hls::stream<bool>& endLenStrm,
    // stream out
    hls::stream<ap_uint<8 * _t> >& tagStrm,
    hls::stream<bool>& endTagStrm) {
    bool end = endLenStrm.read();
    xf::security::aesEnc<_keyWidth> cipher;

    while (!end) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
        // register S0
        ap_uint<128> S0 = S0Strm.read();
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
        std::cout << std::endl << "S0 = " << std::hex << S0 << std::endl;
#endif

        // length of associated data in byte
        ap_uint<64> lenAD = lenADStrm.read();

        // intermediate registers to perform the CTR encryption chain
        ap_uint<128> blockReg = 0;
        ap_uint<128> pld = 0;
        ap_uint<128> pldReg = 0;
        ap_uint<128> inBlock = 0;
        ap_uint<128> inBlockReg = 0;
        ap_uint<128> outBlock = 0;
        ap_uint<128> outBlockReg = 0;

        // header in byte of the encoded AD block
        unsigned char headerLen = 0;
        if ((0 < lenAD) && (lenAD < (0x10000UL - 0x100UL))) {
            headerLen = 2;
        } else if (((0x10000UL - 0x100UL) <= lenAD) && (lenAD < 0x100000000UL)) {
            headerLen = 6;
        } else if (0x100000000UL <= lenAD) {
            headerLen = 10;
        }

        ap_uint<_keyWidth> cipherkey = cipherkeyStrm.read();
        cipher.updateKey(cipherkey);
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
        std::cout << std::endl << "cipherkey = " << std::hex << cipherkey << std::endl;
        std::cout << "key_width = " << std::dec << _keyWidth << std::endl;
#endif

        // is initialized?
        bool isInit = false;

    LOOP_AD_TAG:
        for (ap_uint<64> i = 0;
             i < ((lenAD / 16) + ((lenAD % 16) > 0) +
                  ((((lenAD + headerLen) % 16) > 0) && (((lenAD + headerLen) % 16) <= headerLen)) + 1);
             i++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 10 max = 10 avg = 10
            // read a formatting block
            blockReg = blockStrm.read();
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
            std::cout << "B" << std::dec << i << " = " << std::hex << blockReg << std::endl;
#endif

            if (!isInit) {
                inBlock = blockReg;
                isInit = true;
            } else {
                inBlock = blockReg ^ outBlockReg;
            }

            // change the byte order
            inBlockReg.range(7, 0) = inBlock.range(127, 120);
            inBlockReg.range(15, 8) = inBlock.range(119, 112);
            inBlockReg.range(23, 16) = inBlock.range(111, 104);
            inBlockReg.range(31, 24) = inBlock.range(103, 96);
            inBlockReg.range(39, 32) = inBlock.range(95, 88);
            inBlockReg.range(47, 40) = inBlock.range(87, 80);
            inBlockReg.range(55, 48) = inBlock.range(79, 72);
            inBlockReg.range(63, 56) = inBlock.range(71, 64);
            inBlockReg.range(71, 64) = inBlock.range(63, 56);
            inBlockReg.range(79, 72) = inBlock.range(55, 48);
            inBlockReg.range(87, 80) = inBlock.range(47, 40);
            inBlockReg.range(95, 88) = inBlock.range(39, 32);
            inBlockReg.range(103, 96) = inBlock.range(31, 24);
            inBlockReg.range(111, 104) = inBlock.range(23, 16);
            inBlockReg.range(119, 112) = inBlock.range(15, 8);
            inBlockReg.range(127, 120) = inBlock.range(7, 0);
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
            std::cout << "inBlock  = " << std::hex << inBlock << std::endl;
#endif

            // CIPH_k
            cipher.process(inBlockReg, cipherkey, outBlock);
// xf::security::internal::aesEncrypt<_keyWidth>(inBlockReg, cipherkey, outBlock);
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
            std::cout << "outBlock = " << std::hex << outBlock << std::endl;
#endif

            // change the byte order
            outBlockReg.range(7, 0) = outBlock.range(127, 120);
            outBlockReg.range(15, 8) = outBlock.range(119, 112);
            outBlockReg.range(23, 16) = outBlock.range(111, 104);
            outBlockReg.range(31, 24) = outBlock.range(103, 96);
            outBlockReg.range(39, 32) = outBlock.range(95, 88);
            outBlockReg.range(47, 40) = outBlock.range(87, 80);
            outBlockReg.range(55, 48) = outBlock.range(79, 72);
            outBlockReg.range(63, 56) = outBlock.range(71, 64);
            outBlockReg.range(71, 64) = outBlock.range(63, 56);
            outBlockReg.range(79, 72) = outBlock.range(55, 48);
            outBlockReg.range(87, 80) = outBlock.range(47, 40);
            outBlockReg.range(95, 88) = outBlock.range(39, 32);
            outBlockReg.range(103, 96) = outBlock.range(31, 24);
            outBlockReg.range(111, 104) = outBlock.range(23, 16);
            outBlockReg.range(119, 112) = outBlock.range(15, 8);
            outBlockReg.range(127, 120) = outBlock.range(7, 0);
        }

        // length of payload in byte
        ap_uint<64> lenPld = lenPldStrm.read();

    LOOP_PLD_TAG:
        for (ap_uint<64> i = 0; i < (lenPld / 16 + ((lenPld % 16) > 0)); i++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 10 max = 10 avg = 10
            // read a payload block
            pld = payloadStrm.read();
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
            std::cout << "payload  = " << std::hex << pld << std::endl;
#endif

            // change the byte order
            pldReg.range(7, 0) = pld.range(127, 120);
            pldReg.range(15, 8) = pld.range(119, 112);
            pldReg.range(23, 16) = pld.range(111, 104);
            pldReg.range(31, 24) = pld.range(103, 96);
            pldReg.range(39, 32) = pld.range(95, 88);
            pldReg.range(47, 40) = pld.range(87, 80);
            pldReg.range(55, 48) = pld.range(79, 72);
            pldReg.range(63, 56) = pld.range(71, 64);
            pldReg.range(71, 64) = pld.range(63, 56);
            pldReg.range(79, 72) = pld.range(55, 48);
            pldReg.range(87, 80) = pld.range(47, 40);
            pldReg.range(95, 88) = pld.range(39, 32);
            pldReg.range(103, 96) = pld.range(31, 24);
            pldReg.range(111, 104) = pld.range(23, 16);
            pldReg.range(119, 112) = pld.range(15, 8);
            pldReg.range(127, 120) = pld.range(7, 0);

            inBlock = pldReg ^ outBlockReg;

            // change the byte order
            inBlockReg.range(7, 0) = inBlock.range(127, 120);
            inBlockReg.range(15, 8) = inBlock.range(119, 112);
            inBlockReg.range(23, 16) = inBlock.range(111, 104);
            inBlockReg.range(31, 24) = inBlock.range(103, 96);
            inBlockReg.range(39, 32) = inBlock.range(95, 88);
            inBlockReg.range(47, 40) = inBlock.range(87, 80);
            inBlockReg.range(55, 48) = inBlock.range(79, 72);
            inBlockReg.range(63, 56) = inBlock.range(71, 64);
            inBlockReg.range(71, 64) = inBlock.range(63, 56);
            inBlockReg.range(79, 72) = inBlock.range(55, 48);
            inBlockReg.range(87, 80) = inBlock.range(47, 40);
            inBlockReg.range(95, 88) = inBlock.range(39, 32);
            inBlockReg.range(103, 96) = inBlock.range(31, 24);
            inBlockReg.range(111, 104) = inBlock.range(23, 16);
            inBlockReg.range(119, 112) = inBlock.range(15, 8);
            inBlockReg.range(127, 120) = inBlock.range(7, 0);
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
            std::cout << "inBlock  = " << std::hex << inBlock << std::endl;
#endif

            // CIPH_k
            cipher.process(inBlockReg, cipherkey, outBlock);
// xf::security::internal::aesEncrypt<_keyWidth>(inBlockReg, cipherkey, outBlock);
#if !defined(__SYNTHESIS__) && __XF_SECURITY_CCM_DEBUG__ == 1
            std::cout << "outBlock = " << std::hex << outBlock << std::endl;
#endif

            // change the byte order
            outBlockReg.range(7, 0) = outBlock.range(127, 120);
            outBlockReg.range(15, 8) = outBlock.range(119, 112);
            outBlockReg.range(23, 16) = outBlock.range(111, 104);
            outBlockReg.range(31, 24) = outBlock.range(103, 96);
            outBlockReg.range(39, 32) = outBlock.range(95, 88);
            outBlockReg.range(47, 40) = outBlock.range(87, 80);
            outBlockReg.range(55, 48) = outBlock.range(79, 72);
            outBlockReg.range(63, 56) = outBlock.range(71, 64);
            outBlockReg.range(71, 64) = outBlock.range(63, 56);
            outBlockReg.range(79, 72) = outBlock.range(55, 48);
            outBlockReg.range(87, 80) = outBlock.range(47, 40);
            outBlockReg.range(95, 88) = outBlock.range(39, 32);
            outBlockReg.range(103, 96) = outBlock.range(31, 24);
            outBlockReg.range(111, 104) = outBlock.range(23, 16);
            outBlockReg.range(119, 112) = outBlock.range(15, 8);
            outBlockReg.range(127, 120) = outBlock.range(7, 0);
        }

        // emit MAC
        ap_uint<128> tag;
        tag = outBlock ^ S0;
        tagStrm.write(tag.range(8 * _t - 1, 0));
        endTagStrm.write(false);

        // last message?
        end = endLenStrm.read();
    }

    endTagStrm.write(true);

} // end CBC_MAC

// @brief Duplicate input stream to output streams
template <unsigned int _width>
void dupStrm(
    // stream in
    hls::stream<ap_uint<_width> >& inStrm,
    hls::stream<bool>& endInStrm,
    // stream out
    hls::stream<ap_uint<_width> >& out1Strm,
    hls::stream<bool>& endOut1Strm,
    hls::stream<ap_uint<_width> >& out2Strm,
    hls::stream<bool>& endOut2Strm,
    hls::stream<ap_uint<_width> >& out3Strm,
    hls::stream<bool>& endOut3Strm) {
    ap_uint<_width> inReg;

    bool end = endInStrm.read();

LOOP_DUP_STRM:
    while (!end) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
#pragma HLS PIPELINE II = 1
        inReg = inStrm.read();

        out1Strm.write(inReg);
        endOut1Strm.write(false);
        out2Strm.write(inReg);
        endOut2Strm.write(false);
        out3Strm.write(inReg);
        endOut3Strm.write(false);

        end = endInStrm.read();
    }

    endOut1Strm.write(true);
    endOut2Strm.write(true);
    endOut3Strm.write(true);

} // end dupStrm

/**
 *
 * @brief aesCcmEncrypt is CCM encryption mode with AES single block cipher.
 *
 * The algorithm reference is: "Recommendation for Block Cipher Modes of Operation: The CCM Mode for Authentication and
 * Confidentiality"
 * The implementation is modified for better performance.
 *
 * @tparam _t Length of the MAC in byte, t is an element of {4, 6, 8, 10, 12, 14, 16}.
 * @tparam _q Length in byte of the binary representation of the length of the payload in byte, q is an element of {2,
 * 3, 4, 5, 6, 7, 8}.
 * @tparam _keyWidth The bit-width of the cipher key, which is 128, 192, or 256.
 *
 * @param payloadStrm Input block stream text to be encrypted.
 * @param cipherkeyStrm Input cipher key used in encryption.
 * @param nonceStrm The nonce stream.
 * @param ADStrm Associated data stream.
 * @param lenADStrm Length of associated data in byte.
 * @param lenPldStrm Length of payload in byte.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param cipherStrm Output encrypted block stream text.
 * @param lenCphStrm Length of cipher in byte.
 * @param tagStrm The MAC stream.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

template <unsigned int _t = 16, unsigned int _q = 8, unsigned int _keyWidth = 256>
void aesCcmEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& payloadStrm,
    hls::stream<ap_uint<_keyWidth> >& cipherkeyStrm,
    hls::stream<ap_uint<8 * (15 - _q)> >& nonceStrm,
    hls::stream<ap_uint<128> >& ADStrm,
    hls::stream<ap_uint<64> >& lenADStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<bool>& endLenStrm,
    // stream out
    hls::stream<ap_uint<128> >& cipherStrm,
    hls::stream<ap_uint<64> >& lenCphStrm,
    hls::stream<ap_uint<8 * _t> >& tagStrm,
    hls::stream<bool>& endTagStrm) {
#pragma HLS DATAFLOW

    hls::stream<ap_uint<64> > lenPldStrm1("lenPldStrm1");
    hls::stream<ap_uint<64> > lenPldStrm2("lenPldStrm2");
    hls::stream<ap_uint<64> > lenPldStrm3("lenPldStrm3");
#pragma HLS RESOURCE variable = lenPldStrm1 core = FIFO_LUTRAM
#pragma HLS STREAM variable = lenPldStrm1 depth = 32
#pragma HLS RESOURCE variable = lenPldStrm2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = lenPldStrm2 depth = 32
#pragma HLS RESOURCE variable = lenPldStrm3 core = FIFO_LUTRAM
#pragma HLS STREAM variable = lenPldStrm3 depth = 32

    hls::stream<bool> endLenStrm1("endLenStrm1");
    hls::stream<bool> endLenStrm2("endLenStrm2");
    hls::stream<bool> endLenStrm3("endLenStrm3");
#pragma HLS RESOURCE variable = endLenStrm1 core = FIFO_LUTRAM
#pragma HLS STREAM variable = endLenStrm1 depth = 32
#pragma HLS RESOURCE variable = endLenStrm2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = endLenStrm2 depth = 32
#pragma HLS RESOURCE variable = endLenStrm3 core = FIFO_LUTRAM
#pragma HLS STREAM variable = endLenStrm3 depth = 32

    hls::stream<ap_uint<128> > blockStrm("blockStrm");
#pragma HLS RESOURCE variable = blockStrm core = FIFO_LUTRAM
#pragma HLS STREAM variable = blockStrm depth = 32

    hls::stream<ap_uint<64> > outLenADStrm("outLenADStrm");
#pragma HLS RESOURCE variable = outLenADStrm core = FIFO_LUTRAM
#pragma HLS STREAM variable = outLenADStrm depth = 32

    hls::stream<ap_uint<8 * (15 - _q)> > outNonceStrm("outNonceStrm");
#pragma HLS RESOURCE variable = outNonceStrm core = FIFO_LUTRAM
#pragma HLS STREAM variable = outNonceStrm depth = 32

    hls::stream<ap_uint<128> > outPayloadStrm("outPayloadStrm");
#pragma HLS RESOURCE variable = outPayloadStrm core = FIFO_LUTRAM
#pragma HLS STREAM variable = outPayloadStrm depth = 32

    hls::stream<ap_uint<_keyWidth> > outCipherkeyStrm("outCipherkeyStrm");
#pragma HLS RESOURCE variable = outCipherkeyStrm core = FIFO_LUTRAM
#pragma HLS STREAM variable = outCipherkeyStrm depth = 32

    hls::stream<ap_uint<128> > S0Strm("S0Strm");
#pragma HLS RESOURCE variable = S0Strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = S0Strm depth = 32

    dupStrm<64>(lenPldStrm, endLenStrm, lenPldStrm1, endLenStrm1, lenPldStrm2, endLenStrm2, lenPldStrm3, endLenStrm3);

    formatting<_t, _q>(ADStrm, nonceStrm, lenADStrm, lenPldStrm1, endLenStrm1, blockStrm, outLenADStrm, outNonceStrm);

    aesCtrEncrypt<_q, _keyWidth>(payloadStrm, cipherkeyStrm, outNonceStrm, lenPldStrm2, endLenStrm2, outPayloadStrm,
                                 outCipherkeyStrm, S0Strm, cipherStrm, lenCphStrm);

    CBC_MAC<_t, _keyWidth>(outPayloadStrm, outLenADStrm, lenPldStrm3, blockStrm, outCipherkeyStrm, S0Strm, endLenStrm3,
                           tagStrm, endTagStrm);

} // end aesCcmEncrypt

/**
 *
 * @brief aesCcmDecrypt is CCM decryption mode with AES single block cipher.
 *
 * The algorithm reference is: "Recommendation for Block Cipher Modes of Operation: The CCM Mode for Authentication and
 * Confidentiality"
 * The implementation is modified for better performance.
 *
 * @tparam _t Length of the MAC in byte, t is an element of {4, 6, 8, 10, 12, 14, 16}.
 * @tparam _q Length in byte of the binary representation of the length of the payload in byte, q is an element of {2,
 * 3, 4, 5, 6, 7, 8}.
 * @tparam _keyWidth The bit-width of the cipher key, which is 128, 192, or 256.
 *
 * @param payloadStrm Input block stream text to be decrypted.
 * @param cipherkeyStrm Input cipher key used in decryption.
 * @param nonceStrm The nonce stream.
 * @param ADStrm Associated data stream.
 * @param lenADStrm Length of associated data in byte.
 * @param lenPldStrm Length of payload in byte.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param cipherStrm Output decrypted block stream text.
 * @param lenCphStrm Length of cipher in byte.
 * @param tagStrm The MAC stream.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

template <unsigned int _t = 16, unsigned int _q = 8, unsigned int _keyWidth = 256>
void aesCcmDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& payloadStrm,
    hls::stream<ap_uint<_keyWidth> >& cipherkeyStrm,
    hls::stream<ap_uint<8 * (15 - _q)> >& nonceStrm,
    hls::stream<ap_uint<128> >& ADStrm,
    hls::stream<ap_uint<64> >& lenADStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<bool>& endLenStrm,
    // stream out
    hls::stream<ap_uint<128> >& cipherStrm,
    hls::stream<ap_uint<64> >& lenCphStrm,
    hls::stream<ap_uint<8 * _t> >& tagStrm,
    hls::stream<bool>& endTagStrm) {
#pragma HLS DATAFLOW

    hls::stream<ap_uint<64> > lenPldStrm1("lenPldStrm1");
    hls::stream<ap_uint<64> > lenPldStrm2("lenPldStrm2");
    hls::stream<ap_uint<64> > lenPldStrm3("lenPldStrm3");
#pragma HLS RESOURCE variable = lenPldStrm1 core = FIFO_LUTRAM
#pragma HLS STREAM variable = lenPldStrm1 depth = 32
#pragma HLS RESOURCE variable = lenPldStrm2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = lenPldStrm2 depth = 32
#pragma HLS RESOURCE variable = lenPldStrm3 core = FIFO_LUTRAM
#pragma HLS STREAM variable = lenPldStrm3 depth = 32

    hls::stream<bool> endLenStrm1("endLenStrm1");
    hls::stream<bool> endLenStrm2("endLenStrm2");
    hls::stream<bool> endLenStrm3("endLenStrm3");
#pragma HLS RESOURCE variable = endLenStrm1 core = FIFO_LUTRAM
#pragma HLS STREAM variable = endLenStrm1 depth = 32
#pragma HLS RESOURCE variable = endLenStrm2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = endLenStrm2 depth = 32
#pragma HLS RESOURCE variable = endLenStrm3 core = FIFO_LUTRAM
#pragma HLS STREAM variable = endLenStrm3 depth = 32

    hls::stream<ap_uint<128> > blockStrm("blockStrm");
#pragma HLS RESOURCE variable = blockStrm core = FIFO_LUTRAM
#pragma HLS STREAM variable = blockStrm depth = 32

    hls::stream<ap_uint<64> > outLenADStrm("outLenADStrm");
#pragma HLS RESOURCE variable = outLenADStrm core = FIFO_LUTRAM
#pragma HLS STREAM variable = outLenADStrm depth = 32

    hls::stream<ap_uint<8 * (15 - _q)> > outNonceStrm("outNonceStrm");
#pragma HLS RESOURCE variable = outNonceStrm core = FIFO_LUTRAM
#pragma HLS STREAM variable = outNonceStrm depth = 32

    hls::stream<ap_uint<128> > outPayloadStrm("outPayloadStrm");
#pragma HLS RESOURCE variable = outPayloadStrm core = FIFO_LUTRAM
#pragma HLS STREAM variable = outPayloadStrm depth = 32

    hls::stream<ap_uint<_keyWidth> > outCipherkeyStrm("outCipherkeyStrm");
#pragma HLS RESOURCE variable = outCipherkeyStrm core = FIFO_LUTRAM
#pragma HLS STREAM variable = outCipherkeyStrm depth = 32

    hls::stream<ap_uint<128> > S0Strm("S0Strm");
#pragma HLS RESOURCE variable = S0Strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = S0Strm depth = 32

    dupStrm<64>(lenPldStrm, endLenStrm, lenPldStrm1, endLenStrm1, lenPldStrm2, endLenStrm2, lenPldStrm3, endLenStrm3);

    formatting<_t, _q>(ADStrm, nonceStrm, lenADStrm, lenPldStrm1, endLenStrm1, blockStrm, outLenADStrm, outNonceStrm);

    aesCtrDecrypt<_q, _keyWidth>(payloadStrm, cipherkeyStrm, outNonceStrm, lenPldStrm2, endLenStrm2, outPayloadStrm,
                                 outCipherkeyStrm, S0Strm, cipherStrm, lenCphStrm);

    CBC_MAC<_t, _keyWidth>(outPayloadStrm, outLenADStrm, lenPldStrm3, blockStrm, outCipherkeyStrm, S0Strm, endLenStrm3,
                           tagStrm, endTagStrm);

} // aesCcmDecrypt

} // namespace internal

/**
 *
 * @brief aes128CcmEncrypt is CCM encryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is: "Recommendation for Block Cipher Modes of Operation: The CCM Mode for Authentication and
 * Confidentiality"
 * The implementation is modified for better performance.
 *
 * @tparam _t Length of the MAC in byte, t is an element of {4, 6, 8, 10, 12, 14, 16}.
 * @tparam _q Length in byte of the binary representation of the length of the payload in byte, q is an element of {2,
 * 3, 4, 5, 6, 7, 8}.
 *
 * @param payloadStrm Input block stream text to be encrypted.
 * @param cipherkeyStrm Input cipher key used in encryption.
 * @param nonceStrm The nonce stream.
 * @param ADStrm Associated data stream.
 * @param lenADStrm Length of associated data in byte.
 * @param lenPldStrm Length of payload in byte.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param cipherStrm Output encrypted block stream text.
 * @param lenCphStrm Length of cipher in byte.
 * @param tagStrm The MAC stream.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

template <unsigned int _t = 16, unsigned int _q = 8>
void aes128CcmEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& payloadStrm,
    hls::stream<ap_uint<128> >& cipherkeyStrm,
    hls::stream<ap_uint<8 * (15 - _q)> >& nonceStrm,
    hls::stream<ap_uint<128> >& ADStrm,
    hls::stream<ap_uint<64> >& lenADStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<bool>& endLenStrm,
    // stream out
    hls::stream<ap_uint<128> >& cipherStrm,
    hls::stream<ap_uint<64> >& lenCphStrm,
    hls::stream<ap_uint<8 * _t> >& tagStrm,
    hls::stream<bool>& endTagStrm) {
    internal::aesCcmEncrypt<_t, _q, 128>(payloadStrm, cipherkeyStrm, nonceStrm, ADStrm, lenADStrm, lenPldStrm,
                                         endLenStrm, cipherStrm, lenCphStrm, tagStrm, endTagStrm);

} // end aes128CcmEncrypt

/**
 *
 * @brief aes128CcmDecrypt is CCM decryption mode with AES-128 single block cipher.
 *
 * The algorithm reference is: "Recommendation for Block Cipher Modes of Operation: The CCM Mode for Authentication and
 * Confidentiality"
 * The implementation is modified for better performance.
 *
 * @tparam _t Length of the MAC in byte, t is an element of {4, 6, 8, 10, 12, 14, 16}.
 * @tparam _q Length in byte of the binary representation of the length of the payload in byte, q is an element of {2,
 * 3, 4, 5, 6, 7, 8}.
 *
 * @param payloadStrm Input block stream text to be decrypted.
 * @param cipherkeyStrm Input cipher key used in decryption.
 * @param nonceStrm The nonce stream.
 * @param ADStrm Associated data stream.
 * @param lenADStrm Length of associated data in byte.
 * @param lenPldStrm Length of payload in byte.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param cipherStrm Output decrypted block stream text.
 * @param lenCphStrm Length of cipher in byte.
 * @param tagStrm The MAC stream.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

template <unsigned int _t = 16, unsigned int _q = 8>
void aes128CcmDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& payloadStrm,
    hls::stream<ap_uint<128> >& cipherkeyStrm,
    hls::stream<ap_uint<8 * (15 - _q)> >& nonceStrm,
    hls::stream<ap_uint<128> >& ADStrm,
    hls::stream<ap_uint<64> >& lenADStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<bool>& endLenStrm,
    // stream out
    hls::stream<ap_uint<128> >& cipherStrm,
    hls::stream<ap_uint<64> >& lenCphStrm,
    hls::stream<ap_uint<8 * _t> >& tagStrm,
    hls::stream<bool>& endTagStrm) {
    internal::aesCcmDecrypt<_t, _q, 128>(payloadStrm, cipherkeyStrm, nonceStrm, ADStrm, lenADStrm, lenPldStrm,
                                         endLenStrm, cipherStrm, lenCphStrm, tagStrm, endTagStrm);

} // end aes128CcmDecrypt

/**
 *
 * @brief aes192CcmEncrypt is CCM encryption mode with AES-192 single block cipher.
 *
 * The algorithm reference is: "Recommendation for Block Cipher Modes of Operation: The CCM Mode for Authentication and
 * Confidentiality"
 * The implementation is modified for better performance.
 *
 * @tparam _t Length of the MAC in byte, t is an element of {4, 6, 8, 10, 12, 14, 16}.
 * @tparam _q Length in byte of the binary representation of the length of the payload in byte, q is an element of {2,
 * 3, 4, 5, 6, 7, 8}.
 *
 * @param payloadStrm Input block stream text to be encrypted.
 * @param cipherkeyStrm Input cipher key used in encryption.
 * @param nonceStrm The nonce stream.
 * @param ADStrm Associated data stream.
 * @param lenADStrm Length of associated data in byte.
 * @param lenPldStrm Length of payload in byte.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param cipherStrm Output encrypted block stream text.
 * @param lenCphStrm Length of cipher in byte.
 * @param tagStrm The MAC stream.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

template <unsigned int _t = 16, unsigned int _q = 8>
void aes192CcmEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& payloadStrm,
    hls::stream<ap_uint<192> >& cipherkeyStrm,
    hls::stream<ap_uint<8 * (15 - _q)> >& nonceStrm,
    hls::stream<ap_uint<128> >& ADStrm,
    hls::stream<ap_uint<64> >& lenADStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<bool>& endLenStrm,
    // stream out
    hls::stream<ap_uint<128> >& cipherStrm,
    hls::stream<ap_uint<64> >& lenCphStrm,
    hls::stream<ap_uint<8 * _t> >& tagStrm,
    hls::stream<bool>& endTagStrm) {
    internal::aesCcmEncrypt<_t, _q, 192>(payloadStrm, cipherkeyStrm, nonceStrm, ADStrm, lenADStrm, lenPldStrm,
                                         endLenStrm, cipherStrm, lenCphStrm, tagStrm, endTagStrm);

} // end aes192CcmEncrypt

/**
 *
 * @brief aes192CcmDecrypt is CCM decryption mode with AES-192 single block cipher.
 *
 * The algorithm reference is: "Recommendation for Block Cipher Modes of Operation: The CCM Mode for Authentication and
 * Confidentiality"
 * The implementation is modified for better performance.
 *
 * @tparam _t Length of the MAC in byte, t is an element of {4, 6, 8, 10, 12, 14, 16}.
 * @tparam _q Length in byte of the binary representation of the length of the payload in byte, q is an element of {2,
 * 3, 4, 5, 6, 7, 8}.
 *
 * @param payloadStrm Input block stream text to be decrypted.
 * @param cipherkeyStrm Input cipher key used in decryption.
 * @param nonceStrm The nonce stream.
 * @param ADStrm Associated data stream.
 * @param lenADStrm Length of associated data in byte.
 * @param lenPldStrm Length of payload in byte.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param cipherStrm Output decrypted block stream text.
 * @param lenCphStrm Length of cipher in byte.
 * @param tagStrm The MAC stream.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

template <unsigned int _t = 16, unsigned int _q = 8>
void aes192CcmDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& payloadStrm,
    hls::stream<ap_uint<192> >& cipherkeyStrm,
    hls::stream<ap_uint<8 * (15 - _q)> >& nonceStrm,
    hls::stream<ap_uint<128> >& ADStrm,
    hls::stream<ap_uint<64> >& lenADStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<bool>& endLenStrm,
    // stream out
    hls::stream<ap_uint<128> >& cipherStrm,
    hls::stream<ap_uint<64> >& lenCphStrm,
    hls::stream<ap_uint<8 * _t> >& tagStrm,
    hls::stream<bool>& endTagStrm) {
    internal::aesCcmDecrypt<_t, _q, 192>(payloadStrm, cipherkeyStrm, nonceStrm, ADStrm, lenADStrm, lenPldStrm,
                                         endLenStrm, cipherStrm, lenCphStrm, tagStrm, endTagStrm);

} // end aes192CcmDecrypt

/**
 *
 * @brief aes256CcmEncrypt is CCM encryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is: "Recommendation for Block Cipher Modes of Operation: The CCM Mode for Authentication and
 * Confidentiality"
 * The implementation is modified for better performance.
 *
 * @tparam _t Length of the MAC in byte, t is an element of {4, 6, 8, 10, 12, 14, 16}.
 * @tparam _q Length in byte of the binary representation of the length of the payload in byte, q is an element of {2,
 * 3, 4, 5, 6, 7, 8}.
 *
 * @param payloadStrm Input block stream text to be encrypted.
 * @param cipherkeyStrm Input cipher key used in encryption.
 * @param nonceStrm The nonce stream.
 * @param ADStrm Associated data stream.
 * @param lenADStrm Length of associated data in byte.
 * @param lenPldStrm Length of payload in byte.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param cipherStrm Output encrypted block stream text.
 * @param lenCphStrm Length of cipher in byte.
 * @param tagStrm The MAC stream.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

template <unsigned int _t = 16, unsigned int _q = 8>
void aes256CcmEncrypt(
    // stream in
    hls::stream<ap_uint<128> >& payloadStrm,
    hls::stream<ap_uint<256> >& cipherkeyStrm,
    hls::stream<ap_uint<8 * (15 - _q)> >& nonceStrm,
    hls::stream<ap_uint<128> >& ADStrm,
    hls::stream<ap_uint<64> >& lenADStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<bool>& endLenStrm,
    // stream out
    hls::stream<ap_uint<128> >& cipherStrm,
    hls::stream<ap_uint<64> >& lenCphStrm,
    hls::stream<ap_uint<8 * _t> >& tagStrm,
    hls::stream<bool>& endTagStrm) {
    internal::aesCcmEncrypt<_t, _q, 256>(payloadStrm, cipherkeyStrm, nonceStrm, ADStrm, lenADStrm, lenPldStrm,
                                         endLenStrm, cipherStrm, lenCphStrm, tagStrm, endTagStrm);

} // end aes256CcmEncrypt

/**
 *
 * @brief aes256CcmDecrypt is CCM decryption mode with AES-256 single block cipher.
 *
 * The algorithm reference is: "Recommendation for Block Cipher Modes of Operation: The CCM Mode for Authentication and
 * Confidentiality"
 * The implementation is modified for better performance.
 *
 * @tparam _t Length of the MAC in byte, t is an element of {4, 6, 8, 10, 12, 14, 16}.
 * @tparam _q Length in byte of the binary representation of the length of the payload in byte, q is an element of {2,
 * 3, 4, 5, 6, 7, 8}.
 *
 * @param payloadStrm Input block stream text to be decrypted.
 * @param cipherkeyStrm Input cipher key used in decryption.
 * @param nonceStrm The nonce stream.
 * @param ADStrm Associated data stream.
 * @param lenADStrm Length of associated data in byte.
 * @param lenPldStrm Length of payload in byte.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param cipherStrm Output decrypted block stream text.
 * @param lenCphStrm Length of cipher in byte.
 * @param tagStrm The MAC stream.
 * @param endTagStrm Flag to signal the end of the MAC stream.
 *
 */

template <unsigned int _t = 16, unsigned int _q = 8>
void aes256CcmDecrypt(
    // stream in
    hls::stream<ap_uint<128> >& payloadStrm,
    hls::stream<ap_uint<256> >& cipherkeyStrm,
    hls::stream<ap_uint<8 * (15 - _q)> >& nonceStrm,
    hls::stream<ap_uint<128> >& ADStrm,
    hls::stream<ap_uint<64> >& lenADStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<bool>& endLenStrm,
    // stream out
    hls::stream<ap_uint<128> >& cipherStrm,
    hls::stream<ap_uint<64> >& lenCphStrm,
    hls::stream<ap_uint<8 * _t> >& tagStrm,
    hls::stream<bool>& endTagStrm) {
    internal::aesCcmDecrypt<_t, _q, 256>(payloadStrm, cipherkeyStrm, nonceStrm, ADStrm, lenADStrm, lenPldStrm,
                                         endLenStrm, cipherStrm, lenCphStrm, tagStrm, endTagStrm);

} // end aes256CcmDecrypt

} // namespace security
} // namespace xf

#endif
