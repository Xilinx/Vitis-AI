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
 * @file chacha20.hpp
 * @brief header file for ChaCha20.
 * This file part of Vitis Security Library.
 *
 */

#ifndef _XF_SECURITY_CHACHA20_HPP_
#define _XF_SECURITY_CHACHA20_HPP_

#include <ap_int.h>
#include <hls_stream.h>

#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
#include <iostream>
#endif

namespace xf {
namespace security {
namespace internal {

typedef ap_uint<512> blockTypeChacha;

#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
#define QR(a, b, c, d)   \
    do {                 \
        a += b;          \
        d ^= a;          \
        d = ROTL(d, 16); \
        c += d;          \
        b ^= c;          \
        b = ROTL(b, 12); \
        a += b;          \
        d ^= a;          \
        d = ROTL(d, 8);  \
        c += d;          \
        b ^= c;          \
        b = ROTL(b, 7);  \
    } while (0);
#define ROUNDS 20

/**
 * @brief chacha20 is a function for stream ciphering
 *
 * @param keyStrm initail key
 * @param counterNonceStm initial counter and nonce
 * @param plainStrm input  plain text to be encrypted
 * @param ePlainStrm the end flag of plainStrm
 * @param cipherStrm  output encrypted text
 * @param eCipherStrm the end flag of cipherStrm
 *
 */
void chacha20Imp(hls::stream<ap_uint<256> >& keyStrm,
                 hls::stream<ap_uint<128> >& counterNonceStrm,
                 hls::stream<ap_uint<512> >& plainStrm,
                 hls::stream<bool>& ePlainStrm,
                 hls::stream<ap_uint<512> >& cipherStrm,
                 hls::stream<bool>& eCipherStrm) {
    ap_uint<32> s[16];
#pragma HLS array_partition variable = s complete
    ap_uint<32> x[16];
#pragma HLS array_partition variable = x complete
    ap_uint<32> xs[16];
#pragma HLS array_partition variable = xs complete
    /* sigma constant "expand 32-byte k" in little-endian encoding */
    /*  input[0] = ((u32)'e') | ((u32)'x'<<8) | ((u32)'p'<<16) | ((u32)'a'<<24);
        input[1] = ((u32)'n') | ((u32)'d'<<8) | ((u32)' '<<16) | ((u32)'3'<<24);
        input[2] = ((u32)'2') | ((u32)'-'<<8) | ((u32)'b'<<16) | ((u32)'y'<<24);
        input[3] = ((u32)'t') | ((u32)'e'<<8) | ((u32)' '<<16) | ((u32)'k'<<24);
    */
    s[0] = 0x61707865;
    s[1] = 0x3320646e;
    s[2] = 0x79622d32;
    s[3] = 0x6b206574;
    ap_uint<256> key = keyStrm.read();
    for (int i = 0; i < 8; ++i) {
#pragma HLS unroll
        s[i + 4] = key.range(32 * (i + 1) - 1, i * 32);
    }
    ap_uint<128> counter = counterNonceStrm.read();
    for (int i = 0; i < 4; ++i) {
#pragma HLS unroll
        s[i + 12] = counter.range(i * 32 + 31, i * 32);
    }

    blockTypeChacha cph;
    ap_uint<32> c = s[12];
    while (!ePlainStrm.read()) {
#pragma HLS pipeline II = 1
        for (int i = 0; i < 16; ++i) {
#pragma HLS unroll
            if (i == 12)
                x[12] = c;
            else
                x[i] = s[i];
        } // for
        blockTypeChacha plainData = plainStrm.read();
        // 10 loops * 2 rounds/loop = 20 rounds
        for (int i = 0; i < ROUNDS; i += 2) {
            // Odd round
            QR(x[0], x[4], x[8], x[12]);  // column 0
            QR(x[1], x[5], x[9], x[13]);  // column 1
            QR(x[2], x[6], x[10], x[14]); // column 2
            QR(x[3], x[7], x[11], x[15]); // column 3
            // Even round
            QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
            QR(x[1], x[6], x[11], x[12]); // diagonal 2
            QR(x[2], x[7], x[8], x[13]);  // diagonal 3
            QR(x[3], x[4], x[9], x[14]);  // diagonal 4
        }                                 // for

#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
        std::cout << std::endl << "keystream_" << std::dec << (c - 1) << std::endl;
#endif
        for (int i = 0; i < 16; ++i) {
#pragma HLS unroll
            if (i == 12)
                xs[i] = x[i] + c;
            else
                xs[i] = x[i] + s[i];
#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
            std::cout << std::hex << std::setw(2) << std::setfill('0') << xs[i];
#endif
            cph.range(i * 32 + 31, i * 32) = plainData.range(i * 32 + 31, i * 32) ^ xs[i];
        } // for
        c = c + 1;
        cipherStrm.write(cph);
        eCipherStrm.write(false);
    } // while
    eCipherStrm.write(true);
#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
    std::cout << std::endl;
#endif
}

void hchacha20(hls::stream<ap_uint<256> >& keyStrm,
               hls::stream<ap_uint<192> >& nonceStrm,
               hls::stream<ap_uint<256> >& subKeyStrm,
               hls::stream<ap_uint<128> >& counterNonceStrm) {
    ap_uint<256> key = keyStrm.read();
    ap_uint<192> nonce = nonceStrm.read();

    ap_uint<32> x[16];
#pragma HLS array_partition variable = x complete

    x[0] = 0x61707865;
    x[1] = 0x3320646e;
    x[2] = 0x79622d32;
    x[3] = 0x6b206574;
    for (int i = 0; i < 8; ++i) {
#pragma HLS unroll
        x[i + 4] = key.range(32 * (i + 1) - 1, i * 32);
    }
    for (int i = 0; i < 4; ++i) {
#pragma HLS unroll
        x[i + 12] = nonce.range(i * 32 + 31, i * 32);
    }

    for (int i = 0; i < ROUNDS; i += 2) {
        // Odd round
        QR(x[0], x[4], x[8], x[12]);  // column 0
        QR(x[1], x[5], x[9], x[13]);  // column 1
        QR(x[2], x[6], x[10], x[14]); // column 2
        QR(x[3], x[7], x[11], x[15]); // column 3
        // Even round
        QR(x[0], x[5], x[10], x[15]); // diagonal 1 (main diagonal)
        QR(x[1], x[6], x[11], x[12]); // diagonal 2
        QR(x[2], x[7], x[8], x[13]);  // diagonal 3
        QR(x[3], x[4], x[9], x[14]);  // diagonal 4
    }

    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
        key.range(i * 32 + 31, i * 32) = x[i];
        key.range(i * 32 + 159, i * 32 + 128) = x[i + 12];
    }

    ap_uint<128> counterNonce = 0;
    counterNonce.range(31, 0) = 1;
    counterNonce.range(127, 64) = nonce.range(191, 128);

    subKeyStrm.write(key);
    counterNonceStrm.write(counterNonce);
}

} // end of namespace internal

/**
 * @brief chahcha20 is a basic function for stream ciphering
 * when key is "keylayout-chacha", its layout in a 256-bit ap_uint<> likes this,
 *
 *   0 -  7  bit:   'k'
 *   8 - 15  bit:   'e'
 *  16 - 23  bit:   'y'
 *  24 - 31  bit:   'l'
 *    ...
 *  232- 239 bit:   'c'
 *  240- 247 bit:   'h'
 *  248- 255 bit:   'a'
 *
 * state matrix:
 *  s[0]   s[1]   s[2]   s[3]
 *  s[4]   s[5]   s[6]   s[7]
 *  s[8]   s[9]   s[10]  s[11]
 *  s[12]  s[13]  s[14]  s[15]
 *
 *
 * 128bits counterNonceStrm = counter 32 bits + nonce 96 bits
 *  the layout of the data from counteStrm
 *   0-31  bit: counter  s[12]
 *  32-63  bit: nonce1   s[13]
 *  64-95  bit: nonce2   s[14]
 *  96-127 bit: nonce3   s[15]
 *
 * @param keyStrm initail key
 * @param counterNonceStm initial counter and nonce
 * @param plainStrm input  plain text to be encrypted
 * @param ePlainStrm the end flag of plainStrm
 * @param cipherStrm  output encrypted text
 * @param eCipherStrm the end flag of cipherStrm
 */
void chacha20(hls::stream<ap_uint<256> >& keyStrm,
              hls::stream<ap_uint<128> >& counterNonceStrm,
              hls::stream<ap_uint<512> >& plainStrm,
              hls::stream<bool>& ePlainStrm,
              hls::stream<ap_uint<512> >& cipherStrm,
              hls::stream<bool>& eCipherStrm) {
    internal::chacha20Imp(keyStrm, counterNonceStrm, plainStrm, ePlainStrm, cipherStrm, eCipherStrm);
}

/**
 * @brief xchahcha20 is variant of original chacha20 to support longer nonce of 192bits.
 *
 * @param keyStrm initail key
 * @param nonceStm initial nonce
 * @param plainStrm input  plain text to be encrypted
 * @param ePlainStrm the end flag of plainStrm
 * @param cipherStrm  output encrypted text
 * @param eCipherStrm the end flag of cipherStrm
 */

void xchacha20(hls::stream<ap_uint<256> >& keyStrm,
               hls::stream<ap_uint<192> >& nonceStrm,
               hls::stream<ap_uint<512> >& plainStrm,
               hls::stream<bool>& ePlainStrm,
               hls::stream<ap_uint<512> >& cipherStrm,
               hls::stream<bool>& eCipherStrm) {
#pragma HLS dataflow
    hls::stream<ap_uint<256> > subKeyStrm;
#pragma HLS stream variable = subKeyStrm depth = 2
#pragma HLS resource variable = subKeyStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<128> > counterNonceStrm;
#pragma HLS stream variable = counterNonceStrm depth = 2
#pragma HLS resource variable = counterNonceStrm core = FIFO_LUTRAM

    internal::hchacha20(keyStrm, nonceStrm, subKeyStrm, counterNonceStrm);
    internal::chacha20Imp(subKeyStrm, counterNonceStrm, plainStrm, ePlainStrm, cipherStrm, eCipherStrm);
}

} // end of namespace security
} // end of namespace xf
#endif // _XF_SECURITY_CHACHA20_HPP_
