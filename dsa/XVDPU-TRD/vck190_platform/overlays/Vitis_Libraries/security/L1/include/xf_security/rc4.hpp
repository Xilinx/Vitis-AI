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
 * @file rc4.h
 * @brief header file for Rivest Cipher 4(also known as ARC4 or ARCFOUR).
 * This file part of Vitis Security Library.
 *
 */

#ifndef _XF_SECURITY_RC4_HPP_
#define _XF_SECURITY_RC4_HPP_

#include <ap_int.h>
#include <hls_stream.h>

#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
#include <iostream>
#endif

namespace xf {
namespace security {
namespace internal {

/**
 * @brief Rc_4 is the basic function for stream ciphering
 *  keyLength defined as the number of bytes in the key
 *  and can be in the range 1<= keylength <=256,
 *  typically between 5 and 16, corresponding to a key length of 40-128 bits.
 *
 * @param keyStrm initail key
 * @param eKeyStrm end flag of keyStrm
 * @param plaintStrm input  plain text to be encrypted
 * @param ePlaintStrm the end flag of plaintStrm
 * @param cipherStrm  output encrypted text
 * @param eCipherStrm the end flag of cipherStrm
 *
 */
static void rc4Imp(hls::stream<ap_uint<8> >& keyStrm,
                   hls::stream<bool>& eKeyStrm,
                   hls::stream<ap_uint<8> >& plainStream,
                   hls::stream<bool>& ePlainStream,
                   hls::stream<ap_uint<8> >& cipherStream,
                   hls::stream<bool>& eCipherStream) {
    ap_uint<8> S[256];
#pragma HLS ARRAY_PARTITION variable = S complete dim = 1
    ap_uint<8> keys[256];
#pragma HLS resource variable = keys core = RAM_2P_BRAM

#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
    std::cout << "key:" << std::endl;
#endif
    // initial S and read key, 1<=keyLength<=256
    int keyLength = 0;
LOOP_READ_KEY:
    while (!eKeyStrm.read()) {
#pragma HLS pipeline II = 1
        keys[keyLength] = keyStrm.read();
#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
        std::cout << std::hex << keys[keyLength];
#endif
        ++keyLength;
    }
LOOP_SET_S:
    for (int i = 0; i < 256; ++i) {
#pragma HLS unroll
        S[i] = i;
    }

#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
    std::cout << std::endl << "keyLength=" << std::dec << keyLength << std::endl;
    std::cout << std::endl << "---" << std::endl;
    std::cout << "keystream:" << std::endl;
#endif

    // generate keystream and XOR input data
    ap_uint<8> j8 = 0; //  8-bit width counter to avoid %256
LOOP_UPDATE_S:
    for (int i = 0, ii = 0; i < 256; ++i) {
#pragma HLS pipeline II = 1
        // j = (j + S[i] + keys[i % keyLength]) % 256;
        ap_uint<8> k = keys[ii];
        ii = ((ii + 1) >= keyLength) ? 0 : (ii + 1);
        ap_uint<8> tmp1 = S[i];
        j8 = j8 + tmp1 + k;
        // j = (j + S[i] + k) % 256;
        // S[i] <---> S[j]
        ap_uint<8> tmp2 = S[j8];
        S[i] = tmp2;
        S[j8] = tmp1;
    }

    ap_uint<8> i = 0;
    ap_uint<8> j = 0;
    bool last = ePlainStream.read();
LOOP_EMIT:
    while (!last) {
#pragma HLS pipeline II = 1
        // next ~10 lines equals to
        // i=(i+1)%256; keystream[k++] = S[(S[i] + S[j]) % 256] and S[i]<-->S[j]
        i = i + 1;    //(i + 1) % 256;
        j = j + S[i]; //(j + S[i]) % 256;
        // S[i] <---> S[j]
        ap_uint<8> tmp1 = S[i];
        ap_uint<8> tmp2 = S[j];
        S[i] = tmp2;
        S[j] = tmp1;
        // int p = (S[i] + S[j]) % 256;
        ap_uint<8> p = tmp1 + tmp2;
        ap_uint<8> k = S[p];
        ap_uint<8> txt = plainStream.read();
        ap_uint<8> cptxt = k ^ txt;
        cipherStream.write(cptxt);
        eCipherStream.write(false);
        last = ePlainStream.read();
#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
        std::cout << std::hex << k; // cptxt ;
#endif
    }
    eCipherStream.write(true);
#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
    std::cout << std::endl;
#endif
}

} // end of namespace internal

/**
 * @brief Rc_4 is the basic function for stream ciphering
 *  keyLength defined as the number of bytes in the key
 *  and can be in the range 1<= keylength <=256,
 *  typically between 5 and 16, corresponding to a key length of 40-128 bits.
 *
 * @param keyStrm initail key
 * @param eKeyStrm end flag of keyStrm
 * @param plaintStrm input  plain text to be encrypted
 * @param ePlaintStrm the end flag of plaintStrm
 * @param cipherStrm  output encrypted text
 * @param eCipherStrm the end flag of cipherStrm
 */
static void rc4(hls::stream<ap_uint<8> >& keyStrm,
                hls::stream<bool>& eKeyStrm,
                hls::stream<ap_uint<8> >& plainStream,
                hls::stream<bool>& ePlainStream,
                hls::stream<ap_uint<8> >& cipherStream,
                hls::stream<bool>& eCipherStream) {
    internal::rc4Imp(keyStrm, eKeyStrm, plainStream, ePlainStream, cipherStream, eCipherStream);
}

} // end of namespace security
} // end of namespace xf
#endif // _XF_SECURITY_RC4_HPP_
