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
 * @file poly1305.hpp
 * @brief header file for poly1305.
 * This file part of Vitis Security Library.
 *
 */

#ifndef _XF_SECURITY_POLY1305_HPP_
#define _XF_SECURITY_POLY1305_HPP_

#include <ap_int.h>
#include <hls_stream.h>

#if !defined(__SYNTHESIS__)
#include <iostream>
#endif

namespace xf {
namespace security {
namespace internal {

/**
 *
 * @brief The implementation of large bit-width multiplication, the result is A * B.
 * The optimization goal of this function to improve timing.
 *
 * @param A The input multiplicand.
 * @param B The input multiplier.
 * @return The output product.
 */

ap_uint<260> multOperator(ap_uint<132> A, ap_uint<128> B) {
#pragma HLS inline off
    //#pragma HLS ALLOCATION instances=mul limit=1 operation
    const int W1 = 27;
    const int W2 = 17;
    const int segA = 5; // W1 / 27;
    const int segB = 8; // W2 / 17;
    ap_uint<W1> arrayA[segA];
    ap_uint<W2> arrayB[segB];
#pragma HLS resource variable = arrayA core = RAM_2P_LUTRAM
#pragma HLS resource variable = arrayB core = RAM_2P_LUTRAM
LOOP_INIT_A:
    for (unsigned int i = 0; i < segA - 1; i++) {
#pragma HLS unroll
        arrayA[i] = A.range(W1 * i + W1 - 1, W1 * i);
    }

    // the upper bits will automatically reset to zeros by default
    arrayA[segA - 1] = A.range(131, segA * W1 - W1);

LOOP_INIT_B:
    for (unsigned int i = 0; i < segB - 1; i++) {
#pragma HLS unroll
        arrayB[i] = B.range(W2 * i + W2 - 1, W2 * i);
    }
    // the upper bits will automatically reset to zeros by default
    arrayB[segB - 1] = B.range(127, segB * W2 - W2);

    ap_uint<260> result = 0;
    ap_uint<260> tmp = 0;
LOOP_MULT:
    for (unsigned int i = 0; i < segA; i++) {
        for (unsigned int j = 0; j < segB; j++) {
            tmp = arrayA[i] * arrayB[j];
            result += (tmp << (i * W1 + j * W2));
        }
    }
    return result; // A * B;
}

/**
 *
 * @brief The implementation of large bit-width Module Operation, the result is A % 2^130-5.
 * The optimization goal of this function to improve timing.
 *
 * @param A The input parameter.
 * @return The output result.
 */

ap_uint<132> resOperator(ap_uint<260> A) {
#pragma HLS inline off
    ap_uint<132> P; // 2^130-5
    P.range(131, 128) = 0x3;
    P.range(127, 64) = 0xffffffffffffffff;
    P.range(63, 0) = 0xfffffffffffffffb;
    ap_uint<260> aTmp = A;
    // mod(a,2^130-5)
    ap_uint<130> aHigh = aTmp.range(259, 130);
    ap_uint<136> aHigh2 = 0;
    aHigh2.range(131, 2) = aTmp.range(259, 130);
    aHigh2 += aHigh; // aHigh*5
    aTmp = aTmp.range(129, 0) + aHigh2;
    if (aTmp > P) {
        ap_uint<130> aHigh = aTmp.range(259, 130);
        ap_uint<136> aHigh2 = 0;
        aHigh2.range(131, 2) = aTmp.range(259, 130);
        aHigh2 += aHigh; // aHigh*5
        aTmp = aTmp.range(129, 0) + aHigh2;
    }
    if (aTmp > P) {
        ap_uint<130> aHigh = aTmp.range(259, 130);
        ap_uint<136> aHigh2 = 0;
        aHigh2.range(131, 2) = aTmp.range(259, 130);
        aHigh2 += aHigh; // aHigh*5
        aTmp = aTmp.range(129, 0) + aHigh2;
    }

    return aTmp.range(131, 0);
    // return a % P;
}

/**
 *
 * @brief The implementation of poly1305
 *
 * @param accValue The accumulator's value, initial value is 0, followed by the last output.
 * @param keyValue Corresponding message key
 * @param payload For a massage, input block stream text, 128 bits per block, less than 128 bits, high padding 0
 * @param lenByte Length of a block of payload in byte.
 * @param tagValue return a 16-byte tag to to authenticate the message.
 */

void poly1305Imp(
    ap_uint<132>& accValue, ap_uint<256> keyValue, ap_uint<128> payload, ap_uint<32> lenByte, ap_uint<128>& tagValue) {
    //#pragma HLS pipeline ii = 1
    ap_uint<128> pValue;
    pValue.range(127, 64) = 0x0ffffffc0ffffffc;
    pValue.range(63, 0) = 0x0ffffffc0fffffff;
    ap_uint<128> rValue, sValue;
    rValue = keyValue.range(127, 0) & pValue;
    sValue = keyValue.range(255, 128);

#if !defined(__SYNTHESIS__)
    std::cout << "Acc=" << accValue << std::endl;
#endif
    ap_uint<132> payloadtmp = 0;
    payloadtmp.range(127, 0) = payload;
    payloadtmp.range(lenByte * 8 + 3, lenByte * 8) = 0x1;

    ap_uint<132> tmp1 = accValue + payloadtmp;
    // ap_uint<264> tmp2 = tmp1 * rValue;
    ap_uint<260> tmp2 = multOperator(tmp1, rValue);
    // Acc = tmp2 % P;
    accValue = resOperator(tmp2);

#if !defined(__SYNTHESIS__)
    std::cout << std::hex << "  len=" << lenByte << ", payload=" << payload << ", r=" << rValue << ", tmp1=" << tmp1
              << ", tmp2=" << tmp2 << ", Acc=" << accValue << std::endl;
#endif
    tagValue = accValue + sValue;
}

} // end of namespace internal

/**
 *
 * @brief The poly1305 takes a 32-byte one-time key and a message and produces a 16-byte tag. This tag is used to
 * authenticate the message.
 *
 * @param keyStrm Corresponding message key
 * @param payloadStrm For a massage, input block stream text, 128 bits per block, less than 128 bits, high padding 0
 * @param lenPldStrm Length of a massage in byte.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param tagStrm Return a 16-byte tag to to authenticate the message.
 */

void poly1305(
    // stream in
    hls::stream<ap_uint<256> >& keyStrm,
    hls::stream<ap_uint<128> >& payloadStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<bool>& endLenStrm,
    // stream out
    hls::stream<ap_uint<128> >& tagStrm) {
loop_Len:
    while (endLenStrm.read()) {
#pragma HLS loop_tripcount max = 10 min = 10
        ap_uint<256> keyValue = keyStrm.read();
        ap_int<64> len = lenPldStrm.read();
        ap_uint<32> lenByte = 16;
        ap_uint<132> Acc = 0;
        ap_uint<128> tagValue;
    loop_Block:
        while (len > 0) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline
            if (len < lenByte) lenByte = len;
            ap_uint<128> payload = payloadStrm.read();
            internal::poly1305Imp(Acc, keyValue, payload, lenByte, tagValue);
            len -= lenByte;
        }
        tagStrm.write(tagValue);
    }
}

/**
 *
 * @brief The poly1305MultiChan takes N 32-byte one-time keys and N messages and produces N 16-byte tags. These tags are
 * used to authenticate the corresponding messages.
 *
 * @tparam N Channel number
 * @param keyStrm Corresponding message key
 * @param payloadStrm For a massage, input block stream text, 128 bits per block, less than 128 bits, high padding 0
 * @param lenPldStrm Length of a massage in byte.
 * @param endLenStrm Flag to signal the end of the length streams.
 * @param tagStrm Return a 16-byte tag to to authenticate the message.
 */
template <int N>
void poly1305MultiChan(
    // stream in
    hls::stream<ap_uint<256> >& keyStrm,
    hls::stream<ap_uint<128> >& payloadStrm,
    hls::stream<ap_uint<64> >& lenPldStrm,
    hls::stream<bool>& endLenStrm,
    // stream out
    hls::stream<ap_uint<128> >& tagStrm) {
loop_Len:
    while (endLenStrm.read()) {
#pragma HLS loop_tripcount max = 10 min = 10
        ap_uint<256> keyValue[N];
        ap_int<64> len[N];
        ap_uint<32> lenByte[N];
        ap_uint<128> tagValue[N];
        ap_uint<128> payload[N];
        ap_uint<132> Acc[N];
#pragma HLS resource variable = keyValue core = RAM_2P_LUTRAM
#pragma HLS resource variable = len core = RAM_2P_LUTRAM
#pragma HLS resource variable = lenByte core = RAM_2P_LUTRAM
#pragma HLS resource variable = tagValue core = RAM_2P_LUTRAM
#pragma HLS resource variable = payload core = RAM_2P_LUTRAM
#pragma HLS resource variable = Acc core = RAM_2P_LUTRAM
        bool flag = false;
        for (int i = 0; i < N; i++) {
#pragma HLS pipeline
            keyValue[i] = keyStrm.read();
            len[i] = lenPldStrm.read();
            lenByte[i] = 16;
            Acc[i] = 0;
            if (len[i] > 0) {
                flag = true;
            }
        }
        while (flag) {
#pragma HLS loop_tripcount max = 10 min = 10
            flag = false;
        loop_X:
            for (int i = 0; i < N; i++) {
#pragma HLS pipeline
                if (len[i] < 16) lenByte[i] = len[i];
                payload[i] = payloadStrm.read();
                if (len[i] > 0) internal::poly1305Imp(Acc[i], keyValue[i], payload[i], lenByte[i], tagValue[i]);
                len[i] -= 16;
                if (len[i] > 0) {
                    flag = true;
                }
            }
        }
        for (int i = 0; i < N; i++) {
#pragma HLS pipeline
            tagStrm.write(tagValue[i]);
        }
    }
}

} // end of namespace security
} // end of namespace xf
#endif // _XF_SECURITY_POLY1305_HPP_
