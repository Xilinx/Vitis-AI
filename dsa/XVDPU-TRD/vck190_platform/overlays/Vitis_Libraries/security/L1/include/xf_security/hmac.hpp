
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
 * @file hmac.hpp
 * @brief header file for HMAC.
 * This file part of Vitis Security Library.
 * TODO
 * @detail .
 */

#ifndef _XF_SECURITY_HMAC_HPP_
#define _XF_SECURITY_HMAC_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include <xf_security/types.hpp>

#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
#include <iostream>
#endif
namespace xf {
namespace security {

namespace internal {
// typedef ap_uint<64> u64;

template <int lW, int keyLen>
void expandStrm(hls::stream<bool>& eInStrm, hls::stream<bool>& eOutStrm, hls::stream<ap_uint<lW> >& lenStrm) {
    while (!eInStrm.read()) {
        eOutStrm.write(false);
        lenStrm.write(ap_uint<lW>(keyLen));
    }
    eOutStrm.write(true);
}

template <int dataW, int hshW, int blockSize>
void genPad(hls::stream<ap_uint<hshW> >& keyHashStrm,
            hls::stream<bool>& ekeyHashStrm,
            hls::stream<ap_uint<blockSize * 8> >& kipadStrm,
            hls::stream<ap_uint<blockSize * 8> >& kopadStrm,
            hls::stream<bool>& eKipadStrm) {
    while (!ekeyHashStrm.read()) {
#pragma HLS pipeline II = 1
        ap_uint<blockSize* 8> kipad = 0;
        ap_uint<blockSize* 8> kopad = 0;
        ap_uint<blockSize* 8> k1 = 0;

        ap_uint<hshW> keyHash = keyHashStrm.read();
        for (int i = 0; i < hshW / dataW; i++) {
#pragma HLS unroll
            k1.range(blockSize * 8 - i * dataW - 1, blockSize * 8 - (i + 1) * dataW) =
                keyHash.range(i * dataW + dataW - 1, i * dataW);
        }
        for (int i = 0; i < blockSize; i++) {
#pragma HLS unroll
            kipad.range(i * 8 + 7, i * 8) = k1.range(i * 8 + 7, i * 8) ^ 0x36;
            kopad.range(i * 8 + 7, i * 8) = k1.range(i * 8 + 7, i * 8) ^ 0x5c;
        }
        kipadStrm.write(kipad);
        kopadStrm.write(kopad);
        eKipadStrm.write(false);
    }
    eKipadStrm.write(true);
}

template <int dataW, int lW, int hshW, int keyLen, int blockSize, template <int iW, int ilW, int oW> class F>
void kpadHash(hls::stream<ap_uint<dataW> >& keyStrm,
              hls::stream<bool>& eStrm,
              hls::stream<ap_uint<blockSize * 8> >& kipadStrm,
              hls::stream<ap_uint<blockSize * 8> >& kopadStrm,
              hls::stream<bool>& eKipadStrm) {
#pragma HLS dataflow

    hls::stream<bool> eKeyStrm;
#pragma HLS stream variable = eKeyStrm depth = 4
#pragma HLS resource variable = eKeyStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<lW> > keyLenStrm;
#pragma HLS stream variable = keyLenStrm depth = 4
#pragma HLS resource variable = keyLenStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<hshW> > keyHashStrm;
#pragma HLS stream variable = keyHashStrm depth = 4
#pragma HLS resource variable = keyHashStrm core = FIFO_LUTRAM
    hls::stream<bool> ekeyHashStrm;
#pragma HLS stream variable = ekeyHashStrm depth = 4
#pragma HLS resource variable = ekeyHashStrm core = FIFO_LUTRAM

    expandStrm<lW, keyLen>(eStrm, eKeyStrm, keyLenStrm);

    F<dataW, lW, hshW>::hash(keyStrm, keyLenStrm, eKeyStrm, keyHashStrm, ekeyHashStrm);

    genPad<dataW, hshW, blockSize>(keyHashStrm, ekeyHashStrm, kipadStrm, kopadStrm, eKipadStrm);
}

template <int dataW, int lW, int hshW, int keyLen, int blockSize, template <int iW, int ilW, int oW> class F>
void kpad(hls::stream<ap_uint<dataW> >& keyStrm,
          hls::stream<bool>& eStrm,
          hls::stream<ap_uint<blockSize * 8> >& kipadStrm,
          hls::stream<ap_uint<blockSize * 8> >& kopadStrm,
          hls::stream<bool>& eKipadStrm) {
    if (keyLen > blockSize) {
        kpadHash<dataW, lW, hshW, keyLen, blockSize, F>(keyStrm, eStrm, kipadStrm, kopadStrm, eKipadStrm);
    } else {
        while (!eStrm.read()) {
            ap_uint<blockSize* 8> k1 = 0;
            for (int i = 0; i < ((keyLen * 8 + dataW - 1) / dataW); i++) {
#pragma HLS pipeline II = 1
                ap_uint<dataW> tmp = keyStrm.read();
                k1 <<= dataW;
                k1.range(dataW - 1 + ((blockSize - keyLen) * 8), ((blockSize - keyLen) * 8)) = tmp;
                // k1.range(blockSize * 8 - 1 - i * dataW, blockSize * 8 - (i + 1) * dataW) = tmp;
            }
            ap_uint<blockSize* 8> kipad = 0;
            ap_uint<blockSize* 8> kopad = 0;
            for (int i = 0; i < blockSize; i++) {
#pragma HLS unroll
                kipad(i * 8 + 7, i * 8) = 0x36 ^ k1.range(i * 8 + 7, i * 8);
                kopad(i * 8 + 7, i * 8) = 0x5c ^ k1.range(i * 8 + 7, i * 8);
            }
            kipadStrm.write(kipad);
            kopadStrm.write(kopad);
            eKipadStrm.write(false);
        }
        eKipadStrm.write(true);
    }
}

template <int dataW, int lW, int hshW, int blockSize>
void mergeKipad(hls::stream<ap_uint<blockSize * 8> >& kipadStrm,
                hls::stream<ap_uint<blockSize * 8> >& kopadInStrm,
                hls::stream<ap_uint<dataW> >& msgStrm,
                hls::stream<ap_uint<lW> >& msgLenStrm,
                hls::stream<bool>& eLenStrm2,
                hls::stream<ap_uint<dataW> >& mergeKipadStrm,
                hls::stream<ap_uint<lW> >& mergeKipadLenStrm,
                hls::stream<bool>& eMergeKipadLenStrm,
                hls::stream<ap_uint<blockSize * 8> >& kopadOutStrm) {
    while (!eLenStrm2.read()) {
        eMergeKipadLenStrm.write(false);

        ap_uint<lW> ml = msgLenStrm.read();
        ap_uint<lW> mergeKipadLen = ml + blockSize;

        mergeKipadLenStrm.write(mergeKipadLen);

        ap_uint<blockSize* 8> kipad = kipadStrm.read();

        for (int i = 0; i < ((blockSize * 8 + dataW - 1) / dataW); i++) {
#pragma HLS pipeline II = 1
            // mergeKipadStrm.write(kipad.range(blockSize * 8 - 1 - i * dataW, blockSize * 8 - (i + 1) * dataW));
            mergeKipadStrm.write(kipad.range(blockSize * 8 - 1, blockSize * 8 - dataW));
            kipad <<= dataW;
        }

        kopadOutStrm.write(kopadInStrm.read());

        for (int i = 0; i < ((ml * 8 + dataW - 1) / dataW); i++) {
#pragma HLS pipeline II = 1
            mergeKipadStrm.write(msgStrm.read());
        }
    }
    eMergeKipadLenStrm.write(true);
}

template <int dataW, int lW, int hshW, int keyLen, int blockSize, template <int iW, int ilW, int oW> class F>
void msgHash(hls::stream<ap_uint<blockSize * 8> >& kipadStrm,
             hls::stream<ap_uint<blockSize * 8> >& kopadInStrm,
             hls::stream<ap_uint<dataW> >& msgStrm,
             hls::stream<ap_uint<lW> >& msgLenStrm,
             hls::stream<bool>& eLenStrm,
             hls::stream<ap_uint<blockSize * 8> >& kopadOutStrm,
             hls::stream<ap_uint<hshW> >& msgHashStrm,
             hls::stream<bool>& eMsgHashStrm) {
#pragma HLS dataflow

    hls::stream<ap_uint<dataW> > mergeKipadStrm;
#pragma HLS stream variable = mergeKipadStrm depth = 128
#pragma HLS resource variable = mergeKipadStrm core = FIFO_BRAM
    hls::stream<ap_uint<lW> > mergeKipadLenStrm;
#pragma HLS stream variable = mergeKipadLenStrm depth = 4
#pragma HLS resource variable = mergeKipadLenStrm core = FIFO_LUTRAM
    hls::stream<bool> eMergeKipadLenStrm;
#pragma HLS stream variable = eMergeKipadLenStrm depth = 4
#pragma HLS resource variable = eMergeKipadLenStrm core = FIFO_LUTRAM

    mergeKipad<dataW, lW, hshW, blockSize>(kipadStrm, kopadInStrm, msgStrm, msgLenStrm, eLenStrm, mergeKipadStrm,
                                           mergeKipadLenStrm, eMergeKipadLenStrm, kopadOutStrm);

    F<dataW, lW, hshW>::hash(mergeKipadStrm, mergeKipadLenStrm, eMergeKipadLenStrm, msgHashStrm, eMsgHashStrm);
}

template <int dataW, int lW, int hshW, int keyLen, int blockSize>
void mergeKopad(hls::stream<ap_uint<blockSize * 8> >& kopadStrm,
                hls::stream<ap_uint<hshW> >& msgHashStrm,
                hls::stream<bool>& eMsgHashStrm,
                hls::stream<ap_uint<dataW> >& mergeKopadStrm,
                hls::stream<ap_uint<lW> >& mergeKopadLenStrm,
                hls::stream<bool>& eMergeKopadLenStrm) {
    while (!eMsgHashStrm.read()) {
        eMergeKopadLenStrm.write(false);
        mergeKopadLenStrm.write(ap_uint<lW>(blockSize + hshW / 8));

        ap_uint<blockSize* 8> kopad = kopadStrm.read();
        ap_uint<hshW> msgHash = msgHashStrm.read();

        for (int i = 0; i < ((blockSize * 8 + dataW - 1) / dataW); i++) {
#pragma HLS pipeline II = 1
            mergeKopadStrm.write(kopad.range(blockSize * 8 - 1, blockSize * 8 - dataW));
            kopad <<= dataW;
        }

        for (int i = 0; i < ((hshW + dataW - 1) / dataW); i++) {
#pragma HLS pipeline II = 1
            mergeKopadStrm.write(msgHash.range(dataW - 1, 0));
            msgHash >>= dataW;
        }
    }
    eMergeKopadLenStrm.write(true);
}

template <int dataW, int lW, int hshW, int keyLen, int blockSize, template <int iW, int ilW, int oW> class F>
void resHash(hls::stream<ap_uint<blockSize * 8> >& kopadStrm,
             hls::stream<ap_uint<hshW> >& msgHashStrm,
             hls::stream<bool>& eMsgHashStrm,
             hls::stream<ap_uint<hshW> >& hshStrm,
             hls::stream<bool>& eHshStrm) {
#pragma HLS dataflow

    hls::stream<ap_uint<dataW> > mergeKopadStrm;
#pragma HLS stream variable = mergeKopadStrm depth = 4
#pragma HLS resource variable = mergeKopadStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<lW> > mergeKopadLenStrm;
#pragma HLS stream variable = mergeKopadLenStrm depth = 4
#pragma HLS resource variable = mergeKopadLenStrm core = FIFO_LUTRAM
    hls::stream<bool> eMergeKopadLenStrm;
#pragma HLS stream variable = eMergeKopadLenStrm depth = 4
#pragma HLS resource variable = eMergeKopadLenStrm core = FIFO_LUTRAM

    mergeKopad<dataW, lW, hshW, keyLen, blockSize>(kopadStrm, msgHashStrm, eMsgHashStrm, mergeKopadStrm,
                                                   mergeKopadLenStrm, eMergeKopadLenStrm);

    F<dataW, lW, hshW>::hash(mergeKopadStrm, mergeKopadLenStrm, eMergeKopadLenStrm, hshStrm, eHshStrm);
}

template <int dataW, int lW, int hshW, int keyLen, int blockSize, template <int iW, int ilW, int oW> class F>
void hmacDataflow(hls::stream<ap_uint<dataW> >& keyStrm,
                  hls::stream<ap_uint<dataW> >& msgStrm,
                  hls::stream<ap_uint<lW> >& msgLenStrm,
                  hls::stream<bool>& eLenStrm,
                  hls::stream<ap_uint<hshW> >& hshStrm,
                  hls::stream<bool>& eHshStrm) {
#pragma HLS dataflow
    hls::stream<bool> eKipadStrm;
#pragma HLS stream variable = eKipadStrm depth = 4
#pragma HLS resource variable = eKipadStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<blockSize * 8> > kipadStrm;
#pragma HLS stream variable = kipadStrm depth = 4
#pragma HLS resource variable = kipadStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<blockSize * 8> > kopadStrm;
#pragma HLS stream variable = kopadStrm depth = 4
#pragma HLS resource variable = kopadStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<blockSize * 8> > kopad2Strm;
#pragma HLS stream variable = kopad2Strm depth = 4
#pragma HLS resource variable = kopad2Strm core = FIFO_LUTRAM

    hls::stream<ap_uint<hshW> > msgHashStrm;
#pragma HLS stream variable = msgHashStrm depth = 4
#pragma HLS resource variable = msgHashStrm core = FIFO_LUTRAM
    hls::stream<bool> eMsgHashStrm;
#pragma HLS stream variable = eMsgHashStrm depth = 4
#pragma HLS resource variable = eMsgHashStrm core = FIFO_LUTRAM

    kpad<dataW, lW, hshW, keyLen, blockSize, F>(keyStrm, eLenStrm, kipadStrm, kopadStrm, eKipadStrm);

    msgHash<dataW, lW, hshW, keyLen, blockSize, F>(kipadStrm, kopadStrm, msgStrm, msgLenStrm, eKipadStrm, kopad2Strm,
                                                   msgHashStrm, eMsgHashStrm);

    resHash<dataW, lW, hshW, keyLen, blockSize, F>(kopad2Strm, msgHashStrm, eMsgHashStrm, hshStrm, eHshStrm);
}

} // end of namespace internal

/**
 * @brief Compute HMAC value according to specified hash function and input data.
 *
 *  keyW, keyStrm, keyLenStrm, msgW, msgStrm, and msgLenStrm would be used as
 *  parameters or input for the hash function, so they need to align with the API
 *  of the hash function.
 *
 *  Hash function is wrapped to a template struct which must have a static function named `hash`.
 *
 *  Take md5 for example::
 *
 *  template <int msgW, int lW, int hshW>
 *  struct md5_wrapper {
 *      static void hash(hls::stream<ap_uint<msgW> >& msgStrm,
 *                       hls::stream<lW>& lenStrm,
 *                       hls::stream<bool>& eLenStrm,
 *                       hls::stream<ap_uint<hshW> >& hshStrm,
 *                       hls::stream<bool>& eHshStrm) {
 *          xf::security::md5(msgStrm, lenStrm, eLenStrm, hshStrm, eHshStrm);
 *      }
 *  };
 *
 *  then use hmac like this,
 *
 *   xf::security::hmac<32, 32, 64, 128, 64, md5_wrapper>(...);
 *
 * @tparam dataW the width of input stream keyStrm and msgStrm.
 * @tparam lW the with of input msgLenstrm.
 * @tparam blockSize  the block size (in bytes) of the underlying hash function (e.g. 64 bytes for md5 and SHA-1).
 * @tparam hshW the width of output stream hshStrm.
 * @tparam keyLen lenght of key (in bytes)
 * @tparam F a wrapper of hash function which must have a static fucntion named `hash`.
 *
 * @param keyStrm  input key stream.
 * @param msgStrm  input meassge stream.
 * @param msgLenStrm  the length stream of input message stream.
 * @param eLenStrm  the end flag of length stream.
 * @param hshStrm output stream.
 * @param eHshStrm end flag of output stream hshStrm.
 *
 */
template <int dataW, int lW, int hshW, int keyLen, int blockSize, template <int iW, int ilW, int oW> class F>
void hmac(hls::stream<ap_uint<dataW> >& keyStrm,
          hls::stream<ap_uint<dataW> >& msgStrm,
          hls::stream<ap_uint<lW> >& msgLenStrm,
          hls::stream<bool>& eLenStrm,
          hls::stream<ap_uint<hshW> >& hshStrm,
          hls::stream<bool>& eHshStrm) {
    internal::hmacDataflow<dataW, lW, hshW, keyLen, blockSize, F>(keyStrm, msgStrm, msgLenStrm, eLenStrm, hshStrm,
                                                                  eHshStrm);
}
} // end of namespace security
} // end of namespace xf

#endif // _XF_SECURITY_HMAC_HPP_
