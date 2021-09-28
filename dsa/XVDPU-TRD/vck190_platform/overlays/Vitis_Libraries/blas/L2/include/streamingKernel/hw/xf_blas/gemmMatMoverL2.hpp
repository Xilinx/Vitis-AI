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
#ifndef XF_BLAS_GEMMMATMOVERL2_HPP
#define XF_BLAS_GEMMMATMOVERL2_HPP

#include "ap_int.h"
#include "hls_stream.h"

namespace xf {
namespace blas {

template <unsigned int t_ParEntries, unsigned int t_RowParWords, unsigned int t_ColParWords, unsigned int t_MemBits>
void gemmLoadBlock(ap_uint<t_MemBits>* p_ptr,
                   const unsigned int p_colParWords,
                   hls::stream<ap_uint<t_MemBits> >& p_outStr) {
#pragma HLS INLINE
    static const unsigned int t_Rows = t_ParEntries * t_RowParWords;
    for (unsigned int i = 0; i < t_Rows; ++i) {
#pragma HLS PIPELINE II = t_ColParWords
        for (unsigned int j = 0; j < t_ColParWords; ++j) {
            ap_uint<t_MemBits> l_val = p_ptr[i * p_colParWords + j];
            p_outStr.write(l_val);
        }
    }
}

template <unsigned int t_ParEntries,
          unsigned int t_MparWords,
          unsigned int t_KparWords,
          unsigned int t_NparWords,
          unsigned int t_MemBits>
void gemmLoadDat(ap_uint<t_MemBits>* p_a,
                 ap_uint<t_MemBits>* p_b,
                 ap_uint<t_MemBits>* p_x,
                 unsigned int p_m,
                 unsigned int p_k,
                 unsigned int p_n,
                 hls::stream<ap_uint<t_MemBits> >& p_aStr,
                 hls::stream<ap_uint<t_MemBits> >& p_bStr,
                 hls::stream<ap_uint<t_MemBits> >& p_xStr) {
    static const unsigned int t_AblockRows = t_ParEntries * t_MparWords;
    static const unsigned int t_AblockCols = t_ParEntries * t_KparWords;
    static const unsigned int t_BblockRows = t_ParEntries * t_KparWords;
    static const unsigned int t_BblockCols = t_ParEntries * t_NparWords;

#ifndef __SYNTHESIS
    assert(p_m % (t_ParEntries * t_MparWords) == 0);
    assert(p_k % (t_ParEntries * t_KparWords) == 0);
    assert(p_n % (t_ParEntries * t_NparWords) == 0);
#endif
    unsigned int l_mBlocks = p_m / (t_ParEntries * t_MparWords);
    unsigned int l_kBlocks = p_k / (t_ParEntries * t_KparWords);
    unsigned int l_nBlocks = p_n / (t_ParEntries * t_NparWords);

    unsigned int l_kParWords = p_k / t_ParEntries;
    unsigned int l_nParWords = p_n / t_ParEntries;

    unsigned int l_aBlockOffset = l_kParWords * t_AblockRows;
    unsigned int l_bBlockOffset = l_nParWords * t_BblockRows;
    unsigned int l_xBlockOffset = l_nParWords * t_AblockRows;

    ap_uint<t_MemBits>* l_aPtr = p_a;
    ap_uint<t_MemBits>* l_bPtr = p_b;
    ap_uint<t_MemBits>* l_xPtr = p_x;

    for (unsigned int ar = 0; ar < l_mBlocks; ++ar) {
        for (unsigned int bc = 0; bc < l_nBlocks; ++bc) {
            // load A, B
            for (unsigned int br = 0; br < l_kBlocks; ++br) {
                gemmLoadBlock<t_ParEntries, t_KparWords, t_NparWords, t_MemBits>(l_bPtr + br * l_bBlockOffset,
                                                                                 l_nParWords, p_bStr);
                gemmLoadBlock<t_ParEntries, t_MparWords, t_KparWords, t_MemBits>(l_aPtr + br * t_KparWords, l_kParWords,
                                                                                 p_aStr);
            }

            // loadX
            gemmLoadBlock<t_ParEntries, t_MparWords, t_NparWords, t_MemBits>(l_xPtr + bc * t_NparWords, l_nParWords,
                                                                             p_xStr);
            l_bPtr += t_NparWords;
        }
        l_bPtr = p_b;
        l_aPtr += l_aBlockOffset;
        l_xPtr += l_xBlockOffset;
    }
}

template <unsigned int t_ParEntries, unsigned int t_RowParWords, unsigned int t_ColParWords, unsigned int t_MemBits>
void gemmStoreBlock(hls::stream<ap_uint<t_MemBits> >& p_inStr,
                    const unsigned int p_colParWords,
                    ap_uint<t_MemBits>* p_ptr) {
#pragma HLS INLINE
    static const unsigned int t_Rows = t_ParEntries * t_RowParWords;
    for (unsigned int i = 0; i < t_Rows; ++i) {
#pragma HLS PIPELINE II = t_ColParWords
        for (unsigned int j = 0; j < t_ColParWords; ++j) {
            ap_uint<t_MemBits> l_val = p_inStr.read();
            p_ptr[i * p_colParWords + j] = l_val;
        }
    }
}

template <unsigned int t_ParEntries,
          unsigned int t_MparWords,
          unsigned int t_KparWords,
          unsigned int t_NparWords,
          unsigned int t_MemBits>
void gemmStoreDatABX(hls::stream<ap_uint<t_MemBits> >& p_aStr,
                     hls::stream<ap_uint<t_MemBits> >& p_bStr,
                     hls::stream<ap_uint<t_MemBits> >& p_xStr,
                     unsigned int p_m,
                     unsigned int p_k,
                     unsigned int p_n,
                     ap_uint<t_MemBits>* p_a,
                     ap_uint<t_MemBits>* p_b,
                     ap_uint<t_MemBits>* p_x) {
    static const unsigned int t_AblockRows = t_ParEntries * t_MparWords;
    static const unsigned int t_AblockCols = t_ParEntries * t_KparWords;
    static const unsigned int t_BblockRows = t_ParEntries * t_KparWords;
    static const unsigned int t_BblockCols = t_ParEntries * t_NparWords;

#ifndef __SYNTHESIS
    assert(p_m % (t_ParEntries * t_MparWords) == 0);
    assert(p_k % (t_ParEntries * t_KparWords) == 0);
    assert(p_n % (t_ParEntries * t_NparWords) == 0);
#endif
    unsigned int l_mBlocks = p_m / (t_ParEntries * t_MparWords);
    unsigned int l_kBlocks = p_k / (t_ParEntries * t_KparWords);
    unsigned int l_nBlocks = p_n / (t_ParEntries * t_NparWords);

    unsigned int l_kParWords = p_k / t_ParEntries;
    unsigned int l_nParWords = p_n / t_ParEntries;
    unsigned int l_mParWords = p_m / t_ParEntries;

    unsigned int l_aBlockOffset = l_kParWords * t_AblockRows;
    unsigned int l_bBlockOffset = l_nParWords * t_BblockRows;
    unsigned int l_xBlockOffset = l_nParWords * t_AblockRows;

    ap_uint<t_MemBits>* l_aPtr = p_a;
    ap_uint<t_MemBits>* l_bPtr = p_b;
    ap_uint<t_MemBits>* l_xPtr = p_x;

    for (unsigned int ar = 0; ar < l_mBlocks; ++ar) {
        for (unsigned int bc = 0; bc < l_nBlocks; ++bc) {
            // store A, B
            for (unsigned int br = 0; br < l_kBlocks; ++br) {
                gemmStoreBlock<t_ParEntries, t_MparWords, t_KparWords, t_MemBits>(
                    p_aStr, l_kParWords, l_aPtr + bc * p_m * p_k / t_ParEntries + br * t_KparWords);
                gemmStoreBlock<t_ParEntries, t_KparWords, t_NparWords, t_MemBits>(p_bStr, l_nParWords,
                                                                                  l_bPtr + br * l_bBlockOffset);
            }

            // storeX
            gemmStoreBlock<t_ParEntries, t_MparWords, t_NparWords, t_MemBits>(p_xStr, l_nParWords,
                                                                              l_xPtr + bc * t_NparWords);
            l_bPtr += t_NparWords;
        }
        l_bPtr = p_b + p_k * p_n / t_ParEntries;
        l_aPtr += l_aBlockOffset;
        l_xPtr += l_xBlockOffset;
    }
}
}
}
#endif
