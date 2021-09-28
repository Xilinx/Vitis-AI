/**********
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
 * **********/
/**
 *  @brief GEMM header
 *
 */

#ifndef XF_BLAS_GEMM_KERNEL_HPP
#define XF_BLAS_GEMM_KERNEL_HPP

#include <cassert>
#include <iostream>
#include "xf_blas.hpp"
#include "types.hpp"
#include "transpose.hpp"
#include "matrixBuffer.hpp"
#include "gemm.hpp"

namespace xf {

namespace blas {
// implement C = A*B+X

// Gemm class. t_KparWords defines number of memwords in the columns of one row of buffer_A. Due to the reusability,
// the height of buffer_A is only one memwords. For buffer_B, t_KparWords defines number of memwords in the rows of
// one column in buffer_B, t_NparWords defines number of memwords in the cols of one row in buffer_B. t_MparWords
// and t_NparWords define the height and width of buffer_C in terms of memwords.
template <typename t_FloatType,    // matrix A, B entry data type
          typename t_XDataType,    // matrix X entry data type
          unsigned int t_DdrWidth, // number of matrix elements in one memory word
          unsigned int t_XDdrWidth,
          unsigned int t_KparWords = 1, // number of memory words in one row of the matrix A buffer
          unsigned int t_MparWords = 1, // number of memory words in one column of the matrix A buffer
          unsigned int t_NparWords = 1  // number of memory words in one row of the matrix B buffer
          >
class GemmKernel {
   public:
    static const unsigned int t_aMH = t_DdrWidth * t_MparWords;
    static const unsigned int t_bKD = t_DdrWidth * t_KparWords;
    static const unsigned int t_DdrOverXDdr = t_DdrWidth / t_XDdrWidth;
    static const unsigned int t_xColMemWords = t_NparWords * t_DdrOverXDdr;

    typedef WideType<t_FloatType, t_DdrWidth> DdrWideType;
    typedef typename DdrWideType::t_TypeInt DdrIntType;
    typedef hls::stream<DdrIntType> DdrStream;
    typedef TaggedWideType<t_FloatType, t_DdrWidth> TaggedWideFloat;

    typedef hls::stream<typename TaggedWideType<t_FloatType, t_DdrWidth>::t_TypeInt> EdgeStream;

    typedef WideType<t_XDataType, t_XDdrWidth> XDdrWideType;
    typedef hls::stream<typename WideType<t_XDataType, t_XDdrWidth>::t_TypeInt> XDdrStream;
    typedef WideType<t_XDataType, t_DdrWidth> DdrWideTypeForX;

    // type definitions for enhanced MAC implementation, using 48-bits to store accumulation results.
    typedef t_FloatType MacBitType;
    typedef DdrWideType WideMacBitType;
    typedef DdrStream WideMacBitStream;

   private:
    static const unsigned int t_debug = 0;

   public:
    ///////////////////////////////////////////////////////////////////////////
    // GEMM C Buffering
    //
    ///////////////////////////////////////////////////////////////////////////
    void GemmCBuffer(WideMacBitStream& p_Cs,
                     unsigned int p_aColBlocks,
                     unsigned int p_cBlocks,
                     WideMacBitStream& p_Cout) {
        WideMacBitType l_bufferC[t_aMH * t_NparWords];

        for (int i = 0; i < t_aMH * t_NparWords; i++)
#pragma HLS PIPELINE
            for (int j = 0; j < t_DdrWidth; j++) l_bufferC[i][j] = 0;

        for (int l_block = 0; l_block < p_cBlocks; ++l_block) {
            for (int m = 0; m < p_aColBlocks; ++m) {
                for (int i = 0; i < t_MparWords; ++i) {
                    for (int j = 0; j < t_NparWords; ++j) {
                        for (int l = 0; l < t_DdrWidth; ++l) {
#pragma HLS DEPENDENCE variable = l_bufferC array inter RAW false
#pragma HLS PIPELINE
                            unsigned int l_arrIdx = (l + i * t_DdrWidth) * t_NparWords + j;
                            WideMacBitType l_val = p_Cs.read();
                            for (int k = 0; k < t_DdrWidth; ++k) {
                                l_bufferC[l_arrIdx][k] += l_val[k];
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < t_NparWords * t_MparWords * t_DdrWidth; ++i) {
#pragma HLS PIPELINE
                WideMacBitType l_val = l_bufferC[i];
                p_Cout.write(l_val);
                for (int k = 0; k < t_DdrWidth; k++) l_bufferC[i][k] = 0;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // GEMM Add X
    //
    ///////////////////////////////////////////////////////////////////////////
    void GemmAddX(
        WideMacBitStream& p_Cs, XDdrStream& p_Xs, unsigned int p_cBlocks, int32_t p_postScale, DdrStream& p_Cout) {
#if BLAS_XVEC
        DdrWideTypeForX l_bufferX[t_NparWords];
#else
        DdrWideTypeForX l_bufferX[t_aMH * t_NparWords];
#endif

        ap_uint<32> l_postScale = p_postScale;
        ap_uint<16> l_postScaleVal = l_postScale.range(23, 8);
        ap_uint<8> l_postScaleShift = l_postScale.range(7, 0);

        for (int l_block = 0; l_block < p_cBlocks; ++l_block) {
// read
#if BLAS_XVEC
            for (int xc = 0; xc < t_NparWords; ++xc) {
                DdrWideTypeForX l_wideWordX;
                for (int xw = 0; xw < t_DdrOverXDdr; ++xw) {
                    // #pragma HLS PIPELINE
                    XDdrWideType l_wordX = p_Xs.read();
                    for (int xi = 0; xi < t_XDdrWidth; ++xi) {
                        l_wideWordX[xw * t_XDdrWidth + xi] = l_wordX[xi];
                    }
                }
                l_bufferX[xc] = l_wideWordX;
            }
#else
            for (int xr = 0; xr < t_aMH; ++xr) {
                for (int xc = 0; xc < t_NparWords; ++xc) {
                    DdrWideTypeForX l_wideWordX;
                    for (int xw = 0; xw < t_DdrOverXDdr; ++xw) {
#pragma HLS PIPELINE
                        XDdrWideType l_wordX = p_Xs.read();
                        for (int xi = 0; xi < t_XDdrWidth; ++xi) {
                            l_wideWordX[xw * t_XDdrWidth + xi] = l_wordX[xi];
                        }
                    }
                    l_bufferX[xr * t_NparWords + xc] = l_wideWordX;
                }
            }
#endif
            for (int i = 0; i < t_MparWords * t_DdrWidth; ++i) {
                for (int j = 0; j < t_NparWords; ++j) {
                    WideMacBitType l_val = p_Cs.read();
#if BLAS_XVEC
                    DdrWideTypeForX l_xVal = l_bufferX[j];
#else
                    DdrWideTypeForX l_xVal = l_bufferX[i * t_NparWords + j];
#endif
                    DdrWideType l_cWord;
#pragma HLS PIPELINE
                    for (int w = 0; w < t_DdrWidth; ++w) {
                        t_FloatType l_cEntry;
                        l_cEntry = l_val[w] + l_xVal[w];

                        l_cWord[w] = l_cEntry;
                    }
                    p_Cout.write(l_cWord);
                }
            }
        }
    }

    void GemmBuffers(
        DdrStream& p_As,
        DdrStream& p_Bs,
        XDdrStream& p_Xs,
        DdrStream& p_Cs,
        typename SystolicArray<t_FloatType, t_bKD, t_DdrWidth, t_DdrWidth, t_FloatType>::TaggedWideStreamM& p_tagA,
        DdrStream& p_tagB,
        WideMacBitStream& p_CEdgeS,
        unsigned int p_aColBlocks,
        unsigned int p_aRowBlocks,
        unsigned int p_bColBlocks,
        unsigned int p_transpBlocks,
        int32_t p_postScale) {
        unsigned int l_cBlocks = p_aRowBlocks * p_bColBlocks;
        unsigned int l_abBlocks = l_cBlocks * p_aColBlocks;
#pragma HLS DATAFLOW

        DdrStream p_Bs1, p_AoutS;
        WideMacBitStream p_COutS;

        Transpose<t_FloatType, t_KparWords, t_DdrWidth> l_transp(p_transpBlocks, t_NparWords);
        l_transp.process(p_As, p_AoutS);

        MatrixBuffer<typename DdrWideType::t_TypeInt, t_DdrWidth * t_KparWords, t_NparWords, true, false>().process(
            p_Bs, p_Bs1, l_abBlocks, t_MparWords);

        SystolicArray<t_FloatType, t_bKD, t_DdrWidth, t_DdrWidth, t_FloatType> l_sysArr(l_abBlocks * t_MparWords *
                                                                                        t_NparWords);

        l_sysArr.tagAB(p_AoutS, p_Bs1, p_tagA, p_tagB);

        GemmCBuffer(p_CEdgeS, p_aColBlocks, l_cBlocks, p_COutS);
        GemmAddX(p_COutS, p_Xs, l_cBlocks, p_postScale, p_Cs);
    }

    void GemmTags(DdrStream& p_As,
                  DdrStream& p_Bs,
                  typename SystolicArray<t_FloatType, t_bKD, t_DdrWidth, t_DdrWidth, t_FloatType>::EdgeStreamM& p_aOut,
                  hls::stream<ap_uint<2> >& p_tagOut,
                  typename SystolicArray<t_FloatType, t_bKD, t_DdrWidth, t_DdrWidth, t_FloatType>::EdgeStreamN& p_tagB,
                  unsigned int p_aColBlocks,
                  unsigned int p_aRowBlocks,
                  unsigned int p_bColBlocks,
                  unsigned int p_transpBlocks,
                  int32_t p_postScale) {
        unsigned int l_cBlocks = p_aRowBlocks * p_bColBlocks;
        unsigned int l_abBlocks = l_cBlocks * p_aColBlocks;
#pragma HLS DATAFLOW

        DdrStream p_Bs1, p_AoutS;
        WideMacBitStream p_COutS;

        Transpose<t_FloatType, t_KparWords, t_DdrWidth> l_transp(p_transpBlocks, t_NparWords);
        l_transp.process(p_As, p_AoutS);

        MatrixBuffer<typename DdrWideType::t_TypeInt, t_DdrWidth * t_KparWords, t_NparWords, true, false>().process(
            p_Bs, p_Bs1, l_abBlocks, t_MparWords);

        SystolicArray<t_FloatType, t_bKD, t_DdrWidth, t_DdrWidth, t_FloatType> l_sysArr(l_abBlocks * t_MparWords *
                                                                                        t_NparWords);

        l_sysArr.tagAB(p_AoutS, p_Bs1, p_aOut, p_tagOut, p_tagB);
    }

    void GemmCPlusX(WideMacBitStream& p_CEdgeS,
                    DdrStream& p_Xs,
                    DdrStream& p_Cs,
                    unsigned int p_aColBlocks,
                    unsigned int p_aRowBlocks,
                    unsigned int p_bColBlocks,
                    int32_t p_postScale) {
        unsigned int l_cBlocks = p_aRowBlocks * p_bColBlocks;
        unsigned int l_abBlocks = l_cBlocks * p_aColBlocks;
#pragma HLS DATAFLOW
        WideMacBitStream p_COutS;
        GemmCBuffer(p_CEdgeS, p_aColBlocks, l_cBlocks, p_COutS);
        GemmAddX(p_COutS, p_Xs, l_cBlocks, p_postScale, p_Cs);
    }

    void GemmBlockStream(DdrStream& p_As,
                         DdrStream& p_Bs,
                         XDdrStream& p_Xs,
                         DdrStream& p_Cs,
                         unsigned int p_aColBlocks,
                         unsigned int p_aRowBlocks,
                         unsigned int p_bColBlocks,
                         unsigned int p_transpBlocks,
                         int32_t p_postScale) {
        unsigned int l_cBlocks = p_aRowBlocks * p_bColBlocks;
        unsigned int l_abBlocks = l_cBlocks * p_aColBlocks;
#pragma HLS DATAFLOW

        DdrStream p_Bs1, p_AoutS, p_CBufferS;
        EdgeStream p_AEdgeS0, p_BEdgeS0;
        WideMacBitStream p_CEdgeS, p_COutS;
#pragma HLS STREAM variable = p_CEdgeS depth = t_DdrWidth * t_MparWords * t_NparWords
#pragma HLS RESOURCE variable = p_CEdgeS core = fifo_uram

        /*
           Transp<t_FloatType, t_DdrWidth, t_KparWords, 1> l_transp;
           l_transp.processWithReuse(p_As, p_AoutS, p_transpBlocks, t_NparWords);
           */

        Transpose<t_FloatType, t_KparWords, t_DdrWidth> l_transp(p_transpBlocks, t_NparWords);
        l_transp.process(p_As, p_AoutS);

        MatrixBuffer<typename DdrWideType::t_TypeInt, t_DdrWidth * t_KparWords, t_NparWords, true, false>().process(
            p_Bs, p_Bs1, l_abBlocks, t_MparWords);

        Gemm<t_FloatType, t_bKD, t_DdrWidth>::gemm(p_AoutS, p_Bs1, p_CEdgeS, l_abBlocks * t_MparWords * t_NparWords);

        GemmCBuffer(p_CEdgeS, p_aColBlocks, l_cBlocks, p_COutS);
        GemmAddX(p_COutS, p_Xs, l_cBlocks, p_postScale, p_Cs);
    }
};
}
} // namespace
#endif
