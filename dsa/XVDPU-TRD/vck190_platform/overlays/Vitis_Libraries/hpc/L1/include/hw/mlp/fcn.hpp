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

#ifndef XF_HPC_MLP_FCN_HPP
#define XF_HPC_MLP_FCN_HPP

#include "hls_math.h"
#include "activations.hpp"
#include "gemmKernel.hpp"

using namespace xf::blas;

namespace xf {
namespace hpc {
namespace mlp {

////////////////////////////////////////////////////////////////////////////////
// class FCN (fully connected network)
// add preScale, pRelu and postScal operations to the gemm results
////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Fcn class, implement C = f(A*B+X)
 *
 * @tparam t_FloatType  matrix A B entry data type
 * @tparam t_XDataType  matrix X entry data type
 * @tparam t_DdrWidth  number of matrix elements in one memory word
 * @tparam t_XDdrWidth  number of matrix elements in one memory word
 * @tparam t_aColMemWords  the number of memory words in one row of the matrix A buffer
 * @tparam t_aRowMemWords  the number of memory words in one column of the matrix A buffer
 * @tparam t_bColMemWords  the number of memory words in one row of the matrix B buffer
 * @tparam t_maxWSize  the maximun Weight matrix size 0 for unlimited size
 * @tparam t_maxBSize  the maximun Bias vector size 0 for unlimited size
 *
 */

template <typename t_FloatType,
          typename t_XDataType, // bias matrix entry type
          unsigned int t_DdrWidth,
          unsigned int t_XDdrWidth,
          unsigned int t_aColMemWords = 1,
          unsigned int t_aRowMemWords = 1,
          unsigned int t_bColMemWords = 1,
          unsigned int t_maxWSize = 0,
          unsigned int t_maxBSize = 0>
class Fcn {
   private:
    static const unsigned int t_debug = 0;

   public:
    // type definitions
    typedef typename GemmKernel<t_FloatType,
                                t_XDataType,
                                t_DdrWidth,
                                t_XDdrWidth,
                                t_aColMemWords,
                                t_aRowMemWords,
                                t_bColMemWords>::DdrWideType DdrWideType;
    typedef typename GemmKernel<t_FloatType,
                                t_XDataType,
                                t_DdrWidth,
                                t_XDdrWidth,
                                t_aColMemWords,
                                t_aRowMemWords,
                                t_bColMemWords>::DdrStream DdrStream;
    typedef FcnArgs FcnArgsType;
    typedef typename DdrWideType::t_TypeInt DdrIntType;

#if BLAS_CACHE
    DdrIntType m_matB[t_maxWSize / t_DdrWidth];
    DdrIntType m_bias[t_maxBSize / t_DdrWidth];
#endif

   public:
    Fcn() {
#if BLAS_CACHE
#pragma HLS INLINE
#pragma HLS RESOURCE variable = m_matB core = RAM_2P_URAM
#endif
    }

    /** @brief FcnActivation applies activation function to the FCN output
     *
     *  @param p_inS is the input stream from FCN
     *  @param p_outS is the output stream after applying activation
     *  @param p_blocks is the number of blocks to be processed
     *  @param f_act is the activation function
     *  @param p_args is the arguments passed to the activation function
     */
    void FcnActivation(
        DdrStream& p_inS, DdrStream& p_outS, unsigned int p_blocks, t_FloatType (*f_act)(t_FloatType), int16_t p_args) {
        unsigned l_count = p_blocks * t_aRowMemWords * t_DdrWidth * t_bColMemWords;
        if ((p_args & 0x01) == 1) {
            for (int c = 0; c < l_count; ++c) {
#pragma HLS PIPELINE
                DdrIntType l_val = p_inS.read();
                p_outS.write(l_val);
            }
        } else {
            for (int c = 0; c < l_count; ++c) {
#pragma HLS PIPELINE II = t_aColMemWords
                DdrWideType l_val = p_inS.read();
                DdrWideType l_valOut;
                for (int w = 0; w < t_DdrWidth; ++w) {
                    l_valOut[w] = f_act(l_val[w]);
                }
                p_outS.write(l_valOut);
            }
        }
    }

    /**
     * @brief FcnBlocks runs the FCN for input matrices
     *
     * @param p_aAddr  the base address of matrix A in external memory
     * @param p_bAddr  the base address of matrix B in external memory
     * @param p_cAddr  the base address of matrix C in external memory
     * @param p_xAddr  the base address of matrix X in external memory
     *
     * @param p_aColBlocks  the No. blocks along matrix X cols
     * @param p_aRowBlocks  the No. blocks along matrix X rows
     * @param p_bColBlocks  the No. blocks along matrix X cols
     *
     * @param p_aLd  the matrix A word leading dimention
     * @param p_bLd  the matrix B word leading dimention
     * @param p_cLd  the matrix C word leading dimention
     * @param p_xLd  the matrix X word leading dimention
     *
     * @param f_act  the activation function
     * @param p_Args  the arguments for fcn operation
     * @param p_transpBlocks the number of blocks for compute
     *
     */
    void FcnBlocks(DdrIntType* p_aAddr,
                   DdrIntType* p_bAddr,
                   DdrIntType* p_cAddr,
                   DdrIntType* p_xAddr,
                   unsigned int p_aColBlocks,
                   unsigned int p_aRowBlocks,
                   unsigned int p_bColBlocks,
                   unsigned int p_aLd,
                   unsigned int p_bLd,
                   unsigned int p_cLd,
                   unsigned int p_xLd,
                   unsigned int p_transpBlocks,
                   t_FloatType (*f_act)(t_FloatType),
                   FcnArgsType& p_Args) {
#pragma HLS DATAFLOW

        GemmKernel<t_FloatType, t_XDataType, t_DdrWidth, t_XDdrWidth, t_aColMemWords, t_aRowMemWords, t_bColMemWords>
            l_gemm;
        DdrStream p_C2ScalePRelu;
        DdrStream p_Cs;

#pragma HLS STREAM variable = p_C2ScalePRelu depth = t_DdrWidth * t_aRowMemWords * t_bColMemWords
#pragma HLS RESOURCE variable = p_C2ScalePRelu core = fifo_uram

        l_gemm.GemmReadAndMult(p_aAddr, p_bAddr, p_xAddr, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_aLd, p_bLd, p_xLd,
                               p_transpBlocks, p_Args.m_postScale, p_C2ScalePRelu);
        FcnActivation(p_C2ScalePRelu, p_Cs, p_aRowBlocks * p_bColBlocks, f_act, p_Args.m_PReluVal);
        l_gemm.GemmWriteDdrStream(p_cAddr, p_Cs, p_aRowBlocks, p_bColBlocks, p_cLd);
    }

    /**
     * @brief runFcn launch fcn operation with given arguments
     *
     * @param p_DdrRd  the DDR/HBM address for input data
     * @param p_DdrWr  the DDR/HBM address for output data
     * @param p_Args  the arguments for fcn
     *
     */
    void runFcn(DdrIntType* p_DdrRd, DdrIntType* p_DdrWr, FcnArgsType& p_Args) {
        DdrIntType* l_aAddr = p_DdrRd + p_Args.m_Aoffset * DdrWideType::per4k();
        DdrIntType* l_bAddr = p_DdrRd + p_Args.m_Boffset * DdrWideType::per4k();
        DdrIntType* l_xAddr = p_DdrRd + p_Args.m_Xoffset * DdrWideType::per4k();
        DdrIntType* l_cAddr = p_DdrWr + p_Args.m_Coffset * DdrWideType::per4k();

        const unsigned int l_aColBlocks = p_Args.m_K / (t_DdrWidth * t_aColMemWords);
        const unsigned int l_aRowBlocks = p_Args.m_M / (t_DdrWidth * t_aRowMemWords);
        const unsigned int l_bColBlocks = p_Args.m_N / (t_DdrWidth * t_bColMemWords);
        const unsigned int l_aLd = p_Args.m_Lda / t_DdrWidth;
        const unsigned int l_bLd = p_Args.m_Ldb / t_DdrWidth;
        const unsigned int l_cLd = p_Args.m_Ldc / t_DdrWidth;
        const unsigned int l_xLd = p_Args.m_Ldx / t_XDdrWidth;

        unsigned int l_transpBlocks = l_aColBlocks * l_aRowBlocks * l_bColBlocks * t_aRowMemWords;
#if BLAS_CACHE
#ifndef __SYNTHESIS__
        assert(t_maxWSize >= p_Args.m_K * p_Args.m_N);
        assert(t_maxBSize >= p_Args.m_N);
#endif
    loadB:
        for (int r = 0; r < p_Args.m_K * p_Args.m_N / t_DdrWidth; r++) {
#pragma HLS PIPELINE
            m_matB[r] = l_bAddr[r];
        }

    loadX:
        for (int r = 0; r < p_Args.m_N / t_DdrWidth; r++) {
#pragma HLS PIPELINE
            m_bias[r] = l_xAddr[r];
        }
#if MLP_RELU
        FcnBlocks(l_aAddr, m_matB, l_cAddr, m_bias, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd,
                  l_xLd, l_transpBlocks, relu, p_Args);
#elif MLP_TANSIG
        FcnBlocks(l_aAddr, m_matB, l_cAddr, m_bias, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd,
                  l_xLd, l_transpBlocks, tansig, p_Args);
#else
        FcnBlocks(l_aAddr, m_matB, l_cAddr, m_bias, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd,
                  l_xLd, l_transpBlocks, sigmoid, p_Args);
#endif
#else
#if MLP_RELU
        FcnBlocks(l_aAddr, m_matB, l_cAddr, m_bias, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd,
                  l_xLd, l_transpBlocks, relu, p_Args);
#elif MLP_TANSIG
        FcnBlocks(l_aAddr, m_matB, l_cAddr, m_bias, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd,
                  l_xLd, l_transpBlocks, tansig, p_Args);
#else
        FcnBlocks(l_aAddr, l_bAddr, l_cAddr, l_xAddr, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd,
                  l_xLd, l_transpBlocks, sigmoid, p_Args);
#endif
#endif
    }
};
}
}
}
#endif
