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

#ifndef XF_BLAS_GEMM_HPP
#define XF_BLAS_GEMM_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "ap_int.h"
#include "hls_stream.h"
#include "xf_blas/helpers.hpp"

#include "gemm/systolicArray.hpp"
#include "scal.hpp"
#include "axpy.hpp"

namespace xf {

namespace blas {

template <typename t_DataType,
          unsigned int t_KBufferDim,
          unsigned int t_ParEntriesM,
          unsigned int t_ParEntriesN = t_ParEntriesM,
          typename t_MacDataType = t_DataType>
class Gemm {
    typedef WideType<t_DataType, t_ParEntriesM> WideTypeM;
    typedef WideType<t_DataType, t_ParEntriesN> WideTypeN;
    typedef WideType<t_MacDataType, t_ParEntriesN> WideMacTypeN;

    typedef hls::stream<typename WideTypeM::t_TypeInt> EdgeStreamM;
    typedef hls::stream<typename WideTypeN::t_TypeInt> EdgeStreamN;
    typedef hls::stream<typename WideMacTypeN::t_TypeInt> EdgeMacStreamN;

    typedef TaggedFloat<t_DataType> TaggedDataType;

    typedef TaggedFloat<t_MacDataType> TaggedMacType;
    typedef WideType<typename TaggedMacType::t_TypeInt, t_ParEntriesN> WideTaggedMacType;
    typedef hls::stream<typename WideTaggedMacType::t_TypeInt> TaggedMacTypeStream;

   public:
    static void gemm(EdgeStreamM& p_As, EdgeStreamN& p_Bs, EdgeMacStreamN& p_sum, unsigned int p_blocks) {
#ifndef __SYNTHESIS__
        assert(t_KBufferDim >= t_ParEntriesM + t_ParEntriesN);
#endif

        WindowRm<TaggedDataType, t_ParEntriesM, t_ParEntriesN> l_awin;
        WindowRm<TaggedDataType, t_ParEntriesN, t_ParEntriesM> l_bwin;
        TriangSrl<TaggedDataType, t_ParEntriesM> l_Ta;
        TriangSrl<TaggedDataType, t_ParEntriesN> l_Tb;
        l_awin.clear();
        l_bwin.clear();
        l_Ta.clear();
        l_Tb.clear();

        typedef WideType<TaggedDataType, t_ParEntriesM> TaggedArrayM;
        typedef WideType<TaggedDataType, t_ParEntriesN> TaggedArrayN;

        WideType<t_MacDataType, t_ParEntriesN> l_C[t_ParEntriesM];
#pragma HLS ARRAY_PARTITION variable = l_C dim = 1 complete
        WideType<t_MacDataType, t_ParEntriesN> l_Co[t_ParEntriesM];
#pragma HLS ARRAY_PARTITION variable = l_Co dim = 1 complete

        for (uint32_t l = 0; l <= p_blocks; l++)
            for (int k = 0; k < t_KBufferDim; k++) {
#pragma HLS PIPELINE

                WideType<t_DataType, t_ParEntriesM> l_A = 0;
                WideType<t_DataType, t_ParEntriesN> l_B = 0;

                if (l < p_blocks) {
                    l_A = p_As.read();
                    l_B = p_Bs.read();
                }

                TaggedArrayM l_avec;
                for (int i = 0; i < t_ParEntriesM; i++) l_avec[i] = TaggedDataType(l_A[i], k == 0);
                TaggedArrayN l_bvec;
                for (int i = 0; i < t_ParEntriesN; i++) l_bvec[i] = TaggedDataType(l_B[i], k == 0);

                TaggedArrayM l_avec1 = l_Ta.shift(l_avec);
                TaggedArrayN l_bvec1 = l_Tb.shift(l_bvec);

                (void)l_awin.shift_right(l_avec1);
                (void)l_bwin.shift(l_bvec1);

                if (l > 0 && k >= t_ParEntriesN + 1 && k <= t_ParEntriesM + t_ParEntriesN) {
                    p_sum.write(l_Co[k - t_ParEntriesN - 1]);
                }

                for (unsigned int row = 0; row < t_ParEntriesM; ++row) {
#pragma HLS UNROLL
                    TaggedArrayM l_arow = l_awin[row];
                    TaggedArrayN l_brow = l_bwin[row];
                    for (unsigned int col = 0; col < t_ParEntriesN; ++col) {
#pragma HLS UNROLL
                        t_DataType aval = l_arow[col]();
                        t_DataType bval = l_brow[col]();
                        bool aflush = l_arow[col].getFlush();
#ifndef __SYNTEHSIS__
                        bool bflush = l_brow[col].getFlush();
                        assert(aflush == bflush);
#endif
                        if (aflush) {
                            l_Co[row][col] = l_C[row][col];
                            l_C[row][col] = 0;
                        }
                        l_C[row][col] += aval * bval;
                    }
                }
            }
    }
};

template <unsigned int t_KBufferDim, unsigned int t_ParEntriesM, unsigned int t_ParEntriesN>
class Gemm<float, t_KBufferDim, t_ParEntriesM, t_ParEntriesN, float> {
    typedef float t_DataType;
    typedef float t_MacDataType;
    typedef WideType<t_DataType, t_ParEntriesM> WideTypeM;
    typedef WideType<t_DataType, t_ParEntriesN> WideTypeN;
    typedef WideType<t_MacDataType, t_ParEntriesN> WideMacTypeN;

    typedef hls::stream<typename WideTypeM::t_TypeInt> EdgeStreamM;
    typedef hls::stream<typename WideTypeN::t_TypeInt> EdgeStreamN;
    typedef hls::stream<typename WideMacTypeN::t_TypeInt> EdgeMacStreamN;

    typedef TaggedFloat<t_DataType> TaggedDataType;

    typedef TaggedFloat<t_MacDataType> TaggedMacType;
    typedef WideType<typename TaggedMacType::t_TypeInt, t_ParEntriesN> WideTaggedMacType;
    typedef hls::stream<typename WideTaggedMacType::t_TypeInt> TaggedMacTypeStream;

   public:
    static void gemm(EdgeStreamM& p_As, EdgeStreamN& p_Bs, EdgeMacStreamN& p_sum, unsigned int p_blocks) {
#ifndef __SYNTHESIS__
        assert(t_KBufferDim >= t_ParEntriesM + t_ParEntriesN);
#endif
#pragma HLS DATAFLOW
        SystolicArray<t_DataType, t_KBufferDim, t_ParEntriesM, t_ParEntriesN, t_MacDataType> l_sysArr(p_blocks);
        l_sysArr.process(p_As, p_Bs, p_sum);
    }
};

template <typename t_DataType, unsigned int t_KBufferDim, unsigned int t_ParEntries>
void gemm(const unsigned int p_m,
          const unsigned int p_n,
          const unsigned int p_k,
          hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_A,
          hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_B,
          hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_C) {
#pragma HLS DATAFLOW
#ifndef __SYNTHESIS__
    assert(p_m % t_ParEntries == 0);
    assert(p_n % t_ParEntries == 0);
    assert(p_k == t_KBufferDim);
#endif
    unsigned int l_blocks = p_m * p_n / t_ParEntries / t_ParEntries;
    Gemm<t_DataType, t_KBufferDim, t_ParEntries>::gemm(p_A, p_B, p_C, l_blocks);
}

template <typename t_DataType,
          unsigned int t_KBufferDim,
          unsigned int t_ParEntries,
          unsigned int t_MaxSizeC,
          typename t_IndexType = unsigned int>
void gemm(const unsigned int p_m,
          const unsigned int p_n,
          const unsigned int p_k,
          const t_DataType p_alpha,
          hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_A,
          hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_B,
          const t_DataType p_beta,
          hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_C,
          hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_R) {
#ifndef __SYNTHESIS__
    assert(p_m % t_ParEntries == 0);
    assert(p_n % t_ParEntries == 0);
    assert(p_k == t_KBufferDim);
#endif
    unsigned int l_size = p_m * p_n;
    unsigned int l_blocks = l_size / t_ParEntries / t_ParEntries;

    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_sum, l_mat, l_betaC;

#pragma HLS DATAFLOW

    Gemm<t_DataType, t_KBufferDim, t_ParEntries, t_ParEntries>::gemm(p_A, p_B, l_sum, l_blocks);
    gemmBufferC<t_DataType, t_ParEntries, t_ParEntries, t_MaxSizeC>(p_m, p_n, l_sum, l_mat);
    scal<t_DataType, t_ParEntries, t_IndexType>(l_size, p_beta, p_C, l_betaC);
    axpy<t_DataType, t_ParEntries, t_IndexType>(l_size, p_alpha, l_mat, l_betaC, p_R);
}

} // end namespace blas

} // end namespace xf

#endif
