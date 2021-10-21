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
#ifndef XF_BLAS_SYSTOLICARRAY_HPP
#define XF_BLAS_SYSTOLICARRAY_HPP

#include <cassert>
#include "../helpers/utils/utils.hpp"

namespace xf {

namespace blas {

template <typename t_DataType,
          unsigned int t_KBufferDim,
          unsigned int t_ParEntriesM,
          unsigned int t_ParEntriesN = t_ParEntriesM,
          typename t_MacDataType = t_DataType>
class SystolicArray {
   public:
    typedef WideType<t_DataType, t_ParEntriesM> WideTypeM;
    typedef WideType<t_DataType, t_ParEntriesN> WideTypeN;
    typedef WideType<t_MacDataType, t_ParEntriesN> WideMacTypeN;

    typedef hls::stream<typename WideTypeM::t_TypeInt> EdgeStreamM;
    typedef hls::stream<typename WideTypeN::t_TypeInt> EdgeStreamN;
    typedef hls::stream<typename WideMacTypeN::t_TypeInt> EdgeMacStreamN;

    typedef TaggedWideType<t_DataType, t_ParEntriesM> TaggedWideTypeM;
    typedef hls::stream<typename TaggedWideTypeM::t_TypeInt> TaggedWideStreamM;

    typedef DualTaggedType<t_DataType> DualTagged;
    typedef hls::stream<typename DualTagged::t_TypeInt> DualTaggedStream;

    static const unsigned int t_FloatDelay = AdderDelay<t_DataType>::m_Delays;

   public:
    SystolicArray(unsigned int p_blocks = 0) : m_blocks(p_blocks) {}

    void process(EdgeStreamM& p_As, EdgeStreamN& p_Bs, EdgeMacStreamN& p_sum) {
        TaggedWideStreamM l_aStr;
        EdgeMacStreamN l_sum[t_ParEntriesM];
#pragma HLS ARRAY_PARTITION variable = l_sum complete dim = 1

        DualTaggedStream l_dataA[t_ParEntriesM];
#pragma HLS ARRAY_PARTITION variable = l_dataA complete dim = 1
#pragma HLS stream variable = l_dataA depth = t_ParEntriesM * 2

        EdgeStreamN l_bStr, l_dataB[t_ParEntriesM];
#pragma HLS ARRAY_PARTITION variable = l_dataB complete dim = 1
#pragma HLS stream variable = l_dataB[0] depth = t_ParEntriesM * 2

#pragma HLS DATAFLOW
        tagAB(p_As, p_Bs, l_aStr, l_bStr);
        systolicArray(l_aStr, l_bStr, l_dataA, l_dataB[0]);

        for (unsigned int col = 0; col < t_ParEntriesN - 1; ++col) {
#pragma HLS UNROLL
            macs(l_dataA[col], l_dataB[col], l_dataB[col + 1], l_sum[col]);
        }
        macs(l_dataA[t_ParEntriesN - 1], l_dataB[t_ParEntriesN - 1], l_sum[t_ParEntriesN - 1]);
        Merge<EdgeMacStreamN, t_ParEntriesM>::merge(l_sum, p_sum, m_blocks);
        // mergeSum(l_sum, p_sum);
    }

    void mergeSum(EdgeMacStreamN p_sums[t_ParEntriesM], EdgeMacStreamN& p_sum) {
        for (unsigned int i = 0; i < m_blocks; i++) {
            for (int j = 0; j < t_ParEntriesM; j++) {
#pragma HLS PIPELINE
                p_sum.write(p_sums[j].read());
            }
        }
    }

    void muls(DualTaggedStream& p_aIn, EdgeStreamN& p_bIn, EdgeStreamN& p_bOut, EdgeMacStreamN& p_mul) {
        ap_shift_reg<bool, t_ParEntriesN> l_flush;
        for (int i = 0; i < t_ParEntriesN; i++)
#pragma HLS PIPELINE
            l_flush.shift(false);

        TriangSrl<t_MacDataType, t_ParEntriesN> l_Tc;
        l_Tc.clear();

        uint32_t l_outCount = t_KBufferDim;
        bool l_exit;

        WideTypeN l_aSlr(0);

        do {
#pragma HLS PIPELINE
            DualTagged l_c = p_aIn.read();
            t_DataType l_aVal = l_c.m_val;
            WideTypeN l_bWideVal = p_bIn.read();
            p_bOut.write(l_bWideVal);

            WideMacTypeN l_mul;
            WideMacTypeN l_cVec;
            WideMacTypeN l_mVec;

            bool l_iflush = l_c.m_flush;
            l_exit = l_c.m_exit;

            bool l_oflush = l_flush.shift(l_iflush);
            if (l_oflush) {
                l_outCount = 0;
            }

            for (int i = 0; i < t_ParEntriesN; i++) {
                l_cVec[t_ParEntriesN - i - 1] = l_aSlr[i] * l_bWideVal[i];
            }

            l_mVec = l_Tc.shift(l_cVec);

            for (int i = 0; i < t_ParEntriesN; i++) {
                l_mul[i] = l_mVec[t_ParEntriesN - i - 1];
            }

            if (l_outCount < t_KBufferDim) {
                p_mul.write(l_mul);
                l_outCount++;
            }
            l_aSlr.shift(l_aVal);

        } while (!l_exit);
    }

    void muls(DualTaggedStream& p_aIn, EdgeStreamN& p_bIn, EdgeMacStreamN& p_mul) {
        ap_shift_reg<bool, t_ParEntriesN> l_flush;
        for (int i = 0; i < t_ParEntriesN; i++)
#pragma HLS PIPELINE
            l_flush.shift(false);

        TriangSrl<t_MacDataType, t_ParEntriesN> l_Tc;
        l_Tc.clear();

        uint32_t l_outCount = t_KBufferDim;
        bool l_exit;

        WideTypeN l_aSlr(0);

        do {
#pragma HLS PIPELINE
            DualTagged l_c = p_aIn.read();
            t_DataType l_aVal = l_c.m_val;
            WideTypeN l_bWideVal = p_bIn.read();

            WideMacTypeN l_mul;
            WideMacTypeN l_cVec;
            WideMacTypeN l_mVec;

            bool l_iflush = l_c.m_flush;
            l_exit = l_c.m_exit;

            bool l_oflush = l_flush.shift(l_iflush);
            if (l_oflush) {
                l_outCount = 0;
            }

            for (int i = 0; i < t_ParEntriesN; i++) {
                l_cVec[t_ParEntriesN - i - 1] = l_aSlr[i] * l_bWideVal[i]; // l_c.m_a[i] * l_c.m_b[i];
            }

            l_mVec = l_Tc.shift(l_cVec);

            for (int i = 0; i < t_ParEntriesN; i++) {
                l_mul[i] = l_mVec[t_ParEntriesN - i - 1];
            }

            if (l_outCount < t_KBufferDim) {
                p_mul.write(l_mul);
                l_outCount++;
            }
            l_aSlr.shift(l_aVal);

        } while (!l_exit);
    }
    void adds(EdgeMacStreamN& p_mul, EdgeMacStreamN& p_sum) {
        WideType<t_MacDataType, t_ParEntriesN> l_sum;
        constexpr int l_kIter = t_KBufferDim / t_FloatDelay;

        for (unsigned int n = 0; n < m_blocks; n++) {
            for (int k = 0; k < l_kIter; k++) {
#pragma HLS PIPELINE II = t_FloatDelay
                if (k == 0) l_sum = 0;

                t_MacDataType l_pSum[t_ParEntriesN];
#pragma HLS ARRAY_PARTITION variable = l_pSum complete dim = 1
                WideMacTypeN l_mul = p_mul.read();

                for (int i = 0; i < t_ParEntriesN; i++) {
                    l_pSum[i] = l_mul[i];
                }

                for (int d = 1; d < t_FloatDelay; d++) {
                    WideMacTypeN l_mul = p_mul.read();
                    for (int i = 0; i < t_ParEntriesN; i++) {
                        l_pSum[i] += l_mul[i];
                    }
                }

                for (int i = 0; i < t_ParEntriesN; i++) {
                    l_sum[i] += l_pSum[i];
                }

                if (k == l_kIter - 1) {
                    p_sum.write(l_sum);
                }
            }
        }
    }

    void macs(DualTaggedStream& p_A, EdgeStreamM& p_Bin, EdgeStreamM& p_Bout, EdgeMacStreamN& p_sum) {
#pragma HLS DATAFLOW
        EdgeMacStreamN l_mul;
        muls(p_A, p_Bin, p_Bout, l_mul);
        adds(l_mul, p_sum);
    }

    void macs(DualTaggedStream& p_A, EdgeStreamM& p_Bin, EdgeMacStreamN& p_sum) {
#pragma HLS DATAFLOW
        EdgeMacStreamN l_mul;
        muls(p_A, p_Bin, l_mul);
        adds(l_mul, p_sum);
    }

    void systolicArray(TaggedWideStreamM& p_As,
                       EdgeStreamN& p_Bs,
                       DualTaggedStream p_aOut[t_ParEntriesM],
                       EdgeStreamN& p_bOut) {
        TriangSrl<t_DataType, t_ParEntriesN> l_Tb;
        WideType<bool, 2> l_Tf = 0;
        l_Tb.clear();

        bool l_exit;
        do {
#pragma HLS PIPELINE

            TaggedWideTypeM l_A = p_As.read();
            WideTypeN l_B = p_Bs.read();
            l_exit = l_A.getExit();
            bool l_flush = l_A.getFlush();

            WideTypeN l_bvec1 = l_Tb.shift(l_B);
            l_Tf.shift(l_flush);

            p_bOut.write(l_bvec1);
            for (unsigned int row = 0; row < t_ParEntriesM; ++row) {
#pragma HLS UNROLL
                DualTagged l_dualTag;
                l_dualTag.m_val = l_A[row];
                l_dualTag.m_flush = l_Tf[1];
                l_dualTag.m_exit = l_exit;
                p_aOut[row].write(l_dualTag);
            }
        } while (!l_exit);
    }

    void tagAB(EdgeStreamM& p_a, EdgeStreamN& p_b, TaggedWideStreamM& l_aOut, EdgeStreamN& l_b) {
        for (unsigned int i = 0; i < m_blocks; i++)
            for (int k = 0; k < t_KBufferDim; k++) {
#pragma HLS PIPELINE
                WideTypeM l_valA = p_a.read();
                typename WideTypeN::t_TypeInt l_valB = p_b.read();
                bool l_flush = (k == 0);
                bool l_exit = false;
                TaggedWideTypeM l_taggedValA(l_valA, l_flush, l_exit);
                l_aOut.write(l_taggedValA);
                l_b.write(l_valB);
            }
        const unsigned int l_flushLen = 1 + t_ParEntriesM + t_ParEntriesN;
        for (unsigned i = 0; i < l_flushLen; i++) {
#pragma HLS PIPELINE
            bool l_exit = (i == l_flushLen - 1);
            bool l_flush = false;
            TaggedWideTypeM l_taggedValA(0, l_flush, l_exit);
            typename WideTypeN::t_TypeInt l_valB = 0;
            l_aOut.write(l_taggedValA);
            l_b.write(l_valB);
        }
    }

   private:
    unsigned int m_blocks;
};
}
}

#endif
