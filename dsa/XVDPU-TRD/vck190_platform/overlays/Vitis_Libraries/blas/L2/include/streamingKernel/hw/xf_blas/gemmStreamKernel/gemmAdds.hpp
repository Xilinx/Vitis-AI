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
#ifndef XF_BLAS_GEMM_ADD_HPP
#define XF_BLAS_GEMM_ADD_HPP

#include "blasInstr.hpp"
#include "gemmKernel.hpp"
namespace xf {
namespace blas {

template <typename t_DataType,
          unsigned int t_ParEntriesN,
          unsigned int t_KparWords,
          unsigned int t_FloatDelay = 4,
          typename t_WideType = WideType<t_DataType, t_ParEntriesN> >
#ifdef GemmStreamSeq
void gemmAdds(hls::stream<typename t_WideType::t_TypeInt>& p_mul,
              hls::stream<typename t_WideType::t_TypeInt>& p_sum,
              hls::stream<unsigned int>& p_blocks) {
#else
void gemmAdds(hls::stream<typename t_WideType::t_TypeInt>& p_mul, hls::stream<typename t_WideType::t_TypeInt>& p_sum) {
#endif
    t_WideType l_sum;
    constexpr int l_kIter = t_KparWords * t_ParEntriesN / t_FloatDelay;
#ifdef GemmStreamSeq
    unsigned int m_blocks = p_blocks.read();
    for (unsigned int n = 0; n < m_blocks; n++)
#endif
        for (int k = 0; k < l_kIter; k++) {
#pragma HLS PIPELINE II = t_FloatDelay
            if (k == 0) l_sum = 0;

            t_DataType l_pSum[t_ParEntriesN];
#pragma HLS ARRAY_PARTITION variable = l_pSum complete dim = 1
            t_WideType l_mul = p_mul.read();

            for (int i = 0; i < t_ParEntriesN; i++) {
                l_pSum[i] = l_mul[i];
            }

            for (int d = 1; d < t_FloatDelay; d++) {
                t_WideType l_mul = p_mul.read();
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
}
#endif
