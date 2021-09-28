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
#ifndef XF_BLAS_GEMM_MERGE_HPP
#define XF_BLAS_GEMM_MERGE_HPP

#include "blasInstr.hpp"
namespace xf {
namespace blas {

template <unsigned int t_MemBits, unsigned int t_ParEntries>
void gemmMerge(hls::stream<ap_uint<t_MemBits> > p_sums[t_ParEntries],
               hls::stream<ap_uint<t_MemBits> >& p_sum,
               hls::stream<unsigned int>& p_blocks) {
    unsigned int m_blocks = p_blocks.read();
    for (unsigned int i = 0; i < m_blocks; i++) {
        for (int j = 0; j < t_ParEntries; j++) {
#pragma HLS PIPELINE
            p_sum.write(p_sums[j].read());
        }
    }
}
}
}
#endif
