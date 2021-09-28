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
#ifndef XF_BLAS_TRANSPOSE_HPP
#define XF_BLAS_TRANSPOSE_HPP

namespace xf {
namespace blas {

template <typename t_DataType,
          unsigned int t_ColMemWords,
          unsigned int t_ParEntriesM,
          unsigned int t_ParEntriesN = t_ParEntriesM>
class Transpose {
    typedef WideType<t_DataType, t_ParEntriesM> t_WideTypeM;
    typedef hls::stream<typename t_WideTypeM::t_TypeInt> DdrStreamM;

    typedef WideType<t_DataType, t_ParEntriesN> t_WideTypeN;
    typedef hls::stream<typename t_WideTypeN::t_TypeInt> DdrStreamN;

    static const int t_BufferSize = t_ParEntriesM * t_ColMemWords;

    unsigned int m_iterationNum;
    unsigned int m_reuseNum;

   public:
    Transpose(unsigned int p_iterationNum, unsigned int p_reuseNum) {
        m_iterationNum = p_iterationNum;
        m_reuseNum = p_reuseNum;
    }

    void process(DdrStreamN& p_streamIn, DdrStreamM& p_streamOut) {
        DdrStreamN p_s0_0, p_s0_1;
        DdrStreamM p_s1_0, p_s1_1;
        unsigned int l_iter1 = m_iterationNum >> 1;
        unsigned int l_iter0 = m_iterationNum - l_iter1;
#pragma HLS DATAFLOW
        split(p_streamIn, p_s0_0, p_s0_1);
        buffer(l_iter0, p_s0_0, p_s1_0);
        buffer(l_iter1, p_s0_1, p_s1_1);
        merge(p_s1_0, p_s1_1, p_streamOut);
    }

    void buffer(unsigned int p_iterationNum, DdrStreamN& p_in, DdrStreamM& p_out) {
        t_WideTypeN l_buffer[t_ParEntriesM][t_ColMemWords];
#pragma HLS ARRAY_PARTITION variable = l_buffer dim = 1 complete
        for (int l_block = 0; l_block < p_iterationNum; ++l_block) {
            // read block
            for (int i = 0; i < t_ParEntriesM; ++i) {
                for (int j = 0; j < t_ColMemWords; j++) {
#pragma HLS PIPELINE
                    t_WideTypeN l_word = p_in.read();
                    l_buffer[i][j] = l_word;
                }
            }
            // stream down l_buffer
            for (int r = 0; r < m_reuseNum; ++r) {
                for (int i = 0; i < t_ColMemWords; i++) {
                    for (int j = 0; j < t_ParEntriesN; j++) {
#pragma HLS PIPELINE
                        t_WideTypeM l_word;
                        for (int k = 0; k < t_ParEntriesM; k++) {
                            l_word[k] = l_buffer[k][i][j];
                        }
                        p_out.write(l_word);
                    }
                }
            }
        }
    }
    void split(DdrStreamN& p_in, DdrStreamN& p_out1, DdrStreamN& p_out2) {
        for (int i = 0; i < m_iterationNum; ++i) {
            for (int j = 0; j < t_BufferSize; ++j) {
#pragma HLS PIPELINE
                t_WideTypeN l_word = p_in.read();
                if ((i % 2) == 0) {
                    p_out1.write(l_word);
                } else {
                    p_out2.write(l_word);
                }
            }
        }
    }

    void merge(DdrStreamM& p_in1, DdrStreamM& p_in2, DdrStreamM& p_out) {
        for (int i = 0; i < m_iterationNum; ++i) {
            for (int r = 0; r < m_reuseNum; ++r) {
                for (int j = 0; j < t_BufferSize; ++j) {
#pragma HLS PIPELINE
                    t_WideTypeM l_word;
                    if ((i % 2) == 0) {
                        l_word = p_in1.read();
                    } else {
                        l_word = p_in2.read();
                    }
                    p_out.write(l_word);
                }
            }
        }
    }
};
}
}
#endif
