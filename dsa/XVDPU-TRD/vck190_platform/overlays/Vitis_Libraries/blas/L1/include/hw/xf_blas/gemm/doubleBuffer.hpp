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
#ifndef XF_BLAS_DOUBLEBUFFER_HPP
#define XF_BLAS_DOUBLEBUFFER_HPP
namespace xf {
namespace blas {

template <typename t_DataType,
          unsigned int t_BufferSize> // number of memory words in one row of the matrix B buffer
class DoubleBuffer {
   public:
    typedef hls::stream<t_DataType> DdrStream;

    void process(DdrStream& p_streamIn,
                 DdrStream& p_streamOut,
                 unsigned int p_iterationNum,
                 unsigned int p_reuseNum = 1) {
        DdrStream p_s0_0, p_s0_1, p_s1_0, p_s1_1;
#pragma HLS DATAFLOW
        split(p_iterationNum, p_streamIn, p_s0_0, p_s0_1);
        buffer((p_iterationNum / 2) + (p_iterationNum % 2), p_s0_0, p_s1_0, p_reuseNum);
        buffer((p_iterationNum / 2), p_s0_1, p_s1_1, p_reuseNum);
        merge(p_iterationNum, p_s1_0, p_s1_1, p_streamOut, p_reuseNum);
    }

    void buffer(unsigned int p_iterationNum, DdrStream& p_in, DdrStream& p_out, unsigned int p_reuseNum) {
        t_DataType l_buffer[t_BufferSize];
        for (int l_block = 0; l_block < p_iterationNum; ++l_block) {
            // read block
            for (int i = 0; i < t_BufferSize; ++i) {
#pragma HLS PIPELINE
                t_DataType l_word = p_in.read();
                l_buffer[i] = l_word;
            }
            // stream down l_buffer
            for (int i = 0; i < p_reuseNum; ++i) {
                for (int l = 0; l < t_BufferSize; ++l) {
#pragma HLS PIPELINE
                    t_DataType l_word = l_buffer[l];
                    p_out.write(l_word);
                }
            }
        }
    }

    void split(unsigned int p_iterationNum, DdrStream& p_in, DdrStream& p_out1, DdrStream& p_out2) {
        for (int i = 0; i < p_iterationNum; ++i) {
            for (int j = 0; j < t_BufferSize; ++j) {
#pragma HLS PIPELINE
                t_DataType l_word = p_in.read();
                if ((i % 2) == 0) {
                    p_out1.write(l_word);
                } else {
                    p_out2.write(l_word);
                }
            }
        }
    }

    void merge(
        unsigned int p_iterationNum, DdrStream& p_in1, DdrStream& p_in2, DdrStream& p_out, unsigned int p_reuseNum) {
        for (int i = 0; i < p_iterationNum; ++i) {
            for (int r = 0; r < p_reuseNum; ++r) {
                for (int j = 0; j < t_BufferSize; ++j) {
#pragma HLS PIPELINE
                    t_DataType l_word;
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
