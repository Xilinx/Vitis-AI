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
#ifndef XF_BLAS_MATRIXBUFFER_HPP
#define XF_BLAS_MATRIXBUFFER_HPP
#include "doubleBuffer.hpp"

namespace xf {
namespace blas {

template <typename t_DataType, unsigned int t_bKDim, unsigned int t_bColMemWords, bool t_RowIn, bool t_RowOut>
class MatrixBuffer : public DoubleBuffer<t_DataType, t_bKDim * t_bColMemWords> {};

template <typename t_DataType, unsigned int t_bKDim, unsigned int t_bColMemWords>

class MatrixBuffer<t_DataType, t_bKDim, t_bColMemWords, true, false>
    : public DoubleBuffer<t_DataType, t_bKDim * t_bColMemWords> {
   public:
    typedef hls::stream<t_DataType> DdrStream;
    void process(DdrStream& p_streamIn,
                 DdrStream& p_streamOut,
                 unsigned int p_iterationNum,
                 unsigned int p_reuseNum = 1) {
        DdrStream p_s0_0, p_s0_1, p_s1_0, p_s1_1;
#pragma HLS DATAFLOW
        this->split(p_iterationNum, p_streamIn, p_s0_0, p_s0_1);
        buffer((p_iterationNum / 2) + (p_iterationNum % 2), p_s0_0, p_s1_0, p_reuseNum);
        buffer((p_iterationNum / 2), p_s0_1, p_s1_1, p_reuseNum);
        this->merge(p_iterationNum, p_s1_0, p_s1_1, p_streamOut, p_reuseNum);
    }
    void buffer(unsigned int p_iterationNum, DdrStream& p_in, DdrStream& p_out, unsigned int p_reuseNum) {
        t_DataType l_buffer[t_bKDim][t_bColMemWords];
        for (int l_block = 0; l_block < p_iterationNum; ++l_block) {
            // read block
            for (int i = 0; i < t_bKDim; ++i) {
                for (int j = 0; j < t_bColMemWords; ++j) {
#pragma HLS PIPELINE
                    t_DataType l_word = p_in.read();
                    l_buffer[i][j] = l_word;
                }
            }
            // stream down l_buffer
            for (int i = 0; i < p_reuseNum; ++i) {
                for (int k = 0; k < t_bColMemWords; ++k) {
                    for (int l = 0; l < t_bKDim; ++l) {
#pragma HLS PIPELINE
                        t_DataType l_word = l_buffer[l][k];
                        p_out.write(l_word);
                    }
                }
            }
        }
    }
};
template <typename t_DataType, unsigned int t_bKDim, unsigned int t_bColMemWords>
class MatrixBuffer<t_DataType, t_bKDim, t_bColMemWords, true, true>
    : public DoubleBuffer<t_DataType, t_bKDim * t_bColMemWords> {
   public:
    typedef hls::stream<t_DataType> DdrStream;
    void process(DdrStream& p_streamIn,
                 DdrStream& p_streamOut,
                 unsigned int p_iterationNum,
                 unsigned int p_reuseNum = 1) {
        DdrStream p_s0_0, p_s0_1, p_s1_0, p_s1_1;
#pragma HLS DATAFLOW
        this->split(p_iterationNum, p_streamIn, p_s0_0, p_s0_1);
        buffer((p_iterationNum / 2) + (p_iterationNum % 2), p_s0_0, p_s1_0, p_reuseNum);
        buffer((p_iterationNum / 2), p_s0_1, p_s1_1, p_reuseNum);
        this->merge(p_iterationNum, p_s1_0, p_s1_1, p_streamOut, p_reuseNum);
    }
    void buffer(unsigned int p_iterationNum, DdrStream& p_in, DdrStream& p_out, unsigned int p_reuseNum) {
        t_DataType l_buffer[t_bKDim][t_bColMemWords];
        for (int l_block = 0; l_block < p_iterationNum; ++l_block) {
            // read block
            for (int i = 0; i < t_bKDim; ++i) {
                for (int j = 0; j < t_bColMemWords; ++j) {
#pragma HLS PIPELINE
                    t_DataType l_word = p_in.read();
                    l_buffer[i][j] = l_word;
                }
            }
            // stream down l_buffer
            for (int i = 0; i < p_reuseNum; ++i) {
                for (int l = 0; l < t_bKDim; ++l) {
                    for (int k = 0; k < t_bColMemWords; ++k) {
#pragma HLS PIPELINE
                        t_DataType l_word = l_buffer[l][k];
                        p_out.write(l_word);
                    }
                }
            }
        }
    }
};
template <typename t_DataType, unsigned int t_bKDim, unsigned int t_bColMemWords>
class MatrixBuffer<t_DataType, t_bKDim, t_bColMemWords, false, false>
    : public DoubleBuffer<t_DataType, t_bKDim * t_bColMemWords> {
   public:
    typedef hls::stream<t_DataType> DdrStream;
    void process(DdrStream& p_streamIn,
                 DdrStream& p_streamOut,
                 unsigned int p_iterationNum,
                 unsigned int p_reuseNum = 1) {
        DdrStream p_s0_0, p_s0_1, p_s1_0, p_s1_1;
#pragma HLS DATAFLOW
        this->split(p_iterationNum, p_streamIn, p_s0_0, p_s0_1);
        buffer((p_iterationNum / 2) + (p_iterationNum % 2), p_s0_0, p_s1_0, p_reuseNum);
        buffer((p_iterationNum / 2), p_s0_1, p_s1_1, p_reuseNum);
        this->merge(p_iterationNum, p_s1_0, p_s1_1, p_streamOut, p_reuseNum);
    }
    void buffer(unsigned int p_iterationNum, DdrStream& p_in, DdrStream& p_out, unsigned int p_reuseNum) {
        t_DataType l_buffer[t_bKDim][t_bColMemWords];
        for (int l_block = 0; l_block < p_iterationNum; ++l_block) {
            // read block
            for (int j = 0; j < t_bColMemWords; ++j) {
                for (int i = 0; i < t_bKDim; ++i) {
#pragma HLS PIPELINE
                    t_DataType l_word = p_in.read();
                    l_buffer[i][j] = l_word;
                }
            }
            // stream down l_buffer
            for (int i = 0; i < p_reuseNum; ++i) {
                for (int k = 0; k < t_bColMemWords; ++k) {
                    for (int l = 0; l < t_bKDim; ++l) {
#pragma HLS PIPELINE
                        t_DataType l_word = l_buffer[l][k];
                        p_out.write(l_word);
                    }
                }
            }
        }
    }
};

template <typename t_DataType, unsigned int t_bKDim, unsigned int t_bColMemWords>
class MatrixBuffer<t_DataType, t_bKDim, t_bColMemWords, false, true>
    : public DoubleBuffer<t_DataType, t_bKDim * t_bColMemWords> {
   public:
    typedef hls::stream<t_DataType> DdrStream;
    void process(DdrStream& p_streamIn,
                 DdrStream& p_streamOut,
                 unsigned int p_iterationNum,
                 unsigned int p_reuseNum = 1) {
        DdrStream p_s0_0, p_s0_1, p_s1_0, p_s1_1;
#pragma HLS DATAFLOW
        this->split(p_iterationNum, p_streamIn, p_s0_0, p_s0_1);
        buffer((p_iterationNum / 2) + (p_iterationNum % 2), p_s0_0, p_s1_0, p_reuseNum);
        buffer((p_iterationNum / 2), p_s0_1, p_s1_1, p_reuseNum);
        this->merge(p_iterationNum, p_s1_0, p_s1_1, p_streamOut, p_reuseNum);
    }
    void buffer(unsigned int p_iterationNum, DdrStream& p_in, DdrStream& p_out, unsigned int p_reuseNum) {
        t_DataType l_buffer[t_bKDim][t_bColMemWords];
        for (int l_block = 0; l_block < p_iterationNum; ++l_block) {
            // read block
            for (int j = 0; j < t_bColMemWords; ++j) {
                for (int i = 0; i < t_bKDim; ++i) {
#pragma HLS PIPELINE
                    t_DataType l_word = p_in.read();
                    l_buffer[i][j] = l_word;
                }
            }
            // stream down l_buffer
            for (int i = 0; i < p_reuseNum; ++i) {
                for (int l = 0; l < t_bKDim; ++l) {
                    for (int k = 0; k < t_bColMemWords; ++k) {
#pragma HLS PIPELINE
                        t_DataType l_word = l_buffer[l][k];
                        p_out.write(l_word);
                    }
                }
            }
        }
    }
};
}
}
#endif
