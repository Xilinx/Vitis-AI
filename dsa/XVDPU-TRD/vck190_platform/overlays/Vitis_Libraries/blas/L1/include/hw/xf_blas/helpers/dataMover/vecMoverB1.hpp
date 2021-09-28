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

/**
 * @file vecMoverB1.hpp
 * @brief common data movers for vectors used in BLAS L1 routines.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_VECMOVERB1_HPP
#define XF_BLAS_VECMOVERB1_HPP

#include "hls_stream.h"
#include "ap_int.h"
#include "ap_shift_reg.h"

namespace xf {

namespace blas {

template <unsigned int t_NumStreams, typename t_DataType>
void duplicateStream(unsigned int p_n,
                     hls::stream<t_DataType>& p_inputStream,
                     hls::stream<t_DataType> p_streams[t_NumStreams]) {
    for (unsigned int i = 0; i < p_n; i++) {
#pragma HLS PIPELINE
        t_DataType p_in = p_inputStream.read();
        for (unsigned int j = 0; j < t_NumStreams; j++) {
            p_streams[j].write(p_in);
        }
    }
}

template <unsigned int t_NumStreams, typename t_DataType>
void splitStream(unsigned int p_n,
                 hls::stream<typename WideType<t_DataType, t_NumStreams>::t_TypeInt>& p_wideStream,
                 hls::stream<typename WideType<t_DataType, 1>::t_TypeInt> p_stream[t_NumStreams]) {
    for (unsigned int i = 0; i < p_n; i++) {
#pragma HLS PIPELINE
        WideType<t_DataType, t_NumStreams> p_in = p_wideStream.read();
        for (unsigned int j = 0; j < t_NumStreams; j++) {
            p_stream[j].write(WideType<t_DataType, 1>(p_in[j]));
        }
    }
}

template <unsigned int t_NumStreams, typename t_DataType>
void combineStream(unsigned int p_n,
                   hls::stream<typename WideType<t_DataType, 1>::t_TypeInt> p_stream[t_NumStreams],
                   hls::stream<typename WideType<t_DataType, t_NumStreams>::t_TypeInt>& p_wideStream) {
    for (unsigned int i = 0; i < p_n; i++) {
#pragma HLS PIPELINE
        WideType<t_DataType, t_NumStreams> p_out;
        for (unsigned int j = 0; j < t_NumStreams; j++) {
            p_out[j] = p_stream[j].read().getVal(0);
        }
        p_wideStream.write(p_out);
    }
}

template <typename t_DataType, typename t_DesDataType = t_DataType>
void mem2stream(unsigned int p_n,
                const t_DataType* p_in,
                hls::stream<t_DesDataType>& p_out,
                unsigned int p_repeat = 1,
                unsigned int p_batch = 1) {
    if (p_repeat == 1) {
        for (unsigned int i = 0; i < p_n * p_batch; ++i) {
#pragma HLS PIPELINE
            t_DesDataType l_val = p_in[i];
            p_out.write(l_val);
        }
    } else {
        for (unsigned int b = 0; b < p_batch; b++)
            for (unsigned int r = 0; r < p_repeat; r++)
                for (unsigned int i = 0; i < p_n; ++i) {
#pragma HLS PIPELINE
                    t_DesDataType l_val = p_in[i + b * p_n];
                    p_out.write(l_val);
                }
    }
} // end mem2stream

template <typename t_DataType, typename t_DesDataType = t_DataType>
void stream2mem(unsigned int p_n, hls::stream<t_DataType>& p_in, t_DesDataType* p_out, unsigned int p_batch = 1) {
    for (unsigned int i = 0; i < p_n * p_batch; ++i) {
#pragma HLS PIPELINE
        t_DesDataType l_val = p_in.read();
        p_out[i] = l_val;
    }
} // end stream2mem

/**
 * @brief readVec2Stream function that moves vector from memory to stream
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries number of parallelly processed entries in the matrix
 *
 * @param p_n number of entries in a vectpr
 * @param p_in vector input
 * @param p_out output stream
 */
template <typename t_DataType, unsigned int t_ParEntries>
void readVec2Stream(t_DataType* p_in,
                    unsigned int p_n,
                    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
#ifndef __SYNTHESIS__
    assert((p_n % t_ParEntries) == 0);
#endif
    unsigned int l_parBlocks = p_n / t_ParEntries;
    for (unsigned int i = 0; i < l_parBlocks; ++i) {
#pragma HLS PIPELINE
        BitConv<t_DataType> l_bitConv;
        WideType<t_DataType, t_ParEntries> l_val;
        for (unsigned int j = 0; j < t_ParEntries; ++j) {
            l_val[j] = p_in[i * t_ParEntries + j];
        }
        p_out.write(l_val);
    }
} // end readVec2Stream

/**
 * @brief writeStream2Vec function that moves vector from stream to vector
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries number of parallelly processed entries in the matrix
 *
 * @param p_n number of entries in a vectpr
 * @param p_in vector stream input
 * @param p_out vector output memory
 */
template <typename t_DataType, unsigned int t_ParEntries>
void writeStream2Vec(hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_in,
                     unsigned int p_n,
                     t_DataType* p_out) {
#ifndef __SYNTHESIS__
    assert((p_n % t_ParEntries) == 0);
#endif
    unsigned int l_parBlocks = p_n / t_ParEntries;
    for (unsigned int i = 0; i < l_parBlocks; ++i) {
#pragma HLS PIPELINE
        BitConv<t_DataType> l_bitConv;
        WideType<t_DataType, t_ParEntries> l_val;
        l_val = p_in.read();
        for (unsigned int j = 0; j < t_ParEntries; ++j) {
            p_out[i * t_ParEntries + j] = l_val[j];
        }
    }
} // end writeStream2Vec

} // namespace blas

} // namespace xf
#endif
