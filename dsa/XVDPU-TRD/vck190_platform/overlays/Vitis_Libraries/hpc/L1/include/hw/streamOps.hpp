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

#ifndef XF_HPC_STREAM_OPS_HPP
#define XF_HPC_STREAM_OPS_HPP

/**
 * @file streamOps.hpp
 * @brief stream operations are defined here
 */
#include <type_traits>
#include "xf_blas.hpp"
namespace xf {
namespace hpc {

/**
 * @brief wide2stream converts an integer of wide datawidth to an integer of base datawidth
 *
 * @tparam t_DataWidth is the base datawidth
 * @tparam t_Multi is the factor between two datawidth
 *
 * @param p_n is the number of data to be read
 * @param p_wide is the input stream of wide datawidth
 * @param p_str is the output stream of base datawidth
 */

template <unsigned int t_DataWidth, unsigned int t_Multi>
void wide2stream(unsigned int p_n,
                 hls::stream<ap_uint<t_DataWidth * t_Multi> >& p_wide,
                 hls::stream<ap_uint<t_DataWidth> >& p_str) {
    blas::WideType<ap_uint<t_DataWidth>, t_Multi, t_DataWidth> l_wide;
    for (int i = 0, j = 0; i < p_n * t_Multi; i++) {
#pragma HLS PIPELINE
        if (j == 0) l_wide = p_wide.read();
        p_str.write(l_wide.unshift());
        if (j == t_Multi - 1) {
            j = 0;
        } else
            j++;
    }
}

/**
 * @brief stream2wide converts an integer of base datawidth to an integer of wide datawidth
 *
 * @tparam t_DataWidth is the base datawidth
 * @tparam t_Multi is the factor between two datawidth
 *
 * @param p_n is the number of data to be write
 * @param p_str is the input stream of base datawidth
 * @param p_wide is the output stream of wide datawidth
 */
template <unsigned int t_DataWidth, unsigned int t_Multi>
void stream2wide(unsigned int p_n,
                 hls::stream<ap_uint<t_DataWidth> >& p_str,
                 hls::stream<ap_uint<t_DataWidth * t_Multi> >& p_wide) {
    blas::WideType<ap_uint<t_DataWidth>, t_Multi, t_DataWidth> l_wide;
    for (int i = 0, j = 0; i < p_n * t_Multi; i++) {
#pragma HLS PIPELINE
        ap_uint<t_DataWidth> l_str = p_str.read();
        l_wide.unshift(l_str);
        if (j == t_Multi - 1) {
            j = 0;
        } else
            j++;
        if (j == 0) p_wide.write(l_wide);
    }
}

template <unsigned int t_ParEntries, typename t_DataType>
void wide2stream(unsigned int p_n,
                 hls::stream<blas::WideType<t_DataType, t_ParEntries> >& p_wide,
                 hls::stream<t_DataType>& p_str) {
    blas::WideType<t_DataType, t_ParEntries> l_wide;
    for (int i = 0, j = 0; i < p_n * t_ParEntries; i++) {
#pragma HLS PIPELINE
        if (j == 0) l_wide = p_wide.read();
        p_str.write(l_wide[j]);
        if (j == t_ParEntries - 1) {
            j = 0;
        } else
            j++;
    }
}

template <unsigned int t_ParEntries, typename t_DataType>
void stream2wide(unsigned int p_n,
                 hls::stream<t_DataType>& p_str,
                 hls::stream<blas::WideType<t_DataType, t_ParEntries> >& p_wide) {
    blas::WideType<t_DataType, t_ParEntries> l_wide;
    for (int i = 0, j = 0; i < p_n * t_ParEntries; i++) {
#pragma HLS PIPELINE
        l_wide[j] = p_str.read();
        if (j == t_ParEntries - 1) p_wide.write(l_wide);
        if (j == t_ParEntries - 1) {
            j = 0;
        } else
            j++;
    }
}

template <typename t_DesDataType, typename t_DataType>
void streamConversion(unsigned int p_n, hls::stream<t_DataType>& p_in, hls::stream<t_DesDataType>& p_out) {
    for (int i = 0; i < p_n; i++) {
        p_out.write(p_in.read());
    }
}

template <unsigned int t_NumEntries, unsigned int t_ParEntries, typename t_DataType>
void conv2stream(unsigned int p_n,
                 hls::stream<blas::WideType<t_DataType, t_ParEntries> >& p_wide,
                 hls::stream<blas::WideType<t_DataType, t_NumEntries> >& p_str) {
    blas::WideType<t_DataType, t_ParEntries> l_wide;
    blas::WideType<t_DataType, t_NumEntries> l_str;
#ifndef __SYNTHESIS__
    assert(t_ParEntries % t_NumEntries == 0);
#endif

    for (int i = 0; i < p_n; i++) {
        for (int j = 0; j < t_ParEntries / t_NumEntries; j++) {
#pragma HLS PIPELINE
            if (j == 0) l_wide = p_wide.read();
            for (int k = 0; k < t_NumEntries; k++) {
#pragma HLS UNROLL
                l_str[k] = l_wide[j * t_NumEntries + k];
            }
            p_str.write(l_str);
        }
    }
}

template <unsigned int t_NumEntries, unsigned int t_ParEntries, typename t_DataType>
void conv2wide(unsigned int p_n,
               hls::stream<blas::WideType<t_DataType, t_NumEntries> >& p_str,
               hls::stream<blas::WideType<t_DataType, t_ParEntries> >& p_wide) {
#ifndef __SYNTHESIS__
    assert(t_ParEntries % t_NumEntries == 0);
#endif

    blas::WideType<t_DataType, t_ParEntries> l_wide;
    blas::WideType<t_DataType, t_NumEntries> l_str;
    for (int i = 0; i < p_n; i++) {
        for (int j = 0; j < t_ParEntries / t_NumEntries; j++) {
#pragma HLS PIPELINE
            l_str = p_str.read();
            for (int k = 0; k < t_NumEntries; k++) {
#pragma HLS UNROLL
                l_wide[j * t_NumEntries + k] = l_str[k];
            }
            if (j == t_ParEntries / t_NumEntries - 1) p_wide.write(l_wide);
        }
    }
}

template <typename T>
void duplicate(const unsigned int p_n, hls::stream<T>& p_in, hls::stream<T>& p_out0, hls::stream<T>& p_out1) {
    for (int i = 0; i < p_n; i++) {
#pragma HLS PIPELINE
        T l_in = p_in.read();
        p_out0.write(l_in);
        p_out1.write(l_in);
    }
}

template <typename T>
void dataConsumer(const unsigned int p_n, hls::stream<T>& p_s) {
    for (int i = 0; i < p_n; i++)
#pragma HLS PIPELINE
        p_s.read();
}

template <typename t_DataType,
          unsigned int t_NumStreams,
          unsigned int t_ParEntries = 1,
          typename T0 = typename xf::blas::WideType<t_DataType, 1>,
          typename T1 = typename xf::blas::WideType<t_DataType, t_ParEntries> >
void collectStream(unsigned int p_n,
                   hls::stream<typename T0::t_TypeInt> p_strIn[t_NumStreams],
                   hls::stream<typename T1::t_TypeInt>& p_strOut) {
    static_assert(t_NumStreams % t_ParEntries == 0, "");
    for (unsigned int i = 0; i < p_n; i++) {
        for (unsigned int j = 0; j < t_NumStreams / t_ParEntries; j++) {
#pragma HLS PIPELINE
            T1 out;
            for (unsigned int k = 0; k < t_ParEntries; k++) {
                T0 in = p_strIn[j * t_ParEntries + k].read();
                out[k] = in[0];
            }
            p_strOut.write(out);
        }
    }
}

template <typename T>
void copy(unsigned int p_n, hls::stream<T>& p_in, hls::stream<T>& p_out) {
    for (int t = 0; t < p_n; t++) {
#pragma HLS PIPELINE
        T l_val = p_in.read();
        p_out.write(l_val);
    }
}

template <unsigned int N, typename T, typename std::enable_if<N == 1, int>::type = 0>
void streamFwd(const unsigned int p_n, hls::stream<T>& p_in, hls::stream<T> p_out[N]) {
#pragma HLS DATAFLOW
    copy<T>(p_n, p_in, p_out[0]);
}

template <unsigned int N, typename T, typename std::enable_if<N == 2, int>::type = 0>
void streamFwd(const unsigned int p_n, hls::stream<T>& p_in, hls::stream<T> p_out[N]) {
#pragma HLS DATAFLOW
    duplicate<T>(p_n, p_in, p_out[0], p_out[1]);
}

template <unsigned int N, typename T, typename std::enable_if<(N > 2), int>::type = 0>
void streamFwd(const unsigned int p_n, hls::stream<T>& p_in, hls::stream<T> p_out[N]) {
#pragma HLS DATAFLOW
    hls::stream<T> p_mid[N];
    duplicate<T>(p_n, p_in, p_mid[0], p_out[0]);
    for (int i = 1; i < N - 2; i++)
#pragma HLS UNROLL
        duplicate<T>(p_n, p_mid[i - 1], p_mid[i], p_out[i]);
    duplicate<T>(p_n, p_mid[N - 3], p_out[N - 2], p_out[N - 1]);
}

template <unsigned int t_NumStreams, typename t_DataType>
void splitStream(unsigned int p_n, hls::stream<t_DataType>& p_strIn, hls::stream<t_DataType> p_strOut[t_NumStreams]) {
    for (unsigned int i = 0, j = 0; i < p_n * t_NumStreams; i++, j++) {
#pragma HLS PIPELINE
        j = j % t_NumStreams;
        t_DataType p_in = p_strIn.read();
        p_strOut[j].write(p_in);
    }
}

template <unsigned int t_NumStreams, typename t_DataType>
void mergeStream(unsigned int p_n, hls::stream<t_DataType> p_strIn[t_NumStreams], hls::stream<t_DataType>& p_strOut) {
    for (unsigned int i = 0, j = 0; i < p_n * t_NumStreams; i++, j++) {
#pragma HLS PIPELINE
        j = j % t_NumStreams;
        t_DataType p_in = p_strIn[j].read();
        p_strOut.write(p_in);
    }
}
}
}
#endif
