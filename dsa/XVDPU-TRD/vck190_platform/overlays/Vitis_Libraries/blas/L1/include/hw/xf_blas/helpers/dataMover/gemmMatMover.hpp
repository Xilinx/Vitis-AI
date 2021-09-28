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
#include "ap_int.h"
#include "hls_stream.h"
namespace xf {

namespace blas {

template <typename t_DataType, unsigned int t_M, unsigned int t_N = t_M>
void gemmMatAMover(t_DataType* p_memA,
                   unsigned int p_m,
                   unsigned int p_n,
                   unsigned int p_k,
                   hls::stream<typename WideType<t_DataType, t_M>::t_TypeInt>& p_As) {
#ifndef __SYNTHESIS__
    assert(p_m % t_M == 0);
    assert(p_n % t_N == 0);
#endif
    unsigned int l_iter = p_m / t_M;
    unsigned int l_repeat = p_n / t_N;

    for (int m = 0; m < l_iter; m++) {
        for (int r = 0; r < l_repeat; r++) {
            for (int k = 0; k < p_k; k++) {
#pragma HLS PIPELINE
                WideType<t_DataType, t_M> l_A;
                for (int i = 0; i < t_M; i++) {
#pragma HLS UNROLL
                    l_A[i] = p_memA[(k * l_iter + m) * t_M + i];
                }
                p_As.write(l_A);
            }
        }
    }
}

template <typename t_DataType, unsigned int t_M, unsigned int t_N = t_M>
void gemmMatBMover(t_DataType* p_memB,
                   unsigned int p_m,
                   unsigned int p_n,
                   unsigned int p_k,
                   hls::stream<typename WideType<t_DataType, t_N>::t_TypeInt>& p_Bs) {
#ifndef __SYNTHESIS__
    assert(p_m % t_M == 0);
    assert(p_n % t_N == 0);
#endif
    unsigned int l_repeat = p_m / t_M;
    unsigned int l_iter = p_n / t_N;

    for (int r = 0; r < l_repeat; r++) {
        for (int n = 0; n < l_iter; n++) {
            for (int k = 0; k < p_k; k++) {
#pragma HLS PIPELINE
                WideType<t_DataType, t_N> l_B;
                for (int i = 0; i < t_N; i++) {
#pragma HLS UNROLL
                    l_B[i] = p_memB[(k * l_iter + n) * t_N + i];
                }
                p_Bs.write(l_B);
            }
        }
    }
}

template <typename t_DataType, unsigned int t_M, unsigned int t_N = t_M, unsigned int t_MaxSizeC>
void gemmBufferC(unsigned int p_m,
                 unsigned int p_n,
                 hls::stream<typename WideType<t_DataType, t_N>::t_TypeInt>& p_sum,
                 hls::stream<typename WideType<t_DataType, t_N>::t_TypeInt>& p_C) {
#ifndef __SYNTHESIS__
    assert(p_m % t_M == 0);
    assert(p_n % t_N == 0);
    assert(p_m * p_n <= t_MaxSizeC);
#endif
    t_DataType l_bufferC[t_MaxSizeC];
#pragma HLS ARRAY_PARTITION variable = l_bufferC cyclic factor = t_N dim = 1
    unsigned int l_iterM = p_m / t_M;
    unsigned int l_iterN = p_n / t_N;
    unsigned int index = 0;
    for (int n = 0; n < l_iterN; n++) {
        for (int j = 0; j < t_M; j++) {
#pragma HLS PIPELINE
            WideType<t_DataType, t_N> l_sum = p_sum.read();
            for (int i = 0; i < t_N; i++) {
#pragma HLS UNROLL
                l_bufferC[j * l_iterN * t_N + n * t_N + i] = l_sum[i];
            }
        }
    }
    for (int m = 1; m < l_iterM; m++) {
        for (int n = 0; n < l_iterN; n++) {
            for (int j = 0; j < t_M; j++) {
#pragma HLS PIPELINE
                WideType<t_DataType, t_N> l_sum = p_sum.read();
                WideType<t_DataType, t_N> l_out;
                for (int i = 0; i < t_N; i++) {
#pragma HLS UNROLL
                    l_bufferC[(m * t_M + j) * l_iterN * t_N + n * t_N + i] = l_sum[i];
                    l_out[i] = l_bufferC[index * t_N + i];
                }
                index++;
                p_C.write(l_out);
            }
        }
    }
    for (int n = 0; n < l_iterN * t_M; n++) {
#pragma HLS PIPELINE
        WideType<t_DataType, t_N> l_out;
        for (int i = 0; i < t_N; i++) {
#pragma HLS UNROLL
            l_out[i] = l_bufferC[(n + index) * t_N + i];
        }
        p_C.write(l_out);
    }
}
}
}
