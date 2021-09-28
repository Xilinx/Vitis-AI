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
#include "xf_blas.hpp"
#include "uut_top.hpp"

using namespace xf::blas;

void uut_top(uint32_t p_m,
             uint32_t p_n,
             uint32_t p_kl,
             uint32_t p_ku,
             BLAS_dataType p_alpha,
             BLAS_dataType p_beta,
             BLAS_dataType p_a[BLAS_memorySize],
             BLAS_dataType p_x[BLAS_vectorSize],
             BLAS_dataType p_y[BLAS_vectorSize],
             BLAS_dataType p_aRes[BLAS_memorySize],
             BLAS_dataType p_yRes[BLAS_vectorSize]) {
#ifndef __SYNTHESIS__
    assert(p_m == p_n);
#endif
#pragma HLS DATAFLOW
    hls::stream<typename WideType<BLAS_dataType, BLAS_parEntries>::t_TypeInt> l_strA;
    hls::stream<typename WideType<BLAS_dataType, BLAS_parEntries>::t_TypeInt> l_strX;
    hls::stream<typename WideType<BLAS_dataType, 1>::t_TypeInt> l_strY;
    hls::stream<typename WideType<BLAS_dataType, 1>::t_TypeInt> l_strYR;
#pragma HLS DATAFLOW
    tpmUp2Stream<BLAS_dataType, BLAS_parEntries>(p_n, p_a, l_strA);
    vec2TrmUpStream<BLAS_dataType, BLAS_parEntries>(p_n, p_x, l_strX);
    readVec2Stream<BLAS_dataType, 1>(p_y, p_n, l_strY);
    trmv<BLAS_dataType, BLAS_logParEntries>(true, p_n, p_alpha, l_strA, l_strX, p_beta, l_strY, l_strYR);
    writeStream2Vec<BLAS_dataType, 1>(l_strYR, p_m, p_yRes);
}
