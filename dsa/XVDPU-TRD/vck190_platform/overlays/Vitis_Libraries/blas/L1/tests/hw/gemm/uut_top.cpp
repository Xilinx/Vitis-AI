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
#include "uut_top.hpp"
void uut_top(uint32_t p_m,
             uint32_t p_n,
             uint32_t p_k,
             BLAS_dataType p_alpha,
             BLAS_dataType p_beta,
             BLAS_dataType p_A[BLAS_matrixSizeA],
             BLAS_dataType p_B[BLAS_matrixSizeB],
             BLAS_dataType p_C[BLAS_matrixSizeC],
             BLAS_dataType p_R[BLAS_matrixSizeC]) {
    hls::stream<typename WideType<BLAS_dataType, BLAS_parEntries>::t_TypeInt> l_strA;
    hls::stream<typename WideType<BLAS_dataType, BLAS_parEntries>::t_TypeInt> l_strB;
    hls::stream<typename WideType<BLAS_dataType, BLAS_parEntries>::t_TypeInt> l_strC;
    hls::stream<typename WideType<BLAS_dataType, BLAS_parEntries>::t_TypeInt> l_strSum;
#pragma HLS DATAFLOW
    gemmMatAMover<BLAS_dataType, BLAS_parEntries>(p_A, p_m, p_n, p_k, l_strA);
    gemmMatBMover<BLAS_dataType, BLAS_parEntries>(p_B, p_m, p_n, p_k, l_strB);
    readVec2Stream<BLAS_dataType, BLAS_parEntries>(p_C, p_m * p_n, l_strC);
    gemm<BLAS_dataType, BLAS_k, BLAS_parEntries, BLAS_matrixSizeC>(p_m, p_n, p_k, p_alpha, l_strA, l_strB, p_beta,
                                                                   l_strC, l_strSum);
    writeStream2Vec<BLAS_dataType, BLAS_parEntries>(l_strSum, p_m * p_n, p_R);
}
