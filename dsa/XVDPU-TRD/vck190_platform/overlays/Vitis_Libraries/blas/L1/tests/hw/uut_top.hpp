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
#ifndef UUT_TOP_HPP
#define UUT_TOP_HPP
#if BLAS_L1

void uut_top(uint32_t p_n,
             BLAS_dataType p_alpha,
             BLAS_dataType p_x[BLAS_vectorSize],
             BLAS_dataType p_y[BLAS_vectorSize],
             BLAS_dataType p_xRes[BLAS_vectorSize],
             BLAS_dataType p_yRes[BLAS_vectorSize],
             BLAS_resDataType& p_goldRes);
#endif

#if BLAS_L2
void uut_top(uint32_t p_m,
             uint32_t p_n,
             uint32_t p_kl,
             uint32_t p_ku,
             BLAS_dataType p_alpha,
             BLAS_dataType p_beta,
             BLAS_dataType p_a[BLAS_matrixSize],
             BLAS_dataType p_x[BLAS_vectorSize],
             BLAS_dataType p_y[BLAS_matrixSize / BLAS_vectorSize],
             BLAS_dataType p_aRes[BLAS_matrixSize],
             BLAS_dataType p_yRes[BLAS_matrixSize / BLAS_vectorSize]);
#endif
#endif
