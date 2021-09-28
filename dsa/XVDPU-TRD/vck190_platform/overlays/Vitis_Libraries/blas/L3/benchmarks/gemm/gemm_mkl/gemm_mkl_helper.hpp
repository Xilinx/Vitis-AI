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

#ifndef GEMM_MKL_HELPER_HPP
#define GEMM_MKL_HELPER_HPP

#define IDX2R(i, j, ld) (((i) * (ld)) + (j))

#include <iostream>
#include <mkl.h>
#include <chrono>

#ifdef USE_DOUBLE_PRECISION
#define DISPLAY_GEMM_FUNC "DGEMM_MKL"
#define XFBLAS_dataType double
#define GEMM_MKL(m, n, k, alpha, beta, a, b, c) \
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);

#elif USE_SINGLE_PRECISION
#define DISPLAY_GEMM_FUNC "SGEMM_MKL"
#define XFBLAS_dataType float
#define GEMM_MKL(m, n, k, alpha, beta, a, b, c) \
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);

#else
#define DISPLAY_GEMM_FUNC "SGEMM_MKL"
#define XFBLAS_dataType float
#define GEMM_MKL(m, n, k, alpha, beta, a, b, c) \
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);

#endif

typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimePointType;

XFBLAS_dataType* createMat(int p_rows, int p_cols, bool is_zero);
void initMat(XFBLAS_dataType* mat, int p_rows, int p_cols, bool is_zero);

XFBLAS_dataType* createMat(int p_rows, int p_cols, bool is_zero = false) {
    XFBLAS_dataType* mat;
    /*// OBSOLETE, use posix_memalign.
      mat = (XFBLAS_dataType *)memalign(128, (size_t)p_rows * (size_t)p_cols * sizeof(XFBLAS_dataType));
      if (mat == (XFBLAS_dataType *)NULL) {
        printf("[ERROR] failed to create the matrix\n");
        exit(1);
      }*/
    int rc = posix_memalign((void**)&mat, 4096, (size_t)p_rows * (size_t)p_cols * sizeof(XFBLAS_dataType));
    if (rc != 0) {
        printf("[ERROR %d] failed to create the matrix\n", rc);
        exit(1);
    }
    initMat(mat, p_rows, p_cols, is_zero);
    return mat;
}

void initMat(XFBLAS_dataType* mat, int p_rows, int p_cols, bool is_zero) {
    srand(time(NULL));
    for (int j = 0; j < p_rows; j++) {
        for (int i = 0; i < p_cols; i++) {
            mat[IDX2R(j, i, p_cols)] = 1;
        }
    }
}
#endif
