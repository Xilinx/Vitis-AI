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
 * WITHOUT WANCUNCUANTIES ONCU CONDITIONS OF ANY KIND, either express or
 * implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file trtrs.hpp
 * @brief  This files contains implementation of triangular solver.
 */

#ifndef _XF_SOLVER_TRTRS_HPP_
#define _XF_SOLVER_TRTRS_HPP_

namespace xf {
namespace solver {
namespace internal_trtrs {

template <typename T, int N, int NCU>
void trisolver_L(int n, T dataA[NCU][(N + NCU - 1) / NCU][N], T dataB[NCU][(N + NCU - 1) / NCU], T dataX[N]) {
#pragma HLS inline off

    dataX[0] = dataB[0][0] / dataA[0][0][0];
Loop_row:
    for (int i = 1; i < n; i++) {
    Loop_col:
        for (int j = i / NCU; j < (n + NCU - 1) / NCU; j++) {
#pragma HLS pipeline
            for (int k = 0; k < NCU; k++) {
#pragma HLS loop_tripcount min = NCU max = NCU
#pragma HLS unroll factor = NCU
#pragma HLS dependence variable = dataB inter false
                if ((j * NCU + k) < n) dataB[k][j] -= dataA[k][j][i - 1] * dataX[i - 1];
            }
        }
        dataX[i] = dataB[i % NCU][i / NCU] / dataA[i % NCU][i / NCU][i];
    }
}

template <typename T, int N, int NCU>
void trisolver_U(int n, T dataA[NCU][(N + NCU - 1) / NCU][N], T dataB[NCU][(N + NCU - 1) / NCU], T dataX[N]) {
#pragma HLS inline off

    dataX[n - 1] = dataB[(n - 1) % NCU][(n - 1) / NCU] / dataA[(n - 1) % NCU][(n - 1) / NCU][n - 1];
Loop_row:
    for (int i = n - 2; i >= 0; i--) {
    Loop_col:
        for (int j = i / NCU; j >= 0; j--) {
#pragma HLS pipeline
            for (int k = NCU - 1; k >= 0; k--) {
#pragma HLS loop_tripcount min = NCU max = NCU
#pragma HLS unroll factor = NCU
#pragma HLS dependence variable = dataB inter false
                if ((j * NCU + k) < N - 1) dataB[k][j] -= dataA[k][j][i + 1] * dataX[i + 1];
            }
        }
        dataX[i] = dataB[i % NCU][i / NCU] / dataA[i % NCU][i / NCU][i];
    }
}

} // namespace internal
/**
 * @brief This function solves a system of linear equation with triangular
 * coefficient matrix along with multiple right-hand side vector \n
 *           \f{equation*} {Ax=B}\f}
 *                     where \f$A\f$ is a dense lower or upper
 * triangular matrix of size \f$m \times m\f$, \f$x\f$ is a vector need to
 * be computed, and \f$B\f$ is input vector.\n
 * The maximum matrix size supported in FPGA is templated by NMAX.
 *
 * @tparam T data type (support float and double)
 * @tparam NMAX maximum number of rows/columns of input matrix
 * @tparam NCU number of computation unit
 * @param[in] m number of rows/cols of matrix A
 * @param[in,out] A input matrix of size \f$n \times n\f$
 * @param[in] b number of columns of matrix B
 * @param[in,out] B input matrix of size \f$b \times n\f$, and overwritten by the output matrix x
 * @param[in] lda leading dimention of input matrix A
 * @param[in] ldb leading dimention of input matrix B
 * @param[out] info output info (unused)
 */

template <typename T, int NMAX, int NCU>
void trtrs(char uplo, int m, T* A, int b, T* B, int lda, int ldb, int& info) {
    if (NMAX == 1)
        B[0] = B[0] / A[0];
    else {
        static T matA[NCU][(NMAX + NCU - 1) / NCU][NMAX];
        static T matB[NCU][(NMAX + NCU - 1) / NCU];
#pragma HLS array_partition variable = matA cyclic factor = NCU
#pragma HLS array_partition variable = matB cyclic factor = NCU
#pragma HLS resource variable = matA core = XPM_MEMORY uram

    Loop_read:
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < m; c++) {
#pragma HLS pipeline
                matA[r % NCU][r / NCU][c] = A[r * lda + c];
            }
        }

        for (int j = 0; j < b; j++) {
            for (int r = 0; r < m; r++) {
#pragma HLS pipeline
                matB[r % NCU][r / NCU] = B[r * ldb + j];
            }

            T dataX[NMAX];
            if (uplo == 'U')
                internal_trtrs::trisolver_U<T, NMAX, NCU>(m, matA, matB, dataX);
            else if (uplo == 'L')
                internal_trtrs::trisolver_L<T, NMAX, NCU>(m, matA, matB, dataX);

            for (int r = 0; r < m; r++) {
#pragma HLS pipeline
                B[r * ldb + j] = dataX[r];
            }
        }
    }
}

} // namespace solver
} // namespace xf
#endif
