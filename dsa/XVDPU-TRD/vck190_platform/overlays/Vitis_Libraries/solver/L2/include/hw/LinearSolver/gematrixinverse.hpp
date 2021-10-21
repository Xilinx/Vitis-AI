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
 * @file gematrixinverse.hpp
 * @brief  This files contains implementation of SPD matrix inverse.
 */

#ifndef _XF_SOLVER_GEMATRIXINVERSE_HPP_
#define _XF_SOLVER_GEMATRIXINVERSE_HPP_

namespace xf {
namespace solver {
namespace internal_gemi {

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

template <typename T, int N, int NCU>
void inverse(int n, int P[N], T dataA[NCU][(N + NCU - 1) / NCU][N], T dataX[N][N]) {
    T dataD[NCU][(N + NCU - 1) / NCU];
#pragma HLS resource variable = dataD core = XPM_MEMORY uram
#pragma HLS array_partition variable = dataD cyclic factor = NCU
    T buf[N], buf_i[NCU][(N + NCU - 1) / NCU], buf_o[N];
#pragma HLS array_partition variable = buf_i cyclic factor = NCU
    for (int c = 0; c < n; c++) {
        for (int i = 0; i < (n + NCU - 1) / NCU; i++) {
            for (int k = 0; k < NCU; k++) {
#pragma HLS pipeline
                if ((i * NCU + k) == P[c])
                    dataD[k][i] = 1;
                else
                    dataD[k][i] = 0;
            }
        }

        trisolver_L<T, N, NCU>(n, dataA, dataD, buf);

        for (int i = 0; i < (n + NCU - 1) / NCU; i++) {
            for (int k = 0; k < NCU; k++) {
#pragma HLS pipeline
                if ((i * NCU + k) < n) {
                    buf_i[k][i] = buf[i * NCU + k];
                } else
                    buf_i[k][i] = 0;
            }
        }

        trisolver_U<T, N, NCU>(n, dataA, buf_i, buf_o);

        for (int i = 0; i < n; i++) {
#pragma HLS pipeline
            dataX[i][c] = buf_o[i];
        }
    }
}

template <typename T, int NRCU, int NMAX, int NCU>
void getrf_core(int n, T A[NCU][NRCU][NMAX], int lda, int P[NMAX]) {
    for (int r = 0; r < n; r++) {
#pragma HLS pipeline
        P[r] = r;
    }
    internalgetrf::getrf_core<T, NRCU, NMAX, NCU>(n, n, A, P, n);
};
template <typename T, int N, int NCU>
void inverse_core(int n, T dataA[NCU][(N + NCU - 1) / NCU][N], T dataX[N][N]) {
    const int NRCU = int((N + NCU - 1) / NCU);
    int P[N];
    int info;
    getrf_core<T, NRCU, N, NCU>(n, dataA, n, P);
    inverse<T, N, NCU>(n, P, dataA, dataX);
}

} // namespace internal
/**
 * @brief This function computes the inverse matrix of \f$A\f$ \n
 *           \f{equation*} {A}^{-1}\f}
 *                     where \f$A\f$ is a dense general
 * matrix of size \f$m \times m\f$.
 * The maximum matrix size supported in FPGA is templated by NMAX.
 *
 * @tparam T data type (support float and double)
 * @tparam NMAX maximum number of rows/columns of input matrix
 * @tparam NCU number of computation unit
 * @param[in] m number of rows/cols of matrix A
 * @param[in,out] A input matrix of size \f$n \times n\f$
 * @param[in] lda leading dimention of input matrix A
 * @param[out] info output info (unused)
 */

template <typename T, int NMAX, int NCU>
void gematrixinverse(int m, T* A, int lda, int& info) {
    if (m == 1)
        A[0] = 1.0 / A[0];
    else {
        static T matA[NCU][(NMAX + NCU - 1) / NCU][NMAX];
#pragma HLS resource variable = matA core = XPM_MEMORY uram
#pragma HLS array_partition variable = matA cyclic factor = NCU

    Loop_read:
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < m; c++) {
#pragma HLS pipeline
                matA[r % NCU][r / NCU][c] = A[r * lda + c];
            }
        }

        static T dataXO[NMAX][NMAX];

        internal_gemi::inverse_core<T, NMAX, NCU>(m, matA, dataXO);

    Loop_write:
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < m; c++) {
#pragma HLS pipeline
                A[r * lda + c] = dataXO[r][c];
            }
        }
    }
}

} // namespace solver
} // namespace xf
#endif
