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
 * @file polinearsolver.hpp
 * @brief  This files contains implementation of SPD matrix linear solver.
 */

#ifndef _XF_SOLVER_POLINEAR_HPP_
#define _XF_SOLVER_POLINEAR_HPP_

namespace xf {
namespace solver {
namespace internal_polinear {

template <typename T, int N, int NCU>
void chol_col(int n, T dataA[(N + NCU - 1) / NCU][N], T dataj[N], T tmp1_i, int num, int j) {
Loop_per_Unit:
    for (int p = (j + 1) / NCU; p < (n + NCU - 1) / NCU; p++) {
#pragma HLS loop_tripcount min = 1 max = 8
        T tmp_i[16] = {0}, tmp3_i, tmp1[8], tmp2[4], tmp3[2];
#pragma HLS resource variable = tmp_i core = RAM_2P_LUTRAM

    Loop_vec_mul:
        for (int k = 0; k < j; k++) {
#pragma HLS loop_tripcount min = 1 max = N
#pragma HLS pipeline
#pragma HLS dependence variable = tmp_i inter false
#pragma HLS dependence variable = dataA inter false
            tmp_i[k % 16] += dataA[p][k] * dataj[k];
        }

    Loop_add_1:
        for (int j = 0; j < 8; j++) {
#pragma HLS pipeline
            tmp1[j] = tmp_i[j] + tmp_i[j + 8];
        }

    Loop_add_2:
        for (int j = 0; j < 4; j++) {
#pragma HLS pipeline
            tmp2[j] = tmp1[j] + tmp1[j + 4];
        }

    Loop_add_3:
        for (int j = 0; j < 2; j++) {
#pragma HLS pipeline
            tmp3[j] = tmp2[j] + tmp2[j + 2];
        }

        tmp3_i = tmp3[0] + tmp3[1];

        if (p * NCU + num > j) dataA[p][j] = (dataA[p][j] - tmp3_i) / tmp1_i;
    }
}

template <typename T, int N, int NCU>
void chol_jj(T dataA[NCU][(N + NCU - 1) / NCU][N], T dataj[NCU][N], T& tmp1_j, int& j) {
    T tmp[16] = {0}, tmp3_j, tmp1[8], tmp2[4], tmp3[2];
#pragma HLS resource variable = tmp core = RAM_2P_LUTRAM

Loop_vec_mul_jj:
    for (int k = 0; k < j; k++) {
#pragma HLS pipeline
#pragma HLS dependence variable = tmp inter false
#pragma HLS dependence variable = dataA inter false
        T tmp2_j = dataA[j % NCU][j / NCU][k];
        tmp[k % 16] += tmp2_j * tmp2_j;
        for (int p = 0; p < NCU; p++) {
#pragma HLS unroll
            dataj[p][k] = tmp2_j;
        }
    }

Loop_add_1:
    for (int j = 0; j < 8; j++) {
#pragma HLS pipeline
        tmp1[j] = tmp[j] + tmp[j + 8];
    }

Loop_add_2:
    for (int j = 0; j < 4; j++) {
#pragma HLS pipeline
        tmp2[j] = tmp1[j] + tmp1[j + 4];
    }

Loop_add_3:
    for (int j = 0; j < 2; j++) {
#pragma HLS pipeline
        tmp3[j] = tmp2[j] + tmp2[j + 2];
    }

    tmp3_j = tmp3[0] + tmp3[1];

    tmp1_j = sqrt(dataA[j % NCU][j / NCU][j] - tmp3_j);
    dataA[j % NCU][j / NCU][j] = tmp1_j;
}

template <typename T, int N, int NCU>
void chol_col_wrapper(int n, T dataA[NCU][(N + NCU - 1) / NCU][N], T dataj[NCU][N], T tmp1, int j) {
#pragma HLS DATAFLOW

Loop_row:
    for (int num = 0; num < NCU; num++) {
#pragma HLS unroll factor = NCU

        chol_col<T, N, NCU>(n, dataA[num], dataj[num], tmp1, num, j);
    }
}

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
void solver(int n, T dataA[NCU][(N + NCU - 1) / NCU][N], T dataB[NCU][(N + NCU - 1) / NCU], T dataX[N]) {
    T buf[N], buf_i[NCU][(N + NCU - 1) / NCU], buf_o[N];
#pragma HLS array_partition variable = buf_i cyclic factor = NCU

    trisolver_L<T, N, NCU>(n, dataA, dataB, buf);

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
        dataX[i] = buf_o[i];
    }
}

template <typename T, int N, int NCU>
void solver_core(int n, int b, T dataA[NCU][(N + NCU - 1) / NCU][N], T dataB[NCU][(N + NCU - 1) / NCU], T dataX[N]) {
    T tmp1, dataj[NCU][N];
#pragma HLS array_partition variable = dataj cyclic factor = NCU
    if (b == 0) {
    Loop_col:
        for (int j = 0; j < n; ++j) {
            chol_jj<T, N, NCU>(dataA, dataj, tmp1, j);
            chol_col_wrapper<T, N, NCU>(n, dataA, dataj, tmp1, j);
        }

        for (int i = 0; i < (n + NCU - 1) / NCU; i++) {
            for (int k = 0; k < NCU; k++)
                for (int j = i * NCU + k + 1; j < n; j++) {
#pragma HLS dependence variable = dataA inter false
#pragma HLS pipeline
                    if ((i * NCU + k) < n) {
                        dataA[k][i][j] = dataA[j % NCU][j / NCU][i * NCU + k];
                    }
                }
        }
    }

    solver<T, N, NCU>(n, dataA, dataB, dataX);
}

} // namespace internal
/**
 * @brief This function solves a system of linear equation with symmetric
 * positive definite (SPD) matrix along with multiple right-hand side vector \n
 *           \f{equation*} {Ax=B}\f}
 *                     where \f$A\f$ is a dense SPD triangular matrix
 * of size \f$m \times m\f$, \f$x\f$ is a vector need to be computed, and \f$B\f$
 * is input vector.\n
 * The maximum matrix size supported in FPGA is templated by NMAX.
 *
 * @tparam T data type (support float and double)
 * @tparam NMAX maximum number of rows/columns of input matrix
 * @tparam NCU number of computation unit
 * @param[in] n number of rows/cols of matrix A
 * @param[in,out] A input matrix of size \f$n \times n\f$
 * @param[in] b number of columns of matrix B
 * @param[in,out] B input matrix of size \f$b \times n\f$, and overwritten by the output matrix x
 * @param[in] lda leading dimention of input matrix A
 * @param[in] ldb leading dimention of input matrix B
 * @param[out] info output info (unused)
 */

template <typename T, int NMAX, int NCU>
void polinearsolver(int n, T* A, int b, T* B, int lda, int ldb, int& info) {
    if (NMAX == 1)
        B[0] = B[0] / A[0];
    else {
        static T matA[NCU][(NMAX + NCU - 1) / NCU][NMAX];
        static T matB[NCU][(NMAX + NCU - 1) / NCU];
#pragma HLS array_partition variable = matA cyclic factor = NCU
#pragma HLS array_partition variable = matB cyclic factor = NCU
#pragma HLS resource variable = matA core = XPM_MEMORY uram

    Loop_read:
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
#pragma HLS pipeline
                matA[r % NCU][r / NCU][c] = A[r * lda + c];
            }
        }

        for (int j = 0; j < b; j++) {
            for (int r = 0; r < n; r++) {
#pragma HLS pipeline
                matB[r % NCU][r / NCU] = B[r * ldb + j];
            }

            T dataX[NMAX];
            internal_polinear::solver_core<T, NMAX, NCU>(n, j, matA, matB, dataX);

            for (int r = 0; r < n; r++) {
#pragma HLS pipeline
                B[r * ldb + j] = dataX[r];
            }
        }
    }
}

} // namespace solver
} // namespace xf
#endif //#ifndef XF_SOLVENCU_POTNCUF_HPP
