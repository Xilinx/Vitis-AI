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
 * @file getrf.hpp
 * @brief  This files contains implementation of Cholesky decomposition
 */

#ifndef _XF_SOLVER_GETRF_
#define _XF_SOLVER_GETRF_

#include <hw/math_helper.hpp>

namespace xf {
namespace solver {

namespace internalgetrf {

// update submatrix
template <typename T, int NRCU, int NCMAX>
void subUpdate(T A[NRCU][NCMAX], T rows[NCMAX], T cols[NCMAX], int rs, int re, int cs, int ce) {
    T a00 = rows[cs];

    T Acs[NRCU];

    int nrows = re - rs + 1;
    int ncols = ce - cs;

LoopMulSub:
    for (unsigned int i = 0; i < nrows * ncols; i++) {
#pragma HLS pipeline
#pragma HLS dependence variable = A inter false
// clang-format off
#pragma HLS loop_tripcount min = 1 max = NCMAX*NRCU
        // clang-format on

        int r = i / ncols + rs;
        int c = i % ncols + cs + 1;

        A[r][c] = A[r][c] - cols[r] * rows[c];
    };
}

// core part of getrf (no pivoting)
template <typename T, int NRCU, int NCMAX, int NCU>
void getrf_core(int m, int n, T A[NCU][NRCU][NCMAX], int pivot[NCMAX], int lda) {
LoopSweeps:
    for (int s = 0; s < (m - 1); s++) {
        T rows[NCU][NCMAX];
        T cols[NCU][NCMAX];
#pragma HLS array_partition variable = rows
#pragma HLS array_partition variable = cols
#pragma HLS resource variable = rows core = RAM_2P_BRAM
#pragma HLS resource variable = cols core = RAM_2P_BRAM

        int idscu = s % NCU;
        int idsrow = s / NCU;

        T pmax = -1.0;
        int prow = s;
        int pidcu = 0;
        int pidrow = 0;

    LoopPivot:
        for (int k = s; k < n; k++) {
#pragma HLS pipeline
            int idcu = k % NCU;
            int idrow = k / NCU;
            T absa = xf::solver::internal::m::abs(A[idcu][idrow][s]);
            if (absa > pmax) {
                pmax = absa;
                pidcu = idcu;
                pidrow = idrow;
                prow = k;
            };
        };

        int ptmp = pivot[s];
        pivot[s] = pivot[prow];
        pivot[prow] = ptmp;

    LoopRows:
        for (int k = 0; k < n; k++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 1 max = NCMAX

            for (int r = 0; r < NCU; r++) {
#pragma HLS unroll
                rows[r][k] = A[pidcu][pidrow][k];
            };
            A[pidcu][pidrow][k] = A[idscu][idsrow][k];
            A[idscu][idsrow][k] = rows[0][k];
        };

        T a00 = rows[0][s];

    LoopDiv:
        for (int j = s + 1; j < m; j++) {
#pragma HLS pipeline
#pragma HLS dependence variable = A inter false
#pragma HLS loop_tripcount min = 1 max = NRCU
            int i = j % NCU;
            int r = j / NCU;
            A[i][r][s] = A[i][r][s] / a00;
            cols[i][r] = A[i][r][s];
        };

    LoopMat:
        for (int i = 0; i < NCU; i++) {
#pragma HLS unroll
            int rs, re, cs, ce;
            rs = (i <= (s % NCU)) ? (s / NCU + 1) : (s / NCU);
            re = NRCU - 1;
            cs = s;
            ce = NCMAX - 1;
            subUpdate<T, NRCU, NCMAX>(A[i], rows[i], cols[i], rs, re, cs, ce);
        };
    };
};

}; // end of namespace internalgetrf

/**
 * @brief This function computes the LU decomposition (with partial pivoting) of matrix \f$A\f$ \n
          \f{equation*} {P A = L U}\f}
          where \f$P\f$ is a permutation matrix, \f$A\f$ is a dense matrix of size \f$n \times n\f$, \f$L\f$ is a
   lower triangular matrix with unit diagonal, and \f$U\f$ is an upper triangular matrix. This function does not
 implement
   pivoting.\n
   The maximum matrix size supported in FPGA is templated by NMAX.
 *
 * @tparam T data type (support float and double)
 * @tparam NMAX maximum number of rows/columns of input matrix
 * @tparam NCU number of computation unit
 * @param[in] n number of rows/cols of matrix A
 * @param[in,out] A input matrix, and overwritten by the output upper and lower triangular matrix
 * @param[in] lda leading dimention of input matrix A
 * @param[out] pivot indices, row i of matrix A is stored in row[i]
 * @param[out] info output info (unused)
 */
template <typename T, int NMAX, int NCU>
void getrf(int n, T* A, int lda, int* ipiv, int& info) {
    const int NRCU = int((NMAX + NCU - 1) / NCU);

    T matA[NCU][NRCU][NMAX];
#pragma HLS array_partition variable = matA complete
#pragma HLS resource variable = matA core = XPM_MEMORY uram

LoopRead:
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
#pragma HLS pipeline
            matA[r % NCU][r / NCU][c] = A[lda * r + c];
        };
    };

    int pivot[NMAX];
    for (int r = 0; r < n; r++) {
#pragma HLS pipeline
        pivot[r] = r;
    }

    internalgetrf::getrf_core<T, NRCU, NMAX, NCU>(n, n, matA, pivot, lda);

LoopWrite:
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
#pragma HLS pipeline
            A[r * lda + c] = matA[r % NCU][r / NCU][c];
        };
    };

    for (int r = 0; r < n; r++) {
#pragma HLS pipeline
        ipiv[r] = pivot[r];
    }
};

}; // end of namcespace solver
}; // end of namespace xf

#endif
