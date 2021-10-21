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

#ifndef XF_SOLVER_SYEVJ_H
#define XF_SOLVER_SYEVJ_H

namespace xf {
namespace solver {

/**
 * @brief Symmetric Matrix Jacobi based Eigenvalue Decomposition (SYEVJ) .
   \f{equation*} {A U = U \Sigma, }\f}
   where \f$A\f$ is a dense symmetric matrix of size \f$m \times m\f$, \f$U\f$
   is a \f$m \times m\f$ matrix with orthonormal columns, each column of U is the
   eigenvector \f$v_{i}\f$, and \f$\Sigma\f$ is diagonal matrix, which contains
   the eigenvalues \f$\lambda_{i}\f$ of matrix A.\n
   The maximum matrix size supported in FPGA is templated by NMAX.
 *
 * @tparam T data type (support float and double).
 * @tparam NMAX maximum number of rows/columns of input matrix
 * @tparam NCU number of computation unit
 * @tparam m number of rows/cols of matrix A
 * @param A input matrix of size \f$m \times m\f$
 * @param S decomposed diagonal singular matrix of size \f$m \times m\f$
 * @param U left U matrix of SVD
 * @param lda leading dimension of matrix A
 * @param ldu leading dimension of matrix U
 * @param info output info (unused)
 */
#ifndef __SYNTHESIS__
template <typename T, int NMAX, int NCU>
void syevj(int m, T* A, int lda, T* S, T* U, int ldu, int& info) {
#else
template <typename T, int NMAX, int NCU>
void syevj(int m, T A[NMAX * NMAX], int lda, T S[NMAX], T U[NMAX * NMAX], int ldu, int& info) {
#endif
    const int tmpMax = (NMAX + NCU - 1) / NCU;
    const int odd1 = tmpMax % 2;
    const int NMAXUN = (odd1) ? (tmpMax + 1) : tmpMax;
    const int NMAX2 = NMAXUN * NCU;
    int tmpReal = (lda + NCU - 1) / NCU;
    int oddReal = tmpReal % 2;
    int ldaTmp = (oddReal) ? (tmpReal + 1) : tmpReal;
    int ldaReal = ldaTmp * NCU;
// matrix initialization
#ifndef __SYNTHESIS__
    T**** dataA_2D;
    T**** dataU_2D;
    dataA_2D = new T***[NCU];
    dataU_2D = new T***[NCU];
    for (int i = 0; i < NCU; ++i) {
        dataA_2D[i] = new T**[NCU];
        dataU_2D[i] = new T**[NCU];
        for (int j = 0; j < NCU; ++j) {
            dataA_2D[i][j] = new T*[NMAXUN];
            dataU_2D[i][j] = new T*[NMAXUN];
            for (int k = 0; k < NMAXUN; ++k) {
                dataA_2D[i][j][k] = new T[NMAXUN];
                dataU_2D[i][j][k] = new T[NMAXUN];
            }
        }
    }
#else
    T dataA_2D[NCU][NCU][NMAXUN][NMAXUN];
    T dataU_2D[NCU][NCU][NMAXUN][NMAXUN];
#pragma HLS RESOURCE variable = dataA_2D core = RAM_T2P_URAM
#pragma HLS RESOURCE variable = dataU_2D core = RAM_T2P_URAM
#pragma HLS ARRAY_PARTITION variable = dataA_2D
#pragma HLS ARRAY_PARTITION variable = dataU_2D
#pragma HLS ARRAY_PARTITION variable = dataA_2D
#endif

// Matrix transform from 1D to 2D
Loop_init_A:
    for (int i = 0; i < NMAX2; ++i) {
#pragma HLS loop_tripcount min = NMAX2 max = NMAX2
        for (int j = 0; j < NMAX2; ++j) {
#pragma HLS loop_tripcount min = NMAX2 max = NMAX2
#pragma HLS pipeline
            if ((i < m) && (j < lda)) {
                dataA_2D[(i % NCU)][j % NCU][i / NCU][j / NCU] = A[i * lda + j];
                if (i == j) {
                    S[i] = 0;
                }
            } else if (i == j) {
                dataA_2D[(i % NCU)][j % NCU][i / NCU][j / NCU] = 1;
            } else {
                dataA_2D[(i % NCU)][j % NCU][i / NCU][j / NCU] = 0;
            }
        }
    }

    // Calling for svd core function
    xf::solver::gesvdj_2D<T, NMAX2, NCU, NMAXUN>(dataA_2D, dataU_2D, ldaReal);

// Matrix transform from 2D to 1D
Loop_postcal:
    for (int i = 0; i < NMAX2; ++i) {
        for (int j = 0; j < NMAX2; ++j) {
// clang-format off
#pragma HLS loop_tripcount min = NMAX2*NMAX2 max = NMAX2*NMAX2
// clang-format on
#pragma HLS pipeline
            if ((j < lda) && (i < lda)) {
                U[j * ldu + i] = dataU_2D[(j % NCU)][i % NCU][j / NCU][i / NCU];
                if (j == i) {
                    S[i] = dataA_2D[(i % NCU)][i % NCU][i / NCU][i / NCU];
                }
            }
        }
    }

// Delete buffers of c simulation
#ifndef __SYNTHESIS__
    for (int i1 = 0; i1 < NCU; ++i1) {
        for (int i2 = 0; i2 < NCU; ++i2) {
            for (int j = 0; j < NMAXUN; ++j) {
                delete[] dataA_2D[i1][i2][j];
                delete[] dataU_2D[i1][i2][j];
            }
            delete[] dataA_2D[i1][i2];
            delete[] dataU_2D[i1][i2];
        }
        delete[] dataA_2D[i1];
        delete[] dataU_2D[i1];
    }
    delete[] dataA_2D;
    delete[] dataU_2D;
#endif
}
} // namespace solver
} // namespace xf
#endif //#ifndef XF_SOLVER_SYEVJ_H
