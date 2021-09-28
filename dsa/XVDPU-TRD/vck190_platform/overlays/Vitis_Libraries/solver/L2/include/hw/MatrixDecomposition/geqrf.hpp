/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 * See the License for the specific language governing permissions
 * and
 * limitations under the License.
 */

/**
 * @file geqrf.h
 * @brief This file contains QR decomposition of dense matrix.
 *
 * This file is part of XF Solver Library.
 */

#ifndef _XF_SOLVER_GEQRF_H_
#define _XF_SOLVER_GEQRF_H_

#include "hw/math_helper.hpp"

namespace xf {
namespace solver {
namespace internal {

template <typename T>
class Trait {};

template <>
class Trait<double> {
   public:
    static const int eps = 0;
};

template <>
class Trait<float> {
   public:
    static const int eps = 1;
};

/*
 * @brief Update columns in group with offset cIdx
 * @tparam DataType matrix data type
 * @tparam M maximum matrix row count
 * @tparam N maximum matrix column count
 * @tparam K number of compute unit
 * @param matrix in/out matrix columns with offset cIdx
 * @param m actual matrix row count
 * @param n actual matrix column count
 * @param v elementary reflection vector with M copies
 * @param beta factor used in elementary reflection
 * @param cIdx offset of matrix columns
 * @param baseCol column index of elementary reflection
 */
template <typename DataType, int M, int N, int K>
void updateColumns(
    DataType matrix[M][(N + K - 1) / K], int m, int n, DataType v[M], DataType beta, int cIdx, int baseCol) {
    int length = m - baseCol;
    int size = (n + K - 1) / K;
    int cgStart = baseCol / K;
    int cgEnd = (size - 1) * K + cIdx >= n ? size - 1 : size;

loop_update:
    for (int j = cgStart; j < cgEnd; ++j) {
        DataType dotProduct = 0.0;

        DataType tmp[16];
    loop_initTmp:
        for (int idx = 0; idx < 16; ++idx) {
#pragma HLS unroll
            tmp[idx] = 0.0;
        }

    loop_dp:
        for (int k = 0; k < length; ++k) {
#pragma HLS pipeline
            DataType temp = v[k] * matrix[k + baseCol][j];
#pragma HLS RESOURCE variable = temp core = DMul_meddsp
            int index = k % 16;
            tmp[index] = tmp[index] + temp;
        }

        DataType s1[8];
        for (int idx = 0; idx < 8; ++idx) {
#pragma HLS pipeline
            int id = idx << 1;
            DataType temp = tmp[id] + tmp[id + 1];
#pragma HLS RESOURCE variable = temp core = DMul_meddsp
            s1[idx] = temp;
        }

        DataType s2[4];
        for (int idx = 0; idx < 4; ++idx) {
#pragma HLS pipeline
            int id = idx << 1;
            s2[idx] = s1[id] + s1[id + 1];
        }

        DataType s3[2];
        for (int idx = 0; idx < 2; ++idx) {
#pragma HLS pipeline
            int id = idx << 1;
            s3[idx] = s2[id] + s2[id + 1];
        }

        dotProduct = s3[0] + s3[1];

        if (j * K + cIdx > baseCol) {
            DataType coeff = beta * dotProduct;

        loop_element:
            for (int k = 0; k < length; ++k) {
#pragma HLS pipeline
#pragma HLS dependence variable = matrix inter false
#pragma HLS dependence variable = v inter false
                int index = k + baseCol;
                DataType delta = coeff * v[k];
                DataType temp = matrix[index][j] - delta;
                matrix[index][j] = temp;
            }
        }
    }
}

template <typename DataType, int M, int N, int K>
void update(DataType matrix[K][M][(N + K - 1) / K], int m, int n, DataType v[K][M], DataType& beta, int i) {
loop_columns:
    for (int k = 0; k < K; ++k) {
#pragma HLS unroll factor = K
        updateColumns<DataType, M, N, K>(matrix[k], m, n, v[k], beta, k, i);
    }
}

template <typename DataType, int M, int N, int K>
void qrf(int m, int n, DataType matrix[K][M][(N + K - 1) / K], int lda, DataType tau[N]) {
    DataType v[K][M];
#pragma HLS array_partition variable = v cyclic factor = K
    DataType epsilon = 0.00001;
    if (Trait<DataType>::eps == 0) {
        epsilon = 0.00000000000001;
    }

    const int num = m < n ? m : n;
loop_col:
    for (int i = 0; i < num; ++i) { // i: column index
        int cIdx = i % K;
        int cp = i / K;

        // create Householder vector
        const int length = m - i;
        DataType sum[16];

    loop_init:
        for (int j = 0; j < 16; ++j) {
#pragma HLS unroll
            sum[j] = 0.0;
        }

    loop_sum:
        for (int k = 1; k < length; ++k) {
#pragma HLS pipeline
#pragma HLS dependence variable = sum inter false
            sum[k % 16] += matrix[cIdx][k + i][cp] * matrix[cIdx][k + i][cp];
        }

        DataType s1[8];
        for (int idx = 0; idx < 8; ++idx) {
#pragma HLS pipeline
            int id = idx << 1;
            DataType temp = sum[id] + sum[id + 1];
            s1[idx] = temp;
        }

        DataType s2[4];
        for (int idx = 0; idx < 4; ++idx) {
#pragma HLS pipeline
            int id = idx << 1;
            s2[idx] = s1[id] + s1[id + 1];
        }

        DataType s3[2];
        for (int idx = 0; idx < 2; ++idx) {
#pragma HLS pipeline
            int id = idx << 1;
            s3[idx] = s2[id] + s2[id + 1];
        }

        DataType accum1 = s3[0] + s3[1];

        DataType accum = matrix[cIdx][i][cp] * matrix[cIdx][i][cp] + accum1;

        DataType norm = xf::solver::internal::m::sqrt(accum);

        int sign = matrix[cIdx][i][cp] < 0 ? -1 : 1;
        matrix[cIdx][i][cp] -= sign * norm;

        DataType temp = matrix[cIdx][i][cp];
        if (xf::solver::internal::m::fabs(temp) > epsilon) {
        loop_vScale:
            for (int k = 0; k < length; ++k) {
#pragma HLS pipeline
#pragma HLS dependence variable = matrix inter false
                matrix[cIdx][k + i][cp] = matrix[cIdx][k + i][cp] / temp;
            }
        }

    loop_copyV:
        for (int k = 0; k < length; ++k) {
#pragma HLS pipeline
            for (int kk = 0; kk < K; ++kk) {
                v[kk][k] = matrix[cIdx][k + i][cp];
            }
        }

        DataType accumV = 1;
        if (xf::solver::internal::m::fabs(temp) > epsilon) {
            accumV += accum1 / (temp * temp);
        }

        DataType beta = 0.0;
        if (length > 1 && xf::solver::internal::m::fabs(accumV) > epsilon) {
            beta = 2.0 / accumV;
        }

        // store beta
        tau[i] = beta;

        update<DataType, M, N, K>(matrix, m, n, v, beta, i);

        matrix[cIdx][i][cp] = sign * norm;
    }
}

} // namespace internal

/**
 * @brief This function computes QR decomposition of matrix \f$A\f$ \n
   \f{equation*} {A = Q R}\f}
   where \f$A\f$ is a dense matrix of size \f$m \times n\f$, \f$Q\f$
   is a \f$m \times n\f$ matrix with orthonormal columns, and \f$R\f$ is an
   upper triangular matrix.\n
   The maximum matrix size supported in FPGA is templated by NRMAX and NCMAX.
 *
 *
 * @tparam T data type (support float and double)
 * @tparam NRMAX maximum number of rows of input matrix
 * @tparam NCMAX maximum number of columns of input matrix
 * @tparam NCU number of computation unit
 * @param[in] m number of rows of matrix A
 * @param[in] n number of cols of matrix A
 * @param[in,out] A input matrix of size \f$m \times lda\f$, and overwritten by the output triangular R matrix and
 min(m,n) elementary reflectors
 * @param[in] lda leading dimension of matrix A
 * @param[out] tau scalar factors for elementary reflectors
 */
#ifndef __SYNTHESIS__
template <typename T, int NRMAX, int NCMAX, int NCU>
int geqrf(int m, int n, T* A, int lda, T* tau) {
#else
template <typename T, int NRMAX, int NCMAX, int NCU>
int geqrf(int m, int n, T A[NRMAX * NCMAX], int lda, T tau[NCMAX]) {
#endif

    static T data[NCU][NRMAX][(NCMAX + NCU - 1) / NCU];
#pragma HLS resource variable = data core = XPM_MEMORY uram
#pragma HLS array_partition variable = data cyclic factor = NCU

    // read
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
#pragma HLS pipeline
            data[j % NCU][i][j / NCU] = A[i * lda + j];
        }
    }

    xf::solver::internal::qrf<T, NRMAX, NCMAX, NCU>(m, n, data, lda, tau);

    // write
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
#pragma HLS pipeline
            A[i * lda + j] = data[j % NCU][i][j / NCU];
        }
    }

    return 0;
}

template <typename T, int NRMAX, int NCMAX, int NCU>
int geqrf(int m, int n, T A[NRMAX][NCMAX], int lda, T tau[NCMAX]) {
    static T data[NCU][NRMAX][(NCMAX + NCU - 1) / NCU];
#pragma HLS resource variable = data core = XPM_MEMORY uram
#pragma HLS array_partition variable = data complete

    // read
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
#pragma HLS pipeline
            data[j % NCU][i][j / NCU] = A[i][j];
        }
    }

    xf::solver::internal::qrf<T, NRMAX, NCMAX, NCU>(m, n, data, lda, tau);

    // write
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
#pragma HLS pipeline
            A[i][j] = data[j % NCU][i][j / NCU];
        }
    }

    return 0;
}

} // namespace solver
} // namespace xf
#endif
