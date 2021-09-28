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
 * @file gesvj.hpp
 * @brief  This files contain implementation of SVD using one-sided Jacobi method.
 *
 * This file is part of XF Solver Library.
 */

#ifndef _XF_SOLVER_GESVJ_HPP_
#define _XF_SOLVER_GESVJ_HPP_

#include "ap_fixed.h"
#include "hls_stream.h"
#include "hw/math_helper.hpp"

namespace xf {
namespace solver {
namespace internal {

union double_casting {
    double d;
    uint64_t i;
};

// the 2x2 matrix contains 4 elements
//------beta  gamma-----
//------gamma alpha-----
template <typename T>
void jacobi_rotation_2x2(T alpha, T beta, T gamma, hls::stream<T>& s_strm, hls::stream<T>& c_strm) {
//	double zeta = (beta - alpha) / (2.0 * gamma);
//	double ang_tan = sgn(zeta) / (xf::solver::internal::m::abs(zeta) + xf::solver::internal::m::sqrt(1.0 +
//(zeta*zeta)));//compute tan of angle
//	double ang_cos = 1.0 / (xf::solver::internal::m::sqrt (1.0 + (ang_tan*ang_tan)));       //cos
//	double ang_sin = ang_cos*ang_tan;              // sin

#pragma HLS inline off

#pragma HLS PIPELINE II = 1
    double m00, m01, m11;

    m00 = beta;
    m01 = gamma;
    m11 = alpha;
    double d;
#pragma HLS RESOURCE variable = d core = DAddSub_nodsp
    d = m00 - m11; // calculate the off-diagonal value
    ap_uint<11> exp1;
    ap_uint<52> sig1;
    union double_casting dc;
    // calculate deno = 2*m01
    dc.d = m01;
    ap_uint<64> data = dc.i;
    exp1(10, 0) = data(62, 52);
    exp1 = exp1 + ap_uint<11>(1);
    data(62, 52) = exp1(10, 0);
    dc.i = data;
    double deno = dc.d;
    ///////////////////////////
    // calculate KK = 2*abs(m00 - m11)
    dc.d = d;
    data = dc.i;
    exp1(10, 0) = data(62, 52);
    exp1 = exp1 + ap_uint<11>(1);
    data(62, 52) = exp1(10, 0);
    data[63] = 0;
    dc.i = data;
    double KK = dc.d;
    ///////////////////////////

    double deno2, d2;
#pragma HLS RESOURCE variable = d2 core = DMul_maxdsp
#pragma HLS RESOURCE variable = deno2 core = DMul_maxdsp
    d2 = d * d;          // d2 = (m00 - m11)^2
    deno2 = deno * deno; // deno2 = 4*(m01)^2
    double m;
#pragma HLS RESOURCE variable = m core = DAddSub_nodsp
    m = deno2 + d2;                                  // m = (m00 - m11)^2 + 4*(m01)^2
    double sqrtM = xf::solver::internal::m::sqrt(m); // sqrtM = sqrt((m00-m11)^2 + 4*(m01)^2)
    //////////////////
    // calculate M2
    dc.d = m;
    data = dc.i;
    exp1(10, 0) = data(62, 52);
    exp1 = exp1 + ap_uint<11>(1);
    data(62, 52) = exp1(10, 0);
    dc.i = data;
    double M2 = dc.d; // M2 = 2*m
    ////////////////////////////////////
    double tmpMul, tmpSum, tmpSub;
#pragma HLS RESOURCE variable = tmpMul core = DMul_maxdsp
    tmpMul = KK * sqrtM; // tmpMul = 2*abs(m00 - m11) * sqrt((m00-m11)^2 + 4*(m01)^2)
#pragma HLS RESOURCE variable = tmpSum core = DAddSub_nodsp
    tmpSum = tmpMul + M2;
    double tmpDivider = deno2 / tmpSum;
#pragma HLS RESOURCE variable = tmpSub core = DAddSub_nodsp
    tmpSub = 1 - tmpDivider;
    T c_right = xf::solver::internal::m::sqrt(tmpSub);
    double tmp = xf::solver::internal::m::sqrt(tmpDivider);
    T s_right = (((d > 0) && (deno > 0)) | ((d < 0) && (deno < 0))) ? tmp : -tmp;

    s_strm.write(s_right);
    c_strm.write(c_right);
}
//! calc the converge of next sweep
template <typename T>
void calc_converge(T alpha, T beta, T gamma, hls::stream<T>& conv_strm) {
    T converge =
        xf::solver::internal::m::abs(gamma) / xf::solver::internal::m::sqrt(alpha * beta); // compute convergence
    conv_strm.write(converge);
}

template <typename T>
void svd_and_conv(T alpha, T beta, T gamma, hls::stream<T>& conv_strm, hls::stream<T>& s_strm, hls::stream<T>& c_strm) {
#pragma HLS DATAFLOW
    jacobi_rotation_2x2<T>(alpha, beta, gamma, s_strm, c_strm);
    calc_converge(alpha, beta, gamma, conv_strm);
}

template <typename T, int NRMAX, int NCMAX, int MCU, int ACUM>
void update_A(
    T matA[MCU][ACUM][NCMAX], T A_i[MCU][ACUM], T A_j[MCU][ACUM], int m, int n, int col_i, int col_j, T s, T c) {
#pragma HLS inline off
UPDATE_A:
    for (int k = 0; k < ACUM; k++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = NRMAX max = NRMAX
        for (int itm = 0; itm < MCU; itm++) {
#pragma HLS UNROLL
            T tki = A_i[itm][k];
            T tkj = A_j[itm][k];
            matA[itm][k][col_i] = c * tki - s * tkj; // matA[k][i] = c*tki - s*matA[k][j];
            matA[itm][k][col_j] = s * tki + c * tkj; // matA[k][j] = s*tki + c*matA[k][j];
        }
    }
}

template <typename T, int NCMAX, int NCU, int ACUN>
void update_V(T matV[NCU][ACUN][NCMAX], T V_i[NCU][ACUN], T V_j[NCU][ACUN], int n, int col_i, int col_j, T s, T c) {
#pragma HLS inline off
CALC_V:
    // for(int k=0; k<n; k++){
    for (int k = 0; k < ACUN; k++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = ACUN max = ACUN
        for (int itn = 0; itn < NCU; itn++) {
#pragma HLS UNROLL
            T tki = V_i[itn][k];
            T tkj = V_j[itn][k];
            matV[itn][k][col_i] = c * tki - s * tkj; // matV[k][i] = c*tki - s*matV[k][j];
            matV[itn][k][col_j] = s * tki + c * tkj; // matV[k][j] = s*tki + c*matV[k][j];
        }
    }
}

template <typename T, int NRMAX, int NCMAX, int MCU, int ACUM, int NCU, int ACUN>
void update_AV(T matA[MCU][ACUM][NCMAX],
               T matV[NCU][ACUN][NCMAX],
               T A_i[MCU][ACUM],
               T A_j[MCU][ACUM],
               T V_i[NCU][ACUN],
               T V_j[NCU][ACUN],
               int m,
               int n,
               int col_i,
               int col_j,
               T s,
               T c) {
#pragma HLS DATAFLOW
    update_A<T, NRMAX, NCMAX, MCU, ACUM>(matA, A_i, A_j, m, n, col_i, col_j, s, c);
    update_V<T, NCMAX, NCU, ACUN>(matV, V_i, V_j, n, col_i, col_j, s, c);
}

//! Read two columns of A into two seperate Bram
template <typename T, int NRMAX, int NCMAX, int MCU, int ACUM>
void read_and_gen_2x2(T matA[MCU][ACUM][NCMAX],
                      T A_i[MCU][ACUM],
                      T A_j[MCU][ACUM],
                      int m,
                      int n,
                      int col_i,
                      int col_j,
                      hls::stream<T>& alpha_strm,
                      hls::stream<T>& beta_strm,
                      hls::stream<T>& gamma_strm) {
#pragma HLS inline off
    T alpha = 0;
    T beta = 0;
    T gamma = 0;

    const int DEP = 16;
    // used for accumulate alpha*alpha, beta*beta, gamma*gamma
    T alpha_acc[MCU][DEP];
#pragma HLS resource variable = alpha_acc core = RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable = alpha_acc complete
    T beta_acc[MCU][DEP];
#pragma HLS resource variable = beta_acc core = RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable = beta_acc complete
    T gamma_acc[MCU][DEP];
#pragma HLS resource variable = gamma_acc core = RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable = gamma_acc complete

    T alpha_sum[DEP];
    T beta_sum[DEP];
    T gamma_sum[DEP];

INIT_ACC:
    for (int t = 0; t < DEP; t++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 8
        for (int itm = 0; itm < MCU; itm++) {
#pragma HLS UNROLL
            alpha_acc[itm][t] = 0;
            beta_acc[itm][t] = 0;
            gamma_acc[itm][t] = 0;
        }

        alpha_sum[t] = 0;
        beta_sum[t] = 0;
        gamma_sum[t] = 0;
    }

CALC_ELEMENTS:
    for (int k = 0; k < ACUM; k++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = ACUM max = ACUM
#pragma HLS dependence variable = alpha_acc inter false
#pragma HLS dependence variable = beta_acc inter false
#pragma HLS dependence variable = gamma_acc inter false
        for (int itm = 0; itm < MCU; itm++) {
#pragma HLS UNROLL
            if (k * MCU + itm < m) {
                T Aki = matA[itm][k][col_i];
                T Akj = matA[itm][k][col_j];
                A_i[itm][k] = Aki; // store to extra bram, so no need to read URAM again when updating A
                A_j[itm][k] = Akj;
                alpha_acc[itm][k % DEP] += Aki * Aki;
                beta_acc[itm][k % DEP] += Akj * Akj;
                gamma_acc[itm][k % DEP] += Aki * Akj;
            }
        }
    }

    ap_uint<4> idx = 0;
ACCU:
    for (int k = 0; k < DEP; k++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 8
#pragma HLS PIPELINE II = MCU
        for (int itm = 0; itm < MCU; itm++) {
#pragma HLS LOOP_TRIPCOUNT min = MCU max = MCU
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = alpha_acc inter false
#pragma HLS dependence variable = beta_acc inter false
#pragma HLS dependence variable = gamma_acc inter false
            alpha_sum[idx] += alpha_acc[itm][k];
            beta_sum[idx] += beta_acc[itm][k];
            gamma_sum[idx] += gamma_acc[itm][k];
            idx++;
        }
    }

    // sum 16 data to 8
    T alpha_sum_tmp0[8];
    T beta_sum_tmp0[8];
    T gamma_sum_tmp0[8];
    for (int k = 0; k < 8; k++) {
#pragma HLS PIPELINE
        alpha_sum_tmp0[k] = alpha_sum[2 * k] + alpha_sum[2 * k + 1];
        beta_sum_tmp0[k] = beta_sum[2 * k] + beta_sum[2 * k + 1];
        gamma_sum_tmp0[k] = gamma_sum[2 * k] + gamma_sum[2 * k + 1];
    }

    // sum 8 data to 4
    T alpha_sum_tmp1[4];
    T beta_sum_tmp1[4];
    T gamma_sum_tmp1[4];
    for (int k = 0; k < 4; k++) {
#pragma HLS PIPELINE
        alpha_sum_tmp1[k] = alpha_sum_tmp0[2 * k] + alpha_sum_tmp0[2 * k + 1];
        beta_sum_tmp1[k] = beta_sum_tmp0[2 * k] + beta_sum_tmp0[2 * k + 1];
        gamma_sum_tmp1[k] = gamma_sum_tmp0[2 * k] + gamma_sum_tmp0[2 * k + 1];
    }
    // sum 4 data to 2
    T alpha_sum_tmp2[2];
    T beta_sum_tmp2[2];
    T gamma_sum_tmp2[2];
    for (int k = 0; k < 2; k++) {
#pragma HLS PIPELINE
        alpha_sum_tmp2[k] = alpha_sum_tmp1[2 * k] + alpha_sum_tmp1[2 * k + 1];
        beta_sum_tmp2[k] = beta_sum_tmp1[2 * k] + beta_sum_tmp1[2 * k + 1];
        gamma_sum_tmp2[k] = gamma_sum_tmp1[2 * k] + gamma_sum_tmp1[2 * k + 1];
    }
    // sum 2 data to 1
    alpha = alpha_sum_tmp2[0] + alpha_sum_tmp2[1];
    beta = beta_sum_tmp2[0] + beta_sum_tmp2[1];
    gamma = gamma_sum_tmp2[0] + gamma_sum_tmp2[1];

    alpha_strm.write(alpha);
    beta_strm.write(beta);
    gamma_strm.write(gamma);
}

//! Read two columns (i and j) of V into two seperate Bram V_i[N] and V_j[N]
template <typename T, int NCMAX, int NCU, int ACUN>
void read_V_2cols(T matV[NCU][ACUN][NCMAX], T V_i[NCU][ACUN], T V_j[NCU][ACUN], int n, int col_i, int col_j) {
#pragma HLS inline off
    for (int k = 0; k < ACUN; k++) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = ACUN max = ACUN
        for (int itn = 0; itn < NCU; itn++) {
#pragma HLS UNROLL
            if (k * NCU + itn < n) {
                V_i[itn][k] = matV[itn][k][col_i];
                V_j[itn][k] = matV[itn][k][col_j];
            }
        }
    }
}

//! read 2 columns(i and j) of data from A matrix and V matrix
template <typename T, int NRMAX, int NCMAX, int MCU, int ACUM, int NCU, int ACUN>
void read_to_2cols(T matA[MCU][ACUM][NCMAX],
                   T matV[NCU][ACUN][NCMAX],
                   T A_i[MCU][ACUM],
                   T A_j[MCU][ACUM],
                   T V_i[NCU][ACUN],
                   T V_j[NCU][ACUN],
                   int m,
                   int n,
                   int col_i,
                   int col_j,
                   hls::stream<T>& alpha_strm,
                   hls::stream<T>& beta_strm,
                   hls::stream<T>& gamma_strm) {
#pragma HLS DATAFLOW
    read_and_gen_2x2<T, NRMAX, NCMAX, MCU, ACUM>(matA, A_i, A_j, m, n, col_i, col_j, alpha_strm, beta_strm, gamma_strm);
    read_V_2cols<T, NCMAX, NCU, ACUN>(matV, V_i, V_j, n, col_i, col_j);
}

} // end of namespace internal

/**
 * @brief This function implements singular value decomposition of matrix A using one-sided Jacobi algorihtm.
   \f{equation*} {A = U \Sigma {V}^T}\f}
   where \f$A\f$ is a dense matrix of size \f$m \times n\f$, \f$U\f$ is
   \f$m \times m\f$ matrix with orthonormal columns, \f$V\f$ is \f$n \times n\f$
   matrix with orthonormal columns, and \f$\Sigma\f$ is diagonal matrix.\n
   The maximum matrix size supported in FPGA is templated by NCMAX, NRMAX.
 *
 * @tparam T: the data type of gesvj
 * @tparam NRMAX maximum number of rows of input matrix
 * @tparam NCMAX maximum number of columns of input matrix
 * @tparam MCU number of computation unit of M
 * @tparam NCU number of computation unit of N
 * @param m number of rows of matrix A
 * @param n number of cols of matrix A
 * @param A input matrix of size \f$m \times n\f$
 * @param S decomposed diagonal singular matrix of size n
 * @param U left U matrix of SVD of size \f$m \times m\f$
 * @param V right V matrix of SVD \f$n \times n\f$
 **/
template <typename T, int NRMAX, int NCMAX, int MCU, int NCU>
void gesvj(int m, int n, T* A, T* U, T* S, T* V) {
    // num of elements in each MCU
    const int ACUM = (NRMAX + MCU - 1) / MCU;
    // num of elements in each NCU
    const int ACUN = (NCMAX + NCU - 1) / NCU;

    T matA[MCU][ACUM][NCMAX];
#pragma HLS RESOURCE variable = matA core = RAM_T2P_URAM
#pragma HLS ARRAY_PARTITION variable = matA
    T matU[NRMAX][NRMAX];
#pragma HLS RESOURCE variable = matU core = RAM_T2P_URAM
    T matV[NCU][ACUN][NCMAX];
#pragma HLS RESOURCE variable = matV core = RAM_T2P_URAM
#pragma HLS ARRAY_PARTITION variable = matV
    T A_i[MCU][ACUM];
#pragma HLS RESOURCE variable = A_i core = RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable = A_i
    T A_j[MCU][ACUM];
#pragma HLS RESOURCE variable = A_j core = RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable = A_j
    T V_i[NCU][ACUN];
#pragma HLS RESOURCE variable = V_i core = RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable = V_i
    T V_j[NCU][ACUN];
#pragma HLS RESOURCE variable = V_j core = RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable = V_j

    T sigma[NCMAX];
#pragma HLS RESOURCE variable = sigma core = RAM_S2P_BRAM

INIT_S:
    for (int j = 0; j < n; j++) {
#pragma HLS LOOP_TRIPCOUNT min = NCMAX max = NCMAX
#pragma HLS PIPELINE II = 1
        sigma[j] = 0.0;
    }

INIT_U:
    for (int i = 0; i < m; i++) {
#pragma HLS LOOP_TRIPCOUNT min = NRMAX max = NRMAX
        for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = NRMAX max = NRMAX
            if (i == j) {
                matU[i][j] = 1.0;
            } else {
                matU[i][j] = 0.0;
            }
        }
    }
// initialize V matrix to diagonal
INIT_V:
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = NCMAX max = NCMAX
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = NCMAX max = NCMAX
            if (i == j) {
                matV[i % NCU][i / NCU][j] = 1.0;
            } else {
                matV[i % NCU][i / NCU][j] = 0.0;
            }
        }
    }

// read A from DDR
READ_A:
    for (int i = 0; i < m; i++) {
#pragma HLS LOOP_TRIPCOUNT min = NRMAX max = NRMAX
        for (int j = 0; j < n; j++) {
#pragma HLS LOOP_TRIPCOUNT min = NCMAX max = NCMAX
#pragma HLS PIPELINE II = 1
            matA[i % MCU][i / MCU][j] = A[i * n + j];
#ifndef __SYNTHESIS__
#ifdef __SOLVER_DEBUG__
            std::cout << matA[i % MCU][i / MCU][j] << " ";
#endif
#endif
        }
#ifndef __SYNTHESIS__
#ifdef __SOLVER_DEBUG__
        std::cout << std::endl;
#endif
#endif
    }

    hls::stream<T> alpha_strm;
#pragma HLS STREAM variable = alpha_strm depth = 16
    hls::stream<T> beta_strm;
#pragma HLS STREAM variable = beta_strm depth = 16
    hls::stream<T> gamma_strm;
#pragma HLS STREAM variable = gamma_strm depth = 16
    hls::stream<T> s_strm;
#pragma HLS STREAM variable = s_strm depth = 16
    hls::stream<T> c_strm;
#pragma HLS STREAM variable = c_strm depth = 16
    hls::stream<T> conv_strm;
#pragma HLS STREAM variable = conv_strm depth = 16

    T converge = 1.0;

    T epsilon = 1.e-8;
    if (sizeof(T) == sizeof(float)) {
        epsilon = 1.e-7;
    }

    int sweep_loop = 0;

CONV_WHILE:
    while (converge > epsilon) {
#pragma HLS LOOP_TRIPCOUNT min = 5 max = 15
        converge = 0.0;
        sweep_loop++;

    LOOP_COLS:
        for (int i = 1; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = NCMAX max = NCMAX
        LOOP_i:
            for (int j = 0; j < i; j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min = NCMAX/2 max = NCMAX/2
                // clang-format on
                internal::read_to_2cols<T, NRMAX, NCMAX, MCU, ACUM, NCU, ACUN>(matA, matV, A_i, A_j, V_i, V_j, m, n, i,
                                                                               j, alpha_strm, beta_strm, gamma_strm);

                T alpha = alpha_strm.read();
                T beta = beta_strm.read();
                T gamma = gamma_strm.read();
                internal::svd_and_conv(alpha, beta, gamma, conv_strm, s_strm, c_strm);

                T s = s_strm.read();
                T c = c_strm.read();
                T conv_tmp = conv_strm.read();
                if (c == 1 && s == 0) {
                    conv_tmp = 0;
                }
                converge = hls::max(converge, conv_tmp);

                internal::update_AV<T, NRMAX, NCMAX, MCU, ACUM, NCU, ACUN>(matA, matV, A_i, A_j, V_i, V_j, m, n, i, j,
                                                                           s, c);
            }
        }
    }

    // calculate the sigma diagonal matrix
    const int DEP = 16;
    T accu_s = 0;
    T AUS_accu[DEP];
    int jj = 0;
CALC_US:
    for (int j = 0; j < n; j++) {
#pragma HLS LOOP_TRIPCOUNT min = NCMAX max = NCMAX
        accu_s = 0;
        for (int t = 0; t < DEP; t++) {
#pragma HLS UNROLL
            AUS_accu[t] = 0;
        }
        for (int i = 0; i < m; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = NRMAX max = NRMAX
            T Aij = matA[i % MCU][i / MCU][j];
            AUS_accu[i % DEP] += Aij * Aij;
        }
        // for (int t = 0; t < DEP; t++) {
        //    accu_s += AUS_accu[t];
        //}
        T AUS_accu_tmp0[8];
        for (int t = 0; t < 8; t++) {
#pragma HLS PIPELINE
            AUS_accu_tmp0[t] = AUS_accu[2 * t] + AUS_accu[2 * t + 1];
        }
        // add 8 -> 4
        T AUS_accu_tmp1[4];
        for (int t = 0; t < 4; t++) {
#pragma HLS PIPELINE
            AUS_accu_tmp1[t] = AUS_accu_tmp0[2 * t] + AUS_accu_tmp0[2 * t + 1];
        }
        // add 4 -> 2
        T AUS_accu_tmp2[2];
        for (int t = 0; t < 2; t++) {
#pragma HLS PIPELINE
            AUS_accu_tmp2[t] = AUS_accu_tmp1[2 * t] + AUS_accu_tmp1[2 * t + 1];
        }
        // add 2 -> 1
        accu_s = AUS_accu_tmp2[0] + AUS_accu_tmp2[1];

        accu_s = xf::solver::internal::m::sqrt(accu_s);
#ifdef __SOLVER_DEBUG__
        std::cout << "accu_s = " << std::setprecision(15) << accu_s << std::endl;
#endif

        sigma[j] = accu_s;
        // when accu_s == 0, skip this column with initialized U_i
        if (accu_s > 1.e-15) {
            for (int i = 0; i < m; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = NRMAX max = NRMAX
                matU[i][jj] = matA[i % MCU][i / MCU][j] / accu_s;
            }
            jj++;
        }
    }

//================output================
// output S, U, V and store
OUT_U:
    for (int i = 0; i < m; i++) {
#pragma HLS LOOP_TRIPCOUNT min = NRMAX max = NRMAX
        for (int j = 0; j < m; j++) {
#pragma HLS LOOP_TRIPCOUNT min = NRMAX max = NRMAX
#pragma HLS PIPELINE II = 1
            U[i * m + j] = matU[i][j];
        }
    }

OUT_S:
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = NCMAX max = NCMAX
        S[i] = sigma[i];
    }

OUT_V:
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = NCMAX max = NCMAX
        for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = NCMAX max = NCMAX
            V[i * n + j] = matV[i % NCU][i / NCU][j];
        }
    }

#ifndef __SYNTHESIS__
#ifdef __SOLVER_DEBUG__
    std::cout << "sweep iterations = " << sweep_loop << "--------" << std::endl;
#endif
#endif
}

} // end of namespace sovler
} // end of namespace xf
#endif //#ifndef _XF_SOLVER_GESVJ_HPP_
