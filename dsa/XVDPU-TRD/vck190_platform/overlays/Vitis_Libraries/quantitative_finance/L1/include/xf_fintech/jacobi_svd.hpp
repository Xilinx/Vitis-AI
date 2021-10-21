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
/**
 * @file jacobi_svd.hpp
 * @brief  This files contains implementation of Jascobi SVD decoompostion
 */
#ifndef XF_FINTECH_SVD_H
#define XF_FINTECH_SVD_H
#include <stdint.h>
#include "ap_fixed.h"
#include "hls_math.h"
namespace xf {
namespace fintech {
namespace internal {
// compare matrix by row or column
template <typename dataType, int N>
void subMax(dataType m[N], dataType& n) {
    for (int i = 0; i < N; ++i) {
        n = (n > m[N]) ? n : m[N];
    }
}

// obtain the max value of matrix
template <typename dataType, int M, int N>
void maxMatrix(dataType m[M][N], dataType& maxValue) {
#pragma HLS inline off
    dataType max1[4]; // need dataflow for the seconde for and third for
Max_Loop2:
    for (int i = 0; i < 4; ++i) {
#pragma HLS unroll
        dataType m1 = hls::abs(m[i][0]);
        dataType m2 = hls::abs(m[i][1]);
        dataType m3 = hls::abs(m[i][2]);
        dataType m4 = hls::abs(m[i][3]);
        dataType m11 = (m1 > m2) ? m1 : m2;
        dataType m22 = (m3 > m4) ? m3 : m4;
        max1[i] = (m11 > m22) ? m11 : m22;
    }
    dataType m11 = (max1[0] > max1[1]) ? max1[0] : max1[1];
    dataType m22 = (max1[2] > max1[3]) ? max1[2] : max1[3];
    maxValue = (m11 > m22) ? m11 : m22;
}

template <typename dataType>
void applyJacobi2x2KJL(dataType& m00, dataType& m01, dataType& m001, dataType& m011, dataType& m_c, dataType& m_s) {
#pragma HLS inline off
    dataType m002, m012, m102, m112;
    dataType tmp11s, tmp12s, tmp21s, tmp22s, tmp11c, tmp12c, tmp21c, tmp22c;
    dataType tmp11s2, tmp12s2, tmp21s2, tmp22s2, tmp11c2, tmp12c2, tmp21c2, tmp22c2;
#pragma HLS RESOURCE variable = tmp11s core = DMul_maxdsp
#pragma HLS RESOURCE variable = tmp12s core = DMul_maxdsp
#pragma HLS RESOURCE variable = tmp11c core = DMul_maxdsp
#pragma HLS RESOURCE variable = tmp12c core = DMul_maxdsp
    tmp11s = m00 * m_s;
    tmp12s = m01 * m_s;
    tmp11c = m00 * m_c;
    tmp12c = m01 * m_c;
#pragma HLS RESOURCE variable = m001 core = DAddSub_nodsp
#pragma HLS RESOURCE variable = m011 core = DAddSub_nodsp
    m001 = tmp11c + tmp12s;
    m011 = tmp12c - tmp11s;
}

// apply 2x2 matrix multiplfication to 1 matrix twice, one left and one right
template <typename dataType, int diagSize>
void applyOnMatrix4x4(dataType dataA[diagSize][diagSize],
                      dataType sigma[diagSize][diagSize],
                      int p,
                      int q,
                      int p2,
                      int q2,
                      dataType m_c_left,
                      dataType m_s_left,
                      dataType m_c_right,
                      dataType m_s_right,
                      dataType m_c_left2,
                      dataType m_s_left2,
                      dataType m_c_right2,
                      dataType m_s_right2) {
    if (diagSize % 2 == 0) {
        dataType dataA2[diagSize][diagSize];
#pragma HLS array_partition variable = dataA2 complete dim = 0
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
#pragma HLS pipeline
                int a = (i == 0) ? p : p2;
                int b = (i == 0) ? q : q2;
                int c = (j == 0) ? p : p2;
                int d = (j == 0) ? q : q2;
                dataType mc;
                dataType ms;
                mc = (j == 0) ? m_c_right : m_c_right2;
                ms = (j == 0) ? m_s_right : m_s_right2;
                dataType m00[2], m01[2], m10[2], m11[2];
                m00[0] = dataA[a][c];
                m01[0] = dataA[a][d];
                m10[0] = dataA[b][c];
                m11[0] = dataA[b][d];
                dataType aa, bb, cc, dd;
                dataType aa1, bb1, cc1, dd1;
                aa = m00[0];
                bb = m01[0];
                aa1 = m10[0];
                bb1 = m11[0];
                applyJacobi2x2KJL(aa, bb, cc, dd, mc, ms);
                applyJacobi2x2KJL(aa1, bb1, cc1, dd1, mc, ms);
                dataA2[a][c] = cc;
                dataA2[a][d] = dd;
                dataA2[b][c] = cc1;
                dataA2[b][d] = dd1;
            }
        }
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
#pragma HLS pipeline
                int a = (i == 0) ? p : p2;
                int b = (i == 0) ? q : q2;
                int c = (j == 0) ? p : p2;
                int d = (j == 0) ? q : q2;
                dataType mc;
                dataType ms;
                mc = (i == 0) ? m_c_left : m_c_left2;
                ms = (i == 0) ? m_s_left : m_s_left2;
                dataType m00[2], m01[2], m10[2], m11[2];
                m00[0] = dataA2[a][c];
                m01[0] = dataA2[a][d];
                m10[0] = dataA2[b][c];
                m11[0] = dataA2[b][d];
                dataType aa, bb, cc, dd;
                dataType aa1, bb1, cc1, dd1;
                aa = m00[0];
                bb = m10[0];
                aa1 = m01[0];
                bb1 = m11[0];
                applyJacobi2x2KJL(aa, bb, cc, dd, mc, ms);
                applyJacobi2x2KJL(aa1, bb1, cc1, dd1, mc, ms);
                sigma[a][c] = cc;
                sigma[a][d] = cc1;
                sigma[b][c] = dd;
                sigma[b][d] = dd1;
            }
        }
    }
}

// apply 2x2 matrix multiplfication to 1 matrix
template <typename dataType, int diagSize>
void applyMatrixRight(dataType dataC[diagSize][diagSize],
                      dataType dataC_out[diagSize][diagSize],
                      int p,
                      int q,
                      int p2,
                      int q2,
                      dataType m_c_right,
                      dataType m_s_right,
                      dataType m_c_right2,
                      dataType m_s_right2) {
#pragma HLS inline off
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
#pragma HLS pipeline
            int a = (i == 0) ? p : p2;
            int b = (i == 0) ? q : q2;
            int c = (j == 0) ? p : p2;
            int d = (j == 0) ? q : q2;
            dataType m_c = (j == 0) ? m_c_right : m_c_right2;
            dataType m_s = (j == 0) ? -m_s_right : -m_s_right2;
            dataType m00, m01;
            dataType m001, m011;
            m00 = dataC[a][c];
            m01 = dataC[a][d];

            applyJacobi2x2KJL(m00, m01, m001, m011, m_c, m_s);
            dataC_out[a][c] = m001;
            dataC_out[a][d] = m011;
        }
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
#pragma HLS pipeline
            int a = (i == 0) ? p : p2;
            int b = (i == 0) ? q : q2;
            int c = (j == 0) ? p : p2;
            int d = (j == 0) ? q : q2;
            dataType m_c = (j == 0) ? m_c_right : m_c_right2;
            dataType m_s = (j == 0) ? -m_s_right : -m_s_right2;
            dataType m10, m11;
            dataType m101, m111;
            m10 = dataC[b][c];
            m11 = dataC[b][d];
            applyJacobi2x2KJL(m10, m11, m101, m111, m_c, m_s);
            dataC_out[b][c] = m101;
            dataC_out[b][d] = m111;
        }
    }
}

// apply 2x2 matrix multiplfication to 2 matrix
template <typename dataType, int diagSize>
void applyMatrixAll(dataType dataU[diagSize][diagSize],
                    dataType dataU_out[diagSize][diagSize],
                    dataType dataV[diagSize][diagSize],
                    dataType dataV_out[diagSize][diagSize],
                    int p,
                    int q,
                    int p2,
                    int q2,
                    dataType m_c_right[diagSize],
                    dataType m_s_right[diagSize],
                    dataType m_c_left[diagSize],
                    dataType m_s_left[diagSize]) {
    dataType dataC[diagSize][diagSize];
    dataType dataC_out[diagSize][diagSize];
#pragma HLS array_partition variable = dataC complete dim = 0
#pragma HLS array_partition variable = dataC_out complete dim = 0
    dataType m_c_1, m_c_2, m_s_1, m_s_2;
    applyMatrixRight<double, 4>(dataU, dataU_out, p, q, p2, q2, m_c_left[0], -m_s_left[0], m_c_left[1], -m_s_left[1]);
}

// copy 2 matrixs
template <typename dataType, int diagSize>
void copyMatrix(dataType dataA[diagSize][diagSize],
                dataType dataA_new[diagSize][diagSize],
                dataType dataB[diagSize][diagSize],
                dataType dataB_new[diagSize][diagSize]) {
    for (int k = 0; k < diagSize; ++k) {
#pragma HLS unroll
        for (int l = 0; l < diagSize; ++l) {
#pragma HLS unroll
            dataA_new[k][l] = dataA[k][l];
        }
    }
}

// copy single matrix
template <typename dataType, int diagSize>
void copyMatrix(dataType dataA[diagSize][diagSize], dataType dataA_new[diagSize][diagSize]) {
    for (int k = 0; k < diagSize; ++k) {
#pragma HLS unroll
        for (int l = 0; l < diagSize; ++l) {
#pragma HLS unroll
            dataA_new[k][l] = dataA[k][l];
        }
    }
}

union double_cast_new {
    double d;
    uint64_t i;
};

// 2x2 Jacobi rotation (core function)
// the calculation process can be found in "Jack Dongarra, Mark Gates, Azzam
// Haidar. The Singular Value Decomposition: Anatomy of Optimizing an Algorithm
// for         Extreme Scale. 2018 SIAM Review, vol.60, No.4, pp.808-865"
template <typename dataType, int diagSize>
void jacobi_rotation_2x2(dataType dataA[diagSize][diagSize],
                         dataType considerAsZero,
                         int p,
                         int q,
                         dataType& m_c_left,
                         dataType& m_s_left,
                         dataType& m_c_right,
                         dataType& m_s_right) {
#pragma HLS inline off
    dataType m00, m01, m10, m11;
    // fetch 2X2 matrix from  matrix A
    m00 = dataA[p][p];
    m01 = dataA[p][q];
    m10 = dataA[q][p];
    m11 = dataA[q][q];
    dataType d;
#pragma HLS RESOURCE variable = d core = DAddSub_nodsp
    d = m00 - m11; // calculate the off-diagonal value
    ap_uint<11> exp1;
    ap_uint<52> sig1;
    union double_cast_new dc;
    // calculate deno = 2*m01
    dc.d = m01;
    ap_uint<64> data = dc.i;
    exp1(10, 0) = data(62, 52);
    exp1 = exp1 + ap_uint<11>(1);
    data(62, 52) = exp1(10, 0);
    dc.i = data;
    dataType deno = dc.d;
    ///////////////////////////
    // calculate KK = 2*abs(m00 - m11)
    dc.d = d;
    data = dc.i;
    exp1(10, 0) = data(62, 52);
    exp1 = exp1 + ap_uint<11>(1);
    data(62, 52) = exp1(10, 0);
    data[63] = 0;
    dc.i = data;
    dataType KK = dc.d;
    ///////////////////////////

    dataType deno2, d2;
#pragma HLS RESOURCE variable = d2 core = DMul_maxdsp
#pragma HLS RESOURCE variable = deno2 core = DMul_maxdsp
    d2 = d * d;          // d2 = (m00 - m11)^2
    deno2 = deno * deno; // deno2 = 4*(m01)^2
    dataType m;
#pragma HLS RESOURCE variable = m core = DAddSub_nodsp
    m = deno2 + d2;                // m = (m00 - m11)^2 + 4*(m01)^2
    dataType sqrtM = hls::sqrt(m); // sqrtM = sqrt((m00-m11)^2 + 4*(m01)^2)
    //////////////////
    // calculate M2
    dc.d = m;
    data = dc.i;
    exp1(10, 0) = data(62, 52);
    exp1 = exp1 + ap_uint<11>(1);
    data(62, 52) = exp1(10, 0);
    dc.i = data;
    dataType M2 = dc.d; // M2 = 2*m
    ////////////////////////////////////
    dataType tmpMul, tmpSum, tmpSub;
#pragma HLS RESOURCE variable = tmpMul core = DMul_maxdsp
    tmpMul = KK * sqrtM; // tmpMul = 2*abs(m00 - m11) * sqrt((m00-m11)^2 + 4*(m01)^2)
#pragma HLS RESOURCE variable = tmpSum core = DAddSub_nodsp
    tmpSum = tmpMul + M2;
    dataType tmpDivider = deno2 / tmpSum;
#pragma HLS RESOURCE variable = tmpSub core = DAddSub_nodsp
    tmpSub = 1 - tmpDivider;
    m_c_right = hls::sqrt(tmpSub);
    dataType tmp = hls::sqrt(tmpDivider);
    m_s_right = (((d > 0) && (deno > 0)) | ((d < 0) && (deno < 0))) ? tmp : -tmp;
    m_c_left = m_c_right;
    m_s_left = m_s_right;
}

// 2X2  two-sided JACOBI ROTATION
// the calculation process can be found in "Jack Dongarra, Mark Gates, Azzam
// Haidar. The Singular Value Decomposition: Anatomy of Optimizing an Algorithm
// for         Extreme Scale. 2018 SIAM Review, vol.60, No.4, pp.808-865"
template <typename dataType, int diagSize>
void jacobi_rotation_2_sided_2x2(dataType dataA[diagSize][diagSize],
                                 dataType considerAsZero,
                                 int p,
                                 int q,
                                 dataType& m_c_left,
                                 dataType& m_s_left,
                                 dataType& m_c_right,
                                 dataType& m_s_right) {
    dataType m00, m01, m10, m11;
    // fetch the 2x2 matrix from the whole matrix A
    m00 = dataA[p][p];
    m01 = dataA[p][q];
    m10 = dataA[q][p];
    m11 = dataA[q][q];
    dataType m_c;
    dataType m_s, d;
#pragma HLS RESOURCE variable = d core = DAddSub_nodsp
    d = m10 - m01;
    ap_uint<1> sign1;
    ap_uint<11> exp1;
    ap_uint<52> sig1;
    union double_cast_new dc;
    dc.d = hls::abs(d);
    ap_uint<64> data = dc.i;
    exp1(10, 0) = data(62, 52);
    sig1(51, 0) = data(51, 0);
    dataType sum;
    dataType tmpSqrt = 0;
    if ((exp1 == ap_uint<11>(1)) && (sig1 < 2)) {
        m_s = 0;
        m_c = 1;
    } else {
        dataType t;
#pragma HLS RESOURCE variable = t core = DAddSub_nodsp
        t = m00 + m11;
        dataType t2, d2; //, divider;
#pragma HLS RESOURCE variable = t2 core = DMul_meddsp
        t2 = t * t;
#pragma HLS RESOURCE variable = d2 core = DMul_meddsp
        d2 = d * d;
#pragma HLS RESOURCE variable = sum core = DAddSub_nodsp
        sum = t2 + d2;
        tmpSqrt = hls::sqrt(sum);
        m_s = d; /// tmpSqrt;//* divider;
        m_c = t; // tmpSqrt;//* divider;

        dataType tmp11s, tmp12s, tmp21s, tmp22s, tmp11c, tmp12c, tmp21c, tmp22c;
#pragma HLS RESOURCE variable = tmp11s core = DMul_meddsp
#pragma HLS RESOURCE variable = tmp12s core = DMul_meddsp
#pragma HLS RESOURCE variable = tmp21s core = DMul_meddsp
#pragma HLS RESOURCE variable = tmp11c core = DMul_meddsp
#pragma HLS RESOURCE variable = tmp21c core = DMul_meddsp
#pragma HLS RESOURCE variable = tmp22c core = DMul_meddsp
        tmp11s = m00 * m_s;
        tmp12s = m01 * m_s;
        tmp21s = m10 * m_s;
        tmp11c = m00 * m_c;
        tmp21c = m10 * m_c;
        tmp22c = m11 * m_c;
#pragma HLS RESOURCE variable = m00 core = DAddSub_nodsp
#pragma HLS RESOURCE variable = m10 core = DAddSub_nodsp
#pragma HLS RESOURCE variable = m11 core = DAddSub_nodsp
        m00 = tmp11c + tmp21s;  // tmp11c-tmp21s;
        m10 = -tmp11s + tmp21c; // tmp11s+tmp21c;
        m11 = -tmp12s + tmp22c; // tmp12s+tmp22c;
    }
    dataType deno;
    dc.d = hls::abs(m10);
    data = dc.i;
    exp1(10, 0) = data(62, 52);
    exp1 = exp1 + ap_uint<11>(1);
    sig1(51, 0) = data(51, 0);
    data(62, 52) = exp1(10, 0);
    dc.i = data;
    deno = dc.d;
    dataType mul;
#pragma HLS RESOURCE variable = mul core = DMul_meddsp
    mul = deno * deno;

    if ((exp1 == ap_uint<11>(1)) && (sig1 < 2)) {
        m_c_right = 1;
        m_s_right = 0;
    } else {
        dataType substrate, sum2, substrate2;
        dataType tmpSum, tmpMul, tmpSum2, tmpMul2, tmpSum3;
        dataType n, n2;
#pragma HLS RESOURCE variable = substrate core = DAddSub_nodsp
        substrate = m00 - m11;
#pragma HLS RESOURCE variable = substrate2 core = DMul_meddsp
        substrate2 = substrate * substrate;

#pragma HLS RESOURCE variable = tmpSum core = DAddSub_nodsp
        tmpSum = substrate2 + mul;
        dataType w1 = hls::sqrt(tmpSum);
        dc.d = substrate;
        data = dc.i;
        sign1[0] = data[63];
        dataType wNew = (sign1[0] == 0) ? w1 : -w1;
#pragma HLS RESOURCE variable = tmpMul core = DMul_meddsp
        tmpMul = substrate * wNew;
#pragma HLS RESOURCE variable = tmpSum3 core = DAddSub_nodsp
        tmpSum3 = substrate + wNew;
#pragma HLS RESOURCE variable = tmpSum2 core = DAddSub_nodsp
        tmpSum2 = tmpSum + tmpMul;

#pragma HLS RESOURCE variable = tmpSum2 core = DMul_meddsp
        tmpSum2 *= sum;
        dc.d = tmpSum2;
        data = dc.i;
        exp1(10, 0) = data(62, 52);
        exp1 = exp1 + ap_uint<11>(1);
        data(62, 52) = exp1(10, 0);
        dc.i = data;
        tmpMul2 = dc.d;
        dataType mm = 1 / hls::sqrt(tmpMul2);
#pragma HLS RESOURCE variable = n core = DMul_meddsp
        n = tmpSum3 * mm;
#pragma HLS RESOURCE variable = n2 core = DMul_meddsp
        n2 = deno * mm;
        ap_int<2> mulTmp2;
        dc.d = substrate;
        data = dc.i;
        sign1[0] = data[63];
        ap_uint<1> sign12;
        union double_cast_new dc2;
        dc2.d = m10;
        ap_uint<64> data2 = dc2.i;
        sign12[0] = data2[63];
        if (((sign1[0] == 0) && (sign12[0] == 0)) || ((sign1[0] == 1) && (sign12[0] == 1))) {
            mulTmp2 = 1;
        } else {
            mulTmp2 = -1;
        }
        m_s_right = (mulTmp2 == 1) ? n2 : -n2; // mulTmp1;
        m_c_right = n;
    }
    dataType productCCRight, productSSRight, productSCRight, productCSRight;
#pragma HLS RESOURCE variable = productCCRight core = DMul_meddsp
#pragma HLS RESOURCE variable = productSSRight core = DMul_meddsp
#pragma HLS RESOURCE variable = productSCRight core = DMul_meddsp
#pragma HLS RESOURCE variable = productCSRight core = DMul_meddsp
    productCCRight = m_c * m_c_right;
    productSSRight = m_s * m_s_right;
    productSCRight = m_s * m_c_right;
    productCSRight = m_c * m_s_right;
#pragma HLS RESOURCE variable = m_c_left core = DAddSub_nodsp
    m_c_left = productCCRight + productSSRight;
#pragma HLS RESOURCE variable = m_s_left core = DAddSub_nodsp
    m_s_left = productSCRight - productCSRight;
#pragma HLS RESOURCE variable = m_c_right core = DMul_meddsp
    m_c_right *= tmpSqrt;
#pragma HLS RESOURCE variable = m_s_right core = DMul_meddsp
    m_s_right *= tmpSqrt;
}

template <typename dataType, int m_diagSize>
void Jacobi_svd(dataType dataA[m_diagSize][m_diagSize],
                dataType sigma[m_diagSize][m_diagSize],
                dataType dataU[m_diagSize][m_diagSize],
                dataType dataV[m_diagSize][m_diagSize],
                dataType dataU_out[m_diagSize][m_diagSize],
                dataType dataV_out[m_diagSize][m_diagSize],
                dataType maxValue) {}

template <>
inline void Jacobi_svd<double, 4>(double dataA[4][4],
                                  double sigma[4][4],
                                  double dataU[4][4],
                                  double dataV[4][4],
                                  double dataU_out[4][4],
                                  double dataV_out[4][4],
                                  double maxValue) {
    double precisionValue = 2.22045e-16 * maxValue;
    double considerAsZero = 2.2250738585072014e-308;
    double threshold = (considerAsZero > precisionValue) ? considerAsZero : precisionValue;
    double considerZero = 2.2250738585072014e-308; // 2.22045e-16;//2.2250738585072014e-308;
    bool finished = false;
#ifndef __SYNTHESIS__
    int counters = 0;
#endif
While_Loop:
    while (!finished) {
#pragma HLS loop_tripcount min = 3 max = 3
        finished = true;
        int P[2][3];
        int Q[2][3];

#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
        std::cout << "while loop:" << counters << std::endl;
        counters++;
#endif
#endif
        ////////// ordering for 4X4 matrix
        P[0][0] = 3;
        Q[0][0] = 2;
        P[1][0] = 1;
        Q[1][0] = 0;
        P[0][1] = 3;
        Q[0][1] = 1;
        P[1][1] = 2;
        Q[1][1] = 0;
        P[0][2] = 2;
        Q[0][2] = 1;
        P[1][2] = 3;
        Q[1][2] = 0;
    Loop_innerWhile:
        for (int i = 0; i < 3; ++i) { // iteration loop for annihilation of matrix A
                                      // to diagonal matrix
#pragma HLS loop_tripcount min = 3 max = 3
            int p1, q1, p2, q2;
            p1 = P[0][i];
            q1 = Q[0][i];
            p2 = P[1][i];
            q2 = Q[1][i];
            double m_c_left[2];
            double m_s_left[2];
            double m_c_right[2];
            double m_s_right[2];
            if (hls::abs(dataA[p1][q1]) > threshold || hls::abs(dataA[q1][p1]) > threshold ||
                hls::abs(dataA[p2][q2]) > threshold || hls::abs(dataA[q2][p2]) > threshold) {
                finished = false;
                for (int j = 0; j < 2; ++j) {
//#pragma HLS unroll
#pragma HLS pipeline
                    jacobi_rotation_2x2<double, 4>(dataA, considerZero, P[j][i], Q[j][i], m_c_left[j], m_s_left[j],
                                                   m_c_right[j],
                                                   m_s_right[j]); // 2x2 Givens Rotation or 2x2 Jacobi method
                }
                applyOnMatrix4x4<double, 4>(dataA, sigma, p1, q1, p2, q2, m_c_left[0], m_s_left[0], m_c_right[0],
                                            m_s_right[0], m_c_left[1], m_s_left[1], m_c_right[1],
                                            m_s_right[1]); // calculate the rotated matrix sigma
                applyMatrixAll<double, 4>(dataU, dataU_out, dataV, dataV_out, P[0][i], Q[0][i], P[1][i], Q[1][i],
                                          m_c_right, m_s_right, m_c_left,
                                          m_s_left); // calculate the new matrix U_out and V_out
                copyMatrix<double, 4>(sigma, dataA); // copy sigma to A in order to continue the iteration
                copyMatrix<double, 4>(dataU_out, dataU, dataV_out,
                                      dataV); // copy matrix U_out and V_out back to matrix U and V
            }
        }
    }
}

} // namespace internal

/**
 * @brief Jacobi Singular Value Decomposition (SVD).
 *
 * @tparam dataType data type.
 * @tparam diagSize matrix size.
 * @param dataA diagSize x diagSize matrix
 * @param sigma2 the decomposed diagonal matrix of dataA
 * @param dataU_out2 the left U matrix of dataA
 * @param dataV_out2 the right V matrix of dataA
 */
template <typename dataType, int diagSize>
void svd(dataType dataA[diagSize][diagSize],
         dataType sigma2[diagSize][diagSize],
         dataType dataU_out2[diagSize][diagSize],
         dataType dataV_out2[diagSize][diagSize]) {
    dataType maxValue = 1;
    dataType scale = 0;
    internal::maxMatrix<dataType, diagSize, diagSize>(dataA, scale); // obtain the max value of matrix A
    dataType dataA2[diagSize][diagSize];
    dataType sigma[diagSize][diagSize];
    dataType dataV[diagSize][diagSize];
    dataType dataV_out[diagSize][diagSize];
    dataType dataU[diagSize][diagSize];
    dataType dataU_out[diagSize][diagSize];
#pragma HLS array_partition variable = dataA2 complete dim = 0
#pragma HLS array_partition variable = dataU complete dim = 0
#pragma HLS array_partition variable = dataV complete dim = 0
#pragma HLS array_partition variable = dataV_out complete dim = 0
#pragma HLS array_partition variable = dataU_out complete dim = 0
#pragma HLS array_partition variable = sigma complete dim = 0
Loop_init_I:
    for (int r = 0; r < diagSize; ++r) {
    Loop_init_J:
        for (int j = 0; j < diagSize; ++j) {
#pragma HLS pipeline
            dataType tmp = dataA[r][j] / scale; // normalization of matrix A in order to avoid overflow
            dataA2[r][j] = tmp;
            sigma[r][j] = tmp;
            sigma2[r][j] = 0;
            if (r == j) {
                dataV[r][j] = 1;
                dataU[r][j] = 1;
                dataV_out[r][j] = 1;
                dataU_out[r][j] = 1;
            } else {
                dataV[r][j] = 0;
                dataU[r][j] = 0;
                dataV_out[r][j] = 0;
                dataU_out[r][j] = 0;
            }
        }
    }
    internal::Jacobi_svd<dataType, diagSize>(dataA2, sigma, dataU, dataV, dataU_out, dataV_out,
                                             maxValue); // core function of Jacbi SVD for symmetric matrix
Loop_postcal:
    for (int i = 0; i < diagSize; ++i) {
        for (int j = 0; j < diagSize; ++j) {
#pragma HLS pipeline
            ap_uint<1> sign1;
            union internal::double_cast_new dc;
            dc.d = sigma[i][i];
            ap_uint<64> data = dc.i;
            sign1[0] = data[63];
            dataU_out2[j][i] =
                (sign1[0] == 1) ? (-dataU_out[j][i]) : (dataU_out[j][i]); // change negative value to positif
            if (j == diagSize - 1) {
                dataType tmp1;
#pragma HLS resource variable = tmp1 core = DMul_meddsp
                tmp1 = sigma[i][i] * scale; // multiply back the max value
                sigma2[i][i] = (sign1[0] == 1) ? (-tmp1) : (tmp1);
            }
        }
    }

    internal::copyMatrix<double, 4>(dataU_out2, dataV_out2); // copy Matrix U to Matrix V
}
} // namespace fintech
} // namespace xf
#endif //#ifndef XF_FINTECH_SVD_H
