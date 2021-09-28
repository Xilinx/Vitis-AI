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
 * @file jacobi_svd.h
 * @brief  This files contains implementation of Jascobi SVD decoompostion
 */
#ifndef XF_SOLVER_SVDJ_H
#define XF_SOLVER_SVDJ_H

#include "ap_fixed.h"
#include "hw/math_helper.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace solver {
namespace internal {
union double_cast_new {
    double d;
    uint64_t i;
};

// 2x2 Jacobi rotation (core function)
// the calculation process can be found in "Jack Dongarra, Mark Gates, Azzam Haidar. The Singular Value Decomposition:
// Anatomy of Optimizing an Algorithm for         Extreme Scale. 2018 SIAM Review, vol.60, No.4, pp.808-865"
template <typename T, int diagSize>
void jacobi_rotation_2x2(T matrix[3], T considerAsZero, T& m_c_left, T& m_s_left, T& m_c_right, T& m_s_right) {
#pragma HLS inline off
    T m00, m01, m11;
    // fetch 2X2 matrix from  matrix A
    m00 = matrix[0];
    m01 = matrix[1];
    m11 = matrix[2];
    if ((m00 == 1) && (m11 == 1) && (m01 == 0)) {
        m_c_left = 1;
        m_s_left = 0;
        m_c_right = 1;
        m_s_right = 0;
    } else {
        T d;
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
        T deno = dc.d;

        // calculate KK = 2*abs(m00 - m11)
        dc.d = d;
        data = dc.i;
        exp1(10, 0) = data(62, 52);
        exp1 = exp1 + ap_uint<11>(1);
        data(62, 52) = exp1(10, 0);
        data[63] = 0;
        dc.i = data;
        T KK = dc.d;

        T deno2, d2;
#pragma HLS RESOURCE variable = d2 core = DMul_maxdsp
#pragma HLS RESOURCE variable = deno2 core = DMul_maxdsp
        d2 = d * d;          // d2 = (m00 - m11)^2
        deno2 = deno * deno; // deno2 = 4*(m01)^2
        T m;
#pragma HLS RESOURCE variable = m core = DAddSub_nodsp
        m = deno2 + d2;                             // m = (m00 - m11)^2 + 4*(m01)^2
        T sqrtM = xf::solver::internal::m::sqrt(m); // sqrtM = sqrt((m00-m11)^2 + 4*(m01)^2)

        // calculate M2
        dc.d = m;
        data = dc.i;
        exp1(10, 0) = data(62, 52);
        exp1 = exp1 + ap_uint<11>(1);
        data(62, 52) = exp1(10, 0);
        dc.i = data;
        T M2 = dc.d; // M2 = 2*m

        T tmpMul, tmpSum, tmpSub;
#pragma HLS RESOURCE variable = tmpMul core = DMul_maxdsp
        tmpMul = KK * sqrtM; // tmpMul = 2*abs(m00 - m11) * sqrt((m00-m11)^2 + 4*(m01)^2)
#pragma HLS RESOURCE variable = tmpSum core = DAddSub_nodsp
        tmpSum = tmpMul + M2;
        T tmpDivider = deno2 / tmpSum;
#pragma HLS RESOURCE variable = tmpSub core = DAddSub_nodsp
        tmpSub = 1 - tmpDivider;
        m_c_right = xf::solver::internal::m::sqrt(tmpSub);
        T tmp = xf::solver::internal::m::sqrt(tmpDivider);
        m_s_right = (((d > 0) && (deno > 0)) | ((d < 0) && (deno < 0))) ? tmp : -tmp;
        m_c_left = m_c_right;
        m_s_left = m_s_right;
    }
}

template <typename DT>
void swapTwo(DT& a, DT& b) {
    DT c = a;
    a = b;
    b = c;
}

#ifndef __SYNTHESIS__
template <int maxDim, bool odd>
void GenBlockMat(int dim, int** order) {
#else
template <int maxDim, bool odd>
void GenBlockMat(int dim, int order[maxDim][maxDim]) {
#endif
#ifndef __SYNTHESIS__
    int* tmpOrder = new int[maxDim];
#else
    int tmpOrder[maxDim];
#endif

#pragma HLS resource variable = tmpOrder core = RAM_T2P_BRAM
    int dim_1 = dim - 1;
    int dim_1_2 = dim_1 >> 1; //(dim - 1)/ 2
    int dim_2 = dim >> 1;     //(dim)/ 2
    int dim_3 = maxDim >> 1;  //(dim)/ 2
    for (int i = 0; i < dim; ++i) {
#pragma HLS loop_tripcount min = 512 max = 512
#pragma HLS pipeline
        tmpOrder[i] = dim_1 - i;
    }
Loop_even:
    for (int i = 0; i < dim_1; ++i) {
#pragma HLS loop_tripcount min = 511 max = 511
    Loop_precal:
        for (int j = 0; j < dim_3; ++j) {
#pragma HLS loop_tripcount min = 256 max = 256
#pragma HLS pipeline II = 1
            int counter1 = j << 1; // for even steps
            int counter2 = counter1 + 1;
            if (j < dim_2) {
                int tmpCounter1 = counter1;
                int tmpCounter2 = counter2;
                int tmp1 = tmpOrder[tmpCounter1];
                int tmp2 = tmpOrder[tmpCounter2];
                order[i][counter1] = tmp1;
                order[i][counter2] = tmp2;
            } else {
                order[i][counter1] = maxDim;
                order[i][counter2] = maxDim;
            }
        }
        int exchangeNm = ((int)(i / 2)) * 2; // rolling
        swapTwo<int>(tmpOrder[exchangeNm], tmpOrder[exchangeNm + 1]);
        int tmp3 = tmpOrder[1];
    Loop_cal:
        for (int j = 1; j < dim - 2; j = j + 2) {
#pragma HLS loop_tripcount min = 510 max = 510
#pragma HLS pipeline II = 1
            int tmp4 = tmpOrder[j + 2];
            tmpOrder[j + 2] = tmp3;
            tmp3 = tmp4;
        }
        tmpOrder[1] = tmp3;
    }
//    }
#ifndef __SYNTHESIS__
    delete[] tmpOrder;
#endif
}

#ifndef __SYNTHESIS__
template <typename T, int m_diagSize, int UN, int NMAXUN>
void unrollCol(int lda,
               int Dim_inner_extend,
               int* Order,
               T* m_c_right,
               T* m_s_right,
               T** dataA1,
               T** dataA2,
               T** dataA3,
               T** dataA4,
               T** dataA5,
               T** dataA6,
               T** dataA7,
               T** dataA8,
               T** dataA9,
               T** dataA10,
               T** dataA11,
               T** dataA12,
               T** dataA13,
               T** dataA14,
               T** dataA15,
               T** dataA16) {
#else
template <typename T, int m_diagSize, int UN, int NMAXUN>
void unrollCol(int lda,
               int Dim_inner_extend,
               int Order[m_diagSize],
               T m_c_right[m_diagSize],
               T m_s_right[m_diagSize],
               T dataA1[NMAXUN][NMAXUN],
               T dataA2[NMAXUN][NMAXUN],
               T dataA3[NMAXUN][NMAXUN],
               T dataA4[NMAXUN][NMAXUN],
               T dataA5[NMAXUN][NMAXUN],
               T dataA6[NMAXUN][NMAXUN],
               T dataA7[NMAXUN][NMAXUN],
               T dataA8[NMAXUN][NMAXUN],
               T dataA9[NMAXUN][NMAXUN],
               T dataA10[NMAXUN][NMAXUN],
               T dataA11[NMAXUN][NMAXUN],
               T dataA12[NMAXUN][NMAXUN],
               T dataA13[NMAXUN][NMAXUN],
               T dataA14[NMAXUN][NMAXUN],
               T dataA15[NMAXUN][NMAXUN],
               T dataA16[NMAXUN][NMAXUN]) {
#endif
    for (int f = 0; f < UN; ++f) { // row
    Loop_Inner_Mul33:
        for (int l = 0; l < NMAXUN / 2; ++l) {
        Loop_Inner_Mul333:
            for (int r = 0; r < NMAXUN / 2; ++r) {
#pragma HLS pipeline II = 4
#pragma HLS dependence variable = dataA1 inter false
#pragma HLS dependence variable = dataA2 inter false
#pragma HLS dependence variable = dataA3 inter false
#pragma HLS dependence variable = dataA4 inter false
#pragma HLS dependence variable = dataA5 inter false
#pragma HLS dependence variable = dataA6 inter false
#pragma HLS dependence variable = dataA7 inter false
#pragma HLS dependence variable = dataA8 inter false
#pragma HLS dependence variable = dataA9 inter false
#pragma HLS dependence variable = dataA10 inter false
#pragma HLS dependence variable = dataA11 inter false
#pragma HLS dependence variable = dataA12 inter false
#pragma HLS dependence variable = dataA13 inter false
#pragma HLS dependence variable = dataA14 inter false
#pragma HLS dependence variable = dataA15 inter false
#pragma HLS dependence variable = dataA16 inter false
                int p1, q1, p2, q2;
                T cr, sr, cl, sl;
                p2 = (l << 1) * UN; // 2*l
                q2 = p2 + UN;
                int tmp = r << 1; // 2*r
                int tmpG = f * NMAXUN;
                int tmpSum = tmpG + tmp;
                int tmpDivide2 = tmpG >> 1; // f*NMAXUN/2
                int tmpSumCS = tmpDivide2 + r;
                p1 = Order[tmpSum];     // row
                q1 = Order[tmpSum + 1]; // row
                if ((p1 < lda) && (p2 < lda) && (q1 < lda) && (q2 < lda) && (p1 >= 0) && (p2 >= 0) && (q1 >= 0) &&
                    (q2 >= 0) && (p1 != q1)) {
                    cl = m_c_right[tmpSumCS]; // g*NMAXUN/2+l
                    sl = m_s_right[tmpSumCS]; // g*NMAXUN/2+l
                    T a00, a01, a10, a11;
                    T m00, m01, m10, m11;
                    int tmpP1 = p1 % UN;
                    int tmpQ1 = q1 % UN;
                    int num1 = p1 / UN;
                    int num2 = q1 / UN;
                    int num3 = p2 / UN;
                    int num4 = q2 / UN;
                    if (tmpP1 == 0) {
                        m00 = dataA1[num1][num3];
                        m01 = dataA1[num1][num4];
                    } else if (tmpP1 == 1) {
                        m00 = dataA2[num1][num3];
                        m01 = dataA2[num1][num4];
                    } else if (tmpP1 == 2) {
                        m00 = dataA3[num1][num3];
                        m01 = dataA3[num1][num4];
                    } else if (tmpP1 == 3) {
                        m00 = dataA4[num1][num3];
                        m01 = dataA4[num1][num4];
                    } else if (tmpP1 == 4) {
                        m00 = dataA5[num1][num3];
                        m01 = dataA5[num1][num4];
                    } else if (tmpP1 == 5) {
                        m00 = dataA6[num1][num3];
                        m01 = dataA6[num1][num4];
                    } else if (tmpP1 == 6) {
                        m00 = dataA7[num1][num3];
                        m01 = dataA7[num1][num4];
                    } else if (tmpP1 == 7) {
                        m00 = dataA8[num1][num3];
                        m01 = dataA8[num1][num4];
                    } else if (tmpP1 == 8) {
                        m00 = dataA9[num1][num3];
                        m01 = dataA9[num1][num4];
                    } else if (tmpP1 == 9) {
                        m00 = dataA10[num1][num3];
                        m01 = dataA10[num1][num4];
                    } else if (tmpP1 == 10) {
                        m00 = dataA11[num1][num3];
                        m01 = dataA11[num1][num4];
                    } else if (tmpP1 == 11) {
                        m00 = dataA12[num1][num3];
                        m01 = dataA12[num1][num4];
                    } else if (tmpP1 == 12) {
                        m00 = dataA13[num1][num3];
                        m01 = dataA13[num1][num4];
                    } else if (tmpP1 == 13) {
                        m00 = dataA14[num1][num3];
                        m01 = dataA14[num1][num4];
                    } else if (tmpP1 == 14) {
                        m00 = dataA15[num1][num3];
                        m01 = dataA15[num1][num4];
                    } else if (tmpP1 == 15) {
                        m00 = dataA16[num1][num3];
                        m01 = dataA16[num1][num4];
                    }
                    if (tmpQ1 == 0) {
                        m10 = dataA1[num2][num3];
                        m11 = dataA1[num2][num4];
                    } else if (tmpQ1 == 1) {
                        m10 = dataA2[num2][num3];
                        m11 = dataA2[num2][num4];
                    } else if (tmpQ1 == 2) {
                        m10 = dataA3[num2][num3];
                        m11 = dataA3[num2][num4];
                    } else if (tmpQ1 == 3) {
                        m10 = dataA4[num2][num3];
                        m11 = dataA4[num2][num4];
                    } else if (tmpQ1 == 4) {
                        m10 = dataA5[num2][num3];
                        m11 = dataA5[num2][num4];
                    } else if (tmpQ1 == 5) {
                        m10 = dataA6[num2][num3];
                        m11 = dataA6[num2][num4];
                    } else if (tmpQ1 == 6) {
                        m10 = dataA7[num2][num3];
                        m11 = dataA7[num2][num4];
                    } else if (tmpQ1 == 7) {
                        m10 = dataA8[num2][num3];
                        m11 = dataA8[num2][num4];
                    } else if (tmpQ1 == 8) {
                        m10 = dataA9[num2][num3];
                        m11 = dataA9[num2][num4];
                    } else if (tmpQ1 == 9) {
                        m10 = dataA10[num2][num3];
                        m11 = dataA10[num2][num4];
                    } else if (tmpQ1 == 10) {
                        m10 = dataA11[num2][num3];
                        m11 = dataA11[num2][num4];
                    } else if (tmpQ1 == 11) {
                        m10 = dataA12[num2][num3];
                        m11 = dataA12[num2][num4];
                    } else if (tmpQ1 == 12) {
                        m10 = dataA13[num2][num3];
                        m11 = dataA13[num2][num4];
                    } else if (tmpQ1 == 13) {
                        m10 = dataA14[num2][num3];
                        m11 = dataA14[num2][num4];
                    } else if (tmpQ1 == 14) {
                        m10 = dataA15[num2][num3];
                        m11 = dataA15[num2][num4];
                    } else if (tmpQ1 == 15) {
                        m10 = dataA16[num2][num3];
                        m11 = dataA16[num2][num4];
                    }
                Loop_Inner_Mul3333:
                    for (int ttt = 0; ttt < 4; ++ttt) {
#pragma HLS pipeline II = 1
                        T m0, m1;
                        if (ttt == 0) {
                            m0 = m00;
                            m1 = m10;
                        } else if (ttt == 1) {
                            m0 = m10;
                            m1 = -m00;
                        } else if (ttt == 2) {
                            m0 = m01;
                            m1 = m11;
                        } else if (ttt == 3) {
                            m0 = m11;
                            m1 = -m01;
                        }
                        T tmpMul0, tmpMul1, sum;
#pragma HLS RESOURCE variable = tmpMul0 core = DMul_maxdsp
#pragma HLS RESOURCE variable = tmpMul1 core = DMul_maxdsp
#pragma HLS RESOURCE variable = sum core = DAddSub_nodsp
                        tmpMul0 = m0 * cl;
                        tmpMul1 = m1 * sl;
                        sum = tmpMul0 + tmpMul1;
                        if (ttt == 0) {
                            a00 = sum;
                        } else if (ttt == 1) {
                            a10 = sum;
                        } else if (ttt == 2) {
                            a01 = sum;
                        } else if (ttt == 3) {
                            a11 = sum;
                        }
                    }
                    if (tmpP1 == 0) {
                        dataA1[num1][num3] = a00;
                        dataA1[num1][num4] = a01;
                    } else if (tmpP1 == 1) {
                        dataA2[num1][num3] = a00;
                        dataA2[num1][num4] = a01;
                    } else if (tmpP1 == 2) {
                        dataA3[num1][num3] = a00;
                        dataA3[num1][num4] = a01;
                    } else if (tmpP1 == 3) {
                        dataA4[num1][num3] = a00;
                        dataA4[num1][num4] = a01;
                    } else if (tmpP1 == 4) {
                        dataA5[num1][num3] = a00;
                        dataA5[num1][num4] = a01;
                    } else if (tmpP1 == 5) {
                        dataA6[num1][num3] = a00;
                        dataA6[num1][num4] = a01;
                    } else if (tmpP1 == 6) {
                        dataA7[num1][num3] = a00;
                        dataA7[num1][num4] = a01;
                    } else if (tmpP1 == 7) {
                        dataA8[num1][num3] = a00;
                        dataA8[num1][num4] = a01;
                    } else if (tmpP1 == 8) {
                        dataA9[num1][num3] = a00;
                        dataA9[num1][num4] = a01;
                    } else if (tmpP1 == 9) {
                        dataA10[num1][num3] = a00;
                        dataA10[num1][num4] = a01;
                    } else if (tmpP1 == 10) {
                        dataA11[num1][num3] = a00;
                        dataA11[num1][num4] = a01;
                    } else if (tmpP1 == 11) {
                        dataA12[num1][num3] = a00;
                        dataA12[num1][num4] = a01;
                    } else if (tmpP1 == 12) {
                        dataA13[num1][num3] = a00;
                        dataA13[num1][num4] = a01;
                    } else if (tmpP1 == 13) {
                        dataA14[num1][num3] = a00;
                        dataA14[num1][num4] = a01;
                    } else if (tmpP1 == 14) {
                        dataA15[num1][num3] = a00;
                        dataA15[num1][num4] = a01;
                    } else if (tmpP1 == 15) {
                        dataA16[num1][num3] = a00;
                        dataA16[num1][num4] = a01;
                    }
                    if (tmpQ1 == 0) {
                        dataA1[num2][num3] = a10;
                        dataA1[num2][num4] = a11;
                    } else if (tmpQ1 == 1) {
                        dataA2[num2][num3] = a10;
                        dataA2[num2][num4] = a11;
                    } else if (tmpQ1 == 2) {
                        dataA3[num2][num3] = a10;
                        dataA3[num2][num4] = a11;
                    } else if (tmpQ1 == 3) {
                        dataA4[num2][num3] = a10;
                        dataA4[num2][num4] = a11;
                    } else if (tmpQ1 == 4) {
                        dataA5[num2][num3] = a10;
                        dataA5[num2][num4] = a11;
                    } else if (tmpQ1 == 5) {
                        dataA6[num2][num3] = a10;
                        dataA6[num2][num4] = a11;
                    } else if (tmpQ1 == 6) {
                        dataA7[num2][num3] = a10;
                        dataA7[num2][num4] = a11;
                    } else if (tmpQ1 == 7) {
                        dataA8[num2][num3] = a10;
                        dataA8[num2][num4] = a11;
                    } else if (tmpQ1 == 8) {
                        dataA9[num2][num3] = a10;
                        dataA9[num2][num4] = a11;
                    } else if (tmpQ1 == 9) {
                        dataA10[num2][num3] = a10;
                        dataA10[num2][num4] = a11;
                    } else if (tmpQ1 == 10) {
                        dataA11[num2][num3] = a10;
                        dataA11[num2][num4] = a11;
                    } else if (tmpQ1 == 11) {
                        dataA12[num2][num3] = a10;
                        dataA12[num2][num4] = a11;
                    } else if (tmpQ1 == 12) {
                        dataA13[num2][num3] = a10;
                        dataA13[num2][num4] = a11;
                    } else if (tmpQ1 == 13) {
                        dataA14[num2][num3] = a10;
                        dataA14[num2][num4] = a11;
                    } else if (tmpQ1 == 14) {
                        dataA15[num2][num3] = a10;
                        dataA15[num2][num4] = a11;
                    } else if (tmpQ1 == 15) {
                        dataA16[num2][num3] = a10;
                        dataA16[num2][num4] = a11;
                    }
                }
            }
        }
    }
}

#ifndef __SYNTHESIS__
template <typename T, int m_diagSize, int UN, int NMAXUN>
void unrollRow(int lda,
               int Dim_inner_extend,
               int* Order,
               T* m_c_right,
               T* m_s_right,
               T** dataA1,
               T** dataA2,
               T** dataA3,
               T** dataA4,
               T** dataA5,
               T** dataA6,
               T** dataA7,
               T** dataA8,
               T** dataA9,
               T** dataA10,
               T** dataA11,
               T** dataA12,
               T** dataA13,
               T** dataA14,
               T** dataA15,
               T** dataA16) {
#else
template <typename T, int m_diagSize, int UN, int NMAXUN>
void unrollRow(int lda,
               int Dim_inner_extend,
               int Order[m_diagSize],
               T m_c_right[m_diagSize],
               T m_s_right[m_diagSize],
               T dataA1[NMAXUN][NMAXUN],
               T dataA2[NMAXUN][NMAXUN],
               T dataA3[NMAXUN][NMAXUN],
               T dataA4[NMAXUN][NMAXUN],
               T dataA5[NMAXUN][NMAXUN],
               T dataA6[NMAXUN][NMAXUN],
               T dataA7[NMAXUN][NMAXUN],
               T dataA8[NMAXUN][NMAXUN],
               T dataA9[NMAXUN][NMAXUN],
               T dataA10[NMAXUN][NMAXUN],
               T dataA11[NMAXUN][NMAXUN],
               T dataA12[NMAXUN][NMAXUN],
               T dataA13[NMAXUN][NMAXUN],
               T dataA14[NMAXUN][NMAXUN],
               T dataA15[NMAXUN][NMAXUN],
               T dataA16[NMAXUN][NMAXUN]) {
#endif
    for (int g = 0; g < UN; ++g) { // col
    Loop_Inner_Mul:
        for (int r = 0; r < NMAXUN / 2; ++r) {
        Loop_Inner_Mul111:
            for (int l = 0; l < NMAXUN / 2; ++l) {
#pragma HLS pipeline II = 4
#pragma HLS dependence variable = dataA1 inter false
#pragma HLS dependence variable = dataA2 inter false
#pragma HLS dependence variable = dataA3 inter false
#pragma HLS dependence variable = dataA4 inter false
#pragma HLS dependence variable = dataA5 inter false
#pragma HLS dependence variable = dataA6 inter false
#pragma HLS dependence variable = dataA7 inter false
#pragma HLS dependence variable = dataA8 inter false
#pragma HLS dependence variable = dataA9 inter false
#pragma HLS dependence variable = dataA10 inter false
#pragma HLS dependence variable = dataA11 inter false
#pragma HLS dependence variable = dataA12 inter false
#pragma HLS dependence variable = dataA13 inter false
#pragma HLS dependence variable = dataA14 inter false
#pragma HLS dependence variable = dataA15 inter false
#pragma HLS dependence variable = dataA16 inter false
                int p1, q1, p2, q2;
                T cr, sr, cl, sl;
                p1 = (r << 1) * UN; // 2*r
                q1 = p1 + UN;
                int tmp = l << 1; // 2*l
                int tmpG = g * NMAXUN;
                int tmpSum = tmpG + tmp;
                int tmpDivide2 = tmpG >> 1; // g*NMAXUN/2
                int tmpSumCS = tmpDivide2 + l;
                p2 = Order[tmpSum];     // row
                q2 = Order[tmpSum + 1]; // row
                if ((p1 < lda) && (p2 < lda) && (q1 < lda) && (q2 < lda) && (p1 >= 0) && (p2 >= 0) && (q1 >= 0) &&
                    (q2 >= 0) && (p2 != q2)) {
                    cr = m_c_right[tmpSumCS]; // g*NMAXUN/2+l
                    sr = m_s_right[tmpSumCS]; // g*NMAXUN/2+l
                    T a00, a01, a10, a11;
                    T m00, m01, m10, m11;
                    int tmpP2 = p2 % UN;
                    int tmpQ2 = q2 % UN;
                    int num1 = p1 / UN;
                    int num2 = q1 / UN;
                    int num3 = p2 / UN;
                    int num4 = q2 / UN;
                    if (tmpP2 == 0) {
                        m00 = dataA1[num1][num3];
                        m10 = dataA1[num2][num3];
                    } else if (tmpP2 == 1) {
                        m00 = dataA2[num1][num3];
                        m10 = dataA2[num2][num3];
                    } else if (tmpP2 == 2) {
                        m00 = dataA3[num1][num3];
                        m10 = dataA3[num2][num3];
                    } else if (tmpP2 == 3) {
                        m00 = dataA4[num1][num3];
                        m10 = dataA4[num2][num3];
                    } else if (tmpP2 == 4) {
                        m00 = dataA5[num1][num3];
                        m10 = dataA5[num2][num3];
                    } else if (tmpP2 == 5) {
                        m00 = dataA6[num1][num3];
                        m10 = dataA6[num2][num3];
                    } else if (tmpP2 == 6) {
                        m00 = dataA7[num1][num3];
                        m10 = dataA7[num2][num3];
                    } else if (tmpP2 == 7) {
                        m00 = dataA8[num1][num3];
                        m10 = dataA8[num2][num3];
                    } else if (tmpP2 == 8) {
                        m00 = dataA9[num1][num3];
                        m10 = dataA9[num2][num3];
                    } else if (tmpP2 == 9) {
                        m00 = dataA10[num1][num3];
                        m10 = dataA10[num2][num3];
                    } else if (tmpP2 == 10) {
                        m00 = dataA11[num1][num3];
                        m10 = dataA11[num2][num3];
                    } else if (tmpP2 == 11) {
                        m00 = dataA12[num1][num3];
                        m10 = dataA12[num2][num3];
                    } else if (tmpP2 == 12) {
                        m00 = dataA13[num1][num3];
                        m10 = dataA13[num2][num3];
                    } else if (tmpP2 == 13) {
                        m00 = dataA14[num1][num3];
                        m10 = dataA14[num2][num3];
                    } else if (tmpP2 == 14) {
                        m00 = dataA15[num1][num3];
                        m10 = dataA15[num2][num3];
                    } else if (tmpP2 == 15) {
                        m00 = dataA16[num1][num3];
                        m10 = dataA16[num2][num3];
                    }
                    if (tmpQ2 == 0) {
                        m01 = dataA1[num1][num4];
                        m11 = dataA1[num2][num4];
                    } else if (tmpQ2 == 1) {
                        m01 = dataA2[num1][num4];
                        m11 = dataA2[num2][num4];
                    } else if (tmpQ2 == 2) {
                        m01 = dataA3[num1][num4];
                        m11 = dataA3[num2][num4];
                    } else if (tmpQ2 == 3) {
                        m01 = dataA4[num1][num4];
                        m11 = dataA4[num2][num4];
                    } else if (tmpQ2 == 4) {
                        m01 = dataA5[num1][num4];
                        m11 = dataA5[num2][num4];
                    } else if (tmpQ2 == 5) {
                        m01 = dataA6[num1][num4];
                        m11 = dataA6[num2][num4];
                    } else if (tmpQ2 == 6) {
                        m01 = dataA7[num1][num4];
                        m11 = dataA7[num2][num4];
                    } else if (tmpQ2 == 7) {
                        m01 = dataA8[num1][num4];
                        m11 = dataA8[num2][num4];
                    } else if (tmpQ2 == 8) {
                        m01 = dataA9[num1][num4];
                        m11 = dataA9[num2][num4];
                    } else if (tmpQ2 == 9) {
                        m01 = dataA10[num1][num4];
                        m11 = dataA10[num2][num4];
                    } else if (tmpQ2 == 10) {
                        m01 = dataA11[num1][num4];
                        m11 = dataA11[num2][num4];
                    } else if (tmpQ2 == 11) {
                        m01 = dataA12[num1][num4];
                        m11 = dataA12[num2][num4];
                    } else if (tmpQ2 == 12) {
                        m01 = dataA13[num1][num4];
                        m11 = dataA13[num2][num4];
                    } else if (tmpQ2 == 13) {
                        m01 = dataA14[num1][num4];
                        m11 = dataA14[num2][num4];
                    } else if (tmpQ2 == 14) {
                        m01 = dataA15[num1][num4];
                        m11 = dataA15[num2][num4];
                    } else if (tmpQ2 == 15) {
                        m01 = dataA16[num1][num4];
                        m11 = dataA16[num2][num4];
                    }
                Loop_Inner_Mul1111:
                    for (int ttt = 0; ttt < 4; ++ttt) {
#pragma HLS pipeline II = 1
                        T m0, m1;
                        if (ttt == 0) {
                            m0 = m00;
                            m1 = m01;
                        } else if (ttt == 1) {
                            m0 = m01;
                            m1 = -m00;
                        } else if (ttt == 2) {
                            m0 = m10;
                            m1 = m11;
                        } else if (ttt == 3) {
                            m0 = m11;
                            m1 = -m10;
                        }
                        T tmpMul0, tmpMul1, sum;
#pragma HLS RESOURCE variable = tmpMul0 core = DMul_maxdsp
#pragma HLS RESOURCE variable = tmpMul1 core = DMul_maxdsp
#pragma HLS RESOURCE variable = sum core = DAddSub_nodsp
                        tmpMul0 = m0 * cr;
                        tmpMul1 = m1 * sr;
                        sum = tmpMul0 + tmpMul1;
                        if (ttt == 0) {
                            a00 = sum;
                        } else if (ttt == 1) {
                            a01 = sum;
                        } else if (ttt == 2) {
                            a10 = sum;
                        } else if (ttt == 3) {
                            a11 = sum;
                        }
                    }
                    if (tmpP2 == 0) {
                        dataA1[num1][num3] = a00;
                        dataA1[num2][num3] = a10;
                    } else if (tmpP2 == 1) {
                        dataA2[num1][num3] = a00;
                        dataA2[num2][num3] = a10;
                    } else if (tmpP2 == 2) {
                        dataA3[num1][num3] = a00;
                        dataA3[num2][num3] = a10;
                    } else if (tmpP2 == 3) {
                        dataA4[num1][num3] = a00;
                        dataA4[num2][num3] = a10;
                    } else if (tmpP2 == 4) {
                        dataA5[num1][num3] = a00;
                        dataA5[num2][num3] = a10;
                    } else if (tmpP2 == 5) {
                        dataA6[num1][num3] = a00;
                        dataA6[num2][num3] = a10;
                    } else if (tmpP2 == 6) {
                        dataA7[num1][num3] = a00;
                        dataA7[num2][num3] = a10;
                    } else if (tmpP2 == 7) {
                        dataA8[num1][num3] = a00;
                        dataA8[num2][num3] = a10;
                    } else if (tmpP2 == 8) {
                        dataA9[num1][num3] = a00;
                        dataA9[num2][num3] = a10;
                    } else if (tmpP2 == 9) {
                        dataA10[num1][num3] = a00;
                        dataA10[num2][num3] = a10;
                    } else if (tmpP2 == 10) {
                        dataA11[num1][num3] = a00;
                        dataA11[num2][num3] = a10;
                    } else if (tmpP2 == 11) {
                        dataA12[num1][num3] = a00;
                        dataA12[num2][num3] = a10;
                    } else if (tmpP2 == 12) {
                        dataA13[num1][num3] = a00;
                        dataA13[num2][num3] = a10;
                    } else if (tmpP2 == 13) {
                        dataA14[num1][num3] = a00;
                        dataA14[num2][num3] = a10;
                    } else if (tmpP2 == 14) {
                        dataA15[num1][num3] = a00;
                        dataA15[num2][num3] = a10;
                    } else if (tmpP2 == 15) {
                        dataA16[num1][num3] = a00;
                        dataA16[num2][num3] = a10;
                    }
                    if (tmpQ2 == 0) {
                        dataA1[num1][num4] = a01;
                        dataA1[num2][num4] = a11;
                    } else if (tmpQ2 == 1) {
                        dataA2[num1][num4] = a01;
                        dataA2[num2][num4] = a11;
                    } else if (tmpQ2 == 2) {
                        dataA3[num1][num4] = a01;
                        dataA3[num2][num4] = a11;
                    } else if (tmpQ2 == 3) {
                        dataA4[num1][num4] = a01;
                        dataA4[num2][num4] = a11;
                    } else if (tmpQ2 == 4) {
                        dataA5[num1][num4] = a01;
                        dataA5[num2][num4] = a11;
                    } else if (tmpQ2 == 5) {
                        dataA6[num1][num4] = a01;
                        dataA6[num2][num4] = a11;
                    } else if (tmpQ2 == 6) {
                        dataA7[num1][num4] = a01;
                        dataA7[num2][num4] = a11;
                    } else if (tmpQ2 == 7) {
                        dataA8[num1][num4] = a01;
                        dataA8[num2][num4] = a11;
                    } else if (tmpQ2 == 8) {
                        dataA9[num1][num4] = a01;
                        dataA9[num2][num4] = a11;
                    } else if (tmpQ2 == 9) {
                        dataA10[num1][num4] = a01;
                        dataA10[num2][num4] = a11;
                    } else if (tmpQ2 == 10) {
                        dataA11[num1][num4] = a01;
                        dataA11[num2][num4] = a11;
                    } else if (tmpQ2 == 11) {
                        dataA12[num1][num4] = a01;
                        dataA12[num2][num4] = a11;
                    } else if (tmpQ2 == 12) {
                        dataA13[num1][num4] = a01;
                        dataA13[num2][num4] = a11;
                    } else if (tmpQ2 == 13) {
                        dataA14[num1][num4] = a01;
                        dataA14[num2][num4] = a11;
                    } else if (tmpQ2 == 14) {
                        dataA15[num1][num4] = a01;
                        dataA15[num2][num4] = a11;
                    } else if (tmpQ2 == 15) {
                        dataA16[num1][num4] = a01;
                        dataA16[num2][num4] = a11;
                    }
                }
            }
        }
    }
}

#ifndef __SYNTHESIS__
template <typename T, int m_diagSize, int UN, int NMAXUN>
void funcDataflow(int i,
                  int Dim_inner_extend,
                  int Dim_inner,
                  int** Order,
                  T* m_c_right,
                  T* m_s_right,
                  T**** dataA,
                  T**** dataU_out,
                  int lda) {
#else
template <typename T, int m_diagSize, int UN, int NMAXUN>
void funcDataflow(int i,
                  int Dim_inner_extend,
                  int Dim_inner,
                  int Order[m_diagSize][m_diagSize],
                  T m_c_right[m_diagSize],
                  T m_s_right[m_diagSize],
                  T dataA[UN][UN][NMAXUN][NMAXUN],
                  T dataU_out[UN][UN][NMAXUN][NMAXUN],
                  int lda) {
#endif
#pragma HLS inline off
    int Order1[3 * UN][m_diagSize];
    T m_c_right1[3 * UN][m_diagSize];
    T m_s_right1[3 * UN][m_diagSize];
    int tmp[16];
#pragma HLS array_partition variable = tmp complete
#pragma HLS array_partition variable = Order1
#pragma HLS array_partition variable = m_c_right1
#pragma HLS array_partition variable = m_s_right1
    for (int i = 0; i < 16; ++i) {
#pragma HLS loop_tripcount min = 16 max = 16
        if (i < UN) {
            tmp[i] = i;
        } else {
            tmp[i] = UN - 1;
        }
    }
    for (int k = 0; k < 3 * UN; ++k) {
// clang-format off
#pragma HLS loop_tripcount min = 3*UN max = 3*UN
        // clang-format on
        for (int j = 0; j < m_diagSize; ++j) {
#pragma HLS loop_tripcount min = 512 max = 512
#pragma HLS pipeline
            Order1[k][j] = Order[i][j];
            m_c_right1[k][j] = m_c_right[j];
            m_s_right1[k][j] = m_s_right[j];
        }
    }
    for (int i = 0; i < UN; ++i) {
#pragma HLS loop_tripcount min = UN max = UN
#pragma HLS unroll
        unrollRow<T, m_diagSize, UN, NMAXUN>(
            lda, Dim_inner_extend, Order1[i], m_c_right1[i], m_s_right1[i], dataA[tmp[i]][tmp[0]],
            dataA[tmp[i]][tmp[1]], dataA[tmp[i]][tmp[2]], dataA[tmp[i]][tmp[3]], dataA[tmp[i]][tmp[4]],
            dataA[tmp[i]][tmp[5]], dataA[tmp[i]][tmp[6]], dataA[tmp[i]][tmp[7]], dataA[tmp[i]][tmp[8]],
            dataA[tmp[i]][tmp[9]], dataA[tmp[i]][tmp[10]], dataA[tmp[i]][tmp[11]], dataA[tmp[i]][tmp[12]],
            dataA[tmp[i]][tmp[13]], dataA[tmp[i]][tmp[14]], dataA[tmp[i]][tmp[15]]);
    }
    for (int i = 0; i < UN; ++i) {
#pragma HLS loop_tripcount min = UN max = UN
#pragma HLS unroll
        unrollRow<T, m_diagSize, UN, NMAXUN>(
            lda, Dim_inner_extend, Order1[i + UN], m_c_right1[i + UN], m_s_right1[i + UN], dataU_out[tmp[i]][tmp[0]],
            dataU_out[tmp[i]][tmp[1]], dataU_out[tmp[i]][tmp[2]], dataU_out[tmp[i]][tmp[3]], dataU_out[tmp[i]][tmp[4]],
            dataU_out[tmp[i]][tmp[5]], dataU_out[tmp[i]][tmp[6]], dataU_out[tmp[i]][tmp[7]], dataU_out[tmp[i]][tmp[8]],
            dataU_out[tmp[i]][tmp[9]], dataU_out[tmp[i]][tmp[10]], dataU_out[tmp[i]][tmp[11]],
            dataU_out[tmp[i]][tmp[12]], dataU_out[tmp[i]][tmp[13]], dataU_out[tmp[i]][tmp[14]],
            dataU_out[tmp[i]][tmp[15]]);
    }
    for (int i = 0; i < UN; ++i) {
#pragma HLS loop_tripcount min = UN max = UN
#pragma HLS unroll
        unrollCol<T, m_diagSize, UN, NMAXUN>(
            lda, Dim_inner_extend, Order1[i + 2 * UN], m_c_right1[i + 2 * UN], m_s_right1[i + 2 * UN],
            dataA[tmp[0]][tmp[i]], dataA[tmp[1]][tmp[i]], dataA[tmp[2]][tmp[i]], dataA[tmp[3]][tmp[i]],
            dataA[tmp[4]][tmp[i]], dataA[tmp[5]][tmp[i]], dataA[tmp[6]][tmp[i]], dataA[tmp[7]][tmp[i]],
            dataA[tmp[8]][tmp[i]], dataA[tmp[9]][tmp[i]], dataA[tmp[10]][tmp[i]], dataA[tmp[11]][tmp[i]],
            dataA[tmp[12]][tmp[i]], dataA[tmp[13]][tmp[i]], dataA[tmp[14]][tmp[i]], dataA[tmp[15]][tmp[i]]);
    }
}

#ifndef __SYNTHESIS__
template <typename T, int m_diagSize, int UN, int NMAXUN>
void Jacobi_svd(T**** dataA, T**** dataU_out, int lda) {
#else
template <typename T, int m_diagSize, int UN, int NMAXUN>
void Jacobi_svd(T dataA[UN][UN][NMAXUN][NMAXUN], T dataU_out[UN][UN][NMAXUN][NMAXUN], int lda) {
#endif
    T considerZero = 2.2250738585072014e-308;
    T threshold;
    if (sizeof(T) == sizeof(double)) {
        threshold = 8.22045e-17;
    } else if (sizeof(T) == sizeof(float)) {
        threshold = 8.22045e-8;
    }
    bool finished = false;
#ifndef __SYNTHESIS__
    int counters = 0;
#ifdef _SOLVER_DEBUG_
    std::cout << "threshold = " << threshold << std::endl;
#endif
#endif
#ifndef __SYNTHESIS__
    int** Order;
    Order = new int*[m_diagSize];
    for (int i = 0; i < m_diagSize; ++i) {
        Order[i] = new int[m_diagSize];
    }
#else
    int Order[m_diagSize][m_diagSize];
#pragma HLS resource variable = Order core = RAM_T2P_BRAM
#endif
    int odd = lda % 2;
    int rank;
    if (odd) {
        rank = lda + 1;
    } else {
        rank = lda;
    }
    GenBlockMat<m_diagSize, 0>(rank, Order);
    int Dim_outer;
    int Dim_inner;
    int Dim_inner_extend;
    int tmp1, tmp2, innerTmp;
    tmp1 = m_diagSize - 1;
    tmp2 = m_diagSize;
    Dim_outer = tmp1;
    innerTmp = tmp2;
    Dim_inner = innerTmp >> 1;
    Dim_inner_extend = Dim_inner;
#ifndef __SYNTHESIS__
#ifdef _SOLVER_DEBUG_
    std::cout << "Dim_inner = " << Dim_inner << std::endl;
    std::cout << "Dim_outer = " << Dim_outer << std::endl;
    std::cout << "Dim_inner_extend = " << Dim_inner_extend << std::endl;
    for (int i = 0; i < m_diagSize; ++i) {
        for (int j = 0; j < m_diagSize; ++j) {
            std::cout << "Order i = " << i << " j = " << j << "  " << Order[i][j] << std::endl;
        }
    }
#endif
#endif

While_Loop:
    while (!finished) {
#pragma HLS loop_tripcount min = 12 max = 12
        finished = true;
        int cnt = 0;
#ifndef __SYNTHESIS__
#ifdef _SOLVER_DEBUG_
        std::cout << "while loop:" << counters << std::endl;
        counters++;
#endif
#endif
    Loop_innerWhile:
        for (int i = 0; i < Dim_outer; ++i) {
#pragma HLS loop_tripcount min = 511 max = 511
            int flag = 0;
            for (int j = 0; j < Dim_inner; ++j) {
#pragma HLS loop_tripcount min = 256 max = 256
#pragma HLS pipeline
                if ((i < rank - 1) && (j < rank / 2)) {
                    int p1, q1;
                    p1 = Order[i][2 * j];
                    q1 = Order[i][2 * j + 1];
#ifndef __SYNTHESIS__
#ifdef _SOLVER_DEBUG_
                    std::cout << "p1 = " << p1 << "  q1 = " << q1 << std::endl;
#endif
#endif
                    if ((xf::solver::internal::m::abs(dataA[(p1 % UN)][q1 % UN][p1 / UN][q1 / UN]) > threshold)) {
                        flag = 1;
                    }
                }
            }
            T m_c_left[m_diagSize];
            T m_s_left[m_diagSize];
            T m_c_right[m_diagSize];
            T m_s_right[m_diagSize];
#pragma HLS ARRAY_PARTITION variable = m_c_right
#pragma HLS ARRAY_PARTITION variable = m_s_right
            if (flag) {
                finished = false;
                T matrix[3];
            Loop_jacobi2x2:
                for (int j = 0; j < Dim_inner; ++j) {
#pragma HLS loop_tripcount min = 256 max = 256
#pragma HLS pipeline II = 2
                    if (j < rank / 2) {
                        for (int tt = 0; tt < 3; ++tt) {
#pragma HLS loop_tripcount min = 3 max = 3
                            int aa, bb;
                            aa = (tt == 2) ? 1 : 0;
                            bb = (tt == 0) ? 0 : 1;
                            matrix[tt] = dataA[(Order[i][2 * j + aa] % UN)][Order[i][2 * j + bb] % UN]
                                              [Order[i][2 * j + aa] / UN][Order[i][2 * j + bb] / UN];
                        }
                        jacobi_rotation_2x2<T, m_diagSize>(matrix, considerZero, m_c_left[j], m_s_left[j], m_c_right[j],
                                                           m_s_right[j]);
#ifndef __SYNTHESIS__
#ifdef _SOLVER_DEBUG_
                        std::cout << "m_diaSize = " << m_diagSize << std::endl;
                        std::cout << "m_c_right = " << m_c_right[j] << "  m_s_right = " << m_s_right[j] << std::endl;
#endif
#endif
                    }
                }
                funcDataflow<T, m_diagSize, UN, NMAXUN>(i, Dim_inner_extend, Dim_inner, Order, m_c_right, m_s_right,
                                                        dataA, dataU_out, rank);
            }
        }
    }

#ifndef __SYNTHESIS__
    for (int i = 0; i < m_diagSize; ++i) {
        delete[] Order[i];
    }
    delete[] Order;
#endif
}

} // namespace internal

#ifndef __SYNTHESIS__
template <typename T, int diagSize, int UN, int NMAXUN>
void gesvdj_2D(T**** dataA, T**** dataU_out, int lda) {
#else
template <typename T, int diagSize, int UN, int NMAXUN>
void gesvdj_2D(T dataA[UN][UN][NMAXUN][NMAXUN], T dataU_out[UN][UN][NMAXUN][NMAXUN], int lda) {
#endif
#pragma HLS inline off
Loop_init_I:
    for (int r = 0; r < diagSize; ++r) {
#pragma HLS loop_tripcount min = 512 max = 512
    Loop_init_J:
        for (int j = 0; j < diagSize; ++j) {
#pragma HLS loop_tripcount min = 512 max = 512
#pragma HLS pipeline
            dataU_out[(r % UN)][j % UN][r / UN][j / UN] = (r == j) ? 1 : 0;
        }
    }
    internal::Jacobi_svd<T, diagSize, UN, NMAXUN>(dataA, dataU_out,
                                                  lda); // core function of Jacbi SVD for symmetric matrix
}

/**
 * @brief Symmetric Matrix Jacobi based Singular Value Decomposition (GESVDJ) .
   \f{equation*} {A = U \Sigma {V}^T}\f}
   where \f$A\f$ is a dense symmetric matrix of size \f$m \times m\f$, \f$U\f$ and \f$V\f$
   are \f$m \times m\f$ matrix with orthonormal columns, and \f$\Sigma\f$ is diagonal matrix.\n
   The maximum matrix size supported in FPGA is templated by NMAX.
 *
 * @tparam T data type (support float and double).
 * @tparam NMAX maximum number of rows/columns of input matrix
 * @tparam NCU number of computation unit
 * @tparam m number of rows/cols of matrix A
 * @param A input matrix of size \f$m \times m\f$
 * @param S decomposed diagonal singular matrix of size \f$m \times m\f$
 * @param U left U matrix of SVD
 * @param V right V matrix of SVD
 * @param lda leading dimension of matrix A
 * @param ldu leading dimension of matrix U
 * @param ldv leading dimension of matrix V
 * @param info output info (unused)
 */
#ifndef __SYNTHESIS__
template <typename T, int NMAX, int NCU>
void gesvdj(int m, T* A, int lda, T* S, T* U, int ldu, T* V, int ldv, int& info) {
#else
template <typename T, int NMAX, int NCU>
void gesvdj(int m,
            T A[NMAX * NMAX],
            int lda,
            T S[NMAX * NMAX],
            T U[NMAX * NMAX],
            int ldu,
            T V[NMAX * NMAX],
            int ldv,
            int& info) {
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
                S[i * lda + j] = 0;
            } else if (i == j) {
                dataA_2D[(i % NCU)][j % NCU][i / NCU][j / NCU] = 1;
            } else {
                dataA_2D[(i % NCU)][j % NCU][i / NCU][j / NCU] = 0;
            }
        }
    }

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
                T tmpVal = dataU_2D[(j % NCU)][i % NCU][j / NCU][i / NCU];
                T tmpS = dataA_2D[(i % NCU)][i % NCU][i / NCU][i / NCU];
                V[j * ldv + i] = tmpVal;
                U[j * ldu + i] = (tmpS < 0) ? (-tmpVal) : (tmpVal);
                if (j == i) {
                    S[i * lda + i] = (tmpS < 0) ? (-tmpS) : (tmpS);
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
#endif
