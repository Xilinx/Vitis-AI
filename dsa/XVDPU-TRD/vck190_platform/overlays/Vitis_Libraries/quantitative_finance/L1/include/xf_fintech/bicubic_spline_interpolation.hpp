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
 * @file bicubic_spline_interpolation.hpp
 * @brief This file include the BicubicSplineInterpolation
 *
 */

#ifndef __XF_FINTECH_BICUBIC_SPLINE_INTERPOLATION_HPP_
#define __XF_FINTECH_BICUBIC_SPLINE_INTERPOLATION_HPP_

#include "hls_math.h"
#include "ap_int.h"

#ifndef __SYNTHESIS__
#include "iostream"
#endif

namespace xf {

namespace fintech {
namespace internal {

template <typename DT, int N>
void splineImplPart1(int n_, DT* dx, DT* S, DT* temp, DT* tmp) {
#pragma HLS inline
loop_part1:
    DT bet;
    for (int i = 0; i < n_; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline
        if (i == 0) {
            bet = 2.0;
            temp[i] = 1.0 / bet;
            tmp[i] = 3.0 * S[i] / bet;
        } else if (i == n_ - 1) {
            bet = 2.0 - temp[i - 1];
            tmp[i] = (3.0 * S[n_ - 2] - tmp[i - 1]) / bet;
        } else if (i > 0) {
            bet = 2.0 * dx[i] + 2.0 * dx[i - 1] - dx[i] * temp[i - 1];
            temp[i] = dx[i - 1] / bet;
            tmp[i] = (3.0 * (dx[i] * S[i - 1] + dx[i - 1] * S[i]) - dx[i] * tmp[i - 1]) / bet;
        }
    }
}

template <typename DT, int N>
void splineImplPart2(int n_, int index, DT* dx, DT* S, DT* temp, DT* tmp, DT* a, DT* b, DT* c) {
loop_part2:
    for (int i = n_ - 2; i >= index; i--) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline
        tmp[i] -= temp[i] * tmp[i + 1];
        a[i] = tmp[i];
        b[i] = (3.0 * S[i] - tmp[i + 1] - 2.0 * tmp[i]) / dx[i];
        c[i] = (tmp[i + 1] + tmp[i] - 2.0 * S[i]) / (dx[i] * dx[i]);
    }
}

template <typename DT>
DT splineImplPart3(DT* xArr, DT* yArr, DT* a, DT* b, DT* c, DT x, int i) {
    DT dx = x - xArr[i];
    return yArr[i] + dx * (a[i] + dx * (b[i] + dx * c[i]));
}
} // internal

/**
 * @brief BicubicSplineInterpolation
 *
 * @tparam DT data type supported include float and double
 * @tparam N max length of array
 */
template <typename DT, int N>
class BicubicSplineInterpolation {
   private:
    int n_;
    DT xArr_[N];
    DT yArr_[N];
    DT zArr_[N][N];
    DT a_[N][N];
    DT b_[N][N];
    DT c_[N][N];
    DT dx_[N];
    DT dy_[N];
    DT S_[N][N];
    DT temp[N], tmp[N][N];

   public:
    /**
     * @brief constructor
     */
    BicubicSplineInterpolation() {
#pragma HLS inline
#pragma HLS resource variable = xArr_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = yArr_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = zArr_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = a_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = b_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = c_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = dx_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = dy_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = S_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = temp core = RAM_2P_LUTRAM
#pragma HLS resource variable = tmp core = RAM_2P_LUTRAM
    }

    /**
     * @brief init initialize parameters and calculate
     *
     * @param n actual array length
     * @param xArr array of coordinate x
     * @param yArr array of coordinate y
     * @param zArr array of coordinate z
     */
    void init(int n, DT xArr[N], DT yArr[N], DT zArr[N][N]) {
        n_ = n;
    loop_init1:
        for (int i = 0; i < n; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
        loop_init2:
            for (int j = 0; j < n; j++) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline
                zArr_[i][j] = zArr[i][j];
                xArr_[j] = xArr[j];
                yArr_[j] = yArr[j];
                dx_[j] = xArr[j + 1] - xArr[j];
                dy_[j] = yArr[j + 1] - yArr[j];
                S_[i][j] = (zArr[i][j + 1] - zArr[i][j]) / dx_[j];
            }
        }
    loop_spline1:
        for (int i = 0; i < n_; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
            internal::splineImplPart1<DT, N>(n_, dx_, S_[i], temp, tmp[i]);
        }
    loop_spline2:
        for (int i = 0; i < n_; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
            internal::splineImplPart2<DT, N>(n_, 0, dx_, S_[i], temp, tmp[i], a_[i], b_[i], c_[i]);
        }
    }

    /**
     * @brief calcu calculate interpolation
     *
     * @param x interpolation coordinate x
     * @param y interpolation coordinate y
     * @return return interpolation coordinate z
     */
    DT calcu(DT x, DT y) {
        DT section[N], ds[N];
#pragma HLS resource variable = section core = RAM_2P_LUTRAM
#pragma HLS resource variable = ds core = RAM_2P_LUTRAM
        int xIndex = -1, yIndex = -1;
    loop_index:
        for (int i = 0; i < n_; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline
            if (x < xArr_[i] && xIndex == -1) xIndex = i - 1;
            if (y < yArr_[i] && yIndex == -1) yIndex = i - 1;
        }

    loop_calcu:
        for (int i = 0; i < n_; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline
            section[i] = internal::splineImplPart3<DT>(xArr_, zArr_[i], a_[i], b_[i], c_[i], x, xIndex);
            if (i > 0) ds[i - 1] = (section[i] - section[i - 1]) / dy_[i - 1];
        }

        internal::splineImplPart1<DT, N>(n_, dy_, ds, temp, tmp[0]);
        internal::splineImplPart2<DT, N>(n_, yIndex, dy_, ds, temp, tmp[0], a_[0], b_[0], c_[0]);
        return internal::splineImplPart3<DT>(yArr_, section, a_[0], b_[0], c_[0], y, yIndex);
    }
}; // class
}
}
#endif
