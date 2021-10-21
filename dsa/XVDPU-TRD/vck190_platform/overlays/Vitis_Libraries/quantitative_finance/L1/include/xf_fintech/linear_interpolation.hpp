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
 * @file linear_interpolation.hpp
 * @brief This file include 2 function: linearInterpolation and linearInterpolation2D to implement function of 1D or 2D
 * linear interpolation.
 *
 */

#ifndef __XF_FINTECH_LINEAR_INTERPOLATION_HPP_
#define __XF_FINTECH_LINEAR_INTERPOLATION_HPP_

#include "hls_math.h"
#include "ap_int.h"

#ifndef __SYNTHESIS__
#include "iostream"
using namespace std;
#endif

namespace xf {

namespace fintech {

namespace internal {

template <typename DT>
DT divideOperation(DT a, DT b) {
#pragma HLS inline off
    return a / b;
}

template <typename DT>
DT addOperation(DT a, DT b) {
#pragma HLS inline off
    return a + b;
}

template <typename DT>
DT multOperation(DT a, DT b) {
#pragma HLS inline off
    return a * b;
}

/**
 * @brief linearInterpolation 1D linear interpolation
 *
 * @tparam DT data type supported include float and double.
 *
 * @param x interpolation coordinate x
 * @param len array of length
 * @param arrX array of coordinate x
 * @param arrY array of coordinate y
 * @return return interpolation coordinate y
 *
 */
template <typename DT>
inline DT linearInterpolation(DT x, int len, DT* arrX, DT* arrY) {
    int cnt = 0;
loop_linear1D:
    for (int i = 0; i < len; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline
        if (x >= arrX[i]) cnt = i + 1;
    }
    DT x_last = arrX[cnt - 1];
    DT y_last = arrY[cnt - 1];
    DT slope = divideOperation<DT>((arrY[cnt] - y_last), (arrX[cnt] - x_last));
    return y_last + (x - x_last) * slope;
}

/**
 * @brief linearInterpolation 2D linear interpolation
 *
 * @tparam DT data type supported include float and double.
 *
 * @param x interpolation coordinate x
 * @param y interpolation coordinate y
 * @param xLen array of coordinate x of length
 * @param yLen array of coordinate y of length
 * @param arrX array of coordinate x
 * @param arrY array of coordinate y
 * @param arrZ array of coordinate z
 * @return return interpolation coordinate z
 *
 */
template <typename DT>
inline DT linearInterpolation2D(DT x, DT y, int xLen, int yLen, DT* arrX, DT* arrY, DT* arrZ) {
#pragma HLS ALLOCATION function instances = divideOperation < DT > limit = 1
#pragma HLS ALLOCATION function instances = addOperation < DT > limit = 2
#pragma HLS ALLOCATION function instances = multOperation < DT > limit = 1
    int i, j;
loopx_linear2D:
    for (i = 0; i < xLen; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline
        if (x < arrX[i]) break;
    }

loopy_linear2D:
    for (j = 0; j < yLen; j++) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline
        if (y < arrY[j]) break;
    }

    DT x_last = arrX[i - 1];
    DT y_last = arrY[j - 1];
    DT z1 = arrZ[(j - 1) * xLen + (i - 1)];
    DT z2 = arrZ[(j - 1) * xLen + i];
    DT z3 = arrZ[j * xLen + (i - 1)];
    DT z4 = arrZ[j * xLen + i];

    DT t = divideOperation<DT>(addOperation<DT>(x, -x_last), addOperation<DT>(arrX[i], -x_last));
    DT u = divideOperation<DT>(addOperation<DT>(y, -y_last), addOperation<DT>(arrY[i], -y_last));
    DT a1 = addOperation<DT>(1.0, -t);
    DT a2 = addOperation<DT>(1.0, -u);
    DT b1 = multOperation<DT>(a1, a2);
    DT c1 = multOperation<DT>(b1, z1);
    DT b2 = multOperation<DT>(t, a2);
    DT c2 = multOperation<DT>(b2, z2);
    DT b3 = multOperation<DT>(a1, t);
    DT c3 = multOperation<DT>(b3, z3);
    DT b4 = multOperation<DT>(t, u);
    DT c4 = multOperation<DT>(b4, z4);
    DT d1 = addOperation<DT>(c1, c2);
    DT d2 = addOperation<DT>(c3, c4);
    return addOperation<DT>(d1, d2);
    // DT t = (x - x_last)/(arrX[i]-x_last);
    // DT u = (y - y_last)/(arrY[i]-y_last);

    // return (1.0-t)*(1.0-u)*z1 + t*(1.0-u)*z2 + (1.0-t)*u*z3 + t*u*z4;
}
} // internal

/**
 * @brief linearImpl 1D linear interpolation
 *
 * @tparam DT data type supported include float and double.
 *
 * @param x interpolation coordinate x
 * @param len array of length
 * @param arrX array of coordinate x
 * @param arrY array of coordinate y
 * @return return interpolation coordinate y
 *
 */
template <typename DT>
DT linearImpl(DT x, int len, DT* arrX, DT* arrY) {
    return internal::linearInterpolation<DT>(x, len, arrX, arrY);
}

} // fintech
} // xf
#endif
