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
 * @file cubic_interpolation.hpp
 *
 */

#ifndef __XF_FINTECH_CUBIC_INTERPOLATION_HPP_
#define __XF_FINTECH_CUBIC_INTERPOLATION_HPP_

#include "hls_math.h"
#include "ap_int.h"

#ifndef __SYNTHESIS__
#include "iostream"
using namespace std;
#endif

namespace xf {

namespace fintech {

/**
 * @brief CubicInterpolation
 *
 * @tparam DT data type
 */
template <typename DT>
class CubicInterpolation {
   private:
    DT a, b, c, d;

   public:
    /**
     * @brief constructor
     */
    CubicInterpolation() {
#pragma HLS inline
    }

    /**
     * @brief initialize parameters
     *
     * @param p array p length is 4, Corresponding to the values of x = -1, x = 0, x = 1, and x = 2.
     */
    void init(DT p[4]) {
        DT t0 = -0.5 * p[0];
        a = t0 + 1.5 * (p[1] - p[2]) + 0.5 * p[3];
        b = p[0] - 2.5 * p[1] + 2 * p[2] - 0.5 * p[3];
        c = t0 + 0.5 * p[2];
        d = p[1];
    }

    /**
     * @brief calculate interpolation
     *
     * @param x input
     * @return ouput interpolation
     */
    DT calcu(DT x) {
#pragma HLS pipeline
        DT x2 = x * x;
        DT x3 = x2 * x;
        return a * x3 + b * x2 + c * x + d;
    }
}; // CubicInterpolation
} // fintech
} // xf
#endif
