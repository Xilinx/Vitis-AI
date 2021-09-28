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
 * @file gamma_distribution.hpp
 * @brief This file include the gammaCDF
 *
 */

#ifndef __XF_FINTECH_GAMMA_DIST_HPP_
#define __XF_FINTECH_GAMMA_DIST_HPP_

#include "hls_math.h"
#include "ap_int.h"

#ifndef __SYNTHESIS__
#include "iostream"
#endif

namespace xf {

namespace fintech {

/**
 * @brief gammaCDF it implement a cumulative distribution function for gamma distribution
 *
 * @tparam DT data type supported include float and double
 *
 * @param a is a positive real number
 * @param x is a positive real number
 * @return it belong to [0, 1] and also is a cumulative probability value.
 */
template <typename DT>
DT gammaCDF(DT a, DT x) {
    if (x <= 0.0) return 0.0;
    DT eps = 1e-8;
    DT tmp1 = a * hls::log(x);
    DT dg = hls::exp(-x + tmp1) / hls::tgamma_p_reduce(a);
    if (x < a + 1.0) {
        DT ap = a;
        DT del = 1.0 / a;
        DT sum = del;
        for (int n = 1; n <= 100; n++) {
#pragma HLS pipeline
            del *= x / (a + n);
            sum += del;
            if (hls::fabs(del) < hls::fabs(sum) * 3.0e-7) return sum * dg;
        }
    } else {
        DT b = x + 1.0 - a;
        DT c = 1.0e30; // big number
        DT d = b;
        DT h = 1.0 / b;
        for (int n = 1; n <= 100; n++) {
#pragma HLS pipeline
            DT an = -n * (n - a);
            // b += 2.0;
            DT b2 = b + (n * 2);
            d = an / d + b2;
            if (hls::fabs(d) < eps) d = eps;
            c = an / c + b2;
            if (hls::fabs(c) < eps) c = eps;
            DT del = c / d;
            h *= del;
            if (hls::fabs(del - 1.0) < eps) return 1.0 - h * dg;
        }
    }
#ifndef __SYNTHESIS__
    std::cout << "[ERROR] too few iterations" << std::endl;
#endif
    return -1;
} // gammaCDF
}
}

#endif
