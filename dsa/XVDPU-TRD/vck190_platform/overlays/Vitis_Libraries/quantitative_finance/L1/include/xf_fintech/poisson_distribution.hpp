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
 * @file poisson_distribution.hpp
 * @brief This file include the poissonPMF, poissonCDF and poissonICDF
 *
 */

#ifndef __XF_FINTECH_POISSON_DIST_HPP_
#define __XF_FINTECH_POISSON_DIST_HPP_

#include "hls_math.h"
#include "ap_int.h"

#ifndef __SYNTHESIS__
#include "iostream"
#endif

namespace xf {

namespace fintech {

/**
 * @brief poissonPMF it implement a probability mass function for poisson distribution
 *
 * @tparam DT data type supported include float and double
 *
 * @param k the number of occurrences
 * @param m is a positive real number, it is equal to the expected value of a discrete random variable and also to its
 * variance.
 * @return it belong to [0, 1] and also is a probability value.
 */
template <typename DT>
DT poissonPMF(unsigned int k, DT m) {
    if (m == 0.0) {
        if (k == 0)
            return 1.0;
        else
            return 0.0;
    }
    DT f = hls::tgamma_p_reduce((DT)(k + 1));
    return std::exp(k * hls::log(m) - m) / f;
}
/**
 * @brief poissonCDF it implement a cumulative distribution function for poisson distribution
 *
 * @tparam DT data type supported include float and double
 *
 * @param k the number of occurrences
 * @param m is a positive real number, it is equal to the expected value of a discrete random variable and also to its
 * variance.
 * @return it belong to [0, 1] and also is a cumulative probability value.
 */
template <typename DT>
DT poissonCDF(unsigned int k, DT m) {
    DT eps = 1.0e-8;
    DT accuracy = 1e-16;
    DT maxIteration = 100;
    int a = k + 1;
    DT x = m;
    int b = k + 2;
    DT da = k + 1;
    DT tmp = hls::exp(-x + da * hls::log(x)) / hls::tgamma_p_reduce(da);
    if (m < b) {
        int ap = k + 1;
        DT del = 1.0 / ap;
        DT sum = del;
        for (ap_uint<8> n = 1; n <= maxIteration; n++) {
            ++ap;
            del *= x / ap;
            sum += del;
            if (hls::fabs(del) < hls::fabs(sum) * accuracy) {
                return 1.0 - sum * tmp;
            }
        }
    } else {
        DT b = x - k;
        DT c = 1.0e8;
        DT d = 1.0 / b;
        DT h = d;
        for (ap_int<10> i = 1; i <= maxIteration; i++) {
            ap_int<20> an = -i * (i - a);
            d = (DT)an * d + (k + 2 * (i + 1));
            if (hls::fabs(d) < eps) d = eps;
            c = b + an / c;
            if (hls::fabs(c) < eps) c = eps;
            d = 1.0 / d;
            DT del = d * c;
            h *= del;
            if (hls::fabs(del - 1.0) < accuracy) {
                return 1.0 - tmp * h;
            }
        }
    }
#ifndef __SYNTHESIS__
    std::cout << "[ERROR] due to maxIteration too small, accuracy not reached\n";
#endif
    return -1;
}

/**
 * @brief poissonICDF it implement a inverse cumulative distribution function for poisson distribution
 *
 * @tparam DT data type supported include float and double
 *
 * @param m is a positive real number, it is equal to the expected value of a discrete random variable and also to its
 * variance.
 * @param x belong to [0, 1] and also is a cumulative probability value.
 * @return the number of occurrences.
 */

template <typename DT>
int poissonICDF(DT m, DT x) {
    if (x < 0.0 || x > 1.0) {
#ifndef __SYNTHESIS__
        std::cout << "[ERROR] x = " << x << ",x should belong to [0, 1]!\n";
#endif
        return -1;
    }
    if (x == 1.0) return 2147483647;
    DT sum = 0.0;
    int index = 0;
    DT tmp1 = 1;
    int tmp2 = 1;
    DT tmpX = x * hls::exp(m);
    while (tmpX > sum) {
#pragma HLS pipeline
        sum += tmp1 / tmp2;
        tmp1 *= m;
        index++;
        tmp2 *= index;
    }
    return index - 1;
}
}
}
#endif
