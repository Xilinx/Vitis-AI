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
 * @file binomial_distribution.hpp
 * @brief This file include the binomialCDF and binomialPMF
 *
 */

#ifndef __XF_FINTECH_BINOMIAL_DIST_HPP_
#define __XF_FINTECH_BINOMIAL_DIST_HPP_

#include "hls_math.h"
#include "ap_int.h"

#ifndef __SYNTHESIS__
#include "iostream"
#endif

namespace xf {

namespace fintech {

/**
 * @brief BinomialDistribution binomial distribution
 *
 * @tparam DT data type supported include float and double
 */
template <typename DT>
class BinomialDistribution {
   private:
    DT logP;
    DT log1MP;
    unsigned int n;
    DT p;

   public:
    /**
     * @brief constructor
     */
    BinomialDistribution() {
#pragma HLS inline
    }

    /**
     * @brief init initialize parameters
     *
     * @param n_ n independent Bernoulli trials
     * @param p_ p is the probability of success of a single trial.
     */
    void init(unsigned int n_, DT p_) {
        n = n_;
        p = p_;
        if (p < 0.0 || p > 1.0) {
#ifndef __SYNTHESIS__
            std::cout << "[ERROR] p = " << p << ",p should belong to [0, 1]!\n";
#endif
        }
        logP = hls::log(p);
        log1MP = hls::log(1 - p);
    }

    /**
     * @brief PMF it implement a probability mass function for binomial distribution
     *
     * @param k k successes in n independent Bernoulli trials
     */
    DT PMF(int k) {
        if (p == 0.0) return (k == 0 ? 1.0 : 0.0);
        if (p == 1.0) return (k == n ? 1.0 : 0.0);
        if (k > n || k < 0) return 0.0;

        DT inX[3];
        DT g[3];
        inX[0] = n + 1;
        inX[1] = k + 1;
        inX[2] = n - k + 1;
        for (int i = 0; i < 3; i++) {
#pragma HLS pipeline
            g[i] = hls::tgamma_p_reduce(inX[i]);
        }
        return g[0] / (g[1] * g[2]) * hls::exp(k * logP + (n - k) * log1MP);
    }

    /**
     * @brief CDF it implement a cumulative distribution function for binomial distribution
     *
     * @param k k successes in n independent Bernoulli trials
     * @return it belong to [0, 1] and also is a cumulative probability value.
     */
    DT CDF(unsigned int k) {
        if (p < 0.0 || p > 1.0) {
#ifndef __SYNTHESIS__
            std::cout << "[ERROR] p = " << p << ",p should belong to [0, 1]!\n";
#endif
            return -1.0;
        }
        if (p == 0.0) return 0.0;
        if (p == 1.0) return 1.0;
        if (k > n) return 1.0;

        const DT accuracy = 1e-16;
        const int maxIteration = 100;
        const DT eps = 1.0e-8;

        int a = k + 1;
        int b = n - k;
        DT dp = 1.0 - p;
        DT inX[3];
        DT g[3];
        inX[0] = n + 1;
        inX[1] = a;
        inX[2] = b;
        for (int i = 0; i < 3; i++) {
#pragma HLS pipeline
            g[i] = hls::tgamma_p_reduce(inX[i]);
        }
        DT tmp = g[0] / (g[1] * g[2]) * hls::exp(a * logP + b * log1MP);
        //    DT tmp = hls::tgamma_p_reduce((DT)(n + 1)) / (hls::tgamma_p_reduce((DT)a) * hls::tgamma_p_reduce((DT)b)) *
        //             hls::exp(a * hls::log(p) + b * hls::log(dp));
        DT pp = ((DT)(k + 2)) / (n + 3);
        if (p > pp) {
            int t = a;
            a = b;
            b = t;
            p = dp;
        }
        DT x = p;
        DT aa;
        DT del;
        int qab = a + b;
        int qap = a + 1;
        int qam = a - 1;
        DT c = 1.0;
        DT d = 1.0 - qab * x / qap;
        if (hls::fabs(d) < eps) d = eps;
        d = 1.0 / d;
        DT result = d;
    loop_main:
        for (int m = 1; m <= maxIteration; m++) {
#pragma HLS pipeline
            int m2 = 2 * m;
            int aa1 = m * (b - m);
            int aa2 = (qam + m2) * (a + m2);
            DT aa3 = x / aa2;
            aa = aa3 * aa1;
            // aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (hls::fabs(d) < eps) d = eps;
            c = 1.0 + aa / c;
            if (hls::fabs(c) < eps) c = eps;
            d = 1.0 / d;
            result *= d * c;
            aa1 = -(a + m) * (qab + m);
            aa2 = (a + m2) * (qap + m2);
            aa3 = x / aa2;
            aa = aa3 * aa1;
            // aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (hls::fabs(d) < eps) d = eps;
            c = 1.0 + aa / c;
            if (hls::fabs(c) < eps) c = eps;
            d = 1.0 / d;
            del = d * c;
            result *= del;
            if (hls::fabs(del - 1.0) < accuracy) {
                DT value = tmp * result / a;
                if (p > pp) return value;
                return 1.0 - value;
            };
        }
#ifndef __SYNTHESIS__
        std::cout << "[ERROR] a or b too big, or maxIteration too small.\n";
#endif
        return -1.0;
    }
}; // binomial
}
}
#endif
