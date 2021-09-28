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
 * @file cubic_spline.hpp
 * @brief This file include the cubic spline implementation
 *
 */

#ifndef __XF_FINTECH_CUBIC_SPLINE_HPP_
#define __XF_FINTECH_CUBIC_SPLINE_HPP_

#include "hls_math.h"
#include "ap_int.h"

#ifndef __SYNTHESIS__
#include "iostream"
using namespace std;
#endif

namespace xf {
namespace fintech {
namespace internal {

/*
 * Class: Cubic Spline Implementation
 */

template <typename DT, int LEN2>
class CubicSpline {
   private:
    DT a_[LEN2];
    DT b_[LEN2];
    DT c_[LEN2];
    DT d_[LEN2 + 1];
    DT e_[LEN2];
    int n_;

   public:
    // constructor
    CubicSpline() {
#pragma HLS inline
#pragma HLS array_partition variable = a_ cyclic factor = 3 dim = 1
#pragma HLS array_partition variable = b_ cyclic factor = 3 dim = 1
#pragma HLS array_partition variable = c_ cyclic factor = 3 dim = 1
#pragma HLS array_partition variable = d_ cyclic factor = 3 dim = 1
#pragma HLS array_partition variable = e_ cyclic factor = 3 dim = 1
    }

    void initialization(DT* A, DT* B) {
        DT h[LEN2], alpha[LEN2], l[LEN2 + 1], mu[LEN2 + 1], z[LEN2 + 1];
#pragma HLS array_partition variable = h cyclic factor = 3 dim = 1
#pragma HLS array_partition variable = alpha cyclic factor = 3 dim = 1
#pragma HLS array_partition variable = l cyclic factor = 3 dim = 1
#pragma HLS array_partition variable = mu cyclic factor = 3 dim = 1
#pragma HLS array_partition variable = z cyclic factor = 3 dim = 1

        n_ = LEN2 - 1;

    loop_cs1:
        for (int i = 0; i <= n_; i++) {
            a_[i] = A[i];
            b_[i] = B[i];
        }

    loop_cs2:
        // create empty vectors
        for (int i = 0; i <= n_; i++) {
            l[i] = 0.0;
            mu[i] = 0.0;
            z[i] = 0.0;
            d_[i] = 0.0;

            if (i < n_) {
                h[i] = 0.0;
                alpha[i] = 0.0;
                c_[i] = 0.0;
                e_[i] = 0.0;
            }
        }

    loop_cs3:
        for (int i = 0; i < n_; i++) {
            h[i] = (a_[i + 1] - a_[i]);
        }

    loop_cs4:
        for (int i = 1; i < n_; i++) {
            alpha[i] = (3.0 / h[i] * (b_[i + 1] - b_[i])) - (3.0 / h[i - 1] * (b_[i] - b_[i - 1]));
        }
        l[0] = 1.0;

    loop_cs5:
        for (int i = 1; i < n_; i++) {
            l[i] = (2.0 * (a_[i + 1] - a_[i - 1]) - h[i - 1] * mu[i - 1]);
            mu[i] = (h[i] / l[i]);
            z[i] = ((alpha[i] - h[i - 1] * z[i - 1]) / l[i]);
        }
        l[n_] = 1.0;
        z[n_] = 0.0;

    loop_cs6:
        for (int j = n_ - 1; j > -1; j--) {
            d_[j] = z[j] - mu[j] * d_[j + 1];
            c_[j] = (b_[j + 1] - b_[j]) / h[j] - h[j] * (d_[j + 1] + 2 * d_[j]) / 3;
            e_[j] = (d_[j + 1] - d_[j]) / (3.0 * h[j]);
        }
    }

    DT CS(DT t) {
#pragma HLS INLINE off
        DT val = 0;

        if (t < a_[0]) {
            t = a_[0];
        }

        if (t >= a_[n_]) {
            t = a_[n_];
        }

    loop_cs7:
        for (int j = 0; j < n_; j++) {
            if (t >= a_[j] && t <= a_[j + 1]) {
#ifndef __SYNTHESIS__
                val = b_[j] + c_[j] * (t - a_[j]) + d_[j] * std::pow((t - a_[j]), 2.0) +
                      e_[j] * std::pow((t - a_[j]), 3.0);
#else
                val = b_[j] + c_[j] * (t - a_[j]) + d_[j] * hls::pow((t - a_[j]), 2.0) +
                      e_[j] * hls::pow((t - a_[j]), 3.0);
#endif
                break;
            }
        }

        return (val);
    }

    DT CS1(DT t) {
#pragma HLS INLINE off
        DT rate = 0;

        if (t < a_[0]) t = a_[0];

        if (t >= a_[n_]) t = a_[n_];

    loop_cs8:
        for (int j = 0; j < n_; j++) {
            if (t >= a_[j] && t <= a_[j + 1]) {
#ifndef __SYNTHESIS__
                rate = c_[j] + 2.0 * d_[j] * (t - a_[j]) + 3.0 * e_[j] * std::pow((t - a_[j]), 2.0);
#else
                rate = c_[j] + 2.0 * d_[j] * (t - a_[j]) + 3.0 * e_[j] * hls::pow((t - a_[j]), 2.0);
#endif
                break;
            }
        }
        return (rate);
    }
};

}; // namespace internal
}; // namespace fintech
}; // namespace xf

#endif /* __XF_FINTECH_CUBIC_SPLINE_HPP_ */
