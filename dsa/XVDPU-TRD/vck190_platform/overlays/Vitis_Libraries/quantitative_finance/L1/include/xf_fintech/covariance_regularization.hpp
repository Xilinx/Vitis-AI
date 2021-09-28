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
 * @file covariance_regularization.hpp
 * @brief This file include 4 regularized estimators are included: banding, tapering, hard-thresholding and
 * soft-thresholding.
 *
 */

#ifndef __XF_FINTECH_COV_RE_HPP_
#define __XF_FINTECH_COV_RE_HPP_

#include <hls_math.h>
#include <hls_stream.h>

#ifndef __SYNTHESIS__
#include "iostream"
#endif

namespace xf {

namespace fintech {

/**
 * @brief covReHardThreshold hard-thresholding Covariance Regularization
 *
 * @tparam DT data type supported include float and double
 *
 * @param n n x n covariance matrix
 * @param threshold hard-thresholding parameter
 * @param inMatStrm a covariance matrix
 * @param outMatStrm a regularized covariance matrix after hard-thresholding operation
 */
template <typename DT>
void covReHardThreshold(int n, DT threshold, hls::stream<DT>& inMatStrm, hls::stream<DT>& outMatStrm) {
    for (int i = 0; i < n; i++) {
#pragma HLS loop_tripcount max = 100 min = 100
        for (int j = 0; j < n; j++) {
#pragma HLS loop_tripcount max = 100 min = 100
#pragma HLS pipeline
            DT in = inMatStrm.read();
            DT out;
            if (i == j)
                out = in;
            else if (hls::abs(in) < threshold)
                out = 0;
            else
                out = in;
            outMatStrm.write(out);
        }
    }
} // covReHardThreshold

/**
 * @brief covReSoftThreshold soft-thresholding Covariance Regularization
 *
 * @tparam DT data type supported include float and double
 *
 * @param n n x n covariance matrix
 * @param threshold soft-thresholding parameter
 * @param inMatStrm a covariance matrix
 * @param outMatStrm a regularized covariance matrix after soft-thresholding operation
 */
template <typename DT>
void covReSoftThreshold(int n, DT threshold, hls::stream<DT>& inMatStrm, hls::stream<DT>& outMatStrm) {
    for (int i = 0; i < n; i++) {
#pragma HLS loop_tripcount max = 100 min = 100
        for (int j = 0; j < n; j++) {
#pragma HLS loop_tripcount max = 100 min = 100
#pragma HLS pipeline
            DT in = inMatStrm.read();
            DT in1 = hls::abs(in) - threshold;
            DT out;
            if (in1 < 0) in1 = 0;
            if (i == j)
                out = in;
            else if (in > 0)
                out = in1;
            else
                out = -in1;
            outMatStrm.write(out);
        }
    }
} // covReSoftThreshold

/**
 * @brief covReBand banding Covariance Regularization
 *
 * @tparam DT data type supported include float and double
 *
 * @param n n x n covariance matrix
 * @param k banding parameter
 * @param inMatStrm a covariance matrix
 * @param outMatStrm a regularized covariance matrix after banding operation
 */
template <typename DT>
void covReBand(int n, int k, hls::stream<DT>& inMatStrm, hls::stream<DT>& outMatStrm) {
    for (int i = 0; i < n; i++) {
#pragma HLS loop_tripcount max = 100 min = 100
        for (int j = 0; j < n; j++) {
#pragma HLS loop_tripcount max = 100 min = 100
#pragma HLS pipeline
            DT in = inMatStrm.read();
            DT out;
            if (i == j)
                out = in;
            else if (hls::abs(i - j) > k)
                out = 0;
            else
                out = in;
            outMatStrm.write(out);
        }
    }
} // covReBand

/**
 * @brief covReTaper tapering Covariance Regularization
 *
 * @tparam DT data type supported include float and double
 *
 * @param n n x n covariance matrix
 * @param l tapering parameter
 * @param h the ratio between taper l_h and parameter l
 * @param inMatStrm a covariance matrix
 * @param outMatStrm a regularized covariance matrix after tapering operation
 */
template <typename DT>
void covReTaper(int n, int l, DT h, hls::stream<DT>& inMatStrm, hls::stream<DT>& outMatStrm) {
    DT m = l * h;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
#pragma HLS loop_tripcount max = 100 min = 100
#pragma HLS pipeline
            DT in = inMatStrm.read();
            DT out;
            int d = hls::abs(i - j);
            if (i == j)
                out = in;
            else if (d < m)
                out = in;
            else if (d > l)
                out = 0;
            else
                out = in * (2 - (d / m));
            outMatStrm.write(out);
        }
    }
} // covReTaper
}
}
#endif
