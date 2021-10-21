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
 * @file normal_distribution.hpp
 * @brief This file include the normal_PDF, normal_CDF, normal_ICDF, logNormal_PDF, logNormal_CDF, logNormal_ICDF.
 *
 */

#ifndef __XF_FINTECH_NORMAL_DIST_HPP_
#define __XF_FINTECH_NORMAL_DIST_HPP_

#include "hls_math.h"
#include "rng.hpp"

#ifndef __SYNTHESIS__
#include "iostream"
#endif

namespace xf {

namespace fintech {

/**
 * @brief normalPDF it implement a probability density function for normal distribution
 *
 * @tparam DT data type supported include float and double
 *
 * @param average the expected value of the distribution
 * @param sigma the  standard deviation of the distribution
 * @param x input of the distribution
 * @return return the result of normal distribution
 */
template <typename DT>
DT normalPDF(DT average, DT sigma, DT x) {
    DT d_sigma = 1.0 / sigma;
    DT nf = 0.3989422804014326779 * d_sigma; // 1/sqrt(2*pi) = 0.3989422804014327
    DT dt = (x - average) * d_sigma;
    DT expo = -0.5 * dt * dt;
    return nf * hls::exp(expo);
}

/**
 * @brief normalCDF it implement a cumulative distribution function for normal distribution
 *
 * @tparam DT data type supported include float and double
 *
 * @param average the expected value of the distribution
 * @param sigma the  standard deviation of the distribution
 * @param x input of the distribution
 * @return return the result of normal distribution
 */
template <typename DT>
DT normalCDF(DT average, DT sigma, DT x) {
    x = (x - average) / sigma;
    return internal::CumulativeNormal<DT>(x);
}

/**
 * @brief normalICDF it implement a inverse cumulative distribution function for normal distribution
 *
 * @tparam DT data type supported include float and double
 *
 * @param average the expected value of the distribution
 * @param sigma the  standard deviation of the distribution
 * @param y input of the distribution
 * @return return the result of normal distribution
 */
template <typename DT>
DT normalICDF(DT average, DT sigma, DT y) {
    DT x = inverseCumulativeNormalAcklam<DT>(y);
    return x * sigma + average;
}

/**
 * @brief logNormalPDF it implement a probability density function for log-normal distribution
 *
 * @tparam DT data type supported include float and double
 *
 * @param average the expected value of the distribution
 * @param sigma the  standard deviation of the distribution
 * @param x input of the distribution
 * @return return the result of log-normal distribution
 */
template <typename DT>
DT logNormalPDF(DT average, DT sigma, DT x) {
    DT y;
    if (x <= 0.0) y = 0.0;
    DT lx = hls::log(x);
    y = normalPDF<DT>(average, sigma, lx) / x;
    return y;
}

/**
 * @brief logNormalCDF it implement a cumulative distribution function for log-normal distribution
 *
 * @tparam DT data type supported include float and double
 *
 * @param average the expected value of the distribution
 * @param sigma the  standard deviation of the distribution
 * @param x input of the distribution
 * @return return the result of log-normal distribution
 */
template <typename DT>
DT logNormalCDF(DT average, DT sigma, DT x) {
    DT y;
    if (x <= 0.0) y = 0.0;
    DT lx = hls::log(x);
    y = normalCDF<DT>(average, sigma, lx);
    return y;
}

/**
 * @brief logNormalICDF it implement a inverse cumulative distribution function for log-normal distribution
 *
 * @tparam DT data type supported include float and double
 *
 * @param average the expected value of the distribution
 * @param sigma the  standard deviation of the distribution
 * @param y input of the distribution
 * @return return the result of log-normal distribution
 */
template <typename DT>
DT logNormalICDF(DT average, DT sigma, DT y) {
    DT x;
    if (y < 0.0 || y > 1.0) {
#ifndef __SYNTHESIS__
        std::cout << "[ERROR] y=" << y << ", y should belong [0,1]!\n" << std::endl;
#endif
        return -1.0;
    }
    DT lx = normalICDF<DT>(average, sigma, y);
    x = hls::exp(lx);
    return x;
}
}
}
#endif
