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
 * @file bsm_model.cpp
 * @brief Full precision calculation of BSM price and options for use in
 * comparisons
 */
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

/// @brief Standard calculation of Normal CDF
///
/// This is a straightforward implementation of the Normal CDF as defined by the
/// error function erfc()
/// using the standard library implementation.
///
/// @param[in] x variable
/// @returns   Normal CDF of input variable
double phi(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

/// @brief     Simple helper to return a float within a range
/// @param[in] range_min Lower bound of random value
/// @param[in] range_max Upper bound of random value
/// @returns   Random double in range range_min to range_max
double random_range(double range_min, double range_max) {
    return range_min + (rand() / (RAND_MAX / (range_max - range_min)));
}

/// @brief Single option price plus associated Greeks
///
/// This is a simple implementation of the BSM closed-form solution along with
/// the associated Greeks.  If the flag
/// 'call' is non-zero, a call premium and corresponding Greeks will be
/// calculated, otherwise a put-option and
/// corresponding Greeks will be calculated.   Theta and Rho are returned in
/// their annualized and percentage forms
/// respectively.
///
/// @param[in]  s     underlying
/// @param[in]  v     volatility (decimal form)
/// @param[in]  r     risk-free rate (decimal form)
/// @param[in]  t     time to maturity
/// @param[in]  k     strike price
/// @param[in]  call  control whether call or put is calculated
/// @param[out] price call/put premium
/// @param[out] delta model sensitivity
/// @param[out] gamma model sensitivity
/// @param[out] vega  model sensitivity
/// @param[out] theta model sensitivity
/// @param[out] rho   model sensitivity
void bsm_model(double s,
               double v,
               double r,
               double t,
               double k,
               double q,
               unsigned int call,
               double& price,
               double& delta,
               double& gamma,
               double& vega,
               double& theta,
               double& rho) {
    // Calculate the host reference value
    double d1 = (std::log(s / k) + (r - q + v * v / 2.0) * t) / (v * std::sqrt(t));
    double d2 = d1 - v * std::sqrt(t);

    double pdf_d1 = (1.0 / std::sqrt(2 * M_PI)) * std::exp(-0.5 * d1 * d1);

    if (call) {
        price = s * phi(d1) * std::exp(-q * t) - k * phi(d2) * std::exp(-r * t);
        delta = std::exp(-q * t) * phi(d1);
        theta = (1.0 / 365) * (-v * s * std::exp(-q * t) * pdf_d1 / (2 * std::sqrt(t)) +
                               q * s * std::exp(-q * t) * phi(d1) - r * k * std::exp(-r * t) * phi(d2));
        rho = (1.0 / 100) * k * t * std::exp(-r * t) * phi(d2);
    } else {
        price = phi(-d2) * k * std::exp(-r * t) - phi(-d1) * s * std::exp(-q * t);
        delta = std::exp(-q * t) * (phi(d1) - 1);
        theta = (1.0 / 365) * (-v * s * std::exp(-q * t) * pdf_d1 / (2 * std::sqrt(t)) -
                               q * s * std::exp(-q * t) * phi(-d1) + r * k * std::exp(-r * t) * phi(-d2));
        rho = (-1.0 / 100) * k * t * std::exp(-r * t) * phi(-d2);
    }

    gamma = exp(-q * t) * std::exp(-d1 * d1 / 2) / (s * v * std::sqrt(t) * std::sqrt(2 * M_PI));
    vega = (1.0 / 100) * s * std::exp(-q * t) * std::sqrt(t) * pdf_d1;

    return;
}
