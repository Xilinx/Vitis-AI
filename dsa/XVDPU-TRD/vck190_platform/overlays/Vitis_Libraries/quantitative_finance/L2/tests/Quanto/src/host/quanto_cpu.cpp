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
 * @file quanto_cpu.cpp
 * @brief Full precision calculation of Quanto price and Greeks
 * for use in  comparisons
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
/// This is a simple implementation of the Quanto closed-form solution
/// along with the associated Greeks.  It is based on the closed form Black
/// Scholes Merton Model.
/// If the flag 'call' is non-zero, a call premium and corresponding Greeks will be
/// calculated, otherwise a put-option and
/// corresponding Greeks will be calculated.   Theta and Rho are returned in
/// their annualized and percentage forms
/// respectively.
///
/// @param[in]  s          underlying
/// @param[in]  v          volatility (decimal form)
/// @param[in]  rd         domestic interest rate (decimal form)
/// @param[in]  rf         foreign interest rate (decimal form)
/// @param[in]  t          time to maturity
/// @param[in]  k          strike price
/// @param[in]  q          dividend rate of the underlying
/// @param[in]  E          spot exchange rate (foreign currency per unit domestic currency)
/// @param[in]  fxv        volatility of the domestic exchange rate
/// @param[in]  corr       correlation between underlying and domestic exchange rate
/// @param[in]  call       control whether call or put is calculated
/// @param[out] price      call/put premium
/// @param[out] delta      model sensitivity
/// @param[out] gamma      model sensitivity
/// @param[out] vega       model sensitivity
/// @param[out] theta      model sensitivity
/// @param[out] rho        model sensitivity
void quanto_model(double s,
                  double v,
                  double rd,
                  double t,
                  double k,
                  double rf,
                  double q,
                  double E,
                  double fxv,
                  double corr,
                  unsigned int call,
                  double& price,
                  double& delta,
                  double& gamma,
                  double& vega,
                  double& theta,
                  double& rho) {
    // Map to BSM parameters
    double qq = -rf + rd + q + (corr * v * fxv);

    // Calculate the host reference value
    double d1 = (std::log(s / k) + (rd - qq + v * v / 2.0) * t) / (v * std::sqrt(t));
    double d2 = d1 - v * std::sqrt(t);

    double pdf_d1 = (1.0 / std::sqrt(2 * M_PI)) * std::exp(-0.5 * d1 * d1);

    price = E * (s * phi(d1) * std::exp(-qq * t) - k * phi(d2) * std::exp(-rd * t));
    delta = std::exp(-qq * t) * phi(d1);
    theta = (1.0 / 365) * (-v * s * std::exp(-qq * t) * pdf_d1 / (2 * std::sqrt(t)) +
                           qq * s * std::exp(-qq * t) * phi(d1) - rd * k * std::exp(-rd * t) * phi(d2));
    rho = (1.0 / 100) * k * t * std::exp(-rd * t) * phi(d2);

    gamma = exp(-qq * t) * std::exp(-d1 * d1 / 2) / (s * v * std::sqrt(t) * std::sqrt(2 * M_PI));
    vega = (1.0 / 100) * s * std::exp(-qq * t) * std::sqrt(t) * pdf_d1;

    return;
}
