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
 * @file cf_bsm.hpp
 * @brief Templated implementation of a closed-form BSM solver.
 */

#ifndef _XF_FINTECH_CFB76_H_
#define _XF_FINTECH_CFB76_H_

#include <cmath>
#include <iostream>
#include "ap_fixed.h"
#include "hls_math.h"

namespace xf {
namespace fintech {

#define PI 3.1415926535897932384626433832795f
#define SQRT2 1.4142135623730950488016887242097f
#define SQRT2_RECIP 0.70710678118654752440084436210485f
#define SQRT_2PI 2.506628274631000502415765284811f
#define SQRT_2PI_RECIP 0.39894228040143267793994605993438f
#define ANNUALIZED_SCALE 0.00273972602739726027397260273973f
#define PERCENTAGE_SCALE 0.01000000000000000000000000000000f

namespace internal {
/// @brief Approximation to Normal CDF
///
/// This is an implentation of the Abramowitz and Stegun approximation.
/// Refer to https://en.wikipedia.org/wiki/Error_function under the Numerical
/// Approximations section.
///
/// @tparam DT Data Type used for this function
/// @param[in] xin variable
/// @return Normal CDF of input variable
template <typename DT>
DT phi(DT xin) {
    // Constants of approximation
    DT a1 = 0.254829592f;
    DT a2 = -0.284496736f;
    DT a3 = 1.421413741f;
    DT a4 = -1.453152027f;
    DT a5 = 1.061405429f;
    DT p = 0.3275911f;

    // Save the sign of x
    DT sign = (xin < 0.0f) ? -1.0f : 1.0f;
    DT x = SQRT2_RECIP * hls::fabsf(xin);

    // A&S formula
    DT t = 1.0f / (1.0f + p * x);
    DT y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * hls::expf(-x * x);

    return 0.5f * (1.0f + sign * y);
}
}
/// @brief Single option price plus associated Greeks
///
/// Produces a single price and associated Greeks for the given input
/// parameters.  This function is optimized to be
/// synthesized by the HLS compiler and as such uses the hls namespace for the
/// maths functions.  In addition, the code
/// is structured to calculate common elements (in parallel where possible) and
/// reuse as appropriate.
///
/// @tparam DT Data Type used for this function
/// @param[in]  f     underlying forward price
/// @param[in]  v     volatility (decimal form)
/// @param[in]  r     risk-free rate (decimal form)
/// @param[in]  t     time to maturity
/// @param[in]  k     strike price
/// @param[in]  q     continuous dividend yield rate
/// @param[in]  call  control whether call or put is calculated
/// @param[out] price call/put premium
/// @param[out] delta model sensitivity
/// @param[out] gamma model sensitivity
/// @param[out] vega  model sensitivity
/// @param[out] theta model sensitivity
/// @param[out] rho   model sensitivity
template <typename DT>
void cfB76Engine(DT f,
                 DT v,
                 DT r,
                 DT t,
                 DT k,
                 DT q,
                 unsigned int call,
                 DT* price,
                 DT* delta,
                 DT* gamma,
                 DT* vega,
                 DT* theta,
                 DT* rho) {
    // Intermediate elements for calculating price and Greeks
    DT sqrt_t = hls::sqrtf(t);
    DT sqrt_t_recip = 1.0f / sqrt_t;
    DT d1 = (hls::logf(f / k) + (0.5f * v * v) * t) / (v * sqrt_t);
    DT d2 = d1 - v * sqrt_t;
    DT exp_d1n_sq_div2 = hls::expf(-0.5f * d1 * d1);
    DT pdf_d1 = SQRT_2PI_RECIP * exp_d1n_sq_div2;
    DT phi_d1 = internal::phi<DT>(d1);
    DT phi_d2 = internal::phi<DT>(d2);
    DT phi_d1n = 1.0 - phi_d1; // phi(-d1);
    DT phi_d2n = 1.0 - phi_d2; // phi(-d2);
    // DT exp_rt = hls::expf(-q * t);
    DT exp_rt = hls::expf(-r * t);
    DT k_exp_rt = k * exp_rt;
    DT r_k_exp_rt = r * k_exp_rt;
    DT t_k_exp_rt = t * k_exp_rt;
    DT theta_x = -0.5f * sqrt_t_recip * v * f * exp_rt * pdf_d1;

    // Local working variables as some elements can be reused
    DT price_temp;
    DT delta_temp;
    DT theta_temp;
    DT rho_temp;
    DT gamma_temp;
    DT vega_temp;

    // Calculate price and [some] Greeks for call/put
    if (call) {
        delta_temp = exp_rt * phi_d1;
        DT f_delta_temp = f * delta_temp;
        DT k_exp_re_phi_d2 = k_exp_rt * phi_d2;
        price_temp = f_delta_temp - k_exp_re_phi_d2;
        theta_temp = ANNUALIZED_SCALE * (theta_x + r * f_delta_temp - r * k_exp_re_phi_d2);
        rho_temp = PERCENTAGE_SCALE * t * k_exp_re_phi_d2;
    } else {
        delta_temp = -exp_rt * phi_d1n;
        DT f_delta_temp = f * delta_temp;
        DT k_exp_re_phi_d2n = k_exp_rt * phi_d2n;
        price_temp = f_delta_temp + k_exp_re_phi_d2n;
        theta_temp = ANNUALIZED_SCALE * (theta_x + r * f_delta_temp + r * k_exp_re_phi_d2n);
        rho_temp = PERCENTAGE_SCALE * t * k_exp_re_phi_d2n;
    }

    // Remaining Greeks are put/call independent
    gamma_temp = exp_rt * exp_d1n_sq_div2 / (f * v * sqrt_t * SQRT_2PI);
    vega_temp = PERCENTAGE_SCALE * f * exp_rt * sqrt_t * pdf_d1;

    // Return the price/Greeks
    *price = price_temp;
    *delta = delta_temp;
    *theta = theta_temp;
    *rho = rho_temp;
    *gamma = gamma_temp;
    *vega = vega_temp;
}
}
} // xf::fintech

#endif
