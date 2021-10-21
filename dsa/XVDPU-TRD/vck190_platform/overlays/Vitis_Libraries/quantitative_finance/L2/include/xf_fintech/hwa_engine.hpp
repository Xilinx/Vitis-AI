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

#ifndef _XF_FINTECH_HWA_ENGINE_H_
#define _XF_FINTECH_HWA_ENGINE_H_

#include "xf_fintech/hw_model.hpp"
#include <stdio.h>

namespace xf {
namespace fintech {

#define SQRT2_RECIP 0.70710678118654752440084436210485f

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
    DT x = SQRT2_RECIP * hls::fabs(xin);

    // A&S formula
    DT t = 1.0f / (1.0f + p * x);
    DT y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * hls::exp(-x * x);

    return 0.5f * (1.0f + sign * y);
}
} // namespace internal

/**
 *@brief
 *
 * @tparam TEST_DT
 * @tparam LEN
 *
 */
template <typename DT, int LEN>
class HWAEngine {
    typedef xf::fintech::HWModelAnalytic<DT, LEN> Model;

   private:
    Model model_;
    DT sigma_;
    DT a_;

    static const int MAXPAYOFF = (30 * 4); // 4 payoffs per year for 30 years

   public:
    /**
     * @brief default constructor
     */
    HWAEngine() {
#pragma HLS inline
    }

    /**
     * @brief init create the hullwhite analytic model
     *
     * @param_t a
     * @param_t sigma
     * @param_t times
     * @param_t rates
     *
     */
    void init(DT a, DT sigma, DT times[LEN], DT rates[LEN]) {
        a_ = a;
        sigma_ = sigma;
        model_.initialization(a, sigma, times, rates);
    }

    /**
     * @brief bondPrice calculate the bond price
     *
     * @param_t t
     * @param_t T
     *
     */
    DT bondPrice(DT t, DT T) {
        DT rate = model_.shortRate(t);
        return (model_.discountBond(t, T, rate));
    }

    /**
     * @brief optionPrice calculate the option price
     *
     * @param_t type
     * @param_t t
     * @param_t T
     * @param_t S
     * @param_t K
     * @param_t P
     *
     */
    DT optionPrice(int type, DT t, DT T, DT S, DT K) {
        DT h;
        DT rate;
        DT ptS;
        DT ptT;
        DT bTS;
        DT sigma_p;
        DT P;

#ifndef __SYNTHESIS__
        bTS = 1.0 / a_ * (1.0 - std::exp(-a_ * (S - T)));
        sigma_p = sigma_ * std::sqrt((1.0 - std::exp(-2.0 * a_ * (T - t))) / (2.0 * a_)) * bTS;
#else
        bTS = 1.0 / a_ * (1.0 - hls::exp(-a_ * (S - T)));
        sigma_p = sigma_ * hls::sqrt((1.0 - hls::exp(-2.0 * a_ * (T - t))) / (2.0 * a_)) * bTS;
#endif

        rate = model_.shortRate(t);
        ptS = model_.discountBond(t, S, rate);
        ptT = model_.discountBond(t, T, rate);

#ifndef __SYNTHESIS__
        h = 1.0 / sigma_p * std::log(ptS / ((ptT * K))) + sigma_p / 2.0;
#else
        h = 1.0 / sigma_p * hls::log(ptS / ((ptT * K))) + sigma_p / 2.0;
#endif

        if (type == 1) {
            P = (ptS * internal::phi<DT>(h) - K * ptT * internal::phi<DT>(h - sigma_p));
        } else {
            P = (K * ptT * internal::phi<DT>(-h + sigma_p) - ptS * internal::phi<DT>(-h));
        }

        return P;
    }

    /**
     * @brief CapFloorPrice calculate the cap/floor price
     *
     * @param_t type
     * @param_t start
     * @param_t end
     * @param_t freq
     * @param_t N
     * @param_t X
     * @param_t P
     *
     */
    DT capfloorPrice(int type, DT start, DT end, DT freq, DT N, DT X) {
        DT s = 0;
        DT numPayoffs = ((end - start) * freq);
        DT T[MAXPAYOFF];

    loop_cap_floor_0:
        for (int i = 0; i < numPayoffs; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 120
            T[i] = start + ((1.0 / freq) * (i + 1));
        }

    loop_cap_floor_1:
        for (int i = 1; i < numPayoffs; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 120
            DT ti = T[i] - T[i - 1];
            s += (1 + X * ti) * optionPrice(type, start, T[i - 1], T[i], 1.0 / (1.0 + X * ti));
        }

        return N * s;
    }
};

} // namespace fintech
} // namespace xf

#endif // _XF_FINTECH_HWA_ENGINE_H_
