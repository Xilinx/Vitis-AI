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
 * @file hw_model.hpp
 * @brief This file include the class HWModel
 *
 */

#ifndef __XF_FINTECH_HWMODEL_HPP_
#define __XF_FINTECH_HWMODEL_HPP_

#include "hls_math.h"
#include "ap_int.h"
#include "cubic_spline.hpp"

#ifndef __SYNTHESIS__
#include <chrono>
#include "iostream"
using namespace std;
using namespace std::chrono;
#endif

namespace xf {

namespace fintech {

/**
 * @brief Hull-White model for Tree Engine
 *
 * @tparam DT data type supported include float and double
 * @tparam Tree class TrinomialTree
 * @tparam LEN2 maximum length of node of tree, which affects the latency and resources utilization.
 *
 */

template <typename DT, typename Tree, int LEN2>
class HWModel {
   private:
    // spreads on interest rates
    DT spread_, a_, sigma_;
    DT rate_;

   public:
    /**
     * @brief constructor
     */
    HWModel() {
#pragma HLS inline
    }

    /**
     * @brief initialize parameters
     *
     * @param r floating benchmark annual interest rate
     * @param spread spreads on interest rates
     * @param a initial volatility of stock.
     * @param sigma the volatility of volatility.
     */
    void initialization(DT r, DT spread, DT a, DT sigma) {
#pragma HLS inline
        spread_ = spread;
        a_ = a;
        sigma_ = sigma;
        rate_ = r;
    }

    /**
     * @brief calculate the discount after time dt
     *
     * @param t the current timepoint
     * @param dt The difference between the next timepoint and the current timepoint
     * @param x underlying
     * @param r shortrate
     * @return discount
     */
    DT discount(DT t, DT dt, DT* x, DT r) {
#pragma HLS inline
        DT rate = r + (*x + spread_) * dt;
#ifndef __SYNTHESIS__
        return std::exp(-rate);
#else
        return hls::exp(-rate);
#endif
    }

    /**
     * @brief calcutate short-rate of dt at t for TreeEngine
     *
     * @param tree class TrinomialTree
     * @param endCnt end counter of timepoints
     * @param time array timepoints
     * @param dtime array the difference between the next timepoint and the current timepoint
     * @param tmp_values1 process values
     * @param tmp_values2 process values
     * @param statePrices state prices
     * @param rates array short-rates
     */
    void treeShortRate(Tree& tree,
                       int endCnt,
                       DT* time,
                       DT* dtime,
                       DT tmp_values1[3][LEN2],
                       DT tmp_values2[3][LEN2],
                       DT* statePrices,
                       DT* rates) {
#pragma HLS inline
        int size = 0;
        DT probs[3];
    loop_compute_init:
        for (int j = 0; j < endCnt * 2; j++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 100 max = 100
            tmp_values1[0][j] = 0.0;
            tmp_values1[1][j] = 0.0;
            tmp_values1[2][j] = 0.0;
            tmp_values2[0][j] = 0.0;
            tmp_values2[1][j] = 0.0;
            tmp_values2[2][j] = 0.0;
        }

        DT values16[48];
        DT rate_last = 1.0;
#pragma HLS array_partition variable = values16 block factor = 3 dim = 1
#pragma HLS resource variable = values16 core = RAM_2P_LUTRAM
    loop_init_values16:
        for (int j = 0; j < 16; j++) {
#pragma HLS pipeline
            values16[j] = 0.0;
            values16[16 + j] = 0.0;
            values16[32 + j] = 0.0;
        }

        DT t, dt;
    loop_compute_LEN:
        for (int i = 0; i < endCnt - 1; i++) {
#pragma HLS loop_tripcount min = 50 max = 50
            if (size == 0) {
                tmp_values1[0][0] = 1.0;
            } else {
                // compute state prices
                DT tmp1 = 0.0, tmp2 = 0.0, tmp3 = 0.0;
                int index_d = -1;
                ap_uint<1> flag = 0;
            loop_compute_branch:
                for (int j = 0; j < size; j++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 100 max = 100
#pragma HLS dependence variable = tmp_values1 inter false
#pragma HLS dependence variable = tmp_values2 inter false
                    values16[j % 16] = 0.0;
                    values16[16 + j % 16] = 0.0;
                    values16[32 + j % 16] = 0.0;
                    int index = tree.calculateProbability(j, t, dt, probs);
                    DT x = tree.underlying(j);
                    DT state_price_disc = discount(t, dt, &x, rate_last) * statePrices[j];
                    DT price_tmp1 = state_price_disc * probs[0];
                    DT price_tmp2 = state_price_disc * probs[1];
                    DT price_tmp3 = state_price_disc * probs[2];
                    if (flag == 0) {
                        flag = 1;
                    } else if (index == index_d) {
                        if (i % 2) {
                            tmp_values2[0][index_d] += tmp1 + price_tmp1;
                            tmp_values2[1][index_d + 1] += tmp2 + price_tmp2;
                            tmp_values2[2][index_d + 2] += tmp3 + price_tmp3;
                        } else {
                            tmp_values1[0][index_d] += tmp1 + price_tmp1;
                            tmp_values1[1][index_d + 1] += tmp2 + price_tmp2;
                            tmp_values1[2][index_d + 2] += tmp3 + price_tmp3;
                        }
                        flag = 0;
                    } else {
                        if (i % 2) {
                            tmp_values2[0][index_d] += tmp1 + 0.0;
                            tmp_values2[1][index_d + 1] += tmp2 + 0.0;
                            tmp_values2[2][index_d + 2] += tmp3 + 0.0;
                        } else {
                            tmp_values1[0][index_d] += tmp1 + 0.0;
                            tmp_values1[1][index_d + 1] += tmp2 + 0.0;
                            tmp_values1[2][index_d + 2] += tmp3 + 0.0;
                        }
                    }
                    index_d = index;
                    tmp1 = price_tmp1;
                    tmp2 = price_tmp2;
                    tmp3 = price_tmp3;
                }
                if (flag != 0) {
                    if (i % 2) {
                        tmp_values2[0][index_d] += tmp1;
                        tmp_values2[1][index_d + 1] += tmp2;
                        tmp_values2[2][index_d + 2] += tmp3;
                    } else {
                        tmp_values1[0][index_d] += tmp1;
                        tmp_values1[1][index_d + 1] += tmp2;
                        tmp_values1[2][index_d + 2] += tmp3;
                    }
                }
            }

            // update
            t = time[i];
            dt = dtime[i];
            tree.dxUpdate(i, t, dt);
            size = tree.size(i);
            DT x = tree.underlying(0);
            DT dx = tree.getDx(i);

        loop_hw:
            for (int j = 0; j < size; j++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 100 max = 100
                DT state_price;
                if (i % 2) {
                    state_price = tmp_values2[0][j] + tmp_values2[1][j] + tmp_values2[2][j];
                    tmp_values1[0][j] = 0.0;
                    tmp_values1[1][j] = 0.0;
                    tmp_values1[2][j] = 0.0;
                } else {
                    state_price = tmp_values1[0][j] + tmp_values1[1][j] + tmp_values1[2][j];
                    tmp_values2[0][j] = 0.0;
                    tmp_values2[1][j] = 0.0;
                    tmp_values2[2][j] = 0.0;
                }
                statePrices[j] = state_price;
                DT x_now = dx * j + x;
#ifndef __SYNTHESIS__
                values16[j % 16] += state_price * std::exp(-x_now * dt);
#else
                values16[j % 16] += state_price * hls::exp(-x_now * dt);
#endif
            }

            DT values8[8];
        loop_hw_add8:
            for (ap_uint<4> i = 0; i < 8; i++) {
#pragma HLS unroll factor = 8
                values8[i] = values16[i] + values16[i + 8];
            }

            DT values4[4];
        loop_hw_add4:
            for (ap_uint<3> i = 0; i < 4; i++) {
#pragma HLS unroll factor = 4
                values4[i] = values8[i] + values8[i + 4];
            }

            DT values2_0 = values4[0] + values4[2];
            DT values2_1 = values4[1] + values4[3];
            DT value = values2_0 + values2_1;
            DT tmp_r = rate_ * (t + dt);
            rate_last = (hls::log(value) + tmp_r); /// dt;
            rates[i] = rate_last;
        }
    }

}; // class

/**
 * @brief Hull-White model for FD (finite difference) Engine
 *
 * @tparam DT data type supported include float and double
 *
 */

template <typename DT>
class HWModel<DT, void, 0> {
   private:
    // spreads on interest rates
    DT spread_, a_, sigma_;
    DT rate_;

   public:
    /**
     * @brief constructor
     */
    HWModel() {
#pragma HLS inline
    }

    /**
     * @brief initialize parameters
     *
     * @param r floating benchmark annual interest rate
     * @param spread spreads on interest rates
     * @param a initial volatility of stock.
     * @param sigma the volatility of volatility.
     */
    void initialization(DT r, DT spread, DT a, DT sigma) {
#pragma HLS inline
        spread_ = spread;
        a_ = a;
        sigma_ = sigma;
        rate_ = r;
    }

    /**
     * @brief calcutate short-rate of dt at t for fd Engine
     *
     * @param t the current timepoint
     * @return finite difference short-rates
     */
    DT fdShortRate(DT t) {
#pragma HLS inline off
        DT forwardrate = rate_;
#ifndef __SYNTHESIS__
        DT temp = sigma_ * (1.0 - std::exp(-a_ * t)) / a_;
#else
        DT temp = sigma_ * (1.0 - hls::exp(-a_ * t)) / a_;
#endif
        return (forwardrate + 0.5 * temp * temp);
    }

    /**
     * @brief calculate the discount after time dt
     *
     * @param t the current timepoint
     * @param T The timepoint
     * @param rate shortrate
     * @return discount Bond
     */
    DT discountBond(DT t, DT T, DT rate) {
#pragma HLS inline off
#ifndef __SYNTHESIS__
        DT B_value = (1.0 - std::exp(-a_ * (T - t))) / a_;
        DT forward = rate_;
        DT temp = sigma_ * B_value;
        DT value = B_value * forward - 0.25 * temp * temp * (1.0 - std::exp(-a_ * 2.0 * t)) / a_;
        return std::exp(value + rate_ * (t - T) - B_value * rate);
#else
        DT B_value = (1.0 - hls::exp(-a_ * (T - t))) / a_;
        DT forward = rate_;
        DT temp = sigma_ * B_value;
        DT value = B_value * forward - 0.25 * temp * temp * (1.0 - hls::exp(-a_ * 2.0 * t)) / a_;
        return hls::exp(value + rate_ * (t - T) - B_value * rate);
#endif
    }
}; // class

/**
 * @brief Hull-White model Analytical Engine
 *
 * @tparam DT data type supported include float and double
 *
 */

template <typename DT, int LEN>
class HWModelAnalytic {
    typedef internal::CubicSpline<DT, LEN> CubicSpline;

   private:
    DT a_, sigma_;
    CubicSpline _splineZ, _splineLogZ;

   public:
    /**
     * brief constructor
     */
    HWModelAnalytic() {
#pragma HLS inline
    }

    /**
     * @brief initialize parameters
     *
     * @param a initial volatility of stock.
     * @param sigma the volatility of volatility.
     * @param A
     * @param B
     */

    void initialization(DT a, DT sigma, DT T[LEN], DT R[LEN]) {
        a_ = a;
        sigma_ = sigma;

        DT C[LEN] = {};
        DT D[LEN] = {};

        for (int i = 0; i < LEN; i++) {
#ifndef __SYNTHESIS__
            C[i] = std::exp(-R[i] * T[i]);
#else
            C[i] = hls::exp(-R[i] * T[i]);
#endif
            D[i] = -R[i] * T[i];
        }

        _splineZ.initialization(T, C);
        _splineLogZ.initialization(T, D);
    }

    /**
     * @brief calcutate short-rate of dt at t
     *
     * @param t the current timepoint
     * @return anaylitic short-rate
     */
    DT shortRate(DT t) { return (-_splineLogZ.CS1(t)); }

    /**
     * @brief calculate the discount after time t
     *
     * @param t the current time point
     * @param T the maturity
     * @return discount bond price
     */

    DT discountBond(DT t, DT T, DT rate) {
        // P(0,T) - from spline interpolated yield curve Z at time T
        // P(0,t) - from spline interpolated yield curve Z at time t0
        // F(0,t) - from spline interpolated log yield curve Z at time t0
        DT P_T = _splineZ.CS(T);
        DT P_t = _splineZ.CS(t);
        DT f_t = -_splineLogZ.CS1(t);

#ifndef __SYNTHESIS__

        // std::cout << "P_T:" << P_T << std::endl;
        // std::cout << "P_t:" << P_t << std::endl;
        // std::cout << "f_t:" << f_t << std::endl;

        // B(t,T)
        DT B = 1.0 / a_ * (1.0 - std::exp(-a_ * (T - t)));
        // std::cout << "B:" << B << std::endl;

        // A(t,T)
        DT A =
            P_T / P_t *
            std::exp(B * f_t - std::pow(sigma_, 2.0) / (4.0 * a_) * (1.0 - std::exp(-2.0 * a_ * t)) * std::pow(B, 2.0));
        // std::cout << "A:" << A << std::endl;

        // P(t,T)
        return (A * std::exp(-rate * B));
#else
        // B(t,T)
        DT B = 1.0 / a_ * (1.0 - hls::exp(-a_ * (T - t)));
        // A(t,T)
        DT A =
            P_T / P_t *
            hls::exp(B * f_t - hls::pow(sigma_, 2.0) / (4.0 * a_) * (1.0 - hls::exp(-2.0 * a_ * t)) * hls::pow(B, 2.0));
        // P(t,T)
        return (A * hls::exp(-rate * B));
#endif
    }

}; // class
} // namespace fintech
} // namespace xf

#endif
