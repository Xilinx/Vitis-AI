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
 * @file g2_model.hpp
 * @brief This file include the class G2Model
 *
 */

#ifndef __XF_FINTECH_G2MODEL_HPP_
#define __XF_FINTECH_G2MODEL_HPP_

#include "hls_math.h"
#include "ap_int.h"
#include "utils.hpp"

#ifndef __SYNTHESIS__
#include "iostream"
using namespace std;
#endif

namespace xf {

namespace fintech {

/**
 * @brief Two-additive-factor gaussian model for Tree Engine
 *
 * @tparam DT data type supported include float and double
 * @tparam Tree class TrinomialTree
 * @tparam LEN2 maximum length of node of tree, which affects the latency and resources utilization.
 *
 */
template <typename DT, typename Tree, int LEN2>
class G2Model {
   private:
    // constants
    DT a_[2], tmp_[2], rho_, rate_;
    DT da_[3];

   public:
    /**
     * @brief constructor
     */
    G2Model() {
#pragma HLS inline
    }

    /**
     * @brief initialize parameters
     *
     * @param r floating benchmark annual interest rate
     * @param a initial volatility of stock.
     * @param sigma the volatility of volatility.
     * @param b initial volatility of stock.
     * @param eta the volatility of volatility.
     * @param rho the correlation coefficient between price and variance.
     */
    void initialization(DT r, DT a, DT sigma, DT b, DT eta, DT rho) {
#pragma HLS inline
        a_[0] = a;
        a_[1] = b;
        tmp_[0] = sigma / a;
        tmp_[1] = eta / b;
        rho_ = rho;
        rate_ = r;
    }

    /**
     * @brief calculate the discount after time dt
     *
     * @param t the current timepoint
     * @param dt the difference between the next timepoint and the current timepoint
     * @param x underlying
     * @param r invalid input
     * @return discount
     */
    DT discount(DT t, DT dt, DT* x, DT r = 0.0) {
#pragma HLS inline
#ifndef __SYNTHESIS__
        DT temp1 = tmp_[0] * (1.0 - std::exp(-a_[0] * t));
        DT temp2 = tmp_[1] * (1.0 - std::exp(-a_[1] * t));
#else
        DT temp1 = tmp_[0] * (1.0 - hls::exp(-a_[0] * t));
        DT temp2 = tmp_[1] * (1.0 - hls::exp(-a_[1] * t));
#endif
        DT rate = 0.5 * (temp1 * temp1 + temp2 * temp2) + rho_ * temp1 * temp2 + rate_;
        rate += x[0] + x[1];
#ifndef __SYNTHESIS__
        DT discount = std::exp(-rate * dt);
#else
        DT discount = hls::exp(-rate * dt);
#endif
        return discount;
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
     * @param rates array short-rates
     */
    void treeShortRate(Tree* tree,
                       int endCnt,
                       DT* time,
                       DT* dtime,
                       internal::xf_2D_array<DT, 4, LEN2>& tmp_values1,
                       internal::xf_2D_array<DT, 4, LEN2>& tmp_values2,
                       DT* rates) {
    loop_compute_LEN:
        for (int i = 0; i < endCnt - 1; i++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 50 max = 50
            DT t = time[i];
            DT dt = dtime[i];
            tree[0].dxUpdate(i, t, dt);
            tree[1].dxUpdate(i, t, dt);
        }
    }

}; // class

/**
 * @brief Two-additive-factor gaussian model
 *
 * @tparam DT data type supported include float and double
 *
 */
template <typename DT>
class G2Model<DT, void, 0> {
   private:
    // constants
    DT a_[2], tmp_[2], rho_, rate_;
    DT da_[3];

    DT calcuV(DT t) {
#pragma HLS inline
#ifndef __SYNTHESIS__
        DT expat = std::exp(-a_[0] * t);
        DT expbt = std::exp(-a_[1] * t);
#else
        DT expat = hls::exp(-a_[0] * t);
        DT expbt = hls::exp(-a_[1] * t);
#endif

        DT valuex = tmp_[0] * tmp_[0] * (t + (2.0 * expat - 0.5 * expat * expat - 1.5) * da_[0]);
        DT valuey = tmp_[1] * tmp_[1] * (t + (2.0 * expbt - 0.5 * expbt * expbt - 1.5) * da_[1]);
        DT value = 2.0 * rho_ * tmp_[0] * tmp_[1] *
                   (t + (expat - 1.0) * da_[0] + (expbt - 1.0) * da_[1] - (expat * expbt - 1.0) * da_[2]);
        return valuex + valuey + value;
    }

   public:
    /**
     * @brief constructor
     */
    G2Model() {
#pragma HLS inline
    }

    /**
     * @brief initialize parameter
     *
     * @param r floating benchmark annual interest rate
     * @param a initial volatility of stock.
     * @param sigma the volatility of volatility.
     * @param b initial volatility of stock.
     * @param eta the volatility of volatility.
     * @param rho the correlation coefficient between price and variance.
     */
    void initialization(DT r, DT a, DT sigma, DT b, DT eta, DT rho) {
#pragma HLS inline
        a_[0] = a;
        a_[1] = b;
        tmp_[0] = sigma / a;
        tmp_[1] = eta / b;
        rho_ = rho;
        rate_ = r;

        da_[0] = 1.0 / a;
        da_[1] = 1.0 / b;
        da_[2] = 1.0 / (a + b);
    }

    /**
     * @brief calculate the discount after time dt
     *
     * @param t the current timepoint
     * @param T the timepoint
     * @param x underlying
     * @return discount bond
     */
    DT discountBond(DT t, DT T, DT* x) {
#pragma HLS inline
        DT dt = T - t;
        DT v1 = 0.5 * (calcuV(dt) - calcuV(T) + calcuV(t));
#ifndef __SYNTHESIS__
        DT v2 = (1.0 - std::exp(-a_[0] * dt)) * x[0] * da_[0] + (1.0 - std::exp(-a_[1] * dt)) * x[1] * da_[1];
        return std::exp(rate_ * (-dt) + v1 - v2);
#else
        DT v2 = (1.0 - hls::exp(-a_[0] * dt)) * x[0] * da_[0] + (1.0 - hls::exp(-a_[1] * dt)) * x[1] * da_[1];
        return hls::exp(rate_ * (-dt) + v1 - v2);
#endif
    }

    /**
     * @brief calculate the short-rate
     *
     * @param t the current timepoint
     * @param x underlying
     * @param r float rate
     * @return short-rate
     */
    DT shortRate(DT t, DT* x, DT r) {
#pragma HLS inline
#ifndef __SYNTHESIS__
        DT temp1 = tmp_[0] * (1.0 - std::exp(-a_[0] * t));
        DT temp2 = tmp_[1] * (1.0 - std::exp(-a_[1] * t));
#else
        DT temp1 = tmp_[0] * (1.0 - hls::exp(-a_[0] * t));
        DT temp2 = tmp_[1] * (1.0 - hls::exp(-a_[1] * t));
#endif
        DT rate = 0.5 * (temp1 * temp1 + temp2 * temp2) + rho_ * temp1 * temp2 + rate_;
        return rate + x[0] + x[1];
    }

}; // class
}
}
#endif
