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
 * @file v_model.hpp
 * @brief This file include the class VModel
 *
 */

#ifndef __XF_FINTECH_VMODEL_HPP_
#define __XF_FINTECH_VMODEL_HPP_

#include "hls_math.h"
#include "ap_int.h"

#ifndef __SYNTHESIS__
#include "iostream"
using namespace std;
#endif

namespace xf {

namespace fintech {

/**
 * @brief Vasicek model for Tree Engine
 *
 * @tparam DT data type supported include float and double
 * @tparam Tree class TrinomialTree
 * @tparam LEN2 maximum length of node of tree, which affects the latency and resources utilization.
 *
 */
template <typename DT, typename Tree, int LEN2>
class VModel {
   private:
    // spreads on interest rates
    DT spread_;
    // constant
    DT b_;
    DT rate_;

   public:
    /**
     * @brief default constructor
     */
    VModel() {
#pragma HLS inline
    }
    /**
     * @brief initialize parameters
     *
     * @param r floating benchmark annual interest rate
     * @param spread spreads on interest rates
     * @param a initial volatility of stock.
     * @param sigma the volatility of volatility.
     * @param b the short-rate follows an stochastic process with mean b.
     */
    void initialization(DT r, DT spread, DT a, DT sigma, DT b) {
#pragma HLS inline
        spread_ = spread;
        b_ = b;
        rate_ = r;
    }

    /**
     * @brief calculate the discount after time dt
     *
     * @param t the current timepoint
     * @param dt the difference between the next timepoint and the current timepoint
     * @param x underlying
     * @param r shortrate
     * @return discount
     */
    DT discount(DT t, DT dt, DT* x, DT r) {
#pragma HLS inline
        DT rate = (*x + b_ + spread_) * dt;
        return hls::exp(-rate);
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
    loop_compute_LEN:
        for (int i = 0; i < endCnt - 1; i++) {
            DT t = time[i];
            DT dt = dtime[i];
            tree.dxUpdate(i, t, dt);
        }
    }

}; // class
} // fintech
} // xf
#endif
