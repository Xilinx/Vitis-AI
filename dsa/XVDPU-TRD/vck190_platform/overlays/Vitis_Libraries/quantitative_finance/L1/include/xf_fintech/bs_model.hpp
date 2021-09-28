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
 * @brief bs_model.hpp
 * @brief This file include mainstream Black-Scholes stochastic process.
 */

#ifndef _XF_FINTECH_BSMODEL_H_
#define _XF_FINTECH_BSMODEL_H_
#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "xf_fintech/enums.hpp"
#include "xf_fintech/utils.hpp"
namespace xf {
namespace fintech {

/**
 * @brief Black-Scholes process
 *
 * @tparam DT data type supported include float and double.
 */
template <typename DT>
class BSModel {
   public:
    /**
      *  risk-free interest rate.
      */
    DT riskFreeRate;
    /**
      *  the constant dividend rate for continuous dividends.
      */
    DT dividendYield;
    /**
      *  volatility of stock price.
      */
    DT volatility;
    /**
      *  variance of change in stock price after time interval dt.
      */
    DT var;
    /**
      * standard deviation of change in stock price after time interval dt according to the given discretization.
      */
    DT stdDev;
    /**
      *  drift of stock price after time interval dt.
      */
    DT drift;

    /**
         * @brief constructor
         */
    BSModel() {
#pragma HLS inline
    }

    /**
     * @brief variance calculate the variance
     *
     * @param dt time interval according to the given discretization
     *
     */
    inline void variance(DT dt) {
        // var = volatility * volatility * dt;
        DT sqV = internal::FPTwoMul(volatility, volatility);
        var = internal::FPTwoMul(sqV, dt);
    }

    /**
     * @brief stdDeviation calculate standard variance
     *
     */
    inline void stdDeviation() { stdDev = hls::sqrt(var); }

    /**
     * @brief updateDrift calculate the drfit of expectation
     * @param dt time interval according to the given discretization.
     *
     */
    inline void updateDrift(DT dt) {
        // drift = (riskFreeRate - dividendYield) * dt - 0.5*var;
        DT u = internal::FPTwoSub(riskFreeRate, dividendYield);
        DT u1 = internal::FPTwoMul(u, dt);
        DT h_v = internal::FPTwoMul((DT)0.5, var);
        drift = internal::FPTwoSub(u1, h_v);
    }

    /**
     * @brief calcuate the price value after time dt.
     *
     * @param x0 initial value
     * @param dt time interval
     * @param dw random number
     *
     */
    inline DT evolve(DT x0, DT dt, DT dw) {
        // DT e1 = dw*stdDeviation;
        // DT e2 = drift + e1;
        // DT x1 = hls::exp(e2);
        // DT x = x0 * x1;

        // DT e1 = internal::FPTwoMul(dw, stdDev);
        // DT e2 = internal::FPTwoAdd(drift, e1);
        DT e2 = logEvolve(dw);
        DT x1 = internal::FPExp(e2);
        DT x = internal::FPTwoMul(x0, x1);
        return x;
    }
    /**
     * @brief calcualte the change of logS after time dt.
     *
     * @param dw randon number
     */
    inline DT logEvolve(DT dw) {
        // DT e1 = dw * stdDeviation;
        // DT e2 = drift + e1;
        // DT dLogX = e2;

        DT e1 = internal::FPTwoMul(dw, stdDev);
        DT dLogX = internal::FPTwoAdd(drift, e1);
        return dLogX;
    }
};
} // namespace fintech
} // namespace xf
#endif //_XF_FINTECH_BSMODEL_H_
