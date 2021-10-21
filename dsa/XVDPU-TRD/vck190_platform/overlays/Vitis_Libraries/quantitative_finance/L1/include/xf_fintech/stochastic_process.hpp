
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
 * @file stochastic_process.hpp
 * @brief This class describes the Ornstein-Uhlenbeck process.
 *
 */
#ifndef XF_FINTECH_STOCHASTIC_HPP
#define XF_FINTECH_STOCHASTIC_HPP

#include "hls_math.h"
#include <iostream>
using namespace std;

namespace xf {
namespace fintech {

/**
 * @brief Stochastic process for CIR and ECIR models to simulate rate volatility
 *
 * @tparam DT data type supported include float and double
 *
 */
template <typename DT = double>
class StochasticProcess1D {
   public:
    /*
     * @brief StochasticProcess1D constructor
     */
    StochasticProcess1D() {
#pragma HLS inline
    }

    /**
     * @brief initialize parameters
     *
     * @param speed Spreads on interest rates.
     * @param vola The overall level of volatility.
     * @param x0 The initial value of the state variable.
     * @param level The initial value of level.
     */
    void init(DT speed, DT vola, DT x0, DT level) {
        _speed = speed;
        _volatility = vola;
        theta_ = x0;
        k_ = level;
    }

    /**
     * @brief the expertation E of the process
     *
     * @param t0 the time at the beginning of processing
     * @param x0 the state of current value
     * @param dt the step of processing
     * @return the result of expectation
     */
    DT expectation(DT t0, DT x0, DT dt);

    /**
     * @brief the variance of the process
     *
     * @param t0 the time at the beginning of processing
     * @param x0 the state of current value
     * @param dt the step of processing
     * @return the result of variance
     */
    DT variance(DT t0, DT x0, DT dt);

   private:
    DT _volatility;
    DT theta_;
    DT k_;
    // spreads on interest rates
    DT _speed;

    // the state variable
    DT _x0;

    //
    DT _level;
};

template <typename DT>
DT StochasticProcess1D<DT>::expectation(DT t0, DT x0, DT dt) {
#pragma HLS inline
    DT temp = x0 + ((0.5 * theta_ * k_ - 0.125 * _volatility * _volatility) / x0 - 0.5 * k_ * x0) * dt;
    return temp;
}

template <typename DT>
DT StochasticProcess1D<DT>::variance(DT t0, DT x0, DT dt) {
#pragma HLS inline
    return 0.25 * _volatility * _volatility * dt;
}

} // fintech
} // xf

#endif // XF_FINTECH_TYPES_H
