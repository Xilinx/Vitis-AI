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
 * @file ornstein_uhlenbeck_process.hpp
 * @brief header file for 1-D stochastic process.
 *
 * This file is part of XF Fintech 1.0 Library
 */

#ifndef _XF_FINTECH_ORNSTEIN_UHLENBECK_PROCESS_HPP_
#define _XF_FINTECH_ORNSTEIN_UHLENBECK_PROCESS_HPP_

#include "hls_math.h"

namespace xf {
namespace fintech {

/**
 * @brief Ornstein-Uhlenbeck Process is one of the basic stochastic processes.
 * This class describes the Ornstein-Uhlenbeck process.
 *
 * @tparam DT The data type, which decides the precision of the result, and the
 * default data type is double.
 *
 * @param _x0         The initial value of the state variable.
 *
 */
template <typename DT = double>
class OrnsteinUhlenbeckProcess {
   public:
    /**
         * @brief constructor
         */
    OrnsteinUhlenbeckProcess() {
#pragma HLS inline
    }

    /**
     * @brief Initialize parameters
     *
     * @param speed Spreads on interest rates.
     * @param vola The overall level of volatility.
     * @param x0 The initial value of the state variable.
     * @param level The initial value of level.
     */
    void init(DT speed, DT vola, DT x0, DT level = 0.0) {
#pragma HLS inline
        _speed = speed;
        _volatility = vola;
        _x0 = x0;
        _level = level;
    }

    /**
     * @brief The expertation E of the process
     *
     * @param t0 the time at the beginning of processing
     * @param x0 the state of current value
     * @param dt the step of processing
     * @return the result of expectation
     */
    DT expectation(DT t0, DT x0, DT dt) const;

    /**
     * @brief The standard deviation S of the process
     *
     * @param t0 the time at the beginning of processing
     * @param x0 the state of current value
     * @param dt the step of processing
     * @return the result of standard deviation
     */
    DT stdDeviation(DT t0, DT x0, DT dt) const;

    /**
     * @brief The variance of the process
     *
     * @param t0 the time at the beginning of processing
     * @param x0 the state of current value
     * @param dt the step of processing
     * @return the result of variance
     */
    DT variance(DT t0, DT x0, DT dt) const;

    /**
     * @brief As this process will only be executed for once in the prcing engine and it
     * is not the critical time consumer,then it is optimized for minimum resource
     * utilization while having a reasonable latency.
     *
     * @param dt the step of processing
     * @param dw the step of evoleing
     * @return Returns the asset value after a time interval Δt according to the given
     * discretization.
     */
    DT evolve(DT dt, DT dw) const;

   private:
    DT _speed;
    DT _volatility;
    DT _level;

   public:
    DT _x0;
};

template <typename DT>
DT OrnsteinUhlenbeckProcess<DT>::expectation(DT t0, DT x0, DT dt) const {
#pragma HLS inline
#ifndef __SYNTHESIS__
    DT temp = x0 * std::exp(-_speed * dt);
#else
    DT temp = x0 * hls::exp(-_speed * dt);
#endif
    return temp;
}

template <typename DT>
DT OrnsteinUhlenbeckProcess<DT>::stdDeviation(DT t0, DT x0, DT dt) const {
#pragma HLS inline
#ifndef __SYNTHESIS__
    return std::sqrt(variance(t0, x0, dt));
#else
    return hls::sqrt(variance(t0, x0, dt));
#endif
}

template <typename DT>
DT OrnsteinUhlenbeckProcess<DT>::variance(DT t0, DT x0, DT dt) const {
#pragma HLS inline
#ifndef __SYNTHESIS__
    DT temp = std::exp(-2 * _speed * dt);
#else
    DT temp = hls::exp(-2 * _speed * dt);
#endif
    return 0.5 * _volatility * _volatility / _speed * (1.0 - temp);
}

template <typename DT>
DT OrnsteinUhlenbeckProcess<DT>::evolve(DT dt, DT dw) const {
    DT exps, std, square_exps, tmp;
#pragma HLS allocation operation instances = dmul limit = 1
#pragma HLS resource variable = tmp core = DAddSub_nodsp

#ifndef __SYNTHESIS__
    exps = std::exp(-_speed * dt);
    square_exps = exps * exps;
    tmp = 1 - square_exps;
    // XXX as _level and x always equal to zero, we remove this computation for
    // less resource utilizations
    std = _volatility * std::sqrt(0.5 / _speed * tmp) * dw;
#else
    exps = hls::exp(-_speed * dt);
    square_exps = exps * exps;
    tmp = 1 - square_exps;
    std = _volatility * hls::sqrt(0.5 / _speed * tmp) * dw;
#endif

    return std;
}

} // fintech
} // xf

#endif // _XF_FINTECH_ORNSTEIN_UHLENBECK_PROCESS_HPP_
