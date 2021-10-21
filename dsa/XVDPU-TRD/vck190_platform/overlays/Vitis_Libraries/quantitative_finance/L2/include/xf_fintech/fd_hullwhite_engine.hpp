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
 * @file fd_hullwhite_engine.hpp
 * @brief This header file includes implementation of finite-difference Hull-white bermudan swaption pricing engine.
 * This file is part of XF Fintech 1.0 Library.
 *
 * @detail engineInitialization module is used to read in configurations, create mesher, and build differential
 * equation.
 * rollbackImplementation module is the main evolve back process of the prcing process.
 *
 */

#ifndef _XF_FINTECH_FDHULLWHITEENGINE_HPP_
#define _XF_FINTECH_FDHULLWHITEENGINE_HPP_

#include "hls_math.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "xf_fintech/enums.hpp"
#include "xf_fintech/hw_model.hpp"
#include "xf_fintech/fdmmesher.hpp"

#ifndef __SYNTHESIS__
#include <cmath>
#include <iostream>
using namespace std;
#endif

namespace xf {
namespace fintech {

/**
 * @brief Bermudan swaption pricing engine using Finite-difference methods.
 *
 * @tparam DT Data type supported including float and double, which decides the precision of the price, and the default
 * data type is double.
 * @tparam _ETSizeMax The maximum number of exercise time supported in the bermudan swaption prcing engine.
 * @tparam _xGridMax The maximum number of NPVs for each time step to calculate.
 * @tparam _legPSizeMax The maximum number of times that payer should pay for the swaption in fixed rate. (depends on
 * payment period)
 * @tparam _legRSizeMax The maximum number of times that payer should receive from the swaption in floating rate.
 * (depends on payment period)
 *
 */

template <typename DT,
          unsigned int _ETSizeMax,
          unsigned int _xGridMax,
          unsigned int _legPSizeMax,
          unsigned int _legRSizeMax>
class FdHullWhiteEngine {
   private:
    // @param sqrt_EPSILON The square root of epsilon is the minimum error that ensure we'll hit the final time step,
    // and
    // the default value should be 1.0e-4.
    const DT sqrt_EPSILON = 1.0e-4;
    // @param epsilon The epsilon which is used to create the mesher, and the default value should be 1.0e-5.
    const DT epsilon = 1.0e-05;

    unsigned int _ETSize, _xGrid, _legPSize, _legRSize;

    DT stoppingTimes_[_ETSizeMax + 1];
    DT payerAccrualTime_[_legPSizeMax + 1];
    DT receiverAccrualTime_[_legRSizeMax + 1];
    DT receiverAccrualPeriod_[_legRSizeMax + 1];
    DT iborTime_[_legRSizeMax + 1];
    DT iborPeriod_[_legRSizeMax + 1];

    // @param locations_ The coordinates of the mesher of the prcing engine.
    DT locations_[_xGridMax];

    DT nominal_;

    DT fixedRate_;
    DT floatingRate_;

    // @param NPV_ The result of the prcing engine.
    DT NPV_;

    DT a_;
    DT sigma_;
    DT theta_;

    // @param dzMap_lower_ The lower diagonal of the tridiagonal matrix of the FDM equation.
    DT dzMap_lower_[_xGridMax];
    // @param dzMap_diag_ The main diagonal of the tridiagonal matrix of the FDM equation.
    DT dzMap_diag_[_xGridMax];
    // @param dzMap_upper_ The upper diagonal of the tridiagonal matrix of the FDM equation.
    DT dzMap_upper_[_xGridMax];

    // @brief Calculate the short rate using Hull-White model
    DT HullWhite_shortRate(DT t) {
#pragma HLS inline off
        // ********************************************************************************************
        // XXX the calculation process of this function has been modified for less resource utilization
        // ********************************************************************************************
        // as we give the same argument t to forwardRate function in Hull-White model,
        // it always returns the floatingRate_. Thus, there is no need for us to waste any DSP to calculate this value.
        /*
                HullWhite<DT, Continuous> hw;
                hw.init(a_, sigma_, floatingRate_);
                DT forwardrate = hw.forwardRate(t, t, NoFrequency);
        */
        DT forwardrate = floatingRate_;

// as a_ is always bigger than sqrt(EPSILON), we eliminate the conditional branch for a simpler implementation
//    DT temp = a_ < sqrt_EPSILON ?
//              sigma_ * t :
#if !defined(__SYNTHESIS__)
        DT temp = sigma_ * (1.0 - std::exp(-a_ * t)) / a_;
#else
        DT temp = sigma_ * (1.0 - hls::exp(-a_ * t)) / a_;
#endif

        return (forwardrate + 0.5 * temp * temp);

    } // end HullWhite_shortRate

    // @brief Calculate the discount using Hull-White and Vasicek model
    DT discountBond(DT now, DT maturity, DT rate) {
#pragma HLS inline off
// ********************************************************************************************
// XXX the calculation process of this function has been modified for less resource utilization
// ********************************************************************************************

#if !defined(__SYNTHESIS__)
        // as a_ always bigger than sqrt(EPSILON), we eliminate the conditional branch for a simpler implementation
        //	if (a_ < sqrt_EPSILON) {
        //		DT B_value = maturity - now;
        //	} else {
        //		DT B_value = (1.0 - std::exp(-a_ * (maturity - now))) / a_;
        //	}

        DT B_value = (1.0 - std::exp(-a_ * (maturity - now))) / a_; // Vasicek_B(now, maturity);

        // since we give the same argument t to forwardRate function in Hull-White model,
        // it always returns the floatingRate_. Thus, there is no need for us to waste any DSP to calculate this value.
        /*
                HullWhite<DT, Continuous> hw;
                hw.init(a_, sigma_, floatingRate_);
                DT forward = hw.forwardRate(t, t, NoFrequency);
        */
        DT forward = floatingRate_;

        DT temp = sigma_ * B_value;
        DT value = B_value * forward - 0.25 * temp * temp * (1.0 - std::exp(-a_ * 2.0 * now)) / a_;

        /*
                DT discount1 = 1.0 / std::exp(floatingRate_ * now);
                DT discount2 = 1.0 / std::exp(floatingRate_ * maturity);
                DT hullwhite_A = std::exp(value) * discount2 / discount1;
        */
        /*
                DT hullwhite_A = std::exp(value) * std::exp(floatingRate_ * (now - maturity));
                return hullwhite_A * std::exp(-B_value * rate);
        */
        return std::exp(value + floatingRate_ * (now - maturity) - B_value * rate);

#else

        DT B_value = (1.0 - hls::exp(-a_ * (maturity - now))) / a_;
        DT forward = floatingRate_;
        DT temp = sigma_ * B_value;
        DT value = B_value * forward - 0.25 * temp * temp * (1.0 - hls::exp(-a_ * 2.0 * now)) / a_;

        return hls::exp(value + floatingRate_ * (now - maturity) - B_value * rate);

#endif

    } // end discountBond

    // @brief Calculate the NPVs at each exercise time
    void calculateNPV(
        // inputs
        DT iterExerciseTime,
        DT disRate,
        DT fwdRate,
        // outputs
        DT* cost,
        DT* profit) {
#pragma HLS allocation function instances = discountBond limit = 1

    LOOP_NPV:
        for (unsigned int leg = 0; leg < _legPSizeMax + _legRSizeMax; leg++) {
#pragma HLS loop_tripcount min = 10 max = 10 avg = 10
#pragma HLS pipeline
            if (leg < _legPSize) { // calculate cost using fixed rate
                if (payerAccrualTime_[leg] >= iterExerciseTime) {
                    DT amount = fixedRate_ * (payerAccrualTime_[leg + 1] - payerAccrualTime_[leg]) *
                                nominal_; // rate * period * nominal
                    DT discount = discountBond(iterExerciseTime, payerAccrualTime_[leg + 1],
                                               disRate); // discountBond(now, maturity, rate)
                    *cost += amount * discount;
                }
            } else if (leg < _legPSize + _legRSize) { // calculate profit using floating rate
                if (receiverAccrualTime_[leg - _legPSize] >= iterExerciseTime) {
                    // pre-calculate floatingRate for the calculation of amount
                    DT discount[3];
                // as profit = amount * discount, and in the calculation of amount we need to run discountBond twice
                // times which
                // is the same implementation that used to calculate the dicount, so we pipeline these process to
                // minimize cycle cost
                LOOP_FLOATINGRATE:
                    for (unsigned int i = 0; i < 3; i++) {
#pragma HLS pipeline II = 1
                        if (i < 2) {
                            discount[i] = discountBond(iterExerciseTime, iborTime_[leg - _legPSize + i],
                                                       fwdRate); // discountBond(now, maturity, rate)
                        } else {
                            discount[i] =
                                discountBond(iterExerciseTime, receiverAccrualTime_[leg - _legPSize + 1], fwdRate);
                        }
                    }

                    DT floatingRate =
                        (discount[0] / discount[1] - 1.0) /
                        (iborPeriod_[leg - _legPSize + 1] - iborPeriod_[leg - _legPSize]) /*accrualPeriod*/;
                    DT amount = floatingRate * (receiverAccrualPeriod_[leg - _legPSize + 1] -
                                                receiverAccrualPeriod_[leg - _legPSize]) *
                                nominal_; // rate * period * nominal
                    *profit += amount * discount[2];
                }
            }
        }

    } // end calculateNPV

    // @brief This function is uesd to calculate NPVs for each time step
    DT calculateInnerValue(unsigned int iter, DT t) {
        DT iterExerciseTime = t;
        DT x = locations_[iter];

        // XXX as we give the same argument t to forwardRate function in Hull-White model,
        // it always returns the floatingRate_. Thus, there is no need for us to waste any DSP to calculate this value.
        /*
                HullWhite<DT, Continuous> hw;
                hw.init(a_, sigma_, floatingRate_);
                DT forwardrate = hw.forwardRate(t, t, NoFrequency);
        */
        DT forwardrate = floatingRate_;

        // XXX as a_ always bigger than sqrt(EPSILON), we eliminate the conditional branch for a simpler implementation
        //    DT temp = a_ < sqrt_EPSILON ?
        //              sigma_ * t :
        DT temp =
#if !defined(__SYNTHESIS__)
            sigma_ * (1.0 - std::exp(-a_ * iterExerciseTime)) / a_;
#else
            sigma_ * (1.0 - hls::exp(-a_ * iterExerciseTime)) / a_;
#endif

        DT Rate = x + (forwardrate + 0.5 * temp * temp);

        DT disRate = Rate;
        DT fwdRate = Rate;

        DT cost = 0.0;
        DT profit = 0.0;

        calculateNPV(iterExerciseTime, disRate, fwdRate, &cost, &profit);

#if !defined(__SYNTHESIS__)
        return std::max(0.0, profit - cost); // npv = profit - cost
#else
        return hls::max(0.0, profit - cost);
#endif

    } // end calculateInnerValue

    // @brief Calculate NPVs at each exercise time from maturity to settlement date
    void applyTo(DT array[_xGridMax], DT t) {
    LOOP_LOCATIONS:
        for (unsigned int iter = 0; iter < _xGrid; iter++) {
#pragma HLS loop_tripcount min = 100 max = 100 avg = 100
#pragma HLS pipeline
            // update the NPV array according to the calculation result at specific exercise time
            DT innerValue = calculateInnerValue(iter, t);
            if (innerValue > array[iter]) {
                array[iter] = innerValue;
            }
        }

    } // end applyTo

    // @brief Evolve the NPV array back while using Thomson algorithm to solve the tridiagonal system
    void solveSplitting(
        // in-out ports
        DT array[_xGridMax],
        // inputs
        DT theta_,
        DT dt,
        DT phi) {
        // use Thomson algorithm to solve a tridiagonal system
        DT a = -(theta_ * dt);
        DT b = 1.0;

        DT split[_xGridMax];
#pragma HLS resource variable = split core = RAM_2P_LUTRAM
        DT tmp[_xGridMax];
#pragma HLS resource variable = tmp core = RAM_2P_LUTRAM

        DT a0, a1, a2;
        DT bet, c1, c2;
        DT lower, diag, upper, upper_minus;
    LOOP_SOLVE_SPLITTING:
        for (unsigned int i = 0; i < _xGrid; ++i) {
#pragma HLS loop_tripcount min = 100 max = 100 avg = 100
#pragma HLS pipeline
            if (0 == i) { // the first iteration
                a0 = array[1];
                a1 = array[0];
                a2 = a0;
            } else if (i < _xGrid - 1) { // middle iterations
                a0 = a1;
                a1 = a2;
                a2 = array[i + 1];
            } else { // the last iteration
                a0 = a1;
                a1 = a2;
                a2 = a0;
            }

            // pre-fetch location & dzMap
            DT location = locations_[i];
            DT dzMap_lower = dzMap_lower_[i];
            DT dzMap_diag = dzMap_diag_[i];
            DT dzMap_upper = dzMap_upper_[i];

            // calculate mapT_ according to dzMap_
            if (0 == i) { // the first iteration
                lower = dzMap_lower;
                diag = dzMap_diag - location - phi;
                upper = dzMap_upper;
            } else { // need to save the upper value of the last iteration before updating mapT_
                upper_minus = upper;
                lower = dzMap_lower;
                diag = dzMap_diag - location - phi;
                upper = dzMap_upper;
            }

            // calculate rhs
            DT rhs = a1 + dt * (a0 * lower + a1 * diag + a2 * upper) * (1.0 - theta_);

            // conditional branch is optimized for less DSP utilization
            /*
            if (i > 0) {
                tmp[i] = a * upper_[i-1] * bet;
                bet = b + a * (diag_[i] - tmp[i] * lower_[i]);
                    bet = 1.0 / bet;
                    split[i] = (rhs - a * lower_[i] * split[i-1]) * bet;
            } else {
                    bet = a * diag_[i] + b;
                    bet = 1.0 / bet;
                    split[i] = rhs * bet;
            }
            */
            if (0 == i) {
                c1 = 0.0;
                c2 = 0.0;
            } else {
                c1 = a * upper_minus * bet;
                c2 = a * lower * split[i - 1];
            }
            tmp[i] = c1;
            bet = 1.0 / (b + a * (diag - c1 * lower));
            split[i] = (rhs - c2) * bet;
        }

        array[_xGrid - 1] = split[_xGrid - 1];
    LOOP_REVERSE:
        for (int j = _xGrid - 2; j >= 0; --j) {
#pragma HLS loop_tripcount min = 100 max = 100 avg = 100
#pragma HLS pipeline
            array[j] = split[j] - tmp[j + 1] * array[j + 1];
        }

    } // end solveSplitting

    // @brief Calculate the average rate at a specific time
    DT setTime(DT t1, DT t2) {
#pragma HLS allocation function instances = HullWhite_shortRate limit = 1

        DT t[2] = {t1, t2};
        DT shortRate[2];
    LOOP_SHORTRATE:
        for (unsigned int i = 0; i < 2; i++) {
#pragma HLS pipeline II = 1
            shortRate[i] = HullWhite_shortRate(t[i]); // a_, sigma_, floatingRate_ are needed in this function
        }

        DT phi = 0.5 * (shortRate[0] + shortRate[1]);

        return phi;

    } // end setTime

    // @brief The main evole back process of the FDM
    void douglasSchemeStep(DT array[_xGridMax], DT t, DT dt) {
        // code relevant to bcSet_ in this function won't run due to bcSet_.size() == 0

        // to is setted accordingly
        DT to = t - dt;
        if (to < 0) {
            to = 0;
        }

        // update mapT_ according to t & t-dt(to)
        DT phi = setTime(to, t); // a_, sigma_, floatingRate_, dzMap_, and locations_ are needed in this function

        // solve splitting according to the latest updated mapT_
        solveSplitting(array, theta_, dt, phi);

    } // end douglasSchemeStep

    // @brief Calculate the first & second derivative parts of the FDM equation according to mesher
    void calculateDerivatives(
        // ouputs
        hls::stream<DT>& lower_fd_strm,
        hls::stream<DT>& diag_fd_strm,
        hls::stream<DT>& upper_fd_strm,
        hls::stream<DT>& lower_sd_strm,
        hls::stream<DT>& diag_sd_strm,
        hls::stream<DT>& upper_sd_strm) {
#pragma HLS allocation operation instances = dmul limit = 1
        // pre-fetch & register x(i-1), x, and x(i+1)
        DT xminus = locations_[0];
        DT x = locations_[1];
        DT xplus = 0.0;

        DT hm = 0.0; // dminus
        // calculate the first hm, hp, and coefficient of the first derivative
        DT hp = x - xminus; // dplus
        DT neg_xa = -(xminus * a_);

        // calculate the coefficient of the second derivative
        DT sigma_square = sigma_ * sigma_;

        // intermediate registers for calculating the first derivative part
        DT lower1 = 0.0;
        DT diag1 = 0.0;
        DT upper1 = 0.0;

        // intermediate registers for calculating the second derivative part
        DT lower2 = 0.0;
        DT diag2 = 0.0;
        DT upper2 = 0.0;

        // set the head of the diagonal matrix of second derivative
        lower_sd_strm.write(lower2);
        diag_sd_strm.write(diag2);
        upper_sd_strm.write(upper2);

        // set the head of the diagonal matrix of first derivative
        lower_fd_strm.write(lower1);
        diag1 = -neg_xa / hp;
        diag_fd_strm.write(diag1);
        upper1 = neg_xa / hp;
        upper_fd_strm.write(upper1);

    // calculate the rest of the diagonal matrix
    LOOP_DERIVATIVES:
        for (unsigned int i = 1; i < _xGrid - 1; i++) {
#pragma HLS loop_tripcount min = 100 max = 100 avg = 100
#pragma HLS pipeline
            xplus = locations_[i + 1];
            hm = x - xminus; // dminus
            hp = xplus - x;  // dplus
            neg_xa = -(x * a_);

            // intermediate calculations to minimize the utilization of DSP
            DT hmhm = hm * hm;
            DT hmhp = hm * hp;
            DT hphp = hp * hp;
            DT sum1;
#pragma HLS resource variable = sum1 core = DAddSub_nodsp
            sum1 = hmhm + hmhp;
            DT sum2;
#pragma HLS resource variable = sum2 core = DAddSub_nodsp
            sum2 = hmhp + hphp;

            lower2 = sigma_square / sum1;
            lower_sd_strm.write(lower2);
            lower1 = -hp / sum1 * neg_xa;
            lower_fd_strm.write(lower1);

            diag2 = -sigma_square / hmhp;
            diag_sd_strm.write(diag2);
            diag1 = (hp - hm) / hmhp * neg_xa;
            diag_fd_strm.write(diag1);

            upper2 = sigma_square / sum2;
            upper_sd_strm.write(upper2);
            upper1 = hm / sum2 * neg_xa;
            upper_fd_strm.write(upper1);

            xminus = x;
            x = xplus;
        }

        // calculate the last hm & coefficient of the sequence
        hm = x - xminus;
        neg_xa = -(x * a_);

        // set the tail of the diagonal matrix of second derivative
        lower2 = 0.0;
        lower_sd_strm.write(lower2);
        diag2 = 0.0;
        diag_sd_strm.write(diag2);
        upper2 = 0.0;
        upper_sd_strm.write(upper2);

        // set the tail of the diagonal matrix of first derivative
        lower1 = -neg_xa / hm;
        lower_fd_strm.write(lower1);
        diag1 = neg_xa / hm;
        diag_fd_strm.write(diag1);
        upper1 = 0.0;
        upper_fd_strm.write(upper1);

    } // end calculateDerivatives

    // @brief Add first and second derivatives together to build the FDM equation
    void addDerivatives(
        // inputs
        hls::stream<DT>& lower_fd_strm,
        hls::stream<DT>& diag_fd_strm,
        hls::stream<DT>& upper_fd_strm,
        hls::stream<DT>& lower_sd_strm,
        hls::stream<DT>& diag_sd_strm,
        hls::stream<DT>& upper_sd_strm) {
    LOOP_TRIPLE_ADD:
        for (unsigned int i = 0; i < _xGrid; ++i) {
#pragma HLS loop_tripcount min = 100 max = 100 avg = 100
#pragma HLS pipeline
            DT lower_in1 = lower_fd_strm.read();
            DT lower_in2 = lower_sd_strm.read();
            dzMap_lower_[i] = lower_in1 + lower_in2;

            DT diag_in1 = diag_fd_strm.read();
            DT diag_in2 = diag_sd_strm.read();
            dzMap_diag_[i] = diag_in1 + diag_in2;

            DT upper_in1 = upper_fd_strm.read();
            DT upper_in2 = upper_sd_strm.read();
            dzMap_upper_[i] = upper_in1 + upper_in2;
        }

    } // end addDerivatives

    // @brief Build up the differential equation in tridiagonal maxtrix form
    void buildDefferentialEquation() {
#pragma HLS dataflow

        hls::stream<DT> lower_sd_strm("lower_sd_strm");
#pragma HLS resource variable = lower_sd_strm core = FIFO_LUTRAM
#pragma HLS stream variable = lower_sd_strm depth = 32 dim = 1
        hls::stream<DT> diag_sd_strm("diag_sd_strm");
#pragma HLS resource variable = diag_sd_strm core = FIFO_LUTRAM
#pragma HLS stream variable = diag_sd_strm depth = 32 dim = 1
        hls::stream<DT> upper_sd_strm("upper_sd_strm");
#pragma HLS resource variable = upper_sd_strm core = FIFO_LUTRAM
#pragma HLS stream variable = upper_sd_strm depth = 32 dim = 1

        hls::stream<DT> lower_fd_strm("lower_fd_strm");
#pragma HLS resource variable = lower_fd_strm core = FIFO_LUTRAM
#pragma HLS stream variable = lower_fd_strm depth = 32 dim = 1
        hls::stream<DT> diag_fd_strm("diag_fd_strm");
#pragma HLS resource variable = diag_fd_strm core = FIFO_LUTRAM
#pragma HLS stream variable = diag_fd_strm depth = 32 dim = 1
        hls::stream<DT> upper_fd_strm("upper_fd_strm");
#pragma HLS resource variable = upper_fd_strm core = FIFO_LUTRAM
#pragma HLS stream variable = upper_fd_strm depth = 32 dim = 1

        // calculate the first & second derivative parts of the FDM equation
        calculateDerivatives(lower_fd_strm, diag_fd_strm, upper_fd_strm, lower_sd_strm, diag_sd_strm, upper_sd_strm);

        // add the first & second derivatives together to build the FDM quation (saved in dzMap_)
        addDerivatives(lower_sd_strm, diag_sd_strm, upper_sd_strm, lower_fd_strm, diag_fd_strm, upper_fd_strm);

    } // end buildDefferentialEquation

    // @brief Calculate the mesher using Ornstein-Uhlenbeck process
    void createMesher() {
        // create mesher (locations_)
        OrnsteinUhlenbeckProcess<DT> process;
        process.init(a_, sigma_, 0.0);
        Fdm1dMesher<DT, _xGridMax> mesher1d;
        mesher1d.init(process, stoppingTimes_[_ETSize], epsilon, _xGrid, locations_);

    } // end createMesher

    // @brief Read initial values from DDR to BRAM, and reset the initial values of NPV (saved in array_)
    void readInitialValues(DT stoppingTimes[_ETSizeMax + 1],
                           DT payerAccrualTime[_legPSizeMax + 1],
                           DT receiverAccrualTime[_legRSizeMax + 1],
                           DT receiverAccrualPeriod[_legRSizeMax + 1],
                           DT iborTime[_legRSizeMax + 1],
                           DT iborPeriod[_legRSizeMax + 1]) {
    // initialize several variables
    LOOP_INIT_STOPPINGTIMES:
        for (unsigned int i = 0; i <= _ETSize; i++) {
#pragma HLS loop_tripcount min = 6 max = 6 avg = 6
#pragma HLS pipeline II = 1
            stoppingTimes_[i] = stoppingTimes[i];
        }

    LOOP_INIT_PAYERACCRUALTIME:
        for (unsigned int i = 0; i <= _legPSize; i++) {
#pragma HLS loop_tripcount min = 6 max = 6 avg = 6
#pragma HLS pipeline II = 1
            payerAccrualTime_[i] = payerAccrualTime[i];
        }

    LOOP_INIT_RECEIVERACCRUALTIME:
        for (unsigned int i = 0; i <= _legRSize; i++) {
#pragma HLS loop_tripcount min = 6 max = 6 avg = 6
#pragma HLS pipeline II = 1
            receiverAccrualTime_[i] = receiverAccrualTime[i];
            receiverAccrualPeriod_[i] = receiverAccrualPeriod[i];
            iborTime_[i] = iborTime[i];
            iborPeriod_[i] = iborPeriod[i];
        }

    LOOP_INIT_ARRAY:
        for (unsigned int i = 0; i < _xGrid; i++) {
#pragma HLS loop_tripcount min = 100 max = 100 avg = 100
#pragma HLS pipeline II = 1
            array_[i] = 0.0;
        }

    } // end readInitialValues

   public:
    // @param array_ Storage for NPVs at a specific time point.
    DT array_[_xGridMax];

    // @brief Allocate specific storage for each variable
    FdHullWhiteEngine() {
#pragma HLS inline
#pragma HLS resource variable = locations_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = stoppingTimes_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = payerAccrualTime_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = receiverAccrualTime_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = receiverAccrualPeriod_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = iborTime_ core = RAM_2P_LUTRAM
#pragma HLS resource variable = iborPeriod_ core = RAM_2P_LUTRAM
    }

    /**
     *
     * @brief This function is used to build up the FDM equation, and calculate the initialValues(NPVs at maturity),
 * meanwhile, read several initial values from DDR to RAMs or registers.
     *
     * @param stoppingTimes The array which contains every exercise time in sequence with a 0.99 day time point at the
     * front. (the unit should be year)
     * @param payerAccrualTime The array which contains every payment time in fixed rate. (the unit should be year)
     * @param receiverAccrualTime The array which contains every payment time in floating rate. (the unit should be
 * year)
     * @param receiverAccrualPeriod We support multiple day count convention, so this port is given to users to let them
     * decide what kind of day count convention should be applied when calculating the amount of money in a period.
     * @param iborTime This array is used to calculate the discount at a specific time point.
     * @param iborPeriod This port is also given to users to let them decide what kind of day count convention should be
     * applied when calculating the actual floating rate in a period.
     * @param ETSize The actual number of exercise time in the bermudan swaption.
     * @param xGrid The actual number of NPVs for each time step to calculate.
     * @param legPSize The actual number of times that payer should pay for the swaption in fixed rate. (depends on
 * payment
     * period)
     * @param legRSize The actual number of times that payer should receive from the swaption in floating rate. (depends
 * on
     * payment period)
     * @param a Spreads on interest rates.
     * @param sigma Overall level of volatility on interest rates.
     * @param theta Parameter used to build up the differential equation, the pricing engine uses crank-nicolson
 * algorithm,
     * so default value of theta should be 0.5
     * @param nominal The nominal value of the swap.
     * @param fixedRate Fixed rate of the swaption. (per year)
     * @param floatingRate Floating rate of the swaption. (per yaer)
     *
     */
    void engineInitialization(DT stoppingTimes[_ETSizeMax + 1],
                              DT payerAccrualTime[_legPSizeMax + 1],
                              DT receiverAccrualTime[_legRSizeMax + 1],
                              DT receiverAccrualPeriod[_legRSizeMax + 1],
                              DT iborTime[_legRSizeMax + 1],
                              DT iborPeriod[_legRSizeMax + 1],
                              unsigned int ETSize,
                              unsigned int xGrid,
                              unsigned int legPSize,
                              unsigned int legRSize,
                              DT a,
                              DT sigma,
                              DT theta,
                              DT nominal,
                              DT fixedRate,
                              DT floatingRate) {
        // get the actual size of ETSize, xGrid, legPSize, and legRSize
        _ETSize = ETSize;
        _xGrid = xGrid;
        _legPSize = legPSize;
        _legRSize = legRSize;

        // initialize several variables for FDM
        a_ = a;
        sigma_ = sigma;
        theta_ = theta;
        nominal_ = nominal;
        fixedRate_ = fixedRate;
        floatingRate_ = floatingRate;

        // read initial values from DDR to BRAM, and reset the initial values of NPV (saved in array_)
        readInitialValues(stoppingTimes, payerAccrualTime, receiverAccrualTime, receiverAccrualPeriod, iborTime,
                          iborPeriod);

        // create the mesher
        createMesher();

        // build up the defferential equation in FDMs
        buildDefferentialEquation();

    } // end engineInitialization

    /**
     *
     * @brief This function perform the main rolling back process in bermudan swaption pricing
     *
     * @param array NPVs at a specific time point.
     * @param from Start time point of evolve back process.
     * @param to End time point of evolve back process.
     * @param tGrid Number of time steps from maturity to settlement date.
     *
     */
    DT rollbackImplementation(DT array[_xGridMax], DT from, DT to, unsigned int tGrid) {
        DT dt = (from - to) / tGrid;
        DT t = from;
        int j = _ETSize - 1;

        // calculate the NPVs at the maturity
        if ((_ETSize > 0) && (stoppingTimes_[_ETSize] == from)) {
            applyTo(array, from);
        }

    // roll back from maturity to settlement date
    LOOP_STEPS:
        for (unsigned int i = 0; i < tGrid; ++i) {
#pragma HLS loop_tripcount min = 120 max = 120 avg = 120
            DT now = t;
            DT next = t - dt;
            if (hls::abs(to - next) < sqrt_EPSILON) {
                next = to;
            }
            bool hit = false;

            if (next <= stoppingTimes_[j] && stoppingTimes_[j] < now) {
                // hitting a stop time
                hit = true;

                // perform a step between now to soppingTimes[j]
                dt = now - stoppingTimes_[j];
                douglasSchemeStep(array, now, dt);
                applyTo(array, stoppingTimes_[j]);

                // continue the evolving process
                now = stoppingTimes_[j];
                j--;
            }

            // as we will never hit the exercise times in this conditional branch,
            // so the applyTo process is removed to optimize the performance
            if (hit) {
                // if we did hit a stopping time,
                // continue the latter part of the evolving process
                if (now > next) {
                    dt = now - next;
                    douglasSchemeStep(array, now, dt);
                }
                // reset to default step
                dt = (from - to) / tGrid;
            } else {
                // if we didn't hit,
                // perform the original evolving process
                douglasSchemeStep(array, now, dt);
            }

            t -= dt;
        } // end LOOP_STEPS

        // _xGrid is restrained to an odd number to save resource utilizations from implementing interpolation process
        unsigned int index = (_xGrid - 1) / 2;
        NPV_ = array_[index];

        return NPV_;

    } // end rollbackImplementation

}; // end class FdHullWhiteEngine

} // namespace fintech
} // namespace xf

#endif //_XF_FINTECH_FDHULLWHITEENGINE_H_
