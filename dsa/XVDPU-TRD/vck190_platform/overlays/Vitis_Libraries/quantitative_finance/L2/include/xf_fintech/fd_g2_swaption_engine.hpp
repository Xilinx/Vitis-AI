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
 * @file fd_g2_swaption_engine.hpp
 * @brief This header file includes implementation of finite-difference G2 model
 * bermudan swaption pricing engine.
 *
 * This file is part of XF Fintech 1.0 Library.
 */

#ifndef _XF_FINTECH_FDG2_SWAPTION_ENGINE_HPP_
#define _XF_FINTECH_FDG2_SWAPTION_ENGINE_HPP_

#include "ap_int.h"
#include "xf_fintech/types.hpp"
#include "xf_fintech/fdmmesher.hpp"
#include "xf_fintech/g2_model.hpp"
#include "xf_fintech/ornstein_uhlenbeck_process.hpp"

namespace xf {
namespace fintech {

/**
 * @brief Bermudan swaption pricing engine using Finite-difference methods.
 *
 * @tparam DT Data type supported including float and double, which decides the
 * precision of the price, and the default data type is double.
 * @tparam _exSize The number of exercise time supported in the bermudan
 * swaption prcing engine.
 * @tparam _xGridMax The maximum number of locations for x direction in a
 * layout.
 * @tparam _yGridMax The maximum number of locations for y direction in a
 * layout.
 * @tparam _layoutSizeMax The maximum number of NPVs for each time step to
 * calculate.
 *
 * @param _dt The gobal variable of dt.
 * @param _shortRate The gobal variable of short rate to calculate.
 * @param _locationsX The location of x direction on a mesher.
 * @param _locationsY The location of y direction on a mesher.
 * @param _initialValues The vector of npv results by every rollback.
 * @param _g2Model The model engine based.
 * @param _dxMap The result of first and second derivative on x.
 * @param _dyMap The result of first and second derivative on y.
 * @param _corrMap The result of mixed derivative on x and y direction.
 *
 */

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax = _xGridMax* _yGridMax>
class FdG2SwaptionEngine {
   public:
    // constructor
    FdG2SwaptionEngine() {
#pragma HLS inline
#pragma HLS resource variable = _stoppingTimes core = RAM_2P_LUTRAM
#pragma HLS ARRAY_PARTITION variable = _accrualTime complete dim = 1
#pragma HLS ARRAY_PARTITION variable = _floatingAccrualPeriod complete dim = 1
#pragma HLS ARRAY_PARTITION variable = _iborTime complete dim = 1
#pragma HLS ARRAY_PARTITION variable = _iborPeriod complete dim = 1
#pragma HLS resource variable = _locationsX core = RAM_2P_LUTRAM
#pragma HLS resource variable = _locationsY core = RAM_2P_LUTRAM
#pragma HLS resource variable = _initialValues core = XPM_MEMORY uram
#pragma HLS resource variable = _dxMap.lower core = RAM_2P_LUTRAM
#pragma HLS resource variable = _dxMap.diag core = RAM_2P_LUTRAM
#pragma HLS resource variable = _dxMap.upper core = RAM_2P_LUTRAM
#pragma HLS resource variable = _dyMap.lower core = RAM_2P_LUTRAM
#pragma HLS resource variable = _dyMap.diag core = RAM_2P_LUTRAM
#pragma HLS resource variable = _dyMap.upper core = RAM_2P_LUTRAM
#pragma HLS resource variable = _corrMap.a00 core = XPM_MEMORY uram
#pragma HLS resource variable = _corrMap.a01 core = XPM_MEMORY uram
#pragma HLS resource variable = _corrMap.a02 core = XPM_MEMORY uram
#pragma HLS resource variable = _corrMap.a10 core = XPM_MEMORY uram
#pragma HLS resource variable = _corrMap.a11 core = XPM_MEMORY uram
#pragma HLS resource variable = _corrMap.a12 core = XPM_MEMORY uram
#pragma HLS resource variable = _corrMap.a20 core = XPM_MEMORY uram
#pragma HLS resource variable = _corrMap.a21 core = XPM_MEMORY uram
#pragma HLS resource variable = _corrMap.a22 core = XPM_MEMORY uram
    }

   public:
    /**
     * @brief initilize the private members of class
     *
     * @param a A factor of spreads on interest rates.
     * @param sigma Overall level of volatility on interest rates.
     * @param b A factor of spreads on interest rates.
     * @param eta A factor of interest rates.
     * @param rho A factor of interest rates.
     * @param tGrid The steps of npv to calculate.
     * @param xGrid The numbers of x direction to calculate.
     * @param yGrid The numbers of y direction to calculate.
     * @param invEps The epsilon which is used to create the mesher, and the
     * default value should be 1.0e-5.
     * @param theta Parameter used to build up the differential equation, the
     * pricing engine uses crank-nicolson algorithm, so default value of theta
     * should be 0.5.
     * @param mu A factor for step.
     * @param nominal The nominal value of the swap.
     * @param fixedRate Fixed rate of the swaption. (per year)
     * @param rate Floating rate of the swaption. (per yaer)
     * @param stoppingTimes The array which contains every exercise time in
     * sequence with a 0.99 day time point at the front. (the unit should be year)
     * @param accrualTime The array which contains every payment time in fixed and
     * floating rate. (the unit should be year)
     * @param floatingAccrualPeriod We support multiple day count convention, so
     * this port is given to users to let them decide what kind of day count
     * convention should be applied when calculating the amount of money in a
     * period.
     * @param iborTime This array is used to calculate the discount at a specific
     * time point.
     * @param iborPeriod This port is also given to users to let them decide what
     * kind of day count convention should be applied when calculating the actual
     * floating rate in a period.
     *
     */
    void init(DT a,
              DT sigma,
              DT b,
              DT eta,
              DT rho,
              Size tGrid,
              Size xGrid,
              Size yGrid,
              DT invEps,
              DT theta,
              DT mu,
              DT fixedRate,
              DT rate,
              DT nominal,
              DT stoppingTimes[_exSize + 1],
              DT accrualTime[_exSize + 1],
              DT floatingAccrualPeriod[_exSize + 1],
              DT iborTime[_exSize + 1],
              DT iborPeriod[_exSize + 1]);

    /**
     * @brief the directly related interface for users
     *
     */
    void calculate();

    /**
     * @brief interface of get npv for users
     * @return the result of npv
     */
    DT getNPV();

   private:
    /**
     * @brief both first-derivative and second-derivative operators
     */
    struct TripleBandLinear {
        DT lower[_xGridMax];
        DT diag[_xGridMax];
        DT upper[_xGridMax];
    };

    /**
     * @brief the round up the basic operators using in mixed-derivative
     */
    struct NinePointLinear {
        DT a00[_layoutSizeMax];
        DT a01[_layoutSizeMax];
        DT a02[_layoutSizeMax];
        DT a10[_layoutSizeMax];
        DT a11[_layoutSizeMax];
        DT a12[_layoutSizeMax];
        DT a20[_layoutSizeMax];
        DT a21[_layoutSizeMax];
        DT a22[_layoutSizeMax];
    };

    DT _a;
    DT _sigma;
    DT _b;
    DT _eta;
    DT _rho;
    Size _tGrid;
    Size _xGrid;
    Size _yGrid;
    Size _layoutSize;
    DT _invEps;
    DT _theta;
    DT _mu;
    DT _dt;
    DT _locationsX[_xGridMax];
    DT _locationsY[_yGridMax];
    DT _initialValues[_layoutSizeMax];
    DT _nominal;
    DT _fixedRate;
    DT _rate;
    DT _shortRate;
    DT _stoppingTimes[_exSize + 1];
    DT _accrualTime[_exSize + 1];
    DT _floatingAccrualPeriod[_exSize + 1];
    DT _iborTime[_exSize + 1];
    DT _iborPeriod[_exSize + 1];

    xf::fintech::G2Model<DT, void, 0> _g2Model;

    // TripleBandLinearOp
    TripleBandLinear _dxMap;
    TripleBandLinear _dyMap;
    NinePointLinear _corrMap;

    // private function for engine

    /**
     * @brief get x and y factors for every step
     *
     * @param i the time is now to calculate
     * @param j the index to point the times every iteration
     * @param rates vector for saving two factors
     */
    void getState(Size i, Size j, DT rates[2]);

    /**
     * @brief the calculate npv for every step
     *
     * @param iter the index to point the times every iteration
     * @param t    the current time to calculate
     */
    DT innerValue(Size iter, DT t);

    /**
     * @brief the inner function of derivativeXY
     *
     * @param d the locations of x or y direction
     * @param uFd the coefficients of first derivative
     * @param uSd the coefficients of second derivative
     * @param hm the minus of on x or y direction
     * @param hp the plus of on x or y direction
     * @param iter the index for loop
     * @param map the derivative of x or y direction
     */
    void derivative(DT d, DT uFd, DT uSd, DT hm, DT hp, Size iter, TripleBandLinear& map);

    /**
     * @brief the derivative on x direction
     *
     * @param d0 the locations of x direction
     * @param d1 the locations of y direction
     */
    void derivativeXY(DT d0[_xGridMax], DT d1[_yGridMax]);

    /**
     * @brief the mixed derivative on x and y direction
     *
     * @param d0 the locations of x direction
     * @param d1 the locations of y direction
     * @param u the coefficient of mixed derivative
     */
    void mixedDerivativeXY(DT d0[_xGridMax], DT d1[_yGridMax], DT u);

    /**
     * @brief the subtraction operation betweed solve_splitting on x and
     * apply_direction on
     *
     * @param tmpThe the multi of _theta and  _dt
     * @param b the factor of splitting and default is 1.0
     * @param y the result of splitting on x direction
     * @param hr the result of short rate betweed now and next time
     * @param split the vector for splitting result
     * @param rhs the vector for result calculated
     */
    void subtraction_merge(DT tmpThe,
                           DT b,
                           DT y[_layoutSizeMax],
                           DT apply_y[_layoutSizeMax],
                           DT hr[_layoutSizeMax],
                           DT split[_layoutSizeMax],
                           DT rhs[_layoutSizeMax]);
    /**
     * @brief reverse array of x direction
     *
     * @param r     array of input to reverse
     * @param split the reslut of splitting
     * @param retVal the result reversed
     */
    void solve_reverse_x(DT r[_layoutSizeMax], DT split[_layoutSizeMax], DT retVal[_layoutSizeMax]);

    /**
     * @brief reverse array of y direction
     *
     * @param r array of input to reverse
     * @param split the reslut of splitting
     * @param retVal the result reversed
     */
    void solve_reverse_y(DT r[_layoutSizeMax], DT split[_layoutSizeMax], DT retVal[_layoutSizeMax]);

    /**
     * @brief calculate the average rate at a specific time
     *
     * @param t1 the begin of time
     * @param t2 the end of time
     */
    DT setTime(DT t1, DT t2);

    /**
     * @brief the main evole back process of the FDM
     *
     * @param a the base npvs of every evole
     * @param t the current time of evolving
     */
    void step_merge(DT a[_layoutSizeMax], DT t);

    /**
     * @brief Calculate NPVs at each exercise time from maturity to settlement
     * date
     *
     * @param a the npvs reslut
     * @param t the current time of updating
     */
    void applyTo(DT a[_layoutSizeMax], DT t);

    /**
     * @brief This function perform the main rolling back process in bermudan
     *
     * swaption pricing
     * @param a the npvs reslut
     * @param from begin of time
     * @param to   the end of time
     */
    void rollback(DT a[_layoutSizeMax], DT from, DT to);

    /**
     * @brief This function for two double to multiply
     *
     * @param in1 the first factor
     * @param in2 the second factor
     */
    DT FPTwoMul(DT in1, DT in2);
};

// initialize all parameters for engine
template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
void FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::init(DT a,
                                                                                 DT sigma,
                                                                                 DT b,
                                                                                 DT eta,
                                                                                 DT rho,
                                                                                 Size tGrid,
                                                                                 Size xGrid,
                                                                                 Size yGrid,
                                                                                 DT invEps,
                                                                                 DT theta,
                                                                                 DT mu,
                                                                                 DT fixedRate,
                                                                                 DT rate,
                                                                                 DT nominal,
                                                                                 DT stoppingTimes[_exSize + 1],
                                                                                 DT accrualTime[_exSize + 1],
                                                                                 DT floatingAccrualPeriod[_exSize + 1],
                                                                                 DT iborTime[_exSize + 1],
                                                                                 DT iborPeriod[_exSize + 1]) {
    _a = a;
    _sigma = sigma;
    _b = b;
    _eta = eta;
    _rho = rho;
    _tGrid = tGrid;
    _xGrid = xGrid;
    _yGrid = yGrid;
    _layoutSize = xGrid * yGrid;
    _invEps = invEps;
    _theta = theta;
    _mu = mu;
    _dt = 0;
    _shortRate = 0.0;
    _fixedRate = fixedRate;
    _rate = rate;
    _nominal = nominal;

loop_initialize_time:
    for (int i = 0; i <= _exSize; i++) {
#pragma HLS pipeline II = 1
        _stoppingTimes[i] = stoppingTimes[i];
        _accrualTime[i] = accrualTime[i];
        _floatingAccrualPeriod[i] = floatingAccrualPeriod[i];
        _iborTime[i] = iborTime[i];
        _iborPeriod[i] = iborPeriod[i];
    }
loop_initialize_value:
    for (int i = 0; i < _layoutSize; i++) {
#pragma HLS pipeline II = 1
        _initialValues[i] = 2.22507e-308;
    }
}

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
DT FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::FPTwoMul(DT in1, DT in2) {
#pragma HLS inline off
    DT r = 0;
#pragma HLS resource variable = r core = DMul_fulldsp
    r = in1 * in2;
    return r;
}

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
void FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::derivative(
    DT d, DT uFd, DT uSd, DT hm, DT hp, Size iter, TripleBandLinear& map) {
#pragma HLS allocation function instances = FPTwoMul limit = 1
    //#pragma HLS inline
    DT tmpLower1, tmpDiag1, tmpUpper1;
    DT tmpLower2, tmpDiag2, tmpUpper2;
    DT lower1, diag1, upper1;
    DT lower2, diag2, upper2;
    DT addTmp = hm + hp;
    DT mulTmp = FPTwoMul(hm, hp);
    DT tmpHm = FPTwoMul(hm, addTmp);
    DT tmpHp = FPTwoMul(hp, addTmp);
    lower1 = -hp / tmpHm;
    diag1 = (hp - hm) / mulTmp;
    upper1 = hm / tmpHp;
    lower2 = 2.0 / tmpHm;
    diag2 = -2.0 / mulTmp;
    upper2 = 2.0 / tmpHp;

    tmpLower1 = FPTwoMul(lower1, uFd);
    tmpDiag1 = FPTwoMul(diag1, uFd);
    tmpUpper1 = FPTwoMul(upper1, uFd);
    tmpLower2 = FPTwoMul(lower2, uSd);
    tmpDiag2 = FPTwoMul(diag2, uSd);
    tmpUpper2 = FPTwoMul(upper2, uSd);

    map.lower[iter] = tmpLower1 + tmpLower2;
    map.diag[iter] = tmpDiag1 + tmpDiag2;
    map.upper[iter] = tmpUpper1 + tmpUpper2;
}

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
void FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::derivativeXY(DT d0[_xGridMax],
                                                                                         DT d1[_yGridMax]) {
    //#pragma HLS inline
    DT x_hm, x_hp, y_hm, y_hp;
    Size last = _xGrid - 1;
    DT uxFd, uyFd;
    DT uxSd = 0.5 * _sigma * _sigma;
    DT uySd = 0.5 * _eta * _eta;

    x_hp = d0[1] - d0[0];
    y_hp = d1[1] - d1[0];
    DT hpTmp1 = 1 / x_hp;
    DT hpTmp2 = 1 / y_hp;

loop_derivative:
    // loops for x and y
    for (int i = 0; i < _xGrid; i++) {
#pragma HLS loop_tripcount min = 51 max = 51
        if (i == 0) {
            // x directrion
            // the first point initialization of first-derivative
            uxFd = -d0[0] * _a;

            DT tmp = hpTmp1 * uxFd;
            _dxMap.lower[0] = 0.0;
            _dxMap.diag[0] = -tmp;
            _dxMap.upper[0] = tmp;

            // y direction
            // the first point initialization of first-derivative
            uyFd = -d1[0] * _b;
            tmp = hpTmp2 * uyFd;
            _dyMap.lower[0] = 0.0;
            _dyMap.diag[0] = -tmp;
            _dyMap.upper[0] = tmp;
        } else if (i == last) {
            // x direction
            // the last point initialization of first-derivative
            uxFd = -d0[last] * _a;
            x_hm = d0[last] - d0[_xGrid - 2];
            DT tmp1 = (1 / x_hm) * uxFd;

            // the last dxMap calculation of firsti-derivative
            _dxMap.lower[last] = -tmp1;
            _dxMap.diag[last] = tmp1;
            _dxMap.upper[last] = 0.0;

            // y direction

            uyFd = -d1[last] * _b;
            y_hm = d1[last] - d1[_xGrid - 2];
            DT tmp2 = (1 / y_hm) * uyFd;
            _dyMap.lower[last] = -tmp2;
            _dyMap.diag[last] = tmp2;
            _dyMap.upper[last] = 0.0;
        } else {
            DT x = d0[i];
            uxFd = -x * _a;
            x_hm = x - d0[i - 1];
            x_hp = d0[i + 1] - x;

            derivative(x, uxFd, uxSd, x_hm, x_hp, i, _dxMap);
            // y direction
            DT y = d1[i];
            uyFd = -y * _b;
            y_hm = y - d1[i - 1];
            y_hp = d1[i + 1] - y;

            derivative(y, uyFd, uySd, y_hm, y_hp, i, _dyMap);
        }
    }
}

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
void FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::mixedDerivativeXY(DT d0[_xGridMax],
                                                                                              DT d1[_yGridMax],
                                                                                              DT u) {
//#pragma HLS inline
#pragma HLS allocation function instances = FPTwoMul limit = 1
    DT hm_d0, hp_d0, hm_d1, hp_d1;
    DT zetam1, zeta0, zetap1, phim1, phi0, phip1;
    DT tmpa00, tmpa01, tmpa02, tmpa10, tmpa20, tmpa11, tmpa22, tmpa12, tmpa21;
    DT minus0, x0, plus0, minus1, x1, plus1;

    x0 = d0[0];
    x1 = d1[0];
loop_mixedderivative:
    for (int j = 0; j < _yGrid; j++) {
#pragma HLS loop_tripcount min = 51 max = 51
        for (int i = 0; i < _xGrid; i++) {
#pragma HLS loop_tripcount min = 51 max = 51
            Size at = i + j * _xGrid;
            if (i == 0 && j == 0) {
                // lower left corner
                plus0 = d0[at + 1];
                plus1 = d1[at + 1];
                hp_d0 = plus0 - x0;
                hp_d1 = plus1 - x1;
                DT tmpHp = FPTwoMul(hp_d0, hp_d1);
                DT tmp = 1.0 / tmpHp;
                tmpa00 = 0.0;
                tmpa01 = 0.0;
                tmpa02 = 0.0;
                tmpa10 = 0.0;
                tmpa20 = 0.0;
                tmpa11 = tmp;
                tmpa22 = tmp;
                tmpa12 = -tmp;
                tmpa21 = -tmp;
                minus0 = x0;
                x0 = plus0;
            } else if (i == (_xGrid - 1) && j == 0) {
                // upper left corner
                plus1 = d1[j + 1];
                hm_d0 = x0 - minus0;
                hp_d1 = plus1 - x1;

                DT tmpHmp = FPTwoMul(hm_d0, hp_d1);
                DT tmp = 1.0 / tmpHmp;
                tmpa22 = 0.0;
                tmpa21 = 0.0;
                tmpa20 = 0.0;
                tmpa10 = 0.0;
                tmpa00 = 0.0;
                tmpa01 = tmp;
                tmpa12 = tmp;
                tmpa11 = -tmp;
                tmpa02 = -tmp;

                x0 = d0[0];
                minus1 = x1;
                x1 = plus1;
            } else if (i == 0 && j == (_yGrid - 1)) {
                // lower right corner
                plus0 = d0[i + 1];
                hp_d0 = plus0 - x0;
                hm_d1 = x1 - minus1;

                DT tmpHpm = FPTwoMul(hp_d0, hm_d1);
                DT tmp = 1.0 / tmpHpm;
                tmpa00 = 0.0;
                tmpa01 = 0.0;
                tmpa02 = 0.0;
                tmpa12 = 0.0;
                tmpa22 = 0.0;
                tmpa10 = tmp;
                tmpa21 = tmp;
                tmpa20 = -tmp;
                tmpa11 = -tmp;

                minus0 = x0;
                x0 = plus0;
            } else if (i == (_xGrid - 1) && j == (_yGrid - 1)) {
                // upper right corner
                hm_d0 = x0 - minus0;
                hm_d1 = x1 - minus1;
                DT tmpHm = FPTwoMul(hm_d0, hm_d1);
                DT tmp = 1.0 / tmpHm;
                tmpa20 = 0.0;
                tmpa21 = 0.0;
                tmpa22 = 0.0;
                tmpa12 = 0.0;
                tmpa02 = 0.0;
                tmpa00 = tmp;
                tmpa11 = tmp;
                tmpa10 = -tmp;
                tmpa01 = -tmp;
            } else if (i == 0) {
                // lower side
                plus0 = d0[i + 1];
                plus1 = d1[j + 1];

                hp_d0 = plus0 - x0;
                hp_d1 = plus1 - x1;
                hm_d1 = x1 - minus1;

                phim1 = hm_d1 * (hm_d1 + hp_d1);
                phi0 = hm_d1 * hp_d1;
                phip1 = hp_d1 * (hm_d1 + hp_d1);

                tmpa00 = 0.0;
                tmpa01 = 0.0;
                tmpa02 = 0.0;
                tmpa20 = -(tmpa10 = hp_d1 / (hp_d0 * phim1));
                tmpa11 = -(tmpa21 = (hp_d1 - hm_d1) / (hp_d0 * phi0));
                tmpa12 = -(tmpa22 = hm_d1 / (hp_d0 * phip1));

                minus0 = x0;
                x0 = plus0;
            } else if (i == (_xGrid - 1)) {
                // upper side
                plus1 = d1[j + 1];
                hp_d1 = plus1 - x1;
                hm_d0 = x0 - minus0;
                hm_d1 = x1 - minus1;

                phim1 = hm_d1 * (hm_d1 + hp_d1);
                phi0 = hm_d1 * hp_d1;
                phip1 = hp_d1 * (hm_d1 + hp_d1);
                tmpa20 = 0.0;
                tmpa21 = 0.0;
                tmpa22 = 0.0;
                tmpa10 = -(tmpa00 = hp_d1 / (hm_d0 * phim1));
                tmpa01 = -(tmpa11 = (hp_d1 - hm_d1) / (hm_d0 * phi0));
                tmpa02 = -(tmpa12 = hm_d1 / (hm_d0 * phip1));
                x0 = d0[0];
                minus1 = x1;
                x1 = plus1;
            } else if (j == 0) {
                // left side
                plus0 = d0[i + 1];
                plus1 = d1[j + 1];
                hp_d0 = plus0 - x0;
                hp_d1 = plus1 - x1;
                hm_d0 = x0 - minus0;
                zetam1 = hm_d0 * (hm_d0 + hp_d0);
                zeta0 = hm_d0 * hp_d0;
                zetap1 = hp_d0 * (hm_d0 + hp_d0);
                tmpa00 = 0.0;
                tmpa10 = 0.0;
                tmpa20 = 0.0;
                tmpa02 = -(tmpa01 = hp_d0 / (zetam1 * hp_d1));
                tmpa11 = -(tmpa12 = (hp_d0 - hm_d0) / (zeta0 * hp_d1));
                tmpa21 = -(tmpa22 = hm_d0 / (zetap1 * hp_d1));
                minus0 = x0;
                x0 = plus0;
            } else if (j == (_yGrid - 1)) {
                // right side
                plus0 = d0[i + 1];
                hp_d0 = plus0 - x0;
                hm_d1 = x1 - minus1;
                hm_d0 = x0 - minus0;
                zetam1 = hm_d0 * (hm_d0 + hp_d0);
                zeta0 = hm_d0 * hp_d0;
                zetap1 = hp_d0 * (hm_d0 + hp_d0);
                tmpa22 = 0.0;
                tmpa12 = 0.0;
                tmpa02 = 0.0;
                tmpa01 = -(tmpa00 = hp_d0 / (zetam1 * hm_d1));
                tmpa10 = -(tmpa11 = (hp_d0 - hm_d0) / (zeta0 * hm_d1));
                tmpa20 = -(tmpa21 = hm_d0 / (zetap1 * hm_d1));
                minus0 = x0;
                x0 = plus0;
            } else {
                plus0 = d0[i + 1];
                plus1 = d1[j + 1];
                hp_d0 = plus0 - x0;
                hp_d1 = plus1 - x1;
                hm_d0 = x0 - minus0;
                hm_d1 = x1 - minus1;
                zetam1 = hm_d0 * (hm_d0 + hp_d0);
                phim1 = hm_d1 * (hm_d1 + hp_d1);
                zeta0 = hm_d0 * hp_d0;
                zetap1 = hp_d0 * (hm_d0 + hp_d0);
                phi0 = hm_d1 * hp_d1;
                phip1 = hp_d1 * (hm_d1 + hp_d1);
                tmpa00 = hp_d0 * hp_d1 / (zetam1 * phim1);
                tmpa10 = -(hp_d0 - hm_d0) * hp_d1 / (zeta0 * phim1);
                tmpa20 = -hm_d0 * hp_d1 / (zetap1 * phim1);
                tmpa01 = -hp_d0 * (hp_d1 - hm_d1) / (zetam1 * phi0);
                tmpa11 = (hp_d0 - hm_d0) * (hp_d1 - hm_d1) / (zeta0 * phi0);
                tmpa21 = hm_d0 * (hp_d1 - hm_d1) / (zetap1 * phi0);
                tmpa02 = -hp_d0 * hm_d1 / (zetam1 * phip1);
                tmpa12 = hm_d1 * (hp_d0 - hm_d0) / (zeta0 * phip1);
                tmpa22 = hm_d0 * hm_d1 / (zetap1 * phip1);

                minus0 = x0;
                x0 = plus0;
            }
            _corrMap.a11[at] = tmpa11 * u;
            _corrMap.a00[at] = tmpa00 * u;
            _corrMap.a01[at] = tmpa01 * u;
            _corrMap.a02[at] = tmpa02 * u;
            _corrMap.a10[at] = tmpa10 * u;
            _corrMap.a20[at] = tmpa20 * u;
            _corrMap.a21[at] = tmpa21 * u;
            _corrMap.a12[at] = tmpa12 * u;
            _corrMap.a22[at] = tmpa22 * u;
        }
    }
}

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
void FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::getState(Size i, Size j, DT rates[2]) {
#pragma HLS inline
    rates[0] = _locationsX[i];
    rates[1] = _locationsY[j];
}

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
void FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::applyTo(DT a[_layoutSizeMax], DT t) {
    // because discountBond is already inlined, allocaiton doesn't work.
    //#pragma HLS allocation function instances = &G2Model<DT, void, 0>::discountBond limit = 1
    DT iterExerciseTime = t;
    DT disRate[2], fwdRate[2];
    DT npv[_layoutSizeMax] = {0.0};
    DT cost = 0.0, profit = 0.0;

// exclude first time
loop_applyto:
    for (int iter = 0; iter < _exSize; iter++) {
        if (_accrualTime[iter] >= t) {
        loop_innervalue:
            for (int j = 0; j < _yGrid; j++) {
#pragma HLS loop_tripcount min = 51 max = 51
                for (int i = 0; i < _xGrid; i++) {
#pragma HLS loop_tripcount min = 51 max = 51
#pragma HLS pipeline II = 22
                    Size at = i + j * _xGrid;

                    // get x and y location
                    getState(i, j, disRate);
                    getState(i, j, fwdRate);
                    cost = _g2Model.discountBond(iterExerciseTime, _accrualTime[iter + 1], disRate) * _nominal *
                           _fixedRate;
                    DT accrualPeriod = _floatingAccrualPeriod[iter + 1] - _floatingAccrualPeriod[iter];
                    DT spanningTime = _iborPeriod[iter + 1] - _iborPeriod[iter];
                    DT disc1 = _g2Model.discountBond(iterExerciseTime, _iborTime[iter], fwdRate);
                    DT disc2 = _g2Model.discountBond(iterExerciseTime, _iborTime[iter + 1], fwdRate);
                    DT fixing = (disc1 / disc2 - 1.0);
                    DT amount = fixing / spanningTime * accrualPeriod * _nominal;

                    DT discount = _g2Model.discountBond(iterExerciseTime, _accrualTime[iter + 1], fwdRate);
                    profit = amount * discount;
#ifndef __SYNTHESIS__
                    npv[at] += std::max(0.0, profit - cost);
#else
                    npv[at] += hls::max(0.0, profit - cost);
#endif
                }
            }
        }
    }
// update npv to get better profit
loop_npv_update:
    for (int iter = 0; iter < _layoutSize; iter++) {
#pragma HLS loop_tripcount min = 2601 max = 2601
#pragma HLS pipeline II = 1
        if (npv[iter] > a[iter]) a[iter] = npv[iter];
    }
}

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
void FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::rollback(DT a[_layoutSizeMax],
                                                                                     DT from,
                                                                                     DT to) {
    DT dt = (from - to) / _tGrid, t = from;
    int j = _exSize - 1;
    _dt = dt;

    DT tmpX[2] = {0.0, 0.0};
    _shortRate = _g2Model.shortRate(from, tmpX, 0.0);

    if (_stoppingTimes[_exSize] == from) applyTo(a, from);

loop_rollback:
    for (int i = 0; i < _tGrid; ++i) {
        DT now = t, next = t - dt;
#ifndef __SYNTHESIS__
        if (std::fabs(to - next) < std::sqrt(XF_EPSILON))
#else
        if (hls::fabs(to - next) < hls::sqrt(XF_EPSILON))
#endif
            next = to;
        bool hit = false;

        if (next <= _stoppingTimes[j] && _stoppingTimes[j] < now) {
            // hitting a stopping time
            hit = true;
            _dt = now - _stoppingTimes[j];
            step_merge(a, now);
            if (_stoppingTimes[j] != _stoppingTimes[0]) applyTo(a, _stoppingTimes[j]);
            now = _stoppingTimes[j];
            j--;
        }

        // if we did hit
        if (hit) {
            if (now > next) {
                _dt = now - next;
                step_merge(a, now);
            }
            _dt = dt;
        } else {
            step_merge(a, now);
        }
        t -= dt;
    }
}

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
void FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::step_merge(DT a[_layoutSizeMax], DT t) {
#pragma HLS DEPENDENCE variable = a inter false
    DT addX, addY, addMixed;
    DT tmpX0, tmpX2, tmpY0, tmpY2, tmpi20, tmpi21, tmpi22;
    DT buffer[6], cache[6];
    Size rim1;
#pragma HLS ARRAY_PARTITION variable = buffer complete dim = 1
#pragma HLS ARRAY_PARTITION variable = cache complete dim = 1
    DT y[_layoutSizeMax], y0[_layoutSizeMax], retVal[_layoutSizeMax], split[_layoutSizeMax], hr[_layoutSizeMax];
    DT apply_y[_layoutSizeMax], rhs[_layoutSizeMax], yt[_layoutSizeMax];

    DT tmpThe = _theta * _dt, b = 1.0, tmpMu = _mu * _dt, bet;
    DT to = t - _dt;
    if (to < 0) to = 0.0;
    DT phi = setTime(to, t);

    buffer[0] = a[1 + _yGrid];
    buffer[1] = a[1];
    buffer[2] = buffer[0];
    buffer[3] = a[_yGrid];
    buffer[4] = a[0];
    buffer[5] = buffer[3];

loop_add_1:
    for (int j = 0; j < _yGrid; j++) {
#pragma HLS loop_tripcount min = 51 max = 51
        for (int i = 0; i < _xGrid; i++) {
#pragma HLS loop_tripcount min = 51 max = 51
#pragma HLS pipeline
            DT tmpAdd;
            Size at = i + j * _xGrid;
            hr[at] = -0.5 * (_locationsX[i] + _locationsY[j] + phi);
            DT diagX = _dxMap.diag[i] + hr[at];
            DT diagY = _dyMap.diag[j] + hr[at];
            if (i == (_xGrid - 1)) {
                // right boundary
                tmpi20 = buffer[0];
                tmpi21 = buffer[1];
                tmpi22 = buffer[2];
            } else if (i == 0 && j == 0) {
                // the first point
                tmpi20 = buffer[0];
                tmpi21 = buffer[1];
                tmpi22 = buffer[2];
            } else if (i == 0 && j == (_yGrid - 1)) {
                //
                buffer[3] = a[at - _yGrid];
                buffer[4] = a[at];
                buffer[5] = buffer[3];
                tmpi20 = buffer[0];
                tmpi21 = buffer[1];
                tmpi22 = buffer[2];
            } else if (i == 0) {
                // left boundary
                buffer[3] = a[at - _yGrid];
                buffer[4] = a[at];
                buffer[5] = a[at + _yGrid];
                tmpi20 = buffer[0];
                tmpi21 = buffer[1];
                tmpi22 = buffer[2];
            } else if (j == 0) {
                tmpi20 = a[at + 1 + _yGrid];
                tmpi21 = a[at + 1];
                tmpi22 = tmpi20;
            } else if (j == (_yGrid - 1)) {
                tmpi20 = a[at + 1 - _yGrid];
                tmpi21 = a[at + 1];
                tmpi22 = tmpi20;
            } else {
                tmpi20 = a[at + 1 - _yGrid];
                tmpi21 = a[at + 1];
                tmpi22 = a[at + 1 + _yGrid];
            }
            addMixed = _corrMap.a00[at] * buffer[0] + _corrMap.a01[at] * buffer[1] + _corrMap.a02[at] * buffer[2] +
                       _corrMap.a10[at] * buffer[3] + _corrMap.a11[at] * buffer[4] + _corrMap.a12[at] * buffer[5] +
                       _corrMap.a20[at] * tmpi20 + _corrMap.a21[at] * tmpi21 + _corrMap.a22[at] * tmpi22;
            addY = buffer[3] * _dyMap.lower[j] + buffer[4] * diagY + buffer[5] * _dyMap.upper[j];
            addX = buffer[1] * _dxMap.lower[i] + buffer[4] * diagX + tmpi21 * _dxMap.upper[i];
            tmpAdd = buffer[4] + _dt * (addMixed + addY + addX);
            y0[at] = tmpAdd;
            apply_y[at] = tmpThe * addY;
            // solve_splitting on x  positive sequence
            DT tmpRhs = tmpAdd - tmpThe * addX;
            if (at == 0) {
                bet = 1.0 / (-tmpThe * diagX + b);
                rhs[0] = tmpRhs * bet;
            } else {
                DT tmpLower = -tmpThe * _dxMap.lower[i];
                split[at] = -tmpThe * _dxMap.upper[rim1] * bet;
                DT tmpDiagX = -tmpThe * diagX;
                bet = b + tmpDiagX - split[at] * tmpLower;
                bet = 1.0 / bet;
                DT tmpVal = tmpLower * rhs[at - 1];
                DT tmpSub = tmpRhs - tmpVal;
                rhs[at] = tmpSub * bet;
            }
            rim1 = i;
            //
            if (i == (_xGrid - 1) && j == (_yGrid - 2)) {
                buffer[0] = a[at + 2 - _xGrid];
                buffer[1] = a[at + 2];
                buffer[2] = buffer[0];
            } else if (i == (_xGrid - 1)) {
                buffer[0] = a[at + 2 - _xGrid];
                buffer[1] = a[at + 2];
                buffer[2] = a[at + 2 + _xGrid];
            } else if (i == 0) {
                DT tmp = buffer[0];
                buffer[0] = buffer[3];
                buffer[3] = tmp;
                tmp = buffer[1];
                buffer[1] = buffer[4];
                buffer[4] = tmp;
                tmp = buffer[2];
                buffer[2] = buffer[5];
                buffer[5] = tmp;
            } else {
                buffer[0] = buffer[3];
                buffer[1] = buffer[4];
                buffer[2] = buffer[5];
                buffer[3] = tmpi20;
                buffer[4] = tmpi21;
                buffer[5] = tmpi22;
            }
        }
    }
    solve_reverse_x(rhs, split, y);
    subtraction_merge(tmpThe, b, y, apply_y, hr, split, rhs);
    // solve_splitting on direction y
    solve_reverse_y(rhs, split, y);

    // initialize cache for the first
    cache[0] = y[1 + _yGrid];
    cache[1] = y[1];
    cache[2] = cache[0];
    cache[3] = y[_yGrid];
    cache[4] = y[0];
    cache[5] = cache[3];
    buffer[0] = cache[0] - a[1 + _yGrid];
    buffer[1] = cache[1] - a[1];
    buffer[2] = buffer[0];
    buffer[3] = cache[3] - a[_yGrid];
    buffer[4] = cache[4] - a[0];
    buffer[5] = buffer[3];
loop_add_2:
    for (int j = 0; j < _yGrid; j++) {
#pragma HLS loop_tripcount min = 51 max = 51
        for (int i = 0; i < _xGrid; i++) {
#pragma HLS loop_tripcount min = 51 max = 51
#pragma HLS pipeline
            Size at = i + j * _xGrid;
            DT diagX = _dxMap.diag[i] + hr[at];
            DT diagY = _dyMap.diag[j] + hr[at];
            if (i == (_xGrid - 1)) {
                // right boundary
                tmpi20 = buffer[0];
                tmpi21 = buffer[1];
                tmpi22 = buffer[2];
                tmpY0 = cache[0];
                tmpX2 = cache[1];
                tmpY2 = cache[2];
            } else if (i == 0 && j == 0) {
                tmpi20 = buffer[0];
                tmpi21 = buffer[1];
                tmpi22 = buffer[2];
                tmpY0 = cache[0];
                tmpX2 = cache[1];
                tmpY2 = cache[2];
            } else if (i == 0 && j == (_yGrid - 1)) {
                DT y10 = y[at - _yGrid];
                DT y11 = y[at];

                buffer[3] = y10 - a[at - _yGrid];
                buffer[4] = y11 - a[at];
                buffer[5] = buffer[3];
                tmpi20 = buffer[0];
                tmpi21 = buffer[1];
                tmpi22 = buffer[2];

                cache[3] = y10;
                cache[4] = y11;
                cache[5] = cache[3];
                tmpY0 = cache[0];
                tmpX2 = cache[1];
                tmpY2 = cache[2];
            } else if (i == 0) {
                // left boundary
                DT y10 = y[at - _yGrid];
                DT y11 = y[at];
                DT y12 = y[at + _yGrid];
                buffer[3] = y10 - a[at - _yGrid];
                buffer[4] = y11 - a[at];
                buffer[5] = y12 - a[at + _yGrid];
                tmpi20 = buffer[0];
                tmpi21 = buffer[1];
                tmpi22 = buffer[2];

                cache[3] = y10;
                cache[4] = y11;
                cache[5] = y12;
                tmpY0 = cache[0];
                tmpX2 = cache[1];
                tmpY2 = cache[2];
            } else if (j == 0) {
                DT y20 = y[at + 1 + _yGrid];
                DT y21 = y[at + 1];

                tmpi20 = y20 - a[at + 1 + _yGrid];
                tmpi21 = y21 - a[at + 1];
                tmpi22 = tmpi20;
                tmpY0 = y20;
                tmpX2 = y21;
                tmpY2 = y20;
            } else if (j == (_yGrid - 1)) {
                DT y20 = y[at + 1 - _yGrid];
                DT y21 = y[at + 1];

                tmpi20 = y20 - a[at + 1 - _yGrid];
                tmpi21 = y21 - a[at + 1];
                tmpi22 = tmpi20;
                tmpY0 = y20;
                tmpX2 = y21;
                tmpY2 = y20;
            } else {
                DT y20 = y[at + 1 - _yGrid];
                DT y21 = y[at + 1];
                DT y22 = y[at + 1 + _yGrid];

                tmpi20 = y20 - a[at + 1 - _yGrid];
                tmpi21 = y21 - a[at + 1];
                tmpi22 = y22 - a[at + 1 + _yGrid];
                tmpY0 = y20;
                tmpX2 = y21;
                tmpY2 = y22;
            }
            addMixed = _corrMap.a00[at] * buffer[0] + _corrMap.a01[at] * buffer[1] + _corrMap.a02[at] * buffer[2] +
                       _corrMap.a10[at] * buffer[3] + _corrMap.a11[at] * buffer[4] + _corrMap.a12[at] * buffer[5] +
                       _corrMap.a20[at] * tmpi20 + _corrMap.a21[at] * tmpi21 + _corrMap.a22[at] * tmpi22;
            addY = buffer[3] * _dyMap.lower[j] + buffer[4] * diagY + buffer[5] * _dyMap.upper[j];
            addX = buffer[1] * _dxMap.lower[i] + buffer[4] * diagX + tmpi21 * _dxMap.upper[i];
            DT tmpAdd = y0[at] + tmpMu * (addMixed + addY + addX);
            // apply_direction on X and Y, y as input

            addY = cache[3] * _dyMap.lower[j] + cache[4] * diagY + cache[5] * _dyMap.upper[j];
            addX = cache[1] * _dxMap.lower[i] + cache[4] * diagX + tmpX2 * _dxMap.upper[i];
            apply_y[at] = tmpThe * addY;

            // solve_splitting on x
            DT tmpRhs = tmpAdd - tmpThe * addX;
            // splitting
            if (at == 0) {
                bet = 1.0 / (-tmpThe * diagX + b);
                rhs[0] = tmpRhs * bet;
            } else {
                DT tmpLower;
#pragma HLS RESOURCE variable = tmpLower core = DMul_nodsp
                tmpLower = -tmpThe * _dxMap.lower[i];
                split[at] = -tmpThe * _dxMap.upper[rim1] * bet;
                DT tmpDiagX;
#pragma HLS RESOURCE variable = tmpDiagX core = DMul_nodsp
                tmpDiagX = -tmpThe * diagX;

                bet = b + tmpDiagX - split[at] * tmpLower;
                bet = 1.0 / bet;
                DT tmpVal;
#pragma HLS RESOURCE variable = tmpVal core = DMul_nodsp
                tmpVal = tmpLower * rhs[at - 1];

                DT tmpSub;
#pragma HLS RESOURCE variable = tmpSub core = DAddSub_nodsp
                tmpSub = tmpRhs - tmpVal;
#pragma HLS RESOURCE variable = rhs core = DMul_nodsp
                rhs[at] = tmpSub * bet;
            }
            rim1 = i;
            // do a buffer cache
            if (i == (_xGrid - 1) && j == (_yGrid - 2)) {
                DT y00 = y[at + 2 - _xGrid];
                DT y01 = y[at + 2];
                buffer[0] = y00 - a[at + 2 - _xGrid];
                buffer[1] = y01 - a[at + 2];
                buffer[2] = buffer[0];

                cache[0] = y00;
                cache[1] = y01;
                cache[2] = y00;
            }
            if (i == (_xGrid - 1)) {
                // right boundary
                DT y00 = y[at + 2 - _xGrid];
                DT y01 = y[at + 2];
                DT y02 = y[at + 2 + _xGrid];
                buffer[0] = y00 - a[at + 2 - _xGrid];
                buffer[1] = y01 - a[at + 2];
                buffer[2] = y02 - a[at + 2 + _xGrid];

                cache[0] = y00;
                cache[1] = y01;
                cache[2] = y02;

            } else if (i == 0) {
                // left boundary
                DT tmp = buffer[0];
                buffer[0] = buffer[3];
                buffer[3] = tmp;
                tmp = buffer[1];
                buffer[1] = buffer[4];
                buffer[4] = tmp;
                tmp = buffer[2];
                buffer[2] = buffer[5];
                buffer[5] = tmp;

                tmp = cache[0];
                cache[0] = cache[3];
                cache[3] = tmp;
                tmp = cache[1];
                cache[1] = cache[4];
                cache[4] = tmp;
                tmp = cache[2];
                cache[2] = cache[5];
                cache[5] = tmp;
            } else {
                buffer[0] = buffer[3];
                buffer[1] = buffer[4];
                buffer[2] = buffer[5];
                buffer[3] = tmpi20;
                buffer[4] = tmpi21;
                buffer[5] = tmpi22;

                cache[0] = cache[3];
                cache[1] = cache[4];
                cache[2] = cache[5];
                cache[3] = tmpY0;
                cache[4] = tmpX2;
                cache[5] = tmpY2;
            }
        }
    }
    solve_reverse_x(rhs, split, yt);
    // the last splitting
    subtraction_merge(tmpThe, b, yt, apply_y, hr, split, rhs);
    solve_reverse_y(rhs, split, a);
}

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
void FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::subtraction_merge(
    DT tmpThe,
    DT b,
    DT y[_layoutSizeMax],
    DT apply_y[_layoutSizeMax],
    DT hr[_layoutSizeMax],
    DT split[_layoutSizeMax],
    DT rhs[_layoutSizeMax]) {
#pragma HLS inline
    DT buff, bet;
#pragma HLS RESOURCE variable = buff core = DMul_nodsp
    Size rim1;
loop_subtraction:
    for (int j = 0; j < _yGrid; j++) {
#pragma HLS loop_tripcount min = 51 max = 51
        for (int i = 0; i < _xGrid; i++) {
#pragma HLS loop_tripcount min = 51 max = 51
#pragma HLS pipeline

            Size ri = j + i * _yGrid;
            Size at = i + j * _xGrid;
            DT tmpRhs = y[ri] - apply_y[ri];
            DT diagY = _dyMap.diag[i] + hr[ri];
            if (at == 0) {
                bet = 1.0 / (-tmpThe * diagY + b);
                buff = tmpRhs * bet;
            } else {
                DT tmpLower = -tmpThe * _dyMap.lower[i];
                split[at] = -tmpThe * _dyMap.upper[rim1] * bet;
                DT tmpDiagY = -tmpThe * diagY;
                bet = b + tmpDiagY - split[at] * tmpLower;
                bet = 1.0 / bet;
                DT tmpVal = tmpLower * buff;
                DT tmpSub = tmpRhs - tmpVal;
                buff = tmpSub * bet;
            }
            rhs[ri] = buff;
            rim1 = i;
        }
    }
}

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
void FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::solve_reverse_x(DT r[_layoutSizeMax],
                                                                                            DT split[_layoutSizeMax],
                                                                                            DT retVal[_layoutSizeMax]) {
#pragma HLS inline
    DT buff;
    buff = r[_layoutSize - 1];
    retVal[_layoutSize - 1] = buff;
loop_solve_reverse_x:
    for (int j = _layoutSize - 2; j >= 0; --j) {
#pragma HLS pipeline
        DT temp;
#pragma HLS RESOURCE variable = temp core = DMul_nodsp
        temp = split[j + 1] * buff;
        retVal[j] = r[j] - temp;
        buff = retVal[j];
    }
}

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
void FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::solve_reverse_y(DT r[_layoutSizeMax],
                                                                                            DT split[_layoutSizeMax],
                                                                                            DT retVal[_layoutSizeMax]) {
#pragma HLS inline
    DT buff;
    buff = r[_layoutSize - 1];
    retVal[_layoutSize - 1] = buff;

loop_solve_reverse_y:
    for (int j = (int)_yGrid - 1; j >= 0; --j) {
        for (int i = (int)_xGrid - 1; i >= 0; --i) {
#pragma HLS pipeline
            Size at = i + j * _xGrid;
            Size ri = j + i * _yGrid;
            if (at == _layoutSize - 1) {
                buff = r[at];
                retVal[at] = buff;
            } else {
                DT temp;
#pragma HLS RESOURCE variable = temp core = DMul_nodsp
                temp = split[at + 1] * buff;
                retVal[ri] = r[ri] - temp;
                buff = retVal[ri];
            }
        }
    }
}

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
DT FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::setTime(DT t1, DT t2) {
#pragma HLS inline
    DT tmpX[2] = {0.0, 0.0};
    DT tmpRate = _g2Model.shortRate(t1, tmpX, 0.0);
    const DT phi = 0.5 * (tmpRate + _shortRate);
    _shortRate = tmpRate;

    return phi;
}

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
void FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::calculate() {
    // g2 model
    //_model.init(_a,_sigma,_b,_eta,_rho,_rate);
    _g2Model.initialization(_rate, _a, _sigma, _b, _eta, _rho);
    //_g2Model.preCalcu(_a, _b);
    // process
    xf::fintech::OrnsteinUhlenbeckProcess<DT> process1;
    process1.init(_a, _sigma, 0.0);
    xf::fintech::OrnsteinUhlenbeckProcess<DT> process2;
    process2.init(_b, _eta, 0.0);

    // mes_exerciseTimesher
    DT maturity = _stoppingTimes[_exSize];
    xf::fintech::Fdm1dMesher<DT, _xGridMax> xMesher;
    xMesher.init(process1, maturity, _invEps, _xGrid, _locationsX);
    xf::fintech::Fdm1dMesher<DT, _yGridMax> yMesher;
    yMesher.init(process2, maturity, _invEps, _yGrid, _locationsY);

    // calculate first and second derivative
    derivativeXY(_locationsX, _locationsY);
    // calculate mixed derivative
    DT s = _rho * _sigma * _eta;
    mixedDerivativeXY(_locationsX, _locationsY, s);

    // rollback
    rollback(_initialValues, _stoppingTimes[_exSize], 0.0);
}

template <typename DT, Size _exSize, Size _xGridMax, Size _yGridMax, Size _layoutSizeMax>
DT FdG2SwaptionEngine<DT, _exSize, _xGridMax, _yGridMax, _layoutSizeMax>::getNPV() {
    return _initialValues[(_layoutSize - 1) / 2];
}

} // xf
} // fintech

#endif // _XF_FINTECH_FDG2_SWAPTION_ENGINE_HPP_
