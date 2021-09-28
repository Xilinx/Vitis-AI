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
#ifndef _XF_FINTECH_MCENGINE_H_
#define _XF_FINTECH_MCENGINE_H_
/**
 * @file mc_engine.hpp
 * @brief This file includes implementation of pricing engine for different
 * option.
 */

#include "xf_fintech/bs_model.hpp"
#include "xf_fintech/early_exercise.hpp"
#include "xf_fintech/mc_simulation.hpp"
#include "xf_fintech/rng.hpp"
namespace xf {

namespace fintech {

using namespace internal;
#define MAX_SAMPLE 134217727
/**
 * @brief European Option Pricing Engine using Monte Carlo Method. This
 * implementation uses Black-Scholes valuation model.
 *
 * @tparam DT supported data type including double and float data type, which
 * decides the precision of result, default double-precision data type.
 * @tparam UN number of Monte Carlo Module in parallel, which affects the
 * latency and resources utilization, default 10.
 * @tparam Antithetic antithetic is used  for variance reduction, default this
 * feature is disabled.
 * @param underlying intial value of underlying asset at time 0.
 * @param volatility fixed volatility of underlying asset.
 * @param dividendYield the constant dividend rate for continuous dividends.
 * @param riskFreeRate risk-free interest rate.
 * @param timeLength the time length of contract from start to end.
 * @param strike the strike price also known as exericse price, which is settled
 * in the contract.
 * @param optionType option type. 1: put option, 0: call option.
 * @param seed array to store the inital seed for each RNG.
 * @param output output array.
 * @param requiredTolerance the tolerance required. If requiredSamples is not
 * set, when reaching the required tolerance, simulation will stop, default
 * 0.02.
 * @param requiredSamples the samples number required. When reaching the
 * required number, simulation will stop, default 1024.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time, default 100.
 * @param maxSamples the maximum sample number. When reaching it, the simulation
 * will stop, default 2,147,483,648.
 */
template <typename DT = double, int UN = 10, bool Antithetic = false>
void MCEuropeanEngine(DT underlying,
                      DT volatility,
                      DT dividendYield,
                      DT riskFreeRate, // model parameter
                      DT timeLength,
                      DT strike,
                      bool optionType, // option parameter
                      ap_uint<32>* seed,
                      DT* output,
                      DT requiredTolerance = 0.02,
                      unsigned int requiredSamples = 1024,
                      unsigned int timeSteps = 100,
                      unsigned int maxSamples = MAX_SAMPLE) {
    // number of samples per simulation
    const static int SN = 1024;

    // number of variate
    const static int VN = 1;

    // Step first or sample first for each simulation
    const static bool SF = true;

    // option style
    const OptionStyle sty = European;

    // antithetic enable or not
    // const static bool Antithetic = false;

    // RNG alias name
    typedef MT19937IcnRng<DT> RNG;

    BSModel<DT> BSInst;

    // path generator instance
    BSPathGenerator<DT, SF, SN, Antithetic> pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1

    // path pricer instance
    PathPricer<sty, DT, SF, SN, Antithetic> pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1

    // RNG sequence instance
    RNGSequence<DT, RNG> rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1

    // pre-process for "cold" logic.
    DT dt = timeLength / timeSteps;
    DT f_1 = internal::FPTwoMul(riskFreeRate, timeLength);
    DT discount = internal::FPExp(-f_1);

    BSInst.riskFreeRate = riskFreeRate;
    BSInst.dividendYield = dividendYield;
    BSInst.volatility = volatility;
    //
    BSInst.variance(dt);
    BSInst.stdDeviation();
    BSInst.updateDrift(dt);

    // configure the path generator and path pricer
    for (int i = 0; i < UN; ++i) {
#pragma HLS unroll
        // Path generator
        pathPriInst[i][0].optionType = optionType;
        pathPriInst[i][0].strike = strike;
        pathPriInst[i][0].underlying = underlying;
        pathPriInst[i][0].discount = discount;
        // Path pricer
        pathGenInst[i][0].BSInst = BSInst;
        // RNGSequnce
        rngSeqInst[i][0].seed[0] = seed[i];
    }

    // call monter carlo simulation
    DT price = mcSimulation<DT, RNG, BSPathGenerator<DT, SF, SN, Antithetic>, PathPricer<sty, DT, SF, SN, Antithetic>,
                            RNGSequence<DT, RNG>, UN, VN, SN>(timeSteps, maxSamples, requiredSamples, requiredTolerance,
                                                              pathGenInst, pathPriInst, rngSeqInst);

    // output the price of option
    output[0] = price;
}
/**
 * @brief path pricer bypass variant (interface compatible with standard MCEuropeanEngine)
 *
 * @tparam DT supported data type including double and float data type, which
 * decides the precision of result, default double-precision data type.
 * @tparam UN number of Monte Carlo Module in parallel, which affects the
 * latency and resources utilization, default 10.
 * @param underlying intial value of underlying asset at time 0.
 * @param volatility fixed volatility of underlying asset.
 * @param dividendYield the constant dividend rate for continuous dividends.
 * @param riskFreeRate risk-free interest rate.
 * @param timeLength the time length of contract from start to end.
 * @param strike the strike price also known as exericse price, which is settled
 * in the contract.
 * @param optionType option type. 1: put option, 0: call option.
 * @param seed array to store the inital seed for each RNG.
 * @param output output array.
 * @param requiredTolerance the tolerance required. If requiredSamples is not
 * set, when reaching the required tolerance, simulation will stop, default
 * 0.02.
 * @param requiredSamples the samples number required. When reaching the
 * required number, simulation will stop, default 1024.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time, default 100.
 * @param maxSamples the maximum sample number. When reaching it, the simulation
 * will stop, default 2,147,483,648.
 */
template <typename DT = double, int UN = 10>
void MCEuropeanPriBypassEngine(DT underlying,
                               DT volatility,
                               DT dividendYield,
                               DT riskFreeRate, // model parameter
                               DT timeLength,
                               DT strike,
                               bool optionType, // option parameter
                               ap_uint<32>* seed,
                               DT* output,
                               DT requiredTolerance = 0.02,
                               unsigned int requiredSamples = 1024,
                               unsigned int timeSteps = 100,
                               unsigned int maxSamples = MAX_SAMPLE) {
    // number of samples per simulation
    const static int SN = 1024;

    // number of variate
    const static int VN = 1;

    // Step first or sample first for each simulation
    const static bool SF = true;

    // option style
    const OptionStyle sty = EuropeanBypass;

    // antithetic enable or not
    const static bool Antithetic = false;

    // RNG alias name
    typedef MT19937IcnRng<DT> RNG;

    BSModel<DT> BSInst;

    // path generator instance
    BSPathGenerator<DT, SF, SN, Antithetic> pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1

    // path pricer instance
    PathPricer<sty, DT, SF, SN, Antithetic> pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1

    // RNG sequence instance
    RNGSequence<DT, RNG> rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1

    // pre-process for "cold" logic.
    DT dt = timeLength / timeSteps;
    DT f_1 = internal::FPTwoMul(riskFreeRate, timeLength);
    DT discount = internal::FPExp(-f_1);

    BSInst.riskFreeRate = riskFreeRate;
    BSInst.dividendYield = dividendYield;
    BSInst.volatility = volatility;
    //
    BSInst.variance(dt);
    BSInst.stdDeviation();
    BSInst.updateDrift(dt);

    // configure the path generator and path pricer
    for (int i = 0; i < UN; ++i) {
#pragma HLS unroll
        // Path generator
        pathPriInst[i][0].optionType = optionType;
        pathPriInst[i][0].strike = strike;
        pathPriInst[i][0].underlying = underlying;
        pathPriInst[i][0].discount = discount;
        // Path pricer
        pathGenInst[i][0].BSInst = BSInst;
        // RNGSequnce
        rngSeqInst[i][0].seed[0] = seed[i];
    }

    // call monter carlo simulation
    DT price = mcSimulation<DT, RNG, BSPathGenerator<DT, SF, SN, Antithetic>, PathPricer<sty, DT, SF, SN, Antithetic>,
                            RNGSequence<DT, RNG>, UN, VN, SN>(timeSteps, maxSamples, requiredSamples, requiredTolerance,
                                                              pathGenInst, pathPriInst, rngSeqInst);

    // output the price of option
    output[0] = price;
}

/**
 * @brief European Option Pricing Engine using Monte Carlo Method based on
 * Heston valuation model.
 *
 * @tparam DT supported data type including double and float, which decides the
 * precision of output, default double-precision data type.
 * @tparam UN number of Monte Carlo Module in parallel, which affects the
 * latency and resources utilization, default 10.
 * @tparam DiscreType methods which is used to discrete the stochastic process.
 * Currently, five discrete types, including kDTPartialTruncation,
 * kDTFullTruncation, kDTReflection, kDTQuadraticExponential and
 * kDTQuadraticExponentialMartingale, are supported, default
 * kDTQuadraticExponential.
 * @tparam Antithetic antithetic is used  for variance reduction, default this
 * feature is disabled.
 *
 * @param underlying the initial price of underlying asset at time 0.
 * @param riskFreeRate risk-free interest rate.
 * @param sigma the volatility of volatility
 * @param v0 initial volatility of stock
 * @param theta the long variance, as t tends to infinity, the expected value of
 * vt tends to theta.
 * @param kappa the rate at which vt tends to theta.
 * @param rho the correlation coefficient between price and variance.
 * @param dividendYield the constant dividend rate for continuous dividends.
 * @param optionType option type. 1: put option, 0: call option.
 * @param strike the strike price also known as exericse price, which is settled
 * in the contract.
 * @param timeLength time length from now to expiry date.
 * @param seed the seeds used to initialize RNG.
 * @param outputs pricing result array of this engine.
 * @param requiredTolerance the tolerance required. If requiredSamples is not
 * set, when reaching the required tolerance, simulation will stop.
 * @param requiredSamples the samples number required. When reaching the
 * required number, simulation will stop, default 1024.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time, default 100.
 * @param maxSamples the maximum sample number. When reaching it, the simulation
 * will stop, default 2,147,483,648.
 *
 */

template <typename DT = double,
          int UN = 8,
          DiscreType discretization = kDTQuadraticExponential,
          bool Antithetic = false>
void MCEuropeanHestonEngine(DT underlying,
                            DT riskFreeRate,
                            DT sigma,
                            DT v0,
                            DT theta,
                            DT kappa,
                            DT rho,
                            DT dividendYield,
                            bool optionType,
                            DT strike,
                            DT timeLength,
                            ap_uint<32> seed[UN][2],
                            DT* outputs,
                            DT requiredTolerance = 0.02,
                            unsigned int requiredSamples = 1024,
                            unsigned int timeSteps = 100,
                            unsigned int maxSamples = MAX_SAMPLE) {
    typedef MT19937IcnRng<DT> RNG;
    const static int SN = 1024;   // SampNum
    const static int VN = 2;      // VariateNum
    const static bool SF = false; // StepFirst
    const OptionStyle sty = European;
    // const static bool Antithetic = false;

    // parameters of Model
    DT dt = timeLength / timeSteps;                // dt
    DT sdt = hls::sqrt(dt);                        // sqrt(dt)
    DT kappa_dt = internal::FPTwoMul(kappa, dt);   // kappa * dt
    DT sigma_sdt = internal::FPTwoMul(sigma, sdt); // sigma * sqrt(dt)

    DT rho_sqr = internal::FPTwoMul(rho, rho);
    DT one_rho_sqr = internal::FPTwoSub((DT)1.0, rho_sqr);
    DT hov = hls::sqrt(one_rho_sqr); // sqrt(1 - rho * rho)

    DT ex = hls::exp(-kappa_dt); // exp(-kappa * dt)
    DT r_d_s = rho / sigma;
    DT k0 = -internal::FPTwoMul(internal::FPTwoMul(r_d_s, kappa_dt), theta); // -(rho * kappa * dt * theta) / sigma
    DT k1_1 = internal::FPTwoMul(kappa_dt * r_d_s, (DT)0.5);
    DT k1_2 = internal::FPTwoMul(dt, (DT)0.25);
    DT k1_3 = internal::FPTwoSub(k1_1, k1_2);
    DT k1 = internal::FPTwoSub(k1_3, r_d_s);               // 0.5 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    DT k2 = internal::FPTwoAdd(k1_3, r_d_s);               // 0.5 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    DT k3 = internal::FPTwoMul(dt * one_rho_sqr, (DT)0.5); // 0.5 * dt * (1 - rho * rho)
    DT k4 = k3;                                            // 0.5 * dt * (1 - rho * rho)
    DT A_1 = internal::FPTwoMul(k4, (DT)0.5);
    DT A = internal::FPTwoAdd(k2, A_1); // k2 + 0.5 * k4

    DT discount = hls::exp(-riskFreeRate * timeLength); //(riskFreeRate - dividendYield) * timeLength

    HestonPathGenerator<discretization, DT, SN, Antithetic> pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1

    PathPricer<European, DT, SF, SN, Antithetic> pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1

    // configure model on parameters calculated.
    for (int i = 0; i < UN; ++i) {
#pragma HLS unroll
        pathPriInst[i][0].optionType = optionType;
        pathPriInst[i][0].strike = strike;

        pathPriInst[i][0].discount = discount;
        pathPriInst[i][0].underlying = underlying;

        pathGenInst[i][0].underlying = 0;
        pathGenInst[i][0].dividendYield = dividendYield;
        pathGenInst[i][0].riskFreeRate = riskFreeRate;
        pathGenInst[i][0].v0 = v0;
        pathGenInst[i][0].kappa = kappa;
        pathGenInst[i][0].theta = theta;
        pathGenInst[i][0].sigma = sigma;
        pathGenInst[i][0].rho = rho;

        pathGenInst[i][0].hov = hov;

        pathGenInst[i][0].sdt = sdt;
        pathGenInst[i][0].dt = dt;

        pathGenInst[i][0].kappa_dt = kappa_dt;
        pathGenInst[i][0].sigma_sdt = sigma_sdt;

        pathGenInst[i][0].ex = ex;
        pathGenInst[i][0].k0 = k0;
        pathGenInst[i][0].k1 = k1;
        pathGenInst[i][0].k2 = k2;
        pathGenInst[i][0].k3 = k3;
        pathGenInst[i][0].k4 = k4;
        pathGenInst[i][0].A = A;
#ifdef GREEKS_TEST
        pathGenInst[i][0].updateDrift(0);
#else
        pathGenInst[i][0].updateDrift(dt);
#endif
    }
    // call mcSimulation
    DT price;
    if (discretization == kDTQuadraticExponential || discretization == kDTQuadraticExponentialMartingale) {
        RNGSequence_Heston_QuadraticExponential<DT, RNG> rngSeqInst_1[UN][1];
#pragma HLS array_partition variable = rngSeqInst_1 dim = 1
        for (int i = 0; i < UN; i++) {
#pragma HLS unroll
            rngSeqInst_1[i][0].seed[0] = seed[i][0];
            rngSeqInst_1[i][0].seed[1] = seed[i][1];
        }

        price = mcSimulation<DT, RNG, HestonPathGenerator<discretization, DT, SN, Antithetic>,
                             PathPricer<sty, DT, SF, SN, Antithetic>, RNGSequence_Heston_QuadraticExponential<DT, RNG>,
                             UN, VN, SN>(timeSteps, maxSamples, requiredSamples, requiredTolerance, pathGenInst,
                                         pathPriInst, rngSeqInst_1);
    } else {
        RNGSequence_2<DT, RNG> rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1
        for (int i = 0; i < UN; i++) {
#pragma HLS unroll
            rngSeqInst[i][0].seed[0] = seed[i][0];
            rngSeqInst[i][0].seed[1] = seed[i][1];
        }
        price = mcSimulation<DT, RNG, HestonPathGenerator<discretization, DT, SN, Antithetic>,
                             PathPricer<European, DT, SF, SN, Antithetic>, RNGSequence_2<DT, RNG>, UN, VN, SN>(
            timeSteps, maxSamples, requiredSamples, requiredTolerance, pathGenInst, pathPriInst, rngSeqInst);
    }
    outputs[0] = price;
}

/**
 * @brief Multiple Asset European Option Pricing Engine using Monte Carlo Method
 based on Heston valuation model.
 *
 * @tparam DT supported data type including double and float, which decides the
 precision of output, default double-precision data type.
 * @tparam ASSETS number of underlying assets supported.
 * @tparam UN number of Monte Carlo Module in parallel, which affects the
 latency and resources utilization, default 10.
 * @tparam DiscreType methods which is used to discrete the stochastic process.
 Currently, five discrete types, including kDTPartialTruncation,
 kDTFullTruncation, kDTReflection, kDTQuadraticExponential and
 kDTQuadraticExponentialMartingale,
 are supported, default kDTQuadraticExponential.
 *
 * @param underlying the initial price of underlying asset at time 0.
 * @param riskFreeRate risk-free interest rate.
 * @param sigma the volatility of volatility
 * @param v0 initial volatility of stock
 * @param theta the long variance, as t tends to infinity, the expected value of
 vt tends to theta.
 * @param kappa the rate at which vt tends to theta.
 * @param corrMatrix the LU decomposition result of correlation matrix of all
 stochastic variables, only none-zero elements are stored.
 * @param rho the correlation coefficient between price and variance.
 * @param dividendYield the constant dividend rate for continuous dividends.
 * @param optionType option type. 1: call option, 0: put option.
 * @param strike the strike price also known as exericse price, which is settled
 in the contract.
 * @param timeLength time length from now to expiry date.
 * @param seed the seeds used to initialize RNG.
 * @param outputs pricing result array of this engine.
 * @param requiredTolerance the tolerance required. If requiredSamples is not
 set, when reaching the required tolerance, simulation will stop.
 * @param requiredSamples the samples number required. When reaching the
 required number, simulation will stop, default 1024.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 time, default 100.
 * @param maxSamples the maximum sample number. When reaching it, the simulation
 will stop, default 2,147,483,648.
 *
 */
template <typename DT = double, int ASSETS = 5, int UN = 1, DiscreType discretization = kDTQuadraticExponential>
void MCMultiAssetEuropeanHestonEngine(DT underlying[ASSETS],
                                      DT riskFreeRate,
                                      DT sigma[ASSETS],
                                      DT v0[ASSETS],
                                      DT theta[ASSETS],
                                      DT kappa[ASSETS],
                                      DT corrMatrix[ASSETS * 2 + 1][ASSETS],
                                      DT rho[ASSETS],
                                      DT dividendYield[ASSETS],
                                      bool optionType,
                                      DT strike,
                                      DT timeLength,
                                      ap_uint<32> seed[UN][2],
                                      DT* outputs,
                                      DT requiredTolerance = 0.02,
                                      ap_uint<32> requiredSamples = 0,
                                      ap_uint<32> timeSteps = 100,
                                      ap_uint<32> maxSamples = MAX_SAMPLE) {
    typedef MT19937IcnRng<DT> RNG;
    const static int SN = 512; // SampNum
    const static int VN = 2;
    const static bool SF = false;
    const OptionStyle sty = European;
    const static bool Antithetic = false;

    DT discount = hls::exp(-riskFreeRate * timeLength);

    typedef MultiAssetHestonPathGenerator<DT, SN, ASSETS, discretization> PathGenType;
    PathGenType pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1

    typedef MultiAssetPathPricer<European, DT, ASSETS, SN> PathPricerType;
    PathPricerType pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1

    typedef CORRAND_2_Sequence<DT, RNG, SN, ASSETS, false> RNGSeqType;
    RNGSeqType rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1

    for (int i = 0; i < UN; i++) {
#pragma HLS unroll
        rngSeqInst[i][0].seed[0] = seed[i][0];
        rngSeqInst[i][0].seed[1] = seed[i][1];

        pathPriInst[i][0].optionType = optionType;
        pathPriInst[i][0].strike = strike;
        pathPriInst[i][0].discount = discount;

        for (int j = 0; j < ASSETS; j++) {
            pathPriInst[i][0].underlying[j] = underlying[j];

            pathGenInst[i][0].underlying[j] = 0;
            pathGenInst[i][0].v0[j] = v0[j];

            pathGenInst[i][0].hestonInst.initParam(ASSETS, timeLength, timeSteps, riskFreeRate, dividendYield, kappa,
                                                   theta, sigma, rho);
        }
        for (int j = 0; j < ASSETS * 2 + 1; j++) {
            for (int k = 0; k < ASSETS; k++) {
#pragma HLS PIPELINE II = 1
                rngSeqInst[i][0].corrand.corrMatrix[j][0][k] = corrMatrix[j][k];
            }
        }
    }
    DT price;
    price = mcSimulation<DT, RNG, PathGenType, PathPricerType, RNGSeqType, UN, VN, SN>(
        timeSteps, maxSamples, requiredSamples, requiredTolerance, pathGenInst, pathPriInst, rngSeqInst);

    outputs[0] = price;
}

/**
 * @brief American Option Pricing Engine using Monte Carlo Method.
 * PreSample kernel: this kernel samples some amount of path and store them to
 * external memory
 *
 * @tparam DT supported data type including double and float data type, which
 * decides the precision of result, default double-precision data type.
 * @tparam UN number of Monte Carlo Module in parallel (in path dimension),
 * which affects the latency and resources utilization, default 2.
 * @param underlying intial value of underlying asset at time 0.
 * @param volatility fixed volatility of underlying asset.
 * @param dividendYield the constant dividend rate for continuous dividends.
 * @param riskFreeRate risk-free interest rate.
 * @param timeLength the time length of contract from start to end.
 * @param strike the strike price also known as exericse price, which is settled
 * in the contract.
 * @param optionType option type. 1: put option, 0: call option.
 * @param seed array to store the inital seed for each RNG.
 * @param priceOut price data output, the data can be stored to HBM or DDR
 * @param matOut matrix output, the data can be stored to HBM or DDR
 * @param calibSamples sample numbers that used in calibration, default 4096.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time, default 100.
 */
template <typename DT = double, int UN = 2>
void MCAmericanEnginePreSamples(DT underlying,
                                DT volatility,
                                DT riskFreeRate,
                                DT dividendYield,
                                DT timeLength,
                                DT strike,
                                bool optionType,
                                ap_uint<32>* seed,
                                ap_uint<8 * sizeof(DT) * UN>* priceOut,
                                ap_uint<8 * sizeof(DT)>* matOut,
                                unsigned int calibSamples = 4096,
                                unsigned int timeSteps = 100) {
    // number of samples per simulation
    const static int SN = 1024;

    // number of variate
    const static int VN = 1;

    // Step first or sample first for each simulation
    const static bool SF = false;

    const static int ITER = 5;

    // option style
    const OptionStyle sty = American;

    // number of coefficients
    const static int COEFNM = 4;

    // size of data with datatype DT
    const static int SZ = 8 * sizeof(DT);

    // antithetic enable or not
    const static bool Antithetic = false;

    // RNG alias name
    typedef MT19937IcnRng<DT> RNG;

    // path generator instance
    BSPathGenerator<DT, SF, SN, Antithetic> pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1

    // path pricer instance
    PathPricer<American, DT, SF, SN, Antithetic> pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1

    // RNG sequence instance
    typedef RNGSequence<DT, RNG> RNGSeqT;
    RNGSeqT rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1

    // RNG instance
    RNG rngInst[UN][1];
#pragma HLS array_partition variable = rngInst dim = 1
    // B-S model instance
    BSModel<DT> BSInst;

    // pre-process for "cold" logic
    DT dt = timeLength / timeSteps;
    DT invStk = 1.0 / strike;

    BSInst.riskFreeRate = riskFreeRate;
    BSInst.dividendYield = dividendYield;
    BSInst.volatility = volatility;
    //
    BSInst.variance(dt);
    BSInst.stdDeviation();
    BSInst.updateDrift(dt);

    // configure the path generator and path pricer
    for (int i = 0; i < UN; ++i) {
#pragma HLS unroll
        // rng sqeuence
        rngSeqInst[i][0].seed[0] = seed[i];
        // path generator
        pathGenInst[i][0].BSInst = BSInst;
    }
    // Initialize RNG
    InitWrap<RNG, RNGSeqT, UN, VN>(rngInst, rngSeqInst);

    // calculate the iteraion needs to run = calibsamples/1024/UN
    int tmp = SN * UN;
    int iter = calibSamples / tmp;

    // the output number of mat data
    int mat_nm = timeSteps * (3 * (COEFNM - 1));

    hls::stream<int> phase_end;
#pragma HLS stream variable = &phase_end depth = 1

    // series of monte calro simulation process
    MCIteration<DT, RNG, UN, BSPathGenerator<DT, SF, SN, Antithetic>, PathPricer<sty, DT, SF, SN, Antithetic>,
                RNGSequence<DT, RNG>, VN, SN, COEFNM, ITER, SZ>(timeSteps, underlying, strike, invStk, mat_nm, iter,
                                                                rngInst, pathGenInst, pathPriInst, rngSeqInst, priceOut,
                                                                matOut, phase_end);

    phase_end.read();
}
/**
 * @brief American Option Pricing Engine using Monte Carlo Method.
 * Calibrate kernel: this kernel reads the sample price data from external
 * memory and use them to calculate the coefficient
 *
 * @tparam DT supported data type including double and float data type, which
 * decides the precision of result, default double-precision data type.
 * @tparam UN number of Monte Carlo Module in parallel (in path dimension),
 * which affects the latency and resources utilization, default 2. [this unroll
 * number should be equal to UN in MCAmericanEnginePresample]
 * @tparam UN_STEP number of Monte Carlo Module in parallel (in time steps
 * dimension), which affects the latency and resources utilization, default 2.
 * [this Unroll is completely resource bounded, unrelated to other params]
 * @param timeLength the time length of contract from start to end.
 * @param riskFreeRate risk-free interest rate.
 * @param strike the strike price also known as exericse price, which is settled
 * in the contract.
 * @param optionType option type. 1: put option, 0: call option.
 * @param priceIn the price data, read in from DDR or HBM
 * @param matIn the matrix data, read in from DDR or HBM
 * @param coefOut the coef data that calculated by this kernel, the data can be
 * stored to DDR or HBM
 * @param calibSamples sample numbers that used in calibration, default 4096.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time, default 100.
 */

template <typename DT = double, int UN = 2, int UN_STEP = 2>
void MCAmericanEngineCalibrate(DT timeLength,
                               DT riskFreeRate,
                               DT strike,
                               bool optionType,
                               ap_uint<8 * sizeof(DT) * UN>* priceIn,
                               ap_uint<8 * sizeof(DT)>* matIn,
                               ap_uint<8 * sizeof(DT) * 4>* coefOut,
                               unsigned int calibSamples = 4096,
                               unsigned int timeSteps = 100) {
#pragma HLS inline off

    // number of samples per simulation
    const static int SN = 1024;

    // number of variate
    const static int VN = 1;

    // order of polynomial for LongStaffShwartz
    const static int COEFNM = 4;

    // Max sampels for Caliration
    const static int CalSample = 4096;

    // pre-process of "cold" logic
    DT dt = timeLength / timeSteps;
    DT invStk = 1 / strike;
    DT discount = internal::FPExp(-1.0 * riskFreeRate * dt);

#pragma HLS dataflow
    // intermediate streams used to buffer data between dataflow functions
    hls::stream<ap_uint<8 * sizeof(double) * UN> > pStrm;
#pragma HLS stream variable = &pStrm depth = 8
    hls::stream<DT> inStrm[UN];
#pragma HLS stream variable = inStrm depth = 8
#pragma HLS array_partition variable = inStrm dim = 0
    hls::stream<DT> xStrm;
#pragma HLS stream variable = &xStrm depth = 9
    hls::stream<DT> xStrm_un[UN_STEP];
#pragma HLS stream variable = xStrm_un depth = 9
#pragma HLS array_partition variable = xStrm_un dim = 0
    hls::stream<DT> mUstrm[UN_STEP];
#pragma HLS stream variable = mUstrm depth = 16
#pragma HLS array_partition variable = mUstrm dim = 0
    hls::stream<DT> mVstrm[UN_STEP];
#pragma HLS stream variable = mVstrm depth = 16
#pragma HLS array_partition variable = mVstrm dim = 0
    hls::stream<DT> mSstrm[UN_STEP];
#pragma HLS stream variable = mSstrm depth = 16
#pragma HLS array_partition variable = mSstrm dim = 0
    hls::stream<DT> Ustrm;
#pragma HLS stream variable = &Ustrm depth = 16
    hls::stream<DT> Vstrm;
#pragma HLS stream variable = &Vstrm depth = 16
    hls::stream<DT> Sstrm;
#pragma HLS stream variable = &Sstrm depth = 16
    hls::stream<DT> coefStrm[COEFNM];
#pragma HLS stream variable = coefStrm depth = 16
#pragma HLS array_partition variable = coefStrm dim = 0

    // read price mat data from DDR
    readin_ddr<DT, UN, SN>(calibSamples / UN / SN, timeSteps, priceIn, pStrm);

    // Because the data are stored in differnet locations of DDR, prepare the
    // complete data for calib process
    read_merge<DT, UN>(calibSamples * timeSteps / UN, pStrm, inStrm);

    // read m mat data from DDR
    read_AtA<DT, UN_STEP, COEFNM>(timeSteps, matIn, xStrm);

    SplitStrm<DT, COEFNM, UN_STEP>(timeSteps / UN_STEP, xStrm, xStrm_un);

    // calc SVD
    MultiSVD<DT, COEFNM, UN_STEP>(timeSteps / UN_STEP, xStrm_un, mUstrm, mVstrm, mSstrm);

    MergeStrm<DT, COEFNM, UN_STEP>(timeSteps / UN_STEP, mUstrm, mVstrm, mSstrm, Ustrm, Vstrm, Sstrm);

    // calculate the coeff // TODO iteration should be as a para in CalCoef
    // //iteration = calibratesamples / unroll / 1024
    CalCoef<DT, COEFNM, CalSample, UN>(timeSteps, calibSamples / UN, optionType, discount, strike, invStk, Ustrm, Vstrm,
                                       Sstrm, inStrm, coefStrm);

    // write the coeff data to DDR, the data width is COEFNM* double
    write_ddr<DT, COEFNM, 8 * sizeof(DT)>(timeSteps - 1, coefStrm, coefOut);
}

/**
 * @brief American Option Pricing Engine using Monte Carlo Method.
 * Pricing kernel
 *
 * @tparam DT supported data type including double and float data type, which
 * decides the precision of result, default double-precision data type.
 * @tparam UN number of Monte Carlo Module in parallel (in path dimension),
 * which affects the latency and resources utilization, default 2. [this unroll
 * number should be equal to UN in MCAmericanEnginePresample]
 * @param underlying intial value of underlying asset at time 0.
 * @param volatility fixed volatility of underlying asset.
 * @param dividendYield the constant dividend rate for continuous dividends.
 * @param riskFreeRate risk-free interest rate.
 * @param timeLength the time length of contract from start to end.
 * @param strike the strike price also known as exericse price, which is settled
 * in the contract.
 * @param optionType option type. 1: put option, 0: call option.
 * @param seed array of seed to initialize RNG.
 * @param coefIn the coef data that calculated by this kernel, the data is
 * loaded from DDR or HBM
 * @param output the output price data (size=1), can be stroed in DDR or HBM
 * @param requiredTolerance the tolerance required. If requiredSamples is not
 * set, when reaching the required tolerance, simulation will stop, default
 * 0.02.
 * @param requiredSamples the samples number required. When reaching the
 * required number, simulation will stop, default 1024.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time.
 * @param maxSamples the maximum sample number. When reaching it, the simulation
 * will stop, default 2,147,483,648.
 */

template <typename DT = double, int UN = 2>
void MCAmericanEnginePricing(DT underlying,
                             DT volatility,
                             DT dividendYield,
                             DT riskFreeRate,
                             DT timeLength,
                             DT strike,
                             bool optionType,
                             ap_uint<32>* seed,
                             ap_uint<8 * sizeof(DT) * 4>* coefIn,
                             DT* output,
                             DT requiredTolerance = 0.02,
                             unsigned int requiredSamples = 4096,
                             unsigned int timeSteps = 100,
                             unsigned int maxSamples = MAX_SAMPLE) {
    // Sample number for each simulation
    const static int SN = 1024;

    // number of maximum time steps for pricing
    const int MAXSTEPS = 1024;

    // number of variate
    const int VN = 1;

    // number of coeff
    const int COEFNM = 4;

    // Step first or sample first for each simulation
    const static bool SF = true;

    // option style
    const OptionStyle sty = LongstaffSchwartz;

    // antithetic enable or not
    const static bool Antithetic = false;

    // RNG alias name
    typedef MT19937IcnRng<DT> RNG;
    // B-S model instance
    BSModel<DT> BSInst;

    // path generator instance
    BSPathGenerator<DT, SF, SN, Antithetic> pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1

    // path pricer instance
    PathPricer<sty, DT, SF, SN, Antithetic, MAXSTEPS> pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1

    // RNG sequence instance
    RNGSequence<DT, RNG> rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1

    // the flag that indicates Calibrate ends
    hls::stream<int> phase2_end;
#pragma HLS stream variable = &phase2_end depth = 1
    phase2_end.write(1);

    // read in the coef from BRAM
    read_coef<DT, UN, 8 * sizeof(DT), COEFNM, SF, SN, MAXSTEPS, Antithetic>(phase2_end, timeSteps - 1, coefIn,
                                                                            pathPriInst);

    // pre-process for "cold" logic.
    DT dt = timeLength / timeSteps;
    DT recipUnderLying = 1.0 / underlying;
    DT strike2 = strike / underlying;
    DT recipStrike = 1.0 / strike2;
    DT riskConst = internal::FPTwoMul(riskFreeRate, dt);
    riskConst = -riskConst;
    DT riskrate_noexp = riskConst * timeSteps;
    DT riskRate = hls::exp(riskrate_noexp);

    BSInst.riskFreeRate = riskFreeRate;
    BSInst.dividendYield = dividendYield;
    BSInst.volatility = volatility;
    //
    BSInst.variance(dt);
    BSInst.stdDeviation();
    BSInst.updateDrift(dt);
    // configure the path generator and path pricer
    for (int i = 0; i < UN; ++i) {
#pragma HLS unroll
        // RNG Sequence
        rngSeqInst[i][0].seed[0] = seed[i];
        // path Generator
        pathGenInst[i][0].BSInst = BSInst;
        // path pricer
        pathPriInst[i][0].recipUnderLying = recipUnderLying;
        pathPriInst[i][0].recipStrike = recipStrike;
        pathPriInst[i][0].riskConst = riskConst;
        pathPriInst[i][0].riskRate = riskRate;
        pathPriInst[i][0].optionType = optionType;
        pathPriInst[i][0].strike = strike2;
    }

    DT price = mcSimulation<DT, RNG, BSPathGenerator<DT, SF, SN, Antithetic>,
                            PathPricer<sty, DT, SF, SN, Antithetic, MAXSTEPS>, RNGSequence<DT, RNG>, UN, VN, SN>(
        timeSteps, maxSamples, requiredSamples, requiredTolerance, pathGenInst, pathPriInst, rngSeqInst);

    output[0] = price * underlying;
}

/**
 * @brief American Option Pricing Engine using Monte Carlo Method.
 *  calibration process and pricing process all in one kernel
 *
 * @tparam DT supported data type including double and float data type, which
 * decides the precision of result, default double-precision data type.
 * @tparam UN_PATH number of Monte Carlo Module of calibration process in
 * parallel (in path dimension), which affects the latency and resources
 * utilization, default 1.
 * @tparam UN_STEP number of Monte Carlo Module of calibration process in
 * parallel (in time steps dimension), which affects the latency and resources
 * utilization, default 1.
 * @tparam UN_PRICING number of Monte Carlo Module of pricing process in
 * parallel (in path dimension), which affects the latency and resources
 * utilization, default 2.
 *  Three unroll numbers UN_PATH, UN_STEP and UN_PRICING are independent. Each
 * unroll number affects the execution speed of part of AmericanEngine. On-board
 * test reveals that
 *  the pricing phase takes longest time of execution. Therefore, setting larger
 * UN_PRICING benifits most with limited resources.
 * @param underlying intial value of underlying asset at time 0.
 * @param volatility fixed volatility of underlying asset.
 * @param dividendYield the constant dividend rate for continuous dividends.
 * @param riskFreeRate risk-free interest rate.
 * @param timeLength the time length of contract from start to end.
 * @param strike the strike price also known as exericse price, which is settled
 * in the contract.
 * @param optionType option type. 1: put option, 0: call option.
 * @param seed array of seed to initialize RNG.
 * @param priceData the price data, used as the intermediate result storage
 * location. It should be mapped to an external memory: DDR or HBM.
 * @param matData the matrix data, used as the intermediate result storage
 * location. It should be mapped to an external memory: DDR or HBM.
 * @param output the output price data (size=1), can be stroed in DDR or HBM
 * @param requiredTolerance the tolerance required. If requiredSamples is not
 * set, when reaching the required tolerance, simulation will stop, default
 * 0.02.
 * @param requiredSamples the samples number required. When reaching the
 * required number, simulation will stop, default 1024.
 * @param calibSamples sample numbers that used in calibration, default 4096.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time.
 * @param maxSamples the maximum sample number. When reaching it, the simulation
 * will stop, default 2,147,483,648.
 */

template <typename DT = double, int UN_PATH = 1, int UN_STEP = 1, int UN_PRICING = 2>
void MCAmericanEngine(DT underlying,
                      DT volatility,
                      DT riskFreeRate,
                      DT dividendYield,
                      DT timeLength,
                      DT strike,
                      bool optionType,
                      ap_uint<32>* seed,
                      ap_uint<8 * sizeof(DT) * UN_PATH>* priceData,
                      ap_uint<8 * sizeof(DT)>* matData,
                      DT* output,
                      DT requiredTolerance = 0.02,
                      unsigned int calibSamples = 4096,
                      unsigned int requiredSamples = 4096,
                      unsigned int timeSteps = 100,
                      unsigned int maxSamples = MAX_SAMPLE) {
    // number of samples per simulation
    const static int SN = 1024;

    // number of variate
    const static int VN = 1;

    // Step first or sample first for each simulation
    const static bool SF = false;

    const static int ITER = 5;

    // number of maximum timesteps in pricing
    const int MAXSTEPS = 1024;

    // option style for calibration process
    const OptionStyle sty_calib = American;

    // option style for pricing process
    const OptionStyle sty_price = LongstaffSchwartz;

    // number of coefficients
    const static int COEFNM = 4;

    // size of data with datatype DT
    const static int SZ = 8 * sizeof(DT);

    // antithetic enable or not
    const static bool Antithetic = false;

    // RNG alias name
    typedef MT19937IcnRng<DT> RNG;

    // For the instances that used/shared in both calibration and pricing process,
    // the instance number of unroll equals max(UN_PATH, UN_PRICING)
    const int UN_MAX = UN_PATH > UN_PRICING ? UN_PATH : UN_PRICING;
    // B-S model instance
    BSModel<DT> BSInst;

    // path generator instance used in Calibration process
    BSPathGenerator<DT, SF, SN, Antithetic> pathGenInst[UN_MAX][1];
#pragma HLS array_partition variable = pathGenInst dim = 1

    // path pricer instance used Calibration process
    PathPricer<sty_calib, DT, SF, SN, Antithetic> pathPriCalInst[UN_PATH][1];
#pragma HLS array_partition variable = pathPriCalInst dim = 1

    // path pricer instance used in pricing process
    PathPricer<sty_price, DT, SF, SN, Antithetic, MAXSTEPS> pathPriPriInst[UN_PRICING][1];
#pragma HLS array_partition variable = pathPriPriInst dim = 1

    // RNG instance
    RNG rngInst[UN_MAX][1];
#pragma HLS array_partition variable = rngInst dim = 1

    // RNG Sequence instance
    typedef RNGSequence<DT, RNG> RNGSeqT;
    RNGSeqT rngSeqInst[UN_MAX][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1

    // pre-process for "cold" logic
    DT dt = timeLength / timeSteps;
    DT invStk = 1.0 / strike;

    DT recipUnderLying = 1.0 / underlying;
    DT strike2 = strike / underlying;
    DT recipStrike = 1.0 / strike2;
    DT riskConst = internal::FPTwoMul(riskFreeRate, dt);
    riskConst = -riskConst;
    DT riskrate_noexp = riskConst * timeSteps;
    DT riskRate = hls::exp(riskrate_noexp);

    BSInst.riskFreeRate = riskFreeRate;
    BSInst.dividendYield = dividendYield;
    BSInst.volatility = volatility;
    //
    BSInst.variance(dt);
    BSInst.stdDeviation();
    BSInst.updateDrift(dt);

    // configure the rng sequence and pathgen
    for (int i = 0; i < UN_MAX; ++i) {
#pragma HLS unroll
        // rng sequence
        rngSeqInst[i][0].seed[0] = seed[i];
        // path generator
        pathGenInst[i][0].BSInst = BSInst;
    }

    // Initialize RNG
    InitWrap<RNG, RNGSeqT, UN_MAX, VN>(rngInst, rngSeqInst);

    // calculate the iteraion needs to run = calibsamples/1024/UN
    int tmp = SN * UN_PATH;
    int iter = calibSamples / tmp;

    // the output number of mat data
    int mat_nm = timeSteps * (3 * (COEFNM - 1));

    // the flag that indicates MCIteration ends
    hls::stream<int> phase1_end;
#pragma HLS stream variable = &phase1_end depth = 1
    // the flag that indicates Calibrate ends
    hls::stream<int> phase2_end;
#pragma HLS stream variable = &phase2_end depth = 1

    // calibration process - presamples
    // series of monte calro simulation process
    MCIteration<DT, RNG, UN_PATH, BSPathGenerator<DT, SF, SN, Antithetic>,
                PathPricer<sty_calib, DT, SF, SN, Antithetic>, RNGSequence<DT, RNG>, VN, SN, COEFNM, ITER, SZ>(
        timeSteps, underlying, strike, invStk, mat_nm, iter, rngInst, pathGenInst, pathPriCalInst, rngSeqInst,
        priceData, matData, phase1_end);

    ap_uint<8 * sizeof(DT) * 4> coeffData[MAXSTEPS];

    // calibration process - calibrate
    MCAmericanEngineCalibrateCalc<double, UN_PATH, UN_STEP>(phase1_end, phase2_end, timeLength, riskFreeRate, strike,
                                                            optionType, priceData, matData, coeffData, calibSamples,
                                                            timeSteps);

    // Pricing process
    // read in the coef from DDR
    read_coef<DT, UN_PRICING, 8 * sizeof(DT), COEFNM, SF, SN, MAXSTEPS, Antithetic>(phase2_end, timeSteps - 1,
                                                                                    coeffData, pathPriPriInst);

    // configure the path generator and path pricer
    for (int i = 0; i < UN_PRICING; ++i) {
#pragma HLS unroll
        // path pricer
        pathPriPriInst[i][0].recipUnderLying = recipUnderLying;
        pathPriPriInst[i][0].recipStrike = recipStrike;
        pathPriPriInst[i][0].riskConst = riskConst;
        pathPriPriInst[i][0].riskRate = riskRate;
        pathPriPriInst[i][0].optionType = optionType;
        pathPriPriInst[i][0].strike = strike2;
    }

    // run with mcSimulation framework to obtain the final optimal exercise price
    DT price =
        mcSimulation<DT, RNG, BSPathGenerator<DT, SF, SN, Antithetic>,
                     PathPricer<sty_price, DT, SF, SN, Antithetic, MAXSTEPS>, RNGSequence<DT, RNG>, UN_PRICING, VN, SN>(
            timeSteps, maxSamples, requiredSamples, requiredTolerance, pathGenInst, pathPriPriInst, rngSeqInst);

    output[0] = price * underlying;
}

/**
 * @brief Asian Arithmetic Average Price Engine using Monte Carlo Method Based
 * on Black-Scholes Model.
 * The settlement price of the underlying asset at expiry time is the geomertic
 * average of asset price during the option lifetime.
 * @tparam DT Supported data type including double and float, which decides the
 * precision of output.
 * @tparam UN The number of Monte Carlo Module in parallel, which affects the
 * latency and resources utilization.
 * @param underlying The initial price of underlying asset.
 * @param volatility The market's price volatility.
 * @param dividendYield The dividend yield is the company's total annual
 * dividend payments divided by its market capitalization, or the dividend per
 * share, divided by the price per share.
 * @param riskFreeRate The risk-free interest rate is the rate of return of a
 * hypothetical investment with no risk of financial loss, over a given period
 * of time.
 * @param timeLength The given period of time.
 * @param strike The strike price also known as exericse price, which is settled
 * in the contract.
 * @param optionType Option type. 1: put option, 0: call option.
 * @param seed array of seed to initialize RNG.
 * @param output Output array.
 * @param requiredTolerance  The tolerance required. If requiredSamples is not
 * set, simulation will not stop, unless the requiredTolerance is reached,
 * default 0.02.
 * @param requiredSamples  The samples number required. When reaching the
 * required number, simulation will stop, default 1024.
 * @param timeSteps Number of interval, default 100.
 * @param maxSamples The maximum sample number. When reaching it, the
 * simulation will stop, default 2147483648.
 *
 */
template <typename DT = double, int UN = 16>
void MCAsianGeometricAPEngine(DT underlying,
                              DT volatility,
                              DT dividendYield,
                              DT riskFreeRate, // process
                              DT timeLength,
                              DT strike,
                              bool optionType, // option
                              ap_uint<32>* seed,
                              DT* output,
                              DT requiredTolerance = 0.02,
                              unsigned int requiredSamples = 1024,
                              unsigned int timeSteps = 100,
                              unsigned int maxSamples = MAX_SAMPLE) {
    // Number of Samples per simulation
    const static int SN = 1024; // SampNum

    // Number of Variate
    const static int VN = 1; // VariateNum

    // Step first or Sample first
    const static bool SF = false; // StepFirst

    // RNG alias
    typedef MT19937IcnRng<DT> RNG;

    // Enable Antithetic or not
    // const static bool Antithetic = false;
    const static bool Antithetic = true;

    // Define Asian Geometric Average Price option type
    const OptionStyle sty = Asian_GP;

    // B-S model instatnce
    BSModel<DT> BSInst;

    // Path generator instance
    BSPathGenerator<DT, SF, SN, Antithetic> pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1

    // Path pricer instance
    PathPricer<sty, DT, SF, SN, Antithetic> pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1

    // RNG sequence Instance
    RNGSequence<DT, RNG> rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1

    // Pre-process of "cold" logic
    DT dt = timeLength / (DT)timeSteps; //((int)((360*timeLength/(DT)timeSteps)+0.5))/(DT)360;
    DT tmpExp = internal::FPTwoMul(riskFreeRate, timeLength);
    DT discount = internal::FPExp(-tmpExp);

    BSInst.riskFreeRate = riskFreeRate;
    BSInst.dividendYield = dividendYield;
    BSInst.volatility = volatility;
    //
    BSInst.variance(dt);
    BSInst.stdDeviation();
    BSInst.updateDrift(dt);

    for (int i = 0; i < UN; ++i) {
#pragma HLS unroll
        // Path Pricer
        pathPriInst[i][0].optionType = optionType;
        pathPriInst[i][0].underlying = underlying;
        pathPriInst[i][0].strike = strike;
        pathPriInst[i][0].discount = discount;

        // Path Generator
        pathGenInst[i][0].BSInst = BSInst;

        // RNG Sequence
        rngSeqInst[i][0].seed[0] = seed[i];
    }
    DT price = mcSimulation<DT, RNG, BSPathGenerator<DT, SF, SN, Antithetic>, PathPricer<sty, DT, SF, SN, Antithetic>,
                            RNGSequence<DT, RNG>, UN, VN, SN>(timeSteps, maxSamples, requiredSamples, requiredTolerance,
                                                              pathGenInst, pathPriInst, rngSeqInst);

    output[0] = price;
}

/**
 * @brief Asian Arithmetic Average Price Engine using Monte Carlo Method Based
 * on Black-Scholes Model.
 * The settlement price of the underlying asset at expiry time is the arithmetic
 * average of asset price during the option lifetime.
 * @tparam DT Supported data type including double and float, which decides the
 * precision of output.
 * @tparam UN The number of Monte Carlo Module in parallel, which affects the
 * latency and resources utilization.
 * @param underlying The initial price of underlying asset.
 * @param volatility The market's price volatility.
 * @param dividendYield The dividend yield is the company's total annual
 * dividend payments divided by its market capitalization, or the dividend per
 * share, divided by the price per share.
 * @param riskFreeRate The risk-free interest rate is the rate of return of a
 * hypothetical investment with no risk of financial loss, over a given period
 * of time.
 * @param timeLength The given period of time.
 * @param strike The strike price also known as exericse price, which is settled
 * in the contract.
 * @param optionType Option type. 1: put option, 0: call option.
 * @param seed array of seed to initialize RNG.
 * @param output Output array.
 * @param requiredTolerance  The tolerance required. If requiredSamples is not
 * set, simulation will not stop, unless the requiredTolerance is reached,
 * default 0.02.
 * @param requiredSamples  The samples number required. When reaching the
 * required number, simulation will stop, default 1024.
 * @param timeSteps Number of interval, default 100.
 * @param maxSamples The maximum sample number. When reaching it, the
 * simulation will stop, default 2147483648.
 *
 */

template <typename DT = double, int UN = 16>
void MCAsianArithmeticAPEngine(DT underlying,
                               DT volatility,
                               DT dividendYield,
                               DT riskFreeRate, // process
                               DT timeLength,
                               DT strike,
                               bool optionType, // option
                               ap_uint<32>* seed,
                               DT* output,
                               DT requiredTolerance = 0.02,
                               unsigned int requiredSamples = 1024,
                               unsigned int timeSteps = 100,
                               unsigned int maxSamples = MAX_SAMPLE) {
    // Number of Samples per simulation
    const static int SN = 1024; // SampNum

    // Number of Variate
    const static int VN = 1; // VariateNum

    // Step first or Sample first
    const static bool SF = false; // StepFirst

    // RNG alias
    typedef MT19937IcnRng<DT> RNG;

    // Enable Antithetic or not
    const static bool Antithetic = false;

    // Define Asian Average Strike option type
    const OptionStyle sty = Asian_AP;

    // B-S model instance
    BSModel<DT> BSInst;

    // Path generator instance
    BSPathGenerator<DT, SF, SN, Antithetic> pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1

    // Path pricer instance
    PathPricer<sty, DT, SF, SN, Antithetic> pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1

    // RNG sequence Instance
    RNGSequence<DT, RNG> rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1

    // Pre-process of "cold" logic
    // DT dt           = ((int)((360.0*timeLength/(timeSteps+1))+0.5))/360.0;
    DT dt = timeLength / ((DT)timeSteps);
    //   timeSteps = 329;
    //   dt = 0.0277778;
    DT tmpSub = internal::FPTwoSub(riskFreeRate, dividendYield);
    // DT tmpExp   =   internal::FPTwoMul(tmpSub, timeLength);
    DT tmpExp = internal::FPTwoMul(riskFreeRate, timeLength);
    DT discount = hls::exp(-tmpExp); // DT discount = internal::FPExp(-1.0 * riskFreeRate * timeLength);

    BSInst.riskFreeRate = riskFreeRate;
    BSInst.dividendYield = dividendYield;
    BSInst.volatility = volatility;

    BSInst.variance(dt);
    BSInst.stdDeviation();
    BSInst.updateDrift(dt);
    // Configure path generator,path pricer and RNG sequence.
    for (int i = 0; i < UN; ++i) {
#pragma HLS unroll
        // Path Pricer
        pathPriInst[i][0].optionType = optionType;
        pathPriInst[i][0].underlying = underlying;
        pathPriInst[i][0].strike = strike;
        pathPriInst[i][0].discount = discount;
        pathPriInst[i][0].dt = dt;

        // Path Generator
        pathGenInst[i][0].BSInst = BSInst;

        // RNG Sequence
        rngSeqInst[i][0].seed[0] = seed[i];
    }

    DT price = mcSimulation<DT, xf::fintech::MT19937IcnRng<DT>, BSPathGenerator<DT, SF, SN, Antithetic>,
                            PathPricer<sty, DT, SF, SN, Antithetic>, RNGSequence<DT, xf::fintech::MT19937IcnRng<DT> >,
                            UN, VN, SN>(timeSteps, maxSamples, requiredSamples, requiredTolerance, pathGenInst,
                                        pathPriInst, rngSeqInst);

    // Control variate price ref
    DT fixings = timeSteps + 1;
    DT timeSum = (timeSteps + 1) * timeLength * 0.5;
    // DT temp=(timeSteps-1)*(timeSteps+1)*timeSteps/6.0*dt;
    DT temp = timeSum * (timeSteps - 1) / 3.0;
    DT tempFC = 2 * temp + timeSum;
    DT sqrtFC = hls::sqrt(tempFC);
    DT tempvf = volatility / fixings;

    DT variance = tempvf * tempvf * tempFC;
    DT nu = riskFreeRate - dividendYield - 0.5 * volatility * volatility;
    DT muG = hls::log(underlying) + nu * timeLength * 0.5;
    DT forwardPrice = std::exp(muG + variance * 0.5);
    DT stDev = hls::sqrt(variance);
    DT d1 = hls::log(forwardPrice / strike) / stDev + 0.5 * stDev;
    DT d2 = d1 - stDev;
    DT cum_d1 = CumulativeNormal<DT>(d1);
    DT cum_d2 = CumulativeNormal<DT>(d2);
    DT alpha, beta;
    if (optionType) {
        alpha = -1 + cum_d1;
        beta = 1 - cum_d2;
    } else {
        alpha = cum_d1;
        beta = -cum_d2;
    }
    DT priceRef = discount * (forwardPrice * alpha + strike * beta);
    // output result
    output[0] = price + priceRef;
}

/**
 * @brief Asian Arithmetic Average Strike Engine using Monte Carlo Method Based
 * on Black-Scholes Model.
 * The settlement price of the underlying asset at expiry time is the asset price at expiry time, but the stock price is
 * the arithmetic average of asset price during the option lifetime.
 *
 * @tparam DT Supported data type including double and float, which decides the
 * precision of output.
 * @tparam UN The number of Monte Carlo Module in parallel, which affects the
 * latency and resources utilization.
 * @param underlying The initial price of underlying asset.
 * @param volatility The market's price volatility.
 * @param dividendYield The dividend yield is the company's total annual
 * dividend payments divided by its market capitalization, or the dividend per
 * share, divided by the price per share.
 * @param riskFreeRate The risk-free interest rate is the rate of return of a
 * hypothetical investment with no risk of financial loss, over a given period
 * of time.
 * @param timeLength The given period of time.
 * @param strike The strike price also known as exericse price, which is settled
 * in the contract.
 * @param optionType Option type. 1: put option, 0: call option.
 * @param seed array of seed to initialize RNG.
 * @param output Output array.
 * @param requiredTolerance the tolerance required. If requiredSamples is not
 * set, when reaching the required tolerance, simulation will stop, default
 * 0.02.
 * @param requiredSamples the samples number required. When reaching the
 * required number, simulation will stop, default 1024.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time, default 100.
 * @param maxSamples the maximum sample number. When reaching it, the
 * simulation will stop, default 2,147,483,648.
 *
 */
template <typename DT = double, int UN = 16>
void MCAsianArithmeticASEngine(DT underlying,
                               DT volatility,
                               DT dividendYield,
                               DT riskFreeRate, // process
                               DT timeLength,
                               DT strike,
                               bool optionType, // option
                               ap_uint<32>* seed,
                               DT* output,
                               DT requiredTolerance = 0.02,
                               unsigned int requiredSamples = 1024,
                               unsigned int timeSteps = 100,
                               unsigned int maxSamples = MAX_SAMPLE) {
    // Number of Samples per simulation
    const static int SN = 1024; // SampNum

    // Number of Variate
    const static int VN = 1; // VariateNum

    // Step first or Sample first
    const static bool SF = false; // StepFirst

    // RNG alias
    typedef MT19937IcnRng<DT> RNG;

    // Enable Antithetic or not
    const static bool Antithetic = true;

    // Define Asian Average Strike option type
    const OptionStyle sty = Asian_AS;

    // B-S Model Instance
    BSModel<DT> BSInst;

    // Path generator instance
    BSPathGenerator<DT, SF, SN, Antithetic> pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1

    // Path pricer instance
    PathPricer<sty, DT, SF, SN, Antithetic> pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1

    // RNG sequence Instance
    RNGSequence<DT, RNG> rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1

    // Pre-process of "cold" logic
    DT dt = timeLength / (DT)timeSteps; //((int)((360*timeLength/(DT)timeSteps)+0.5))/(DT)360;
    DT tmpExp = internal::FPTwoMul(riskFreeRate, timeLength);
    DT discount = hls::exp(-tmpExp); // DT discount = internal::FPExp(-1.0 * riskFreeRate * timeLength);

    BSInst.riskFreeRate = riskFreeRate;
    BSInst.dividendYield = dividendYield;
    BSInst.volatility = volatility;
    //
    BSInst.variance(dt);
    BSInst.stdDeviation();
    BSInst.updateDrift(dt);

    // Configure path generator,path pricer and RNG sequence.
    for (int i = 0; i < UN; ++i) {
#pragma HLS unroll
        // Path Pricer
        pathPriInst[i][0].optionType = optionType;
        pathPriInst[i][0].underlying = underlying;
        pathPriInst[i][0].strike = strike;
        pathPriInst[i][0].discount = discount;

        // Path Generator
        pathGenInst[i][0].BSInst = BSInst;

        // RNG Sequence
        rngSeqInst[i][0].seed[0] = seed[i];
    }

    DT price = mcSimulation<DT, RNG, BSPathGenerator<DT, SF, SN, Antithetic>, PathPricer<sty, DT, SF, SN, Antithetic>,
                            RNGSequence<DT, RNG>, UN, VN, SN>(timeSteps, maxSamples, requiredSamples, requiredTolerance,
                                                              pathGenInst, pathPriInst, rngSeqInst);

    // Output the option price
    output[0] = price;
}

/**
 * @brief Barrier Option Pricing Engine using Monte Carlo Simulation. Using
 * brownian bridge to generate the non-biased result.
 *
 * @tparam DT supported data type including double and float data type, which
 * decides the precision of result, default double-precision data type.
 * @tparam UN number of Monte Carlo Module in parallel, which affects the
 * latency and resources utilization, default 10.
 * @param underlying intial value of underlying asset at time 0.
 * @param volatility fixed volatility of underlying asset.
 * @param dividendYield the constant dividend rate for continuous dividends.
 * @param riskFreeRate risk-free interest rate.
 * @param timeLength the time length of contract from start to end.
 * @param barrier single barrier value.
 * @param strike the strike price also known as exericse price, which is settled
 * in the contract.
 * @param barrierType barrier type including: DownIn(0), DownOut(1), UpIn(2),
 * UpOut(3).
 * @param optionType option type. 1: put option, 0: call option.
 * @param seed array to store the inital seeds of RNG.
 * @param output output array.
 * @param rebate rebate value which is paid when the option is not triggered,
 * default 0.
 * @param requiredTolerance the tolerance required. If requiredSamples is not
 * set, when reaching the required tolerance, simulation will stop, default
 * 0.02.
 * @param requiredSamples the samples number required. When reaching the
 * required number, simulation will stop, default 1024.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time, default 100.
 * @param maxSamples the maximum sample number. When reaching it, the
 * simulation will stop, default 2,147,483,648.
 */
template <typename DT = double, int UN = 10>
void MCBarrierNoBiasEngine(DT underlying,
                           DT volatility,
                           DT dividendYield,
                           DT riskFreeRate,
                           DT timeLength, // Model Parameter
                           DT barrier,
                           DT strike,
                           ap_uint<2> barrierType,
                           bool optionType, // option parameter.
                           ap_uint<32>* seed,
                           DT* output,
                           DT rebate = 0,
                           DT requiredTolerance = 0.02,
                           unsigned int requiredSamples = 1024,
                           unsigned int timeSteps = 100,
                           unsigned int maxSamples = MAX_SAMPLE) {
    // Number of Samples per simulation
    const static int SN = 1024; // SampNum

    // Number of Variate
    const static int VN = 2; // VariateNum

    // Step first or Sample first
    const static bool SF = false; // StepFirst

    // RNG alias
    typedef MT19937IcnRng<DT> RNG;

    // Enable Antithetic or not
    const static bool Antithetic = true;

    // Path generator instance
    BSPathGenerator<DT, SF, SN, Antithetic> pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1

    // Path pricer instance
    PathPricer<BarrierNoBiased, DT, SF, SN, Antithetic> pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1

    // RNG sequence Instance
    GaussUniformSequence<DT, RNG, Antithetic> rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1

    // pre-process "cold" logic.
    DT dt = timeLength / timeSteps;
    DT u = internal::FPTwoSub(riskFreeRate, dividendYield);
    DT ut = internal::FPTwoMul(u, dt);
    DT volSq = internal::FPTwoMul(volatility, volatility);
    DT var = internal::FPTwoMul(volSq, dt);
    DT varH = internal::FPTwoMul((DT)0.5, var);
    DT drift = internal::FPTwoSub(ut, varH);
    DT sqrtVar = hls::sqrt(var);
    DT varDoub = 2 * var;
    DT disDt = -internal::FPTwoMul(riskFreeRate, dt);

    // Configure path generator,path pricer and RNG sequence.
    for (int i = 0; i < UN; ++i) {
#pragma HLS unroll
        // Path Pricer
        pathPriInst[i][0].barrier = barrier;
        pathPriInst[i][0].strike = strike;
        pathPriInst[i][0].rebate = rebate;
        pathPriInst[i][0].underlying = underlying;
        pathPriInst[i][0].optionType = optionType;
        pathPriInst[i][0].drift = drift;
        pathPriInst[i][0].sqrtVar = sqrtVar;
        pathPriInst[i][0].varDoub = varDoub;
        pathPriInst[i][0].disDt = disDt;
        pathPriInst[i][0].barrierType = BarrierType((int)barrierType);
        // Rng Sequence
        rngSeqInst[i][0].seed[0] = seed[i];
        rngSeqInst[i][0].seed[1] = 5;
    }
    // Monte Carlo Simulation
    DT price = mcSimulation<DT, RNG, BSPathGenerator<DT, SF, SN, Antithetic>,
                            PathPricer<BarrierNoBiased, DT, SF, SN, Antithetic>,
                            GaussUniformSequence<DT, RNG, Antithetic>, UN, VN, SN>(
        timeSteps, maxSamples, requiredSamples, requiredTolerance, pathGenInst, pathPriInst, rngSeqInst);

    // Output the option price
    output[0] = price;
}
/**
 * @brief Barrier Option Pricing Engine using Monte Carlo Simulation.
 *
 * @tparam DT supported data type including double and float data type, which
 * decides the precision of result, default double-precision data type.
 * @tparam UN number of Monte Carlo Module in parallel, which affects the
 * latency and resources utilization, default 10.
 * @param underlying intial value of underlying asset at time 0.
 * @param volatility fixed volatility of underlying asset.
 * @param dividendYield the constant dividend rate for continuous dividends.
 * @param riskFreeRate risk-free interest rate.
 * @param timeLength the time length of contract from start to end.
 * @param barrier single barrier value.
 * @param strike the strike price also known as exericse price, which is settled
 * in the contract.
 * @param barrierType barrier type including: DownIn(0), DownOut(1), UpIn(2),
 * UpOut(3).
 * @param optionType option type. 1: put option, 0: call option.
 * @param seed array to store the inital seeds for each RNG.
 * @param output output array.
 * @param rebate rebate value which is paid when the option is not triggered,
 * default 0.
 * @param requiredTolerance the tolerance required. If requiredSamples is not
 * set, when reaching the required tolerance, simulation will stop, default
 * 0.02.
 * @param requiredSamples the samples number required. When reaching the
 * required number, simulation will stop, default 1024.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time, default 100.
 * @param maxSamples the maximum sample number. When reaching it, the
 * simulation will stop, default 2,147,483,648.
 */
template <typename DT = double, int UN = 10>
void MCBarrierEngine(DT underlying,
                     DT volatility,
                     DT dividendYield,
                     DT riskFreeRate,
                     DT timeLength, // Model parameter
                     DT barrier,
                     DT strike,
                     ap_uint<2> barrierType,
                     bool optionType, // option parameter
                     ap_uint<32>* seed,
                     DT* output,
                     DT rebate = 0,
                     DT requiredTolerance = 0.02,
                     unsigned int requiredSamples = 1024,
                     unsigned int timeSteps = 100,
                     unsigned int maxSamples = MAX_SAMPLE) {
    // number of samples per simulation
    const static int SN = 1024; // SampNum

    // number of variate
    const static int VN = 1; // VariateNum

    // step first or sample first
    const static bool SF = false; // StepFirst

    // Antithetic enable or not.
    const static bool Antithetic = false;

    // RNG alias.
    typedef MT19937IcnRng<DT> RNG;

    // B-S model instance
    BSModel<DT> BSInst;

    // Path generator instance.
    BSPathGenerator<DT, SF, SN, Antithetic> pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1

    // Path Pricer instance
    PathPricer<BarrierBiased, DT, SF, SN, Antithetic> pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1

    // RGn sequence generator instance
    RNGSequence<DT, RNG> rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1

    // pre-process for "cold" logic
    DT dt = timeLength / timeSteps;
    DT disDt = -internal::FPTwoMul(riskFreeRate, dt);

    BSInst.riskFreeRate = riskFreeRate;
    BSInst.dividendYield = dividendYield;
    BSInst.volatility = volatility;
    //
    BSInst.variance(dt);
    BSInst.stdDeviation();
    BSInst.updateDrift(dt);
    // configure path generator, paht pricer and RNG sequence generator.
    for (int i = 0; i < UN; ++i) {
#pragma HLS unroll
        // Path Pricer
        pathPriInst[i][0].underlying = underlying;
        pathPriInst[i][0].barrier = barrier;
        pathPriInst[i][0].strike = strike;
        pathPriInst[i][0].rebate = rebate;
        pathPriInst[i][0].optionType = optionType;
        pathPriInst[i][0].disDt = disDt;
        pathPriInst[i][0].barrierType = BarrierType(int(barrierType));
        // Path generator
        pathGenInst[i][0].BSInst = BSInst;
        // RNG sequence
        rngSeqInst[i][0].seed[0] = seed[i];
    }
    // Monte Carlo simulation
    DT price = mcSimulation<DT, RNG, BSPathGenerator<DT, SF, SN, Antithetic>,
                            PathPricer<BarrierBiased, DT, SF, SN, Antithetic>, RNGSequence<DT, RNG>, UN, VN, SN>(
        timeSteps, maxSamples, requiredSamples, requiredTolerance, pathGenInst, pathPriInst, rngSeqInst);
    // output the option price
    output[0] = price;
}

/**
 * @brief Cliquet Option Pricing Engine using Monte Carlo Simulation.
 * The B-S model used to describe the dynamics of undelying asset.
 *
 * @tparam DT supported data type including double and float data type, which
 * decides the precision of result, default double-precision data type.
 * @tparam UN number of Monte Carlo Module in parallel, which affects the
 * latency and resources utilization, default 10.
 * @param underlying intial value of underlying asset.
 * @param volatility fixed volatility of underlying asset.
 * @param dividendYield the constant dividend rate for continuous dividends.
 * @param riskFreeRate risk-free interest rate.
 * @param timeLength the time length of contract from start to end.
 * @param strike the strike price also known as exericse price, which is settled
 * in the contract.
 * @param optionType option type. 1: put option, 0: call option.
 * @param resetDates array for reset dates, such as Semiannual, Quarterly.
 * @param seed array to store the inital seeds of RNG.
 * @param output output array.
 * @param requiredTolerance the tolerance required. If requiredSamples is not
 * set, when reaching the required tolerance, simulation will stop. default
 * 0.02.
 * @param requiredSamples the samples number required. When reaching the
 * required number, simulation will stop, default 1024.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time, default 100.
 * @param maxSamples the maximum sample number. When reaching it, the
 * simulation will stop, default 2,147,483,648.
 */
template <typename DT = double, int UN = 10>
void MCCliquetEngine(DT underlying,
                     DT volatility,
                     DT dividendYield,
                     DT riskFreeRate,
                     DT timeLength, // Process
                     DT strike,
                     bool optionType,
                     DT* resetDates,
                     ap_uint<32>* seed,
                     DT* output,
                     DT requiredTolerance = 0.02,
                     unsigned int timeSteps = 100,
                     unsigned int requiredSamples = 1024,
                     unsigned int maxSamples = MAX_SAMPLE) {
    // number of samples per simulation
    const static int SN = 1024; // SampNum

    // number of variate
    const static int VN = 1; // VariateNum

    // Step first or Sample first
    const static bool SF = false; // StepFirst

    // Antithetic enable or not.
    const static bool Antithetic = false;

    // RNG alias
    typedef MT19937IcnRng<DT> RNG;
    // path generator instance
    BSPathGenerator<DT, SF, SN, Antithetic> pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1
    // Path pricer instance
    PathPricer<Cliquet, DT, SF, SN, Antithetic> pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1
    // RNG sequence instance
    RNGSequence<DT, RNG> rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1
    // pre-process "cold" logic
    DT volSqr = internal::FPTwoMul(volatility, volatility);
    DT u = internal::FPTwoSub(riskFreeRate, dividendYield);
    DT volSqrH = internal::FPTwoMul((DT)0.5, volSqr);
    DT drift = internal::FPTwoSub(u, volSqrH);
    DT riskFreeRateNeg = -riskFreeRate;
    // configure path pricer and path generator.
    for (int i = 0; i < UN; ++i) {
#pragma HLS unroll
        pathPriInst[i][0].optionType = optionType;
        pathPriInst[i][0].strike = strike;
        pathPriInst[i][0].riskFreeRateNeg = riskFreeRateNeg;
        pathPriInst[i][0].driftRate = drift;
        pathPriInst[i][0].volSq = volSqr;
        // Rng Sequence
        rngSeqInst[i][0].seed[0] = seed[i];
    }
Init_Option_Para:
    for (int i = 0; i < timeSteps; ++i) {
#pragma HLS loop_tripcount min = 8 max = 8
#pragma HLS pipeline II = 1
        for (int j = 0; j < UN; ++j) {
#pragma HLS unroll
            pathPriInst[j][0].resetDates[i] = resetDates[i];
        }
    }
    // Monte Carlo Simulation
    DT price = mcSimulation<DT, RNG, BSPathGenerator<DT, SF, SN, Antithetic>,
                            PathPricer<Cliquet, DT, SF, SN, Antithetic>, RNGSequence<DT, RNG>, UN, VN, SN>(
        timeSteps, maxSamples, requiredSamples, requiredTolerance, pathGenInst, pathPriInst, rngSeqInst);
    // out put option price
    output[0] = price;
}
/**
 * @brief Digital Option Pricing Engine using Monte Carlo Simulation.
 * The B-S model is used to describe dynamics of undelying asset price.
 *
 * @tparam DT supported data type including double and float data type, which
 * decides the precision of result, default double-precision data type.
 * @tparam UN number of Monte Carlo Module in parallel, which affects the
 * latency and resources utilization, default 10.
 * @param underlying intial value of underlying asset at time 0.
 * @param volatility fixed volatility of underlying asset.
 * @param dividendYield the constant dividend rate for continuous dividends.
 * @param riskFreeRate risk-free interest rate.
 * @param timeLength the time length of contract from start to end.
 * @param strike the strike price also known as exericse price, which is settled
 * in the contract.
 * @param cashPayoff fixed payoff when option is exercised.
 * @param exEarly exercise early or not, true: option exercise at anytime.
 * false: option only exericse at expiry time.
 * @param optionType option type. 1: put option, 0: call option.
 * @param seed array to store the inital seeds for each RNG.
 * @param output output array.
 * @param requiredTolerance the tolerance required. If requiredSamples is not
 * set, when reaching the required tolerance, simulation will stop, default
 * 0.02.
 * @param requiredSamples the samples number required. When reaching the
 * required number, simulation will stop, default 1024.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time, default 100.
 * @param maxSamples the maximum sample number. When reaching it, the
 * simulation will stop, default 2,147,483,648.
 */
template <typename DT = double, int UN = 10>
void MCDigitalEngine(DT underlying,
                     DT volatility,
                     DT dividendYield,
                     DT riskFreeRate,
                     DT timeLength,
                     DT strike,
                     DT cashPayoff,
                     bool optionType,
                     bool exEarly,
                     ap_uint<32>* seed,
                     DT* output,
                     DT requiredTolerance = 0.02,
                     unsigned int timeSteps = 100,
                     unsigned int requiredSamples = 1024,
                     unsigned int maxSamples = MAX_SAMPLE) {
    // Sample number for each simulation
    const static int SN = 1024; // SampNumW
    // VariateNum
    const static int VN = 2; // VariateNum
    // Step first or sample first
    const static bool SF = false; // StepFirst
    // Antithetic enable or not
    const static bool Antithetic = true;
    // RNG aliase
    typedef MT19937IcnRng<DT> RNG;
    // path generator instance
    BSPathGenerator<DT, SF, SN, Antithetic> pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1
    // path pricer instance
    PathPricer<Digital, DT, SF, SN, Antithetic> pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1
    // RNG sequence instance
    GaussUniformSequence<DT, RNG, Antithetic> rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1
    // pre-process "cold" logic
    DT dt = timeLength / timeSteps;
    DT u = internal::FPTwoSub(riskFreeRate, dividendYield);
    DT ut = internal::FPTwoMul(u, dt);
    DT volSq = internal::FPTwoMul(volatility, volatility);
    DT var = internal::FPTwoMul(volSq, dt);
    DT varH = internal::FPTwoMul((DT)0.5, var);
    DT drift = internal::FPTwoSub(ut, varH);
    DT sqrtVar = hls::sqrt(var);
    DT varDoub = 2 * var;
    DT disDt = -internal::FPTwoMul(riskFreeRate, dt);
    DT log_spot = hls::log(underlying);
    DT log_strike = hls::log(strike);
    for (int i = 0; i < UN; ++i) {
#pragma HLS unroll
        pathPriInst[i][0].cashPayoff = cashPayoff;
        pathPriInst[i][0].optionType = optionType;
        pathPriInst[i][0].exEarly = exEarly;
        pathPriInst[i][0].log_strike = log_strike;
        pathPriInst[i][0].log_spot = log_spot;
        pathPriInst[i][0].drift = drift;
        pathPriInst[i][0].varSqrt = sqrtVar;
        pathPriInst[i][0].varDoub = varDoub;
        pathPriInst[i][0].disDt = disDt;
        // RNG sequence
        rngSeqInst[i][0].seed[0] = seed[i];
        rngSeqInst[i][0].seed[1] = 76;
    }
    // Monte Carlo simulation
    DT price =
        mcSimulation<DT, RNG, BSPathGenerator<DT, SF, SN, Antithetic>, PathPricer<Digital, DT, SF, SN, Antithetic>,
                     GaussUniformSequence<DT, RNG, Antithetic>, UN, VN, SN>(
            timeSteps, maxSamples, requiredSamples, requiredTolerance, pathGenInst, pathPriInst, rngSeqInst);

    output[0] = price;
}
/**
 * @brief European Option Greeks Calculating Engine using Monte Carlo Method
 * based on Heston valuation model.
 *
 * @tparam DT supported data type including double and float, which decides the
 * precision of output, default double-precision data type.
 * @tparam UN number of Monte Carlo Module in parallel, which affects the
 * latency and resources utilization, default 10.
 * @tparam DiscreType methods which is used to discrete the stochastic process.
 * Currently, five discrete types, including kDTPartialTruncation,
 * kDTFullTruncation, kDTReflection, kDTQuadraticExponential and
 * kDTQuadraticExponentialMartingale, are supported, default
 * kDTQuadraticExponential.
 *
 * @param underlying the initial price of underlying asset at time 0.
 * @param riskFreeRate risk-free interest rate.
 * @param sigma the volatility of volatility
 * @param v0 initial volatility of stock
 * @param theta the long variance, as t tends to infinity, the expected value of
 * vt tends to theta.
 * @param kappa the rate at which vt tends to theta.
 * @param rho the correlation coefficient between price and variance.
 * @param dividendYield the constant dividend rate for continuous dividends.
 * @param optionType option type. 1: put option, 0: call option.
 * @param strike the strike price also known as exericse price, which is settled
 * in the contract.
 * @param timeLength time length from now to expiry date.
 * @param seed the seeds used to initialize RNG.
 * @param greeks output calculated greeks.
 * @param requiredTolerance the tolerance required. If requiredSamples is not
 * set, when reaching the required tolerance, simulation will stop.
 * @param requiredSamples the samples number required. When reaching the
 * required number, simulation will stop, default 1024.
 * @param timeSteps the number of discrete steps from 0 to T, T is the expiry
 * time, default 100.
 * @param maxSamples the maximum sample number. When reaching it, the simulation
 * will stop, default 2,147,483,648.
 *
 */

template <typename DT = double, int UN = 1, DiscreType discretization = kDTQuadraticExponential>
void MCEuropeanHestonGreeksEngine(DT underlying,
                                  DT riskFreeRate,
                                  DT sigma,
                                  DT v0,
                                  DT theta,
                                  DT kappa,
                                  DT rho,
                                  DT dividendYield,
                                  bool optionType,
                                  DT strike,
                                  DT timeLength,
                                  ap_uint<32> seed[UN][2],
                                  DT* greeks,
                                  DT requiredTolerance = 0.02,
                                  unsigned int requiredSamples = 1024,
                                  unsigned int timeSteps = 100,
                                  unsigned int maxSamples = MAX_SAMPLE) {
    DT d_S = 0.3;  // 0.01 * underlying;
    DT d_r = 0.01; // 0.01 * riskFreeRate;
    DT d_T = 0.01 * timeLength;
    DT d_v0 = 0.3;  // 0.01 * v0;
    DT d_kap = 0.3; // 0.01 * kappa;
    DT d_the = 0.3; // 0.01 * theta;
    DT d2_S = 0.6;  // 0.02 * underlying;
    DT d_xi = 0.3;  // 0.01 * sigma;

    DT priceBuff[11];

    // calculate base
    MCEuropeanHestonEngine<DT, UN, discretization>(underlying, riskFreeRate, sigma, v0, theta, kappa, rho,
                                                   dividendYield, optionType, strike, timeLength, seed, priceBuff,
                                                   requiredTolerance, requiredSamples, timeSteps, maxSamples);
    // calculate theta
    DT T = timeLength - d_T;
    MCEuropeanHestonEngine<DT, UN, discretization>(underlying, riskFreeRate, sigma, v0, theta, kappa, rho,
                                                   dividendYield, optionType, strike, T, seed, priceBuff + 1,
                                                   requiredTolerance, requiredSamples, timeSteps, maxSamples);
    // calculate rho
    DT r = riskFreeRate + d_r;
    MCEuropeanHestonEngine<DT, UN, discretization>(underlying, r, sigma, v0, theta, kappa, rho, dividendYield,
                                                   optionType, strike, timeLength, seed, priceBuff + 2,
                                                   requiredTolerance, requiredSamples, timeSteps, maxSamples);
    // calculate delta
    DT S0_0 = underlying + d_S;
    DT S0_1 = underlying - d_S;
    MCEuropeanHestonEngine<DT, UN, discretization>(S0_0, riskFreeRate, sigma, v0, theta, kappa, rho, dividendYield,
                                                   optionType, strike, timeLength, seed, priceBuff + 3,
                                                   requiredTolerance, requiredSamples, timeSteps, maxSamples);
    MCEuropeanHestonEngine<DT, UN, discretization>(S0_1, riskFreeRate, sigma, v0, theta, kappa, rho, dividendYield,
                                                   optionType, strike, timeLength, seed, priceBuff + 4,
                                                   requiredTolerance, requiredSamples, timeSteps, maxSamples);
    // calcuate gamma
    DT S0_2 = underlying + d2_S;
    DT S0_3 = underlying - d2_S;
    MCEuropeanHestonEngine<DT, UN, discretization>(S0_2, riskFreeRate, sigma, v0, theta, kappa, rho, dividendYield,
                                                   optionType, strike, timeLength, seed, priceBuff + 5,
                                                   requiredTolerance, requiredSamples, timeSteps, maxSamples);
    MCEuropeanHestonEngine<DT, UN, discretization>(S0_3, riskFreeRate, sigma, v0, theta, kappa, rho, dividendYield,
                                                   optionType, strike, timeLength, seed, priceBuff + 6,
                                                   requiredTolerance, requiredSamples, timeSteps, maxSamples);
    // calcuate modelvega
    DT kap = kappa + d_kap;
    MCEuropeanHestonEngine<DT, UN, discretization>(underlying, riskFreeRate, sigma, v0, theta, kap, rho, dividendYield,
                                                   optionType, strike, timeLength, seed, priceBuff + 7,
                                                   requiredTolerance, requiredSamples, timeSteps, maxSamples);
    DT tha = theta + d_the;
    MCEuropeanHestonEngine<DT, UN, discretization>(underlying, riskFreeRate, sigma, v0, tha, kappa, rho, dividendYield,
                                                   optionType, strike, timeLength, seed, priceBuff + 8,
                                                   requiredTolerance, requiredSamples, timeSteps, maxSamples);

    DT xi = sigma + d_xi;
    MCEuropeanHestonEngine<DT, UN, discretization>(underlying, riskFreeRate, xi, v0, theta, kappa, rho, dividendYield,
                                                   optionType, strike, timeLength, seed, priceBuff + 9,
                                                   requiredTolerance, requiredSamples, timeSteps, maxSamples);
    DT v0_0 = v0 + d_v0;
    MCEuropeanHestonEngine<DT, UN, discretization>(underlying, riskFreeRate, sigma, v0_0, theta, kappa, rho,
                                                   dividendYield, optionType, strike, timeLength, seed, priceBuff + 10,
                                                   requiredTolerance, requiredSamples, timeSteps, maxSamples);

    // calculate thetha
    greeks[0] = (priceBuff[1] - priceBuff[0]) / d_T;
    greeks[1] = (priceBuff[2] - priceBuff[0]) / d_r;
    greeks[2] = (priceBuff[3] - priceBuff[4]) / d2_S;
    DT dP = d2_S * d2_S;
    greeks[3] = (priceBuff[5] + priceBuff[6] - 2 * priceBuff[0]) / dP;
    greeks[4] = (priceBuff[7] - priceBuff[0]) / d_kap;
    greeks[5] = (priceBuff[8] - priceBuff[0]) / d_the;
    greeks[6] = (priceBuff[9] - priceBuff[0]) / d_xi;
    greeks[7] = (priceBuff[10] - priceBuff[0]) / d_v0;
}

/**
 * @brief Cap/Floor Pricing Engine using Monte Carlo Simulation.
 * The Hull-White model is used to describe dynamics of short-term interest.
 * This engine assume a flat term structure.
 *
 * @tparam DT supported data type including double and float data type, which decides the precision of result, default
 * double-precision data type.
 * @tparam UN number of Monte Carlo Module in parallel, which affects the latency and resources utilization, default 10.
 *
 * @param nomial Nomial of capfloor contract.
 * @param initRate Current spot rate and forward rate (flat term structure).
 * @param strike Strike rate of capfloor contract
 * @param isCap True for cap, false for floor
 * @param singlePeriod period between each settlement date.
 * @param alpha Hull White model parameter
 * @param sigma Hull White model parameter
 * @param seed Array to store the inital seed for each RNG.
 * @param output Array to store result
 * @param requiredTolerance the tolerance required. If requiredSamples is not set, when reaching the required
 * tolerance, simulation will stop, default 0.02.
 * @param requiredSamples the samples number required. When reaching the required number, simulation will stop, default
 * 1024.
 * @param timeSteps the number of cap/floor settlement date.
 * @param maxSamples the maximum sample number. When reaching it, the simulation will stop, default 2,147,483,648.
 */

template <typename DT = double, int UN = 1>
void MCHullWhiteCapFloorEngine(DT nomial,
                               DT initRate,
                               DT strike,
                               bool isCap,
                               DT singlePeriod,
                               DT alpha,
                               DT sigma,
                               ap_uint<32>* seed,
                               DT* output,
                               DT requiredTolerance = 0.02,
                               unsigned int requiredSamples = 0,
                               unsigned int timeSteps = 2,
                               unsigned int maxSamples = MAX_SAMPLE) {
    // number of samples per simulation
    const static int SN = 1024;

    // number of variate
    const static int VN = 1;

    // Step first or sample first for each simulation
    const static bool SF = false;

    // RNG alias name
    typedef MT19937IcnRng<DT> RNG;

    // path generator instance
    HullWhitePathGen<DT, SN> pathGenInst[UN][1];
#pragma HLS array_partition variable = pathGenInst dim = 1

    // path pricer instance
    CapFloorPathPricer<DT, SN> pathPriInst[UN][1];
#pragma HLS array_partition variable = pathPriInst dim = 1

    // RNG sequence instance
    RNGSequence<DT, RNG> rngSeqInst[UN][1];
#pragma HLS array_partition variable = rngSeqInst dim = 1

    // initialization of pathgen and pathpricer
    for (int i = 0; i < UN; i++) {
        pathGenInst[i][0].init(alpha, sigma, initRate, singlePeriod);
        pathPriInst[i][0].init(isCap, strike, singlePeriod, initRate, alpha, sigma, nomial);
        rngSeqInst[i][0].seed[0] = seed[i];
    }

    // call monter carlo simulation
    DT price =
        mcSimulation<DT, RNG, HullWhitePathGen<DT, SN>, CapFloorPathPricer<DT, SN>, RNGSequence<DT, RNG>, UN, VN, SN>(
            timeSteps + 1, maxSamples, requiredSamples, requiredTolerance, pathGenInst, pathPriInst, rngSeqInst);

    // output the price of option
    output[0] = price;
}

#undef MAX_SAMPLE
} // namespace fintech
} // namespace xf
#endif // ifndef _XF_FINTECH_MCENGINE_H_
