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
 * @brief pathgenerator.h
 * @brief This file includes two mainstream mathematical model which desribes
 * the dynamics of financial derivation
 */

#ifndef _XF_FINTECH_PATH_GENERATOR_H_
#define _XF_FINTECH_PATH_GENERATOR_H_
#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "lmm.hpp"
#include "xf_fintech/bs_model.hpp"
#include "xf_fintech/enums.hpp"
#include "xf_fintech/heston_model.hpp"
#include "xf_fintech/hjm_model.hpp"
#include "xf_fintech/rng.hpp"
#include "xf_fintech/utils.hpp"
namespace xf {
namespace fintech {
namespace internal {

using namespace fintech::enums;

template <typename DT, int InN, int OutN, int SampNum>
void _antithetic(ap_uint<16> steps, hls::stream<DT> randNumberStrmIn[InN], hls::stream<DT> rand_strm_pair[OutN][InN]) {
    for (int t = 0; t < steps; ++t) {
        for (int p = 0; p < SampNum; ++p) {
#pragma HLS pipeline II = 1
            if (InN == 1) {
                DT in0 = randNumberStrmIn[0].read(); // z0,
                rand_strm_pair[0][0].write(in0);
                if (OutN == 2) {
                    rand_strm_pair[1][0].write(-in0);
                }
            }
            if (InN == 2) {
                DT in0 = randNumberStrmIn[0].read(); // z0,
                DT in1 = randNumberStrmIn[1].read(); // z1,
                rand_strm_pair[0][0].write(in0);
                rand_strm_pair[0][1].write(in1);
                if (OutN == 2) {
                    rand_strm_pair[1][0].write(-in0);
                    rand_strm_pair[1][1].write(-in1);
                }
            }
            if (InN == 3) {
                DT in0 = randNumberStrmIn[0].read(); // z0,
                DT in1 = randNumberStrmIn[1].read(); // z1,
                DT in2 = randNumberStrmIn[2].read(); // u
                rand_strm_pair[0][0].write(in0);
                rand_strm_pair[0][1].write(in1);
                rand_strm_pair[0][2].write(in2);
                if (OutN == 2) {
                    rand_strm_pair[1][0].write(-in0);
                    rand_strm_pair[1][1].write(-in1);
                    rand_strm_pair[1][2].write(1 - in2);
                }
            }
        }
    }
}
template <typename DT, bool StepFirst, int SampNum, bool WithAntithetic>
class BSPathGenerator {
   public:
    static const int OutN = WithAntithetic ? 2 : 1;

    xf::fintech::BSModel<DT> BSInst;
    // Constructor
    BSPathGenerator() {
#pragma HLS inline
    }

    void NextPath(ap_uint<16> steps,
                  ap_uint<16> paths,
                  hls::stream<DT> randNumberStrmIn[1],
                  hls::stream<DT> pathStrmOut[OutN]) {
#pragma HLS inline off
#ifndef __SYNTHESIS__
        static int cnt = 0;
#endif
        if (StepFirst) {
            for (int i = 0; i < paths; ++i) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
                for (int j = 0; j < steps; ++j) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 8 max = 8
                    DT dw = randNumberStrmIn[0].read();
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    if (cnt < 30) std::cout << "dw=" << dw << std::endl;
                    cnt++;
#endif
#endif
                    DT dLogS = BSInst.logEvolve(dw);
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    if (cnt < 30) std::cout << "s=" << s << std::endl;
#endif
#endif
                    pathStrmOut[0].write(dLogS);
                    if (WithAntithetic) {
                        DT dLogS_1 = BSInst.logEvolve(-dw);
                        pathStrmOut[1].write(dLogS_1);
                    }
                }
            }
        } else {
            for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 8 max = 8
                for (int j = 0; j < paths; ++j) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = SampNum max = SampNum
                    DT dw = randNumberStrmIn[0].read();
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    if (cnt < 30) std::cout << "dw=" << dw << std::endl;
                    cnt++;
#endif
#endif
                    DT dLogS = BSInst.logEvolve(dw);
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    if (cnt < 30) std::cout << "s=" << s << std::endl;
#endif
#endif
                    pathStrmOut[0].write(dLogS);
                    if (WithAntithetic) {
                        DT dLogS_1 = BSInst.logEvolve(-dw);
                        pathStrmOut[1].write(dLogS_1);
                    }
                }
            }
        }
    }
};
/*-----------------------------------------Multi Asset Heston
 * Model-----------------------------------------------*/

/**
 * @brief MultiAssetHestonPathGenerator logs of prices of multiple underlying assets, based on
 * kDTQuadraticExponential variation of Heston Model
 *
 * @tparam DT supported data type including double and float.
 * @tparam SampNum Number of path supported.
 * @tparam ASSETS Max assets supported.
 * @tparam discrT variation of Heston model
 */
template <typename DT, int SampNum, int ASSETS, DiscreType discrT>
class MultiAssetHestonPathGenerator {
   public:
    const static unsigned int OutN = 1;

    xf::fintech::HestonModel<ASSETS, DT, discrT> hestonInst;

    DT underlying[ASSETS];
    DT v0[ASSETS];

    DT s_buff[ASSETS][SampNum];
    DT v_buff[ASSETS][SampNum];

    MultiAssetHestonPathGenerator() {
#pragma HLS inline
#pragma HLS array_partition variable = s_buff dim = 1
#pragma HLS array_partition variable = v_buff dim = 1
    }

    /**
     * @param steps total timesteps of Heston Model
     * @param paths number of path of single call
     * @param randNumberStrmIn input random number stream
     * @param pathStrmOut stream of result generated.
     */
    void NextPath(ap_uint<16> steps,
                  ap_uint<16> paths,
                  hls::stream<DT> randNumberStrmIn[2],
                  hls::stream<DT> pathStrmOut[OutN]) {
#pragma HLS inline off

        for (ap_uint<16> t_itr = 0; t_itr < steps; t_itr++) {
#pragma HLS loop_tripcount min = 8 max = 8
            for (int a_itr = 0; a_itr < ASSETS; a_itr++) {
#pragma HLS loop_tripcount min = 9 max = 9
                for (int p_itr = 0; p_itr < paths; p_itr++) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = s_buff inter false
#pragma HLS dependence variable = v_buff inter false

                    DT s, v;
                    if (t_itr == 0) {
                        s = underlying[a_itr];
                        v = v0[a_itr];
                    } else {
                        s = s_buff[a_itr][p_itr];
                        v = v_buff[a_itr][p_itr];
                    }
                    DT z0 = randNumberStrmIn[0].read();
                    DT z1 = randNumberStrmIn[1].read();

                    DT s_next, v_next;
                    hestonInst.logEvolve(a_itr, s, v, s_next, v_next, z0, z1);

                    s_buff[a_itr][p_itr] = s_next;
                    v_buff[a_itr][p_itr] = v_next;

                    pathStrmOut[0].write(s_next);
                }
            }
        }
    }
};
/*-----------------------------------------Heston
 * Model-----------------------------------------------------------*/

/**
 * @brief Generator log of price of underlying asset, based on Heston Model
 *
 * @tparam discrT variation of Heston model
 * @tparam DT supported data type including double and float.
 * @tparam SampNum Number of path supported.
 */
template <DiscreType discrT, typename DT, int SampNum, bool WithAntithetic>
class HestonPathGenerator {
   public:
    const static unsigned int OutN = 1;

    HestonPathGenerator() {
#pragma HLS inline
    }

    /**
     * @param steps total timesteps of Heston Model
     * @param paths number of path of single call
     * @param randNumberStrmIn input random number stream
     * @param pathStrmOut stream of result generated.
     */
    void NextPath(ap_uint<16> steps,
                  ap_uint<16> paths,
                  hls::stream<DT> randNumberStrmIn[2],
                  hls::stream<DT> pathStrmOut[OutN]) {
#ifndef __SYNTHESIS__
        printf("Discrete Type is not supported!\n");
#endif
    }
};
/**
 * @brief HestonPathGenerator log of price of underlying asset, based on Heston Model
 *
 * @tparam discrT variation of Heston model
 * @tparam DT supported data type including double and float.
 * @tparam SampNum Number of path supported.
 */

template <typename DT, int SampNum, bool WithAntithetic>
class HestonPathGenerator<kDTPartialTruncation, DT, SampNum, WithAntithetic> {
   public:
    const static unsigned int OutN = WithAntithetic ? 2 : 1;

    /// initial value of log of price of underlying asset
    DT underlying;
    /// dividendYield of underlying asset
    DT dividendYield;
    /// riskFreeRate or expected anual growth rate of stock price
    DT riskFreeRate;
    /// initial value of volatility process
    DT v0;
    /// rate of volatility returns to theta
    DT kappa;
    /// long term variance
    DT theta;
    /// volatility of volatility
    DT sigma;
    /// correlation of two random walks
    DT rho;
    /// Time difference
    DT dt;
    /// square root of time difference
    DT sdt;
    /// kappa * dt
    DT kappa_dt;
    /// sigma * sdt
    DT sigma_sdt;
    /// sqrt(1 - rho * rho)
    DT hov;
    /// Other paratmers pre-calculated for Heston model. May not be used in
    /// certain variation.
    DT ex, k0, k1, k2, k3, k4, A;

    // drift of price.
    DT drift;

    // Constructor
    HestonPathGenerator() {
#pragma HLS inline
    }

    inline void updateDrift(DT dt) {
        DT u = FPTwoSub(riskFreeRate, dividendYield);
        drift = FPTwoMul(u, dt);
    }
    /**
     * @param steps total timesteps of Heston Model
     * @param paths number of path of single call
     * @param randNumberStrmIn input random number stream
     * @param pathStrmOut stream of result generated.
     */
    void NextPath(ap_uint<16> steps,
                  ap_uint<16> paths,
                  hls::stream<DT> randNumberStrmIn[2],
                  hls::stream<DT> pathStrmOut[OutN]) {
#pragma HLS dataflow
        hls::stream<DT> rand_strm_pair[OutN][2];
#pragma HLS stream variable = rand_strm_pair depth = 64
        _antithetic<DT, 2, OutN, SampNum>(steps, randNumberStrmIn, rand_strm_pair);
        for (int i = 0; i < OutN; ++i) {
#pragma HLS unroll
            _next_body(steps, rand_strm_pair[i], pathStrmOut[i]);
        }
    }
    void _next_body(ap_uint<16> steps, hls::stream<DT> randNumberStrmIn[2], hls::stream<DT>& pathStrmOut) {
#pragma HLS inline off

        DT s = underlying;
        DT v = v0;

        DT z0, z1, z0_rho, z1_hov, z_mix;
        DT v_act, v_abs, v_abs_half, v_bias, v_sqrt, mu, mu_dt, nu_dt;
        DT drift_G, drift_v, dG, ds, dv;

        DT array_s[SampNum];
        DT array_v[SampNum];

        for (ap_uint<16> j = 0; j < steps; j++) {
#pragma HLS loop_tripcount min = 8 max = 8
            for (ap_uint<27> i = 0; i < SampNum; i++) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = array_s inter false
#pragma HLS dependence variable = array_v inter false
                if (j == 0) {
                    s = underlying;
                    v = v0;
                } else {
                    s = array_s[i];
                    v = array_v[i];
                }

                z0 = randNumberStrmIn[0].read();
                z1 = randNumberStrmIn[1].read();

                z0_rho = FPTwoMul(rho, z0);
                z1_hov = FPTwoMul(hov, z1);
                z_mix = FPTwoAdd(z0_rho, z1_hov);

                if (v > 0.0) {
                    v_sqrt = hls::sqrt(v);
                    v_abs = v;
                } else {
                    v_sqrt = 0.0;
                    v_abs = -v;
                }

                v_act = v;
                v_bias = FPTwoSub(theta, v_act);
                //#pragma HLS resource variable=v_abs_half core=FMul_nodsp
                v_abs_half = divide_by_2(v_act);
                // v_abs_half = divide_by_2(v_abs);

                mu = -v_abs_half;
                mu_dt = FPTwoMul(mu, dt);
                nu_dt = kappa_dt * v_bias;

                drift_G = v_sqrt * sdt * z0;
                drift_v = v_sqrt * sigma_sdt * z_mix;

                dG = FPTwoAdd(mu_dt, drift_G);
                dv = FPTwoAdd(nu_dt, drift_v);

                s = FPTwoAdd(s, dG);
                v = FPTwoAdd(v, dv);
                DT logS = FPTwoAdd(drift, s);

                array_s[i] = logS;
                array_v[i] = v;
                if (j == steps - 1) pathStrmOut.write(logS);
            }
        }
    }
};

/**
 * @brief HestonPathGenerator log of price of underlying asset, based on Heston Model
 *
 * @tparam DT supported data type including double and float.
 * @tparam SampNum Number of path supported.
 */
template <typename DT, int SampNum, bool WithAntithetic>
class HestonPathGenerator<kDTFullTruncation, DT, SampNum, WithAntithetic> {
   public:
    const static unsigned int OutN = WithAntithetic ? 2 : 1;

    /// initial value of log of price of underlying asset
    DT underlying;
    /// dividendYield of underlying asset
    DT dividendYield;
    /// riskFreeRate or expected anual growth rate of stock price
    DT riskFreeRate;
    /// initial value of volatility process
    DT v0;
    /// rate of volatility returns to theta
    DT kappa;
    /// long term variance
    DT theta;
    /// volatility of volatility
    DT sigma;
    /// correlation of two random walks
    DT rho;
    /// Time difference
    DT dt;
    /// square root of time difference
    DT sdt;
    /// kappa * dt
    DT kappa_dt;
    /// sigma * sdt
    DT sigma_sdt;
    /// sqrt(1 - rho * rho)
    DT hov;
    /// Other paratmers pre-calculated for Heston model. May not be used in
    /// certain variation.
    DT ex, k0, k1, k2, k3, k4, A;

    // drift of price.
    DT drift;

    // Constructor
    HestonPathGenerator() {
#pragma HLS inline
    }
    inline void updateDrift(DT dt) {
        DT u = FPTwoSub(riskFreeRate, dividendYield);
        drift = FPTwoMul(u, dt);
    }
    /**
     * @param steps total timesteps of Heston Model
     * @param paths number of path of single call
     * @param randNumberStrmIn input random number stream
     * @param pathStrmOut stream of result generated.
     */
    void NextPath(ap_uint<16> steps,
                  ap_uint<16> paths,
                  hls::stream<DT> randNumberStrmIn[2],
                  hls::stream<DT> pathStrmOut[OutN]) {
#pragma HLS dataflow
        hls::stream<DT> rand_strm_pair[OutN][2];
#pragma HLS stream variable = rand_strm_pair depth = 64
        _antithetic<DT, 2, OutN, SampNum>(steps, randNumberStrmIn, rand_strm_pair);
        for (int i = 0; i < OutN; ++i) {
#pragma HLS unroll
            _next_body(steps, rand_strm_pair[i], pathStrmOut[i]);
        }
    }

    void _next_body(ap_uint<16> steps, hls::stream<DT> randNumberStrmIn[2], hls::stream<DT>& pathStrmOut) {
#pragma HLS inline off
        DT s = underlying;
        DT v = v0;

        DT z0, z1, z0_rho, z1_hov, z_mix;
        DT v_act, v_abs, v_abs_half, v_bias, v_sqrt, mu, mu_dt, nu_dt;
        DT drift_G, drift_v, dG, ds, dv;

        DT array_s[SampNum];
        DT array_v[SampNum];

        for (ap_uint<16> j = 0; j < steps; j++) {
#pragma HLS loop_tripcount min = 8 max = 8
            for (ap_uint<27> i = 0; i < SampNum; i++) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = array_s inter false
#pragma HLS dependence variable = array_v inter false
                if (j == 0) {
                    s = underlying;
                    v = v0;
                } else {
                    s = array_s[i];
                    v = array_v[i];
                }

                z0 = randNumberStrmIn[0].read();
                z1 = randNumberStrmIn[1].read();

                z0_rho = rho * z0;
                z1_hov = hov * z1;
                z_mix = FPTwoAdd(z0_rho, z1_hov);

                if (v > 0.0) {
                    v_sqrt = hls::sqrt(v);
                    v_abs = v;
                } else {
                    v_sqrt = 0.0;
                    v_abs = 0;
                }

                v_act = v_abs;
                v_bias = FPTwoSub(theta, v_act);
                v_abs_half = divide_by_2(v_abs);
                // v_abs_half = divide_by_2(v_abs);

                mu = -v_abs_half;
                mu_dt = mu * dt;
                nu_dt = kappa_dt * v_bias;

                drift_G = v_sqrt * sdt * z0;
                drift_v = v_sqrt * sigma_sdt * z_mix;

                dG = FPTwoAdd(mu_dt, drift_G);
                dv = FPTwoAdd(nu_dt, drift_v);

                s = FPTwoAdd(s, dG);
                v = FPTwoAdd(v, dv);

                DT logS = FPTwoAdd(s, drift);
                array_s[i] = logS;
                array_v[i] = v;
                if (j == steps - 1) pathStrmOut.write(logS);
            }
        }
    }
};

/**
 * @brief HestonPathGenerator log of price of underlying asset, based on Heston Model
 *
 * @tparam discrT variation of Heston model
 * @tparam DT supported data type including double and float.
 * @tparam SampNum Number of path supported.
 */
template <typename DT, int SampNum, bool WithAntithetic>
class HestonPathGenerator<kDTReflection, DT, SampNum, WithAntithetic> {
   public:
    const static unsigned int OutN = WithAntithetic ? 2 : 1;

    /// initial value of log of price of underlying asset
    DT underlying;
    /// dividendYield of underlying asset
    DT dividendYield;
    /// riskFreeRate or expected anual growth rate of stock price
    DT riskFreeRate;
    /// initial value of volatility process
    DT v0;
    /// rate of volatility returns to theta
    DT kappa;
    /// long term variance
    DT theta;
    /// volatility of volatility
    DT sigma;
    /// correlation of two random walks
    DT rho;
    /// Time difference
    DT dt;
    /// square root of time difference
    DT sdt;
    /// kappa * dt
    DT kappa_dt;
    /// sigma * sdt
    DT sigma_sdt;
    /// sqrt(1 - rho * rho)
    DT hov;
    /// Other paratmers pre-calculated for Heston model. May not be used in
    /// certain variation.
    DT ex, k0, k1, k2, k3, k4, A;
    // drift of price.
    DT drift;

    // Constructor
    HestonPathGenerator() {
#pragma HLS inline
    }

    inline void updateDrift(DT dt) {
        DT u = FPTwoSub(riskFreeRate, dividendYield);
        drift = FPTwoMul(u, dt);
    }

    /**
     * @param steps total timesteps of Heston Model
     * @param paths number of path of single call
     * @param randNumberStrmIn input random number stream
     * @param pathStrmOut stream of result generated.
     */
    void NextPath(ap_uint<16> steps,
                  ap_uint<16> paths,
                  hls::stream<DT> randNumberStrmIn[2],
                  hls::stream<DT> pathStrmOut[OutN]) {
#pragma HLS dataflow
        hls::stream<DT> rand_strm_pair[OutN][2];
#pragma HLS stream variable = rand_strm_pair depth = 64
        _antithetic<DT, 2, OutN, SampNum>(steps, randNumberStrmIn, rand_strm_pair);
        for (int i = 0; i < OutN; ++i) {
#pragma HLS unroll
            _next_body(steps, rand_strm_pair[i], pathStrmOut[i]);
        }
    }

    void _next_body(ap_uint<16> steps, hls::stream<DT> randNumberStrmIn[2], hls::stream<DT>& pathStrmOut) {
#pragma HLS inline off
        DT s = underlying;
        DT v = v0;

        DT z0, z1, z0_rho, z1_hov, z_mix;
        DT v_act, v_abs, v_abs_half, v_bias, v_sqrt, mu, mu_dt, nu_dt;
        DT drift_G, drift_v, dG, ds;

        DT array_s[SampNum];
        DT array_v[SampNum];

        for (ap_uint<16> j = 0; j < steps; j++) {
#pragma HLS loop_tripcount min = 8 max = 8
            for (ap_uint<27> i = 0; i < SampNum; i++) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = array_s inter false
#pragma HLS dependence variable = array_v inter false
                if (j == 0) {
                    s = underlying;
                    v = v0;
                } else {
                    s = array_s[i];
                    v = array_v[i];
                }

                z0 = randNumberStrmIn[0].read();
                z1 = randNumberStrmIn[1].read();

                z0_rho = rho * z0;
                z1_hov = hov * z1;
                z_mix = FPTwoAdd(z0_rho, z1_hov);

                if (v > 0.0) {
                    v_abs = v;
                } else {
                    v_abs = -v;
                }

                v_act = v_abs;
                v_sqrt = hls::sqrt(v_abs);
                v_bias = FPTwoSub(theta, v_act);
                v_abs_half = divide_by_2(v_abs);
                // v_abs_half = divide_by_2(v_abs);

                // mu  = riskFreeRate - dividendYield - v_abs_half;
                mu = -v_abs_half;
                mu_dt = mu * dt;
                nu_dt = kappa_dt * v_bias;

                drift_G = v_sqrt * sdt * z0;
                drift_v = v_sqrt * sigma_sdt * z_mix;

                dG = FPTwoAdd(mu_dt, drift_G);

                s = FPTwoAdd(s, dG);
                v = FPTwoAdd(v_abs, nu_dt);
                v = FPTwoAdd(v, drift_v);
                DT logS = FPTwoAdd(s, drift);
                array_s[i] = logS;
                array_v[i] = v;
                if (j == steps - 1) pathStrmOut.write(logS);
            }
        }
    }
};

/**
 * @brief HestonPathGenerator log of price of underlying asset, based on Heston Model
 *
 * @tparam discrT variation of Heston model
 * @tparam DT supported data type including double and float.
 * @tparam SampNum Number of path supported.
 */
template <typename DT, int SampNum, bool WithAntithetic>
class HestonPathGenerator<kDTQuadraticExponential, DT, SampNum, WithAntithetic> {
   public:
    const static unsigned int OutN = WithAntithetic ? 2 : 1;

    /// initial value of log of price of underlying asset
    DT underlying;
    /// dividendYield of underlying asset
    DT dividendYield;
    /// riskFreeRate or expected anual growth rate of stock price
    DT riskFreeRate;
    /// initial value of volatility process
    DT v0;
    /// rate of volatility returns to theta
    DT kappa;
    /// long term variance
    DT theta;
    /// volatility of volatility
    DT sigma;
    /// correlation of two random walks
    DT rho;
    /// Time difference
    DT dt;
    /// square root of time difference
    DT sdt;
    /// kappa * dt
    DT kappa_dt;
    /// sigma * sdt
    DT sigma_sdt;
    /// sqrt(1 - rho * rho)
    DT hov;
    /// Other paratmers pre-calculated for Heston model. May not be used in
    /// certain variation.
    DT ex, k0, k1, k2, k3, k4, A;

    // drift of price.
    DT drift;

    // Constructor
    HestonPathGenerator() {
#pragma HLS inline
    }

    inline void updateDrift(DT dt) {
        DT u = FPTwoSub(riskFreeRate, dividendYield);
        drift = FPTwoMul(u, dt);
    }
    /**
     * @param steps total timesteps of Heston Model
     * @param paths number of path of single call
     * @param randNumberStrmIn input random number stream
     * @param pathStrmOut stream of result generated.
     */
    void NextPath(ap_uint<16> steps,
                  ap_uint<16> paths,
                  hls::stream<DT> randNumberStrmIn[3],
                  hls::stream<DT> pathStrmOut[OutN]) {
        hls::stream<DT> rand_strm_pair[2][3];
#pragma HLS stream variable = rand_strm_pair depth = 32
#pragma HLS dataflow
        _antithetic<DT, 3, OutN, SampNum>(steps, randNumberStrmIn, rand_strm_pair);
        for (int i = 0; i < OutN; ++i) {
#pragma HLS unroll
            _next_body(steps, rand_strm_pair[i], pathStrmOut[i]);
        }
    }
    void _next_body(ap_uint<16> steps, hls::stream<DT> randNumberStrmIn[3], hls::stream<DT>& pathStrmOut) {
#pragma HLS inline off

        DT s, v, v_old, z0, z1, u1;
        DT m, m2, s2, psi;
        DT bias_v, ex_bias_v, s2_pre_1, s2_pre_2, s2_pre, one_ex;
        DT td1, td2, td3, td4, td5, td6, tm1, tm2, tm3, tm4, tm5, tm6, tm7, tm8, tm9;
        DT psi_2inv, psi_2inv_1, sqrt_psi_tail, b2, b2_1, b_z1, b;
        DT p, one_u, u, u_p;
        DT kpart1, kpart2, kpart3, kpart4, kpart3_4, kpart3_4_sqrt, k_total;
        bool lt_15;

        // XXXx
        DT array_s[SampNum];
        DT array_v[SampNum];

        for (ap_uint<16> j = 0; j < steps; j++) {
#pragma HLS loop_tripcount min = 8 max = 8
            for (ap_uint<27> i = 0; i < SampNum; i++) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = array_s inter false
#pragma HLS dependence variable = array_v inter false
                if (j == 0) {
                    s = underlying;
                    v = v0;
                } else {
                    s = array_s[i];
                    v = array_v[i];
                }
                v_old = v;

                z0 = randNumberStrmIn[0].read();
                z1 = randNumberStrmIn[1].read();
                u1 = randNumberStrmIn[2].read();
                u = u1;
                one_u = FPTwoSub((DT)1.0, u);

                bias_v = FPTwoSub(v, theta);
                ex_bias_v = ex * bias_v;
                m = FPTwoAdd(theta, ex_bias_v);

                one_ex = FPTwoSub((DT)1.0, ex);
                s2_pre_1 = v * ex;
                s2_pre_2 = divide_by_2(theta * one_ex);
                s2_pre = FPTwoAdd(s2_pre_1, s2_pre_2);
                s2 = sigma * sigma * one_ex * s2_pre / kappa;

                psi = s2 / m / m;

                if (psi < 1.5) {
                    lt_15 = true;
                } else {
                    lt_15 = false;
                }

                td2 = 2.0;
                if (lt_15) {
                    td1 = psi;
                } else {
                    td1 = FPTwoAdd(psi, (DT)1.0);
                }
                td3 = td2 / td1;

                psi_2inv = td3;
                psi_2inv_1 = FPTwoAdd(td3, (DT)1.0);

                if (lt_15) {
                    tm2 = psi_2inv;
                    tm1 = psi_2inv_1;
                } else {
                    tm2 = td1;
                    tm1 = 0.5;
                }
                tm3 = tm2 * tm1;

                sqrt_psi_tail = hls::sqrt(tm3);
                b2_1 = FPTwoAdd(psi_2inv, sqrt_psi_tail);
                b2 = FPTwoSub(b2_1, (DT)1.0);
                b = hls::sqrt(b2);

                if (lt_15) {
                    td5 = m;
                    td4 = b2_1;
                } else {
                    p = FPTwoSub((DT)1.0, td3);
                    td5 = td3;
                    td4 = one_u;
                }
                td6 = td5 / td4;

                b_z1 = FPTwoAdd(b, z1);

                if (lt_15) {
                    tm5 = td6;
                    tm4 = b_z1;
                } else {
                    tm5 = tm3;
                    tm4 = m;
                }
                tm6 = tm5 * tm4;

                if (lt_15) {
                    tm8 = tm6;
                    tm7 = b_z1;
                } else {
                    tm8 = tm6;
                    tm7 = hls::log(td6);
                }
                tm9 = tm8 * tm7;

                u_p = FPTwoSub(u, p); // u-p
                if (!lt_15 && u_p < 0.0) {
                    v = 0.0;
                } else {
                    v = tm9;
                }

                kpart1 = k1 * v_old;
                kpart2 = k2 * v;
                kpart3 = k3 * v_old;
                kpart4 = k4 * v;
                kpart3_4 = FPTwoAdd(kpart3, kpart4);
                kpart3_4_sqrt = hls::sqrt(kpart3_4) * z0;
                k_total = FPTwoAdd(k0, kpart1);
                k_total = FPTwoAdd(k_total, kpart2);
                k_total = FPTwoAdd(k_total, kpart3_4_sqrt);
                s = FPTwoAdd(s, k_total);
                DT logS = FPTwoAdd(drift, s);

                array_s[i] = logS;
                array_v[i] = v;
                if (j == steps - 1) pathStrmOut.write(logS);
            }
        }
    }
};

/**
 * @brief HestonPathGenerator log of price of underlying asset, based on Heston Model
 *
 * @tparam discrT variation of Heston model
 * @tparam DT supported data type including double and float.
 * @tparam SampNum Number of path supported.
 */
template <typename DT, int SampNum, bool WithAntithetic>
class HestonPathGenerator<kDTQuadraticExponentialMartingale, DT, SampNum, WithAntithetic> {
   public:
    const static unsigned int OutN = WithAntithetic ? 2 : 1;

    /// initial value of log of price of underlying asset
    DT underlying;
    /// dividendYield of underlying asset
    DT dividendYield;
    /// riskFreeRate or expected anual growth rate of stock price
    DT riskFreeRate;
    /// initial value of volatility process
    DT v0;
    /// rate of volatility returns to theta
    DT kappa;
    /// long term variance
    DT theta;
    /// volatility of volatility
    DT sigma;
    /// correlation of two random walks
    DT rho;
    /// Time difference
    DT dt;
    /// square root of time difference
    DT sdt;
    /// kappa * dt
    DT kappa_dt;
    /// sigma * sdt
    DT sigma_sdt;
    /// sqrt(1 - rho * rho)
    DT hov;
    /// Other paratmers pre-calculated for Heston model. May not be used in
    /// certain variation.
    DT ex, k0, k1, k2, k3, k4, A;

    // drift of price.
    DT drift;

    // Constructor
    HestonPathGenerator() {
#pragma HLS inline
    }

    inline void updateDrift(DT dt) {
        DT u = FPTwoSub(riskFreeRate, dividendYield);
        drift = FPTwoMul(u, dt);
    }

    /**
     * @param steps total timesteps of Heston Model
     * @param paths number of path of single call
     * @param randNumberStrmIn input random number stream
     * @param pathStrmOut stream of result generated.
     */
    void NextPath(ap_uint<16> steps,
                  ap_uint<16> paths,
                  hls::stream<DT> randNumberStrmIn[3],
                  hls::stream<DT> pathStrmOut[OutN]) {
#pragma HLS dataflow
        hls::stream<DT> rand_strm_pair[OutN][3];
        _antithetic<DT, 3, OutN, SampNum>(steps, randNumberStrmIn, rand_strm_pair);
        for (int i = 0; i < OutN; ++i) {
#pragma HLS unroll
            _next_body(steps, rand_strm_pair[i], pathStrmOut[i]);
        }
    }
    void _next_body(ap_uint<16> steps, hls::stream<DT> randNumberStrmIn[3], hls::stream<DT>& pathStrmOut) {
#pragma HLS inline off

        DT s, v, v_old, z0, z1, u1;
        DT m, m2, s2, psi;
        DT bias_v, ex_bias_v, s2_pre_1, s2_pre_2, s2_pre, one_ex;
        DT td1, td2, td3, td4, td5, td6, tm1, tm2, tm3, tm4, tm5, tm6, tm7, tm8, tm9;
        DT kd1, kd2, kd3, kd4, kd5, kd6, km1, km2, km3, km4, km5, km6;
        DT Bv, one_2Aa, pre_log, log_tmp;
        DT psi_2inv, psi_2inv_1, sqrt_psi_tail, b2, b2_1, b_z1, b;
        DT p, one_u, u, u_p;
        DT kpart1, kpart2, kpart3, kpart4, kpart3_4, kpart3_4_sqrt, k_total;
        bool lt_15;

        // XXXx

        DT array_s[SampNum];
        DT array_v[SampNum];

        for (ap_uint<16> j = 0; j < steps; j++) {
#pragma HLS loop_tripcount min = 8 max = 8
            for (ap_uint<27> i = 0; i < SampNum; i++) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = array_s inter false
#pragma HLS dependence variable = array_v inter false
                if (j == 0) {
                    s = underlying;
                    v = v0;
                } else {
                    s = array_s[i];
                    v = array_v[i];
                }
                v_old = v;

                z0 = randNumberStrmIn[0].read();
                z1 = randNumberStrmIn[1].read();
                u1 = randNumberStrmIn[2].read();
                u = u1;
                one_u = FPTwoSub((DT)1.0, u);

                bias_v = FPTwoSub(v, theta);
                ex_bias_v = ex * bias_v;
                m = FPTwoAdd(theta, ex_bias_v);

                one_ex = FPTwoSub((DT)1.0, ex);
                s2_pre_1 = v * ex;
                s2_pre_2 = divide_by_2(theta * one_ex);
                s2_pre = FPTwoAdd(s2_pre_1, s2_pre_2);
                s2 = sigma * sigma * one_ex * s2_pre / kappa;

                psi = s2 / m / m;

                if (psi < 1.5) {
                    lt_15 = true;
                } else {
                    lt_15 = false;
                }

                td2 = 2.0;
                if (lt_15) {
                    td1 = psi;
                } else {
                    td1 = FPTwoAdd(psi, (DT)1.0);
                }
                td3 = td2 / td1;

                psi_2inv = td3;
                psi_2inv_1 = FPTwoSub(td3, (DT)1.0);

                if (lt_15) {
                    tm2 = psi_2inv;
                    tm1 = psi_2inv_1;
                } else {
                    tm2 = td1;
                    tm1 = 0.5;
                }
                tm3 = tm2 * tm1;

                sqrt_psi_tail = hls::sqrt(tm3);
                b2_1 = FPTwoAdd(psi_2inv, sqrt_psi_tail);
                b2 = FPTwoSub(b2_1, (DT)1.0);
                b = hls::sqrt(b2);

                if (lt_15) {
                    td5 = m;
                    td4 = b2_1;
                } else {
                    p = FPTwoSub((DT)1.0, td3);
                    td5 = td3;
                    td4 = one_u;
                }
                td6 = td5 / td4;

                b_z1 = FPTwoAdd(b, z1);

                if (lt_15) {
                    tm5 = td6;
                    tm4 = b_z1;
                } else {
                    tm5 = tm3;
                    tm4 = m;
                }
                tm6 = tm5 * tm4;

                if (lt_15) {
                    tm8 = tm6;
                    tm7 = b_z1;
                } else {
                    tm8 = tm6;
                    tm7 = hls::log(td6);
                }
                tm9 = tm8 * tm7;

                u_p = u - p;
                if (!lt_15 && u_p < 0.0) {
                    v = 0.0;
                } else {
                    v = tm9;
                }

                kpart1 = k1 * v_old;
                kpart2 = k2 * v;
                kpart3 = k3 * v_old;
                kpart4 = k4 * v;

                Bv = FPTwoAdd(kpart1, divide_by_2(kpart3));
                kd1 = m;
                kd2 = td3;
                kd3 = kd2 / kd1;
                if (lt_15) {
                    km1 = A;
                    km2 = td6;
                } else {
                    km1 = kd3;
                    km2 = td3;
                }
                km3 = km2 * km1;

                km4 = km3;
                km5 = b2;
                km6 = km5 * km4;

                if (lt_15) {
                    kd4 = FPTwoSub((DT)1, mul_by_2(km3));
                    kd5 = km6;
                } else {
                    kd4 = FPTwoSub(kd3, A);
                    kd5 = kd3 * td3;
                }
                kd6 = kd5 / kd4;

                if (lt_15) {
                    pre_log = kd4;
                } else {
                    pre_log = FPTwoSub((DT)1, td3);
                    pre_log = FPTwoAdd(pre_log, kd6);
                }
                log_tmp = hls::log(pre_log);

                DT op1, op2;
                if (lt_15) {
                    op1 = divide_by_2(log_tmp);
                    op2 = -Bv;
                } else {
                    op1 = -log_tmp;
                    op2 = Bv;
                }
                DT k0_1;
                if (lt_15) {
                    k0_1 = FPTwoSub(divide_by_2(log_tmp), Bv);
                    k0_1 = FPTwoSub(k0_1, kd6);
                } else {
                    k0_1 = FPTwoSub(-log_tmp, Bv);
                }

                kpart3_4 = FPTwoAdd(kpart3, kpart4);
                kpart3_4_sqrt = hls::sqrt(kpart3_4) * z0;
                k_total = FPTwoAdd(k0_1, kpart1);
                k_total = FPTwoAdd(k_total, kpart2);
                k_total = FPTwoAdd(k_total, kpart3_4_sqrt);
                s = FPTwoAdd(s, k_total);
                DT logS = FPTwoAdd(s, drift);
                array_s[i] = logS;
                array_v[i] = v;
                if (j == steps - 1) pathStrmOut.write(logS);
            }
        }
    }
};

template <typename DT, int SampNum>
class HullWhitePathGen {
   public:
    static const int OutN = 1;

    DT alpha;
    DT theta;
    DT sigma;
    DT r0;
    DT singlePeriod;
    DT rateBuffer[SampNum];

    DT drift_factor1;
    DT drift_factor2;
    DT volatility;

    HullWhitePathGen() {
#pragma HLS inline
    }

    void init(DT input_a, DT input_sigma, DT input_r, DT input_period) {
        DT alpha = input_a;
        DT sigma = input_sigma;
        r0 = input_r;
        DT singlePeriod = input_period;
        DT theta = r0 * alpha;

        DT alpha_dt = FPTwoMul(alpha, singlePeriod);
        DT exp1t = FPExp(-alpha_dt);
        DT exp2t = exp1t * exp1t;
        DT one_exp1t = FPTwoSub(1.0, exp1t);
        DT one_exp2t = FPTwoSub(1.0, exp2t);
        DT inv_alpha = 1.0 / alpha;
        DT theta_div_alpha = inv_alpha * theta;
        DT sigmasqr = FPTwoMul(sigma, sigma);
        DT sigmasqr_div_2alpha = FPTwoMul(FPTwoMul(sigmasqr, inv_alpha), 0.5);

        drift_factor1 = exp1t;
        drift_factor2 = FPTwoMul(theta_div_alpha, one_exp1t);
        DT variance = FPTwoMul(sigmasqr_div_2alpha, one_exp2t);
        volatility = hls::sqrt(variance);
    }

    void NextPath(ap_uint<16> steps,
                  ap_uint<16> paths,
                  hls::stream<DT> randNumberStrmIn[1],
                  hls::stream<DT> pathStrmOut[OutN]) {
        for (int i = 0; i < steps; i++) {
            for (int j = 0; j < paths; j++) {
#pragma HLS Pipeline II = 1
#pragma HLS dependence variable = rateBuffer inter false
                DT x0;
                if (i == 0) {
                    x0 = r0;
                } else {
                    x0 = rateBuffer[j];
                }

                DT dw = randNumberStrmIn[0].read();

                DT r_part1 = FPTwoMul(drift_factor1, x0);
                DT r_part2 = drift_factor2;
                DT r_part3 = volatility * dw;

                DT r_next = FPTwoAdd(FPTwoAdd(r_part1, r_part2), r_part3);

                rateBuffer[j] = r_next;
                pathStrmOut[0].write(r_next);
            }
        }
    }
};

/**
 * @brief Path generation for Heath-Jarrow-Morton framework. Each path corresponds to a matrix of Instantaneous
 * Forward Rates 'tenors' wide and 'simYears/dt' deep.
 *
 * @tparam DT - Internal DataType of the path generator
 * @tparam MAX_TENORS - Maximum number of tenors supported
 */
template <typename DT, unsigned int MAX_TENORS>
class hjmPathGenerator {
   public:
    xf::fintech::hjmModelData<DT, MAX_TENORS> m_hjmImpl;
    DT prevPath[MAX_TENORS];

    hjmPathGenerator() {
#pragma HLS inline
    }

    void init(xf::fintech::hjmModelData<DT, MAX_TENORS>& hjmImpl) {
        m_hjmImpl = hjmImpl;
#ifndef __SYNTHESIS__
        assert(m_hjmImpl.tenors <= MAX_TENORS &&
               "Provided tenors are larger than the synthetisable MAX_TENORS parameter.");
#endif
    }

    void NextPath(ap_uint<16> steps,
                  ap_uint<16> paths,
                  hls::stream<DT> rngStream[hjmModelParams::N],
                  hls::stream<DT> pathOut[1]) {
#pragma HLS INLINE off
        const ap_uint<16> tenors = m_hjmImpl.tenors;

    HJM_Path_It_Loop:
        for (ap_uint<16> p = 0; p < paths; p++) {
        HJM_Steps_Loop:
            for (ap_uint<16> s = 0; s < steps; s++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = prevPath inter false

                const DT z0 = rngStream[0].read();
                const DT z1 = rngStream[1].read();
                const DT z2 = rngStream[2].read();

            HJM_Path_Tenors_Loop:
                for (ap_uint<16> t = 0; t < tenors; t++) {
                    if (s == 0) {
                        pathOut[0] << m_hjmImpl.m_initialFc[t];
                        prevPath[t] = m_hjmImpl.m_initialFc[t];
                    } else {
                        const DT prevVal = prevPath[t];
                        const DT drift = m_hjmImpl.RnD[t] * hjmModelParams::dt;
                        const DT vola = (m_hjmImpl.vol1[t] * z0 + m_hjmImpl.vol2[t] * z1 + m_hjmImpl.vol3[t] * z2) *
                                        hjmModelParams::sqrtDt;
                        const DT prevValN = (t == tenors - 1) ? prevPath[t - 1] : prevPath[t + 1];
                        const DT deltaIr = (prevValN - prevVal) * (hjmModelParams::dt / hjmModelParams::tau);
                        const DT nextT = prevVal + drift + vola + deltaIr;

                        prevPath[t] = nextT;
                        pathOut[0] << nextT;
                    }
                }
            }
        }
    }
};

template <typename DT, unsigned int NF, unsigned int MAX_TENORS>
class lmmPathGenerator {
    using lmm_data_t = xf::fintech::lmmModelData<DT, NF, MAX_TENORS>;
    using idx_t = ap_uint<8>;

    static constexpr std::size_t triMatSize(unsigned T) { return (T * (T + 1)) / 2; }

    constexpr static unsigned N_MATURITIES = MAX_TENORS - 1;

    lmm_data_t m_lmmData;
    idx_t driftIdxs[triMatSize(N_MATURITIES)];
    idx_t volaIdxs[triMatSize(MAX_TENORS)];

    void initVolaIdx(ap_uint<16> noTenors) {
#pragma HLS INLINE
        idx_t offset = 0, idx = 0;
    LMM_triIdx_vola_outer_loop:
        for (idx_t i = 0; i < noTenors; i++) {
        LMM_triIdx_vola_inner_loop:
            for (idx_t j = i; j < noTenors; j++) {
#pragma HLS PIPELINE
                volaIdxs[idx++] = offset + j;
            }
            offset += ((N_MATURITIES - 1) - i);
        }
    }

    void initDriftIdx(ap_uint<16> noTenors) {
#pragma HLS INLINE
        idx_t offset = 0, idx = 0;
    LMM_triIdx_drift_outer_loop:
        for (idx_t i = 0; i < noTenors - 1; i++) {
        LMM_triIdx_drift_inner_loop:
            for (idx_t j = noTenors - 1; j > i; j--) {
#pragma HLS PIPELINE
                driftIdxs[idx++] = offset + j - 1;
            }
            offset += ((N_MATURITIES - 1) - i);
        }
    }

   public:
    lmmPathGenerator() {
#pragma HLS inline
    }

    void init(lmm_data_t& lmmData, ap_uint<16> noTenors) {
#pragma HLS DATAFLOW
        m_lmmData = lmmData;
        initVolaIdx(noTenors);
        initDriftIdx(noTenors);
    }

    void NextPath(ap_uint<16> steps, ap_uint<16> paths, hls::stream<DT> rngStream[NF], hls::stream<DT> pathOut[1]) {
        const unsigned noTenors = steps + 1;
        DT prevLiborPath[MAX_TENORS] = {0.0f};
        DT cumDrift[N_MATURITIES] = {0.0f};
        idx_t dIdx = 0, vIdx = 0;
        constexpr DT tau = static_cast<DT>(lmmModelParams::tau);

    LMM_Path_Init_Loop:
        for (idx_t i = 0; i < noTenors; i++) {
#pragma HLS LOOP_TRIPCOUNT min = MAX_TENORS max = MAX_TENORS
#pragma HLS PIPELINE
            const DT initRate = m_lmmData.m_presentRate[i];
            prevLiborPath[i] = initRate;
            pathOut[0] << initRate;
        }

    LMM_Path_Sim_Loop:
        for (idx_t s = 1; s < noTenors; s++) {
#pragma HLS LOOP_TRIPCOUNT min = MAX_TENORS max = MAX_TENORS

            DT wRng[NF];
            for (unsigned i = 0; i < NF; i++) {
#pragma HLS UNROLL
                wRng[i] = rngStream[i].read();
            }

            // Compute cumsum of drift
            DT prevDrift = 0.0f;
        LMM_Path_CumDrift_Loop:
            for (idx_t m = steps; m > s; m--) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = MAX_TENORS
#pragma HLS PIPELINE
                const idx_t idx = driftIdxs[dIdx++];
                DT drift = (m_lmmData.m_rho[idx] * tau * prevLiborPath[m] * m_lmmData.m_sigma[idx + 1]) /
                           (1 + tau * prevLiborPath[m]);

                const DT nextDrift = drift + prevDrift;
                cumDrift[m - s - 1] = nextDrift;
                prevDrift = nextDrift;
            }

        // Compute forward rates for every time T for every path
        LMM_Path_Tenor_Loop:
            for (idx_t i = s; i < noTenors; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = MAX_TENORS
#pragma HLS PIPELINE
                DT dW = 0.0f;
                for (unsigned w = 0; w < NF; w++) {
#pragma HLS UNROLL
                    dW += m_lmmData.m_eta[i][w] * wRng[w];
                }
                const DT vol = m_lmmData.m_sigma[volaIdxs[vIdx++]];
                const DT drift = ((-vol * cumDrift[i - 1]) - 0.5 * vol * vol) * tau;
                const DT rng = vol * static_cast<DT>(lmmModelParams::sqrtTau) * dW;

                DT nextRate = prevLiborPath[i] * hls::exp(drift + rng);

                prevLiborPath[i] = nextRate;
                pathOut[0] << nextRate;
            }
        }
    }
};
} // namespace internal
} // namespace fintech
} // namespace xf

#endif //#ifndef XF_FINTECH_PATH_GENERATOR_H
