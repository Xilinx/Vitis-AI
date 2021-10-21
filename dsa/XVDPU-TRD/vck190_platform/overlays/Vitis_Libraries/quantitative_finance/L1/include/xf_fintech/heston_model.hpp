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
 * @brief bs_model.h
 * @brief This file include mainstream Heston stochastic process.
 */

#ifndef _XF_FINTECH_HESTONMODEL_H_
#define _XF_FINTECH_HESTONMODEL_H_
#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "xf_fintech/enums.hpp"
#include "xf_fintech/rng.hpp"
#include "xf_fintech/utils.hpp"
namespace xf {
namespace fintech {

/**
 * @brief Heston process
 *
 * @tparam ASSETS max asset number supported.
 * @tparam DT data type supported include float and double.
 * @tparam discrT variation of Heston model.
 */
template <int ASSETS, typename DT, enums::DiscreType discrT>
class HestonModel {
   public:
    DT riskFreeRate;
    DT dt;
    DT sdt;

    DT kappa_vec[ASSETS];
    DT theta_vec[ASSETS];
    DT sigma_vec[ASSETS];
    DT kappa_dt_vec[ASSETS];
    DT sigma_sdt_vec[ASSETS];
    DT drift_vec[ASSETS];
    DT ex_vec[ASSETS];
    DT k0_vec[ASSETS];
    DT k1_vec[ASSETS];
    DT k2_vec[ASSETS];
    DT k3_vec[ASSETS];
    DT k4_vec[ASSETS];
    DT A_vec[ASSETS];

    /**
     * @brief constructor
     */
    HestonModel() {
#pragma HLS inline
    }

    /**
     * @brief initParam Init model parameters.
     *
     * @param asset_nm Number of asset, should be no  more than ASSETS
     * @param timeLength Length of time of contract
     * @param timeSteps Number of timesteps in timeLength
     * @param riskFreeRate_input  Risk Free rate
     * @param dividendYield Dividend Yield
     * @param kappa Kappa of Heston Model
     * @param theta Theta of Heston Model
     * @param sigma Sigma of Heston Model
     * @param rho Rho of Heston Model
     */
    void initParam(int asset_nm,
                   DT timeLength,
                   DT timeSteps,
                   DT riskFreeRate_input,
                   DT* dividendYield,
                   DT* kappa,
                   DT* theta,
                   DT* sigma,
                   DT* rho);
    /**
     * @brief logEvolve Log Evolve of Heston Model
     *
     * @param asset_itr Which asset to evolve.
     * @param s Current s(log price).
     * @param v Current v(volatility).
     * @param s_next Next s(log price).
     * @param v_next Next v(volatilty).
     * @param z0 Random number input for s.
     * @param z1 Random number input for v.
     */
    void logEvolve(int asset_itr, DT s, DT v, DT& s_next, DT& v_next, DT z0, DT z1);
};

/**
 * @brief Heston process
 *
 * @tparam ASSETS max asset number supported.
 * @tparam DT data type supported include float and double.
 * @tparam discrT variation of Heston model.
 */
template <int ASSETS, typename DT>
class HestonModel<ASSETS, DT, enums::kDTFullTruncation> {
   public:
    DT riskFreeRate;
    DT dt;
    DT sdt;

    DT kappa_vec[ASSETS];
    DT theta_vec[ASSETS];
    DT sigma_vec[ASSETS];
    DT kappa_dt_vec[ASSETS];
    DT sigma_sdt_vec[ASSETS];
    DT drift_vec[ASSETS];
    DT ex_vec[ASSETS];
    DT k0_vec[ASSETS];
    DT k1_vec[ASSETS];
    DT k2_vec[ASSETS];
    DT k3_vec[ASSETS];
    DT k4_vec[ASSETS];
    DT A_vec[ASSETS];

    /**
     * @brief constructor
     */
    HestonModel() {
#pragma HLS inline
    }

    /**
     * @brief initParam Init Model Parameters.
     *
     * @param asset_nm Number of asset, should be no  more than ASSETS
     * @param timeLength Length of time of contract
     * @param timeSteps Number of timesteps in timeLength
     * @param riskFreeRate_input  Risk Free rate
     * @param dividendYield Dividend Yield
     * @param kappa Kappa of Heston Model
     * @param theta Theta of Heston Model
     * @param sigma Sigma of Heston Model
     * @param rho Rho of Heston Model
     */
    void initParam(int asset_nm,
                   DT timeLength,
                   DT timeSteps,
                   DT riskFreeRate_input,
                   DT* dividendYield,
                   DT* kappa,
                   DT* theta,
                   DT* sigma,
                   DT* rho) {
        dt = timeLength / timeSteps;
        sdt = hls::sqrt(dt);
        riskFreeRate = riskFreeRate_input;

        for (int i = 0; i < asset_nm; i++) {
            kappa_vec[i] = kappa[i];
            theta_vec[i] = theta[i];
            sigma_vec[i] = sigma[i];
            kappa_dt_vec[i] = internal::FPTwoMul(kappa[i], dt);
            sigma_sdt_vec[i] = internal::FPTwoMul(sigma[i], sdt);
            drift_vec[i] = internal::FPTwoMul(dt, internal::FPTwoSub(riskFreeRate, dividendYield[i]));
        }
    }

    /**
     * @brief logEvolve Log Evolve of Heston Model
     *
     * @param asset_itr Which asset to evolve.
     * @param s Current s(log price).
     * @param v Current v(volatility).
     * @param s_next Next s(log price).
     * @param v_next Next v(volatilty).
     * @param z0 Random number input for s.
     * @param z1 Random number input for v.
     */
    void logEvolve(int asset_itr, DT s, DT v, DT& s_next, DT& v_next, DT z0, DT z1) {
        DT z_mix;
        DT v_act, v_abs, v_abs_half, v_bias, v_sqrt, mu, mu_dt, nu_dt;
        DT drift_G, drift_v, dG, ds, dv;

        z_mix = z1;

        if (v > 0.0) {
            v_sqrt = hls::sqrt(v);
            v_abs = v;
        } else {
            v_sqrt = 0.0;
            v_abs = 0;
        }

        v_act = v_abs;
        v_bias = internal::FPTwoSub(theta_vec[asset_itr], v_act);
        v_abs_half = divide_by_2(v_abs);

        mu = -v_abs_half;
        mu_dt = mu * dt;
        nu_dt = kappa_dt_vec[asset_itr] * v_bias;

        drift_G = v_sqrt * sdt * z0;
        drift_v = v_sqrt * sigma_sdt_vec[asset_itr] * z_mix;

        dG = internal::FPTwoAdd(mu_dt, drift_G);
        dv = internal::FPTwoAdd(nu_dt, drift_v);

        s_next = internal::FPTwoAdd(internal::FPTwoAdd(s, dG), drift_vec[asset_itr]);
        v_next = internal::FPTwoAdd(v, dv);
    }
};

/**
 * @brief Heston process
 *
 * @tparam ASSETS max asset number supported.
 * @tparam DT data type supported include float and double.
 * @tparam discrT variation of Heston model.
 */
template <int ASSETS, typename DT>
class HestonModel<ASSETS, DT, enums::kDTPartialTruncation> {
   public:
    DT riskFreeRate;
    DT dt;
    DT sdt;

    DT kappa_vec[ASSETS];
    DT theta_vec[ASSETS];
    DT sigma_vec[ASSETS];
    DT kappa_dt_vec[ASSETS];
    DT sigma_sdt_vec[ASSETS];
    DT drift_vec[ASSETS];
    DT ex_vec[ASSETS];
    DT k0_vec[ASSETS];
    DT k1_vec[ASSETS];
    DT k2_vec[ASSETS];
    DT k3_vec[ASSETS];
    DT k4_vec[ASSETS];
    DT A_vec[ASSETS];

    /**
     * @brief constructor
     */
    HestonModel() {
#pragma HLS inline
    }

    /**
     * @brief initParam Init Model Parameters.
     *
     * @param asset_nm Number of asset, should be no  more than ASSETS
     * @param timeLength Length of time of contract
     * @param timeSteps Number of timesteps in timeLength
     * @param riskFreeRate_input  Risk Free rate
     * @param dividendYield Dividend Yield
     * @param kappa Kappa of Heston Model
     * @param theta Theta of Heston Model
     * @param sigma Sigma of Heston Model
     * @param rho Rho of Heston Model
     */
    void initParam(int asset_nm,
                   DT timeLength,
                   DT timeSteps,
                   DT riskFreeRate_input,
                   DT* dividendYield,
                   DT* kappa,
                   DT* theta,
                   DT* sigma,
                   DT* rho) {
        dt = timeLength / timeSteps;
        sdt = hls::sqrt(dt);
        riskFreeRate = riskFreeRate_input;

        for (int i = 0; i < asset_nm; i++) {
            kappa_vec[i] = kappa[i];
            theta_vec[i] = theta[i];
            sigma_vec[i] = sigma[i];
            kappa_dt_vec[i] = internal::FPTwoMul(kappa[i], dt);
            sigma_sdt_vec[i] = internal::FPTwoMul(sigma[i], sdt);
            drift_vec[i] = internal::FPTwoMul(dt, internal::FPTwoSub(riskFreeRate, dividendYield[i]));
        }
    }

    /**
     * @brief logEvolve Log Evolve of Heston Model
     *
     * @param asset_itr Which asset to evolve.
     * @param s Current s(log price).
     * @param v Current v(volatility).
     * @param s_next Next s(log price).
     * @param v_next Next v(volatilty).
     * @param z0 Random number input for s.
     * @param z1 Random number input for v.
     */
    void logEvolve(int asset_itr, DT s, DT v, DT& s_next, DT& v_next, DT z0, DT z1) {
        DT z_mix;
        DT v_act, v_abs, v_abs_half, v_bias, v_sqrt, mu, mu_dt, nu_dt;
        DT drift_G, drift_v, dG, ds, dv;

        z_mix = z1;

        if (v > 0.0) {
            v_sqrt = hls::sqrt(v);
            v_abs = v;
        } else {
            v_sqrt = 0.0;
            v_abs = -v;
        }

        v_act = v;
        v_bias = internal::FPTwoSub(theta_vec[asset_itr], v_act);
        v_abs_half = divide_by_2(v_act);

#pragma HLS resource variable = mu core = FAddSub_nodsp
        mu = -v_abs_half;
        mu_dt = mu * dt;
        nu_dt = kappa_dt_vec[asset_itr] * v_bias;

        drift_G = v_sqrt * sdt * z0;
        drift_v = v_sqrt * sigma_sdt_vec[asset_itr] * z_mix;

        dG = internal::FPTwoAdd(mu_dt, drift_G);
        dv = internal::FPTwoAdd(nu_dt, drift_v);

        s_next = internal::FPTwoAdd(internal::FPTwoAdd(s, dG), drift_vec[asset_itr]);
        v_next = internal::FPTwoAdd(v, dv);
    }
};

/**
 * @brief Heston process
 *
 * @tparam ASSETS max asset number supported.
 * @tparam DT data type supported include float and double.
 * @tparam discrT variation of Heston model.
 */
template <int ASSETS, typename DT>
class HestonModel<ASSETS, DT, enums::kDTReflection> {
   public:
    DT riskFreeRate;
    DT dt;
    DT sdt;

    DT kappa_vec[ASSETS];
    DT theta_vec[ASSETS];
    DT sigma_vec[ASSETS];
    DT kappa_dt_vec[ASSETS];
    DT sigma_sdt_vec[ASSETS];
    DT drift_vec[ASSETS];
    DT ex_vec[ASSETS];
    DT k0_vec[ASSETS];
    DT k1_vec[ASSETS];
    DT k2_vec[ASSETS];
    DT k3_vec[ASSETS];
    DT k4_vec[ASSETS];
    DT A_vec[ASSETS];

    /**
     * @brief constructor
     */
    HestonModel() {
#pragma HLS inline
    }

    /**
     * @brief initParam Init Model Parameters.
     *
     * @param asset_nm Number of asset, should be no  more than ASSETS
     * @param timeLength Length of time of contract
     * @param timeSteps Number of timesteps in timeLength
     * @param riskFreeRate_input  Risk Free rate
     * @param dividendYield Dividend Yield
     * @param kappa Kappa of Heston Model
     * @param theta Theta of Heston Model
     * @param sigma Sigma of Heston Model
     * @param rho Rho of Heston Model
     */
    void initParam(int asset_nm,
                   DT timeLength,
                   DT timeSteps,
                   DT riskFreeRate_input,
                   DT* dividendYield,
                   DT* kappa,
                   DT* theta,
                   DT* sigma,
                   DT* rho) {
        dt = timeLength / timeSteps;
        sdt = hls::sqrt(dt);
        riskFreeRate = riskFreeRate_input;

        for (int i = 0; i < asset_nm; i++) {
            kappa_vec[i] = kappa[i];
            theta_vec[i] = theta[i];
            sigma_vec[i] = sigma[i];
            kappa_dt_vec[i] = internal::FPTwoMul(kappa[i], dt);
            sigma_sdt_vec[i] = internal::FPTwoMul(sigma[i], sdt);
            drift_vec[i] = internal::FPTwoMul(dt, internal::FPTwoSub(riskFreeRate, dividendYield[i]));
        }
    }

    /**
     * @brief logEvolve Log Evolve of Heston Model
     *
     * @param asset_itr Which asset to evolve.
     * @param s Current s(log price).
     * @param v Current v(volatility).
     * @param s_next Next s(log price).
     * @param v_next Next v(volatilty).
     * @param z0 Random number input for s.
     * @param z1 Random number input for v.
     */
    void logEvolve(int asset_itr, DT s, DT v, DT& s_next, DT& v_next, DT z0, DT z1) {
        DT z_mix;
        DT v_act, v_abs, v_abs_half, v_bias, v_sqrt, mu, mu_dt, nu_dt;
        DT drift_G, drift_v, dG, ds;

        z_mix = z1;

        if (v > 0.0) {
            v_abs = v;
        } else {
            v_abs = -v;
        }

        v_act = v_abs;
        v_sqrt = hls::sqrt(v_abs);
        v_bias = internal::FPTwoSub(theta_vec[asset_itr], v_act);
        v_abs_half = divide_by_2(v_abs);

        mu = -v_abs_half;
        mu_dt = mu * dt;
        nu_dt = kappa_dt_vec[asset_itr] * v_bias;

        drift_G = v_sqrt * sdt * z0;
        drift_v = v_sqrt * sigma_sdt_vec[asset_itr] * z_mix;

        dG = internal::FPTwoAdd(mu_dt, drift_G);

        s_next = internal::FPTwoAdd(internal::FPTwoAdd(s, dG), drift_vec[asset_itr]);
        v_next = internal::FPTwoAdd(internal::FPTwoAdd(v_abs, nu_dt), drift_v);
    }
};

/**
 * @brief Heston process
 *
 * @tparam ASSETS max asset number supported.
 * @tparam DT data type supported include float and double.
 * @tparam discrT variation of Heston model.
 */
template <int ASSETS, typename DT>
class HestonModel<ASSETS, DT, enums::kDTQuadraticExponential> {
   public:
    DT riskFreeRate;
    DT dt;
    DT sdt;

    DT kappa_vec[ASSETS];
    DT theta_vec[ASSETS];
    DT sigma_vec[ASSETS];
    DT kappa_dt_vec[ASSETS];
    DT sigma_sdt_vec[ASSETS];
    DT drift_vec[ASSETS];
    DT ex_vec[ASSETS];
    DT k0_vec[ASSETS];
    DT k1_vec[ASSETS];
    DT k2_vec[ASSETS];
    DT k3_vec[ASSETS];
    DT k4_vec[ASSETS];
    DT A_vec[ASSETS];

    /**
     * @brief constructor
     */
    HestonModel() {
#pragma HLS inline
    }

    /**
     * @brief initParam Init Model Parameters.
     *
     * @param asset_nm Number of asset, should be no  more than ASSETS
     * @param timeLength Length of time of contract
     * @param timeSteps Number of timesteps in timeLength
     * @param riskFreeRate_input  Risk Free rate
     * @param dividendYield Dividend Yield
     * @param kappa Kappa of Heston Model
     * @param theta Theta of Heston Model
     * @param sigma Sigma of Heston Model
     * @param rho Rho of Heston Model
     */
    void initParam(int asset_nm,
                   DT timeLength,
                   DT timeSteps,
                   DT riskFreeRate_input,
                   DT* dividendYield,
                   DT* kappa,
                   DT* theta,
                   DT* sigma,
                   DT* rho) {
        dt = timeLength / timeSteps;
        sdt = hls::sqrt(dt);
        riskFreeRate = riskFreeRate_input;

        for (int i = 0; i < asset_nm; i++) {
            kappa_vec[i] = kappa[i];
            theta_vec[i] = theta[i];
            sigma_vec[i] = sigma[i];
            kappa_dt_vec[i] = internal::FPTwoMul(kappa[i], dt);
            sigma_sdt_vec[i] = internal::FPTwoMul(sigma[i], sdt);
            drift_vec[i] = internal::FPTwoMul(dt, internal::FPTwoSub(riskFreeRate, dividendYield[i]));

            DT one_rho_sqr = internal::FPTwoSub((DT)1.0, internal::FPTwoMul(rho[i], rho[i]));
            DT r_d_s = rho[i] / sigma[i];

            ex_vec[i] = hls::exp(-kappa_dt_vec[i]);
            k0_vec[i] = -internal::FPTwoMul(internal::FPTwoMul(r_d_s, kappa_dt_vec[i]), theta_vec[i]);

            DT k1_1 = internal::FPTwoMul(kappa_dt_vec[i] * r_d_s, (DT)0.5);
            DT k1_2 = internal::FPTwoMul(dt, (DT)0.25);
            DT k1_3 = internal::FPTwoSub(k1_1, k1_2);
            k1_vec[i] = internal::FPTwoSub(k1_3, r_d_s);
            k2_vec[i] = internal::FPTwoAdd(k1_3, r_d_s);
            k3_vec[i] = internal::FPTwoMul(dt * one_rho_sqr, (DT)0.5);
            k4_vec[i] = k3_vec[i];

            DT A_1 = internal::FPTwoMul(k4_vec[i], (DT)0.5);
            A_vec[i] = internal::FPTwoAdd(k2_vec[i], A_1);
        }
    }

    /**
     * @brief logEvolve Log Evolve of Heston Model
     *
     * @param asset_itr Which asset to evolve.
     * @param s Current s(log price).
     * @param v Current v(volatility).
     * @param s_next Next s(log price).
     * @param v_next Next v(volatilty).
     * @param z0 Random number input for s.
     * @param z1 Random number input for v.
     */
    void logEvolve(int asset_itr, DT s, DT v, DT& s_next, DT& v_next, DT z0, DT z1) {
        DT v_old, u1;
        DT m, m2, s2, psi;
        DT bias_v, ex_bias_v, s2_pre_1, s2_pre_2, s2_pre, one_ex;
        DT td1, td2, td3, td4, td5, td6, tm1, tm2, tm3, tm4, tm5, tm6, tm7, tm8, tm9;
        DT psi_2inv, psi_2inv_1, sqrt_psi_tail, b2, b2_1, b_z1, b;
        DT p, one_u, u, u_p;
        DT kpart1, kpart2, kpart3, kpart4, kpart3_4, kpart3_4_sqrt, k_total;
        bool lt_15;

        v_old = v;

        u1 = xf::fintech::internal::CumulativeNormal(z1);
        u = u1;
        one_u = internal::FPTwoSub((DT)1.0, u);
        bias_v = internal::FPTwoSub(v, theta_vec[asset_itr]);
        ex_bias_v = internal::FPTwoMul(ex_vec[asset_itr], bias_v);
        m = internal::FPTwoAdd(theta_vec[asset_itr], ex_bias_v);
        one_ex = internal::FPTwoSub((DT)1.0, ex_vec[asset_itr]);
        s2_pre_1 = internal::FPTwoMul(v, ex_vec[asset_itr]);
        DT s2_pre_2_tmp;
        s2_pre_2_tmp = internal::FPTwoMul(theta_vec[asset_itr], one_ex);
        s2_pre_2 = internal::divide_by_2(s2_pre_2_tmp);
        s2_pre = internal::FPTwoAdd(s2_pre_1, s2_pre_2);
        DT s2_tmp_mul_1, s2_tmp_mul_2, s2_tmp_mul_3;
        s2_tmp_mul_1 = internal::FPTwoMul(sigma_vec[asset_itr], sigma_vec[asset_itr]);
        s2_tmp_mul_2 = internal::FPTwoMul(s2_tmp_mul_1, one_ex);
        s2_tmp_mul_3 = internal::FPTwoMul(s2_tmp_mul_2, s2_pre);
        s2 = s2_tmp_mul_3 / kappa_vec[asset_itr];

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
            td1 = internal::FPTwoAdd(psi, (DT)1.0);
        }
        td3 = td2 / td1;

        psi_2inv = td3;
        psi_2inv_1 = internal::FPTwoSub(td3, (DT)1.0);

        if (lt_15) {
            tm2 = psi_2inv;
            tm1 = psi_2inv_1;
        } else {
            tm2 = td1;
            tm1 = 0.5;
        }
        tm3 = internal::FPTwoMul(tm2, tm1);

        sqrt_psi_tail = hls::sqrt(tm3);
        b2_1 = internal::FPTwoAdd(psi_2inv, sqrt_psi_tail);
        b2 = internal::FPTwoSub(b2_1, (DT)1.0);
        b = hls::sqrt(b2);

        if (lt_15) {
            td5 = m;
            td4 = b2_1;
        } else {
            p = internal::FPTwoSub((DT)1.0, td3);
            td5 = td3;
            td4 = one_u;
        }
        td6 = td5 / td4;
        b_z1 = internal::FPTwoAdd(b, z1);

        if (lt_15) {
            tm5 = td6;
            tm4 = b_z1;
        } else {
            tm5 = tm3;
            tm4 = m;
        }
        tm6 = internal::FPTwoMul(tm5, tm4);

        if (lt_15) {
            tm8 = tm6;
            tm7 = b_z1;
        } else {
            tm8 = tm6;
            tm7 = hls::log(td6);
        }
        tm9 = internal::FPTwoMul(tm8, tm7);
        u_p = internal::FPTwoSub(u, p);
        if (!lt_15 && u_p < 0.0) {
            v = 0.0;
        } else {
            v = tm9;
        }
        kpart1 = internal::FPTwoMul(k1_vec[asset_itr], v_old);
        kpart2 = internal::FPTwoMul(k2_vec[asset_itr], v);
        kpart3 = internal::FPTwoMul(k3_vec[asset_itr], v_old);
        kpart4 = internal::FPTwoMul(k4_vec[asset_itr], v);
        kpart3_4 = internal::FPTwoAdd(kpart3, kpart4);
        DT sqrt_kpart3_4 = hls::sqrt(kpart3_4);
        kpart3_4_sqrt = internal::FPTwoMul(sqrt_kpart3_4, z0);
        k_total = internal::FPTwoAdd(k0_vec[asset_itr], kpart1);
        k_total = internal::FPTwoAdd(k_total, kpart2);
        k_total = internal::FPTwoAdd(k_total, kpart3_4_sqrt);
        s = internal::FPTwoAdd(s, k_total);

        s_next = s + drift_vec[asset_itr];
        v_next = v;
    }
};

/**
 * @brief Heston process
 *
 * @tparam ASSETS max asset number supported.
 * @tparam DT data type supported include float and double.
 * @tparam discrT variation of Heston model.
 */
template <int ASSETS, typename DT>
class HestonModel<ASSETS, DT, enums::kDTQuadraticExponentialMartingale> {
   public:
    DT riskFreeRate;
    DT dt;
    DT sdt;

    DT kappa_vec[ASSETS];
    DT theta_vec[ASSETS];
    DT sigma_vec[ASSETS];
    DT kappa_dt_vec[ASSETS];
    DT sigma_sdt_vec[ASSETS];
    DT drift_vec[ASSETS];
    DT ex_vec[ASSETS];
    DT k0_vec[ASSETS];
    DT k1_vec[ASSETS];
    DT k2_vec[ASSETS];
    DT k3_vec[ASSETS];
    DT k4_vec[ASSETS];
    DT A_vec[ASSETS];

    /**
     * @brief constructor
     */
    HestonModel() {
#pragma HLS inline
    }

    /**
     * @brief initParam Init Model Parameters.
     *
     * @param asset_nm Number of asset, should be no  more than ASSETS
     * @param timeLength Length of time of contract
     * @param timeSteps Number of timesteps in timeLength
     * @param riskFreeRate_input  Risk Free rate
     * @param dividendYield Dividend Yield
     * @param kappa Kappa of Heston Model
     * @param theta Theta of Heston Model
     * @param sigma Sigma of Heston Model
     * @param rho Rho of Heston Model
     */
    void initParam(int asset_nm,
                   DT timeLength,
                   DT timeSteps,
                   DT riskFreeRate_input,
                   DT* dividendYield,
                   DT* kappa,
                   DT* theta,
                   DT* sigma,
                   DT* rho) {
        dt = timeLength / timeSteps;
        sdt = hls::sqrt(dt);
        riskFreeRate = riskFreeRate_input;

        for (int i = 0; i < asset_nm; i++) {
            kappa_vec[i] = kappa[i];
            theta_vec[i] = theta[i];
            sigma_vec[i] = sigma[i];
            kappa_dt_vec[i] = internal::FPTwoMul(kappa[i], dt);
            sigma_sdt_vec[i] = internal::FPTwoMul(sigma[i], sdt);
            drift_vec[i] = internal::FPTwoMul(dt, internal::FPTwoSub(riskFreeRate, dividendYield[i]));

            DT one_rho_sqr = internal::FPTwoSub((DT)1.0, internal::FPTwoMul(rho[i], rho[i]));
            DT r_d_s = rho[i] / sigma[i];

            ex_vec[i] = hls::exp(-kappa_dt_vec[i]);
            k0_vec[i] = -internal::FPTwoMul(internal::FPTwoMul(r_d_s, kappa_dt_vec[i]), theta_vec[i]);

            DT k1_1 = internal::FPTwoMul(kappa_dt_vec[i] * r_d_s, (DT)0.5);
            DT k1_2 = internal::FPTwoMul(dt, (DT)0.25);
            DT k1_3 = internal::FPTwoSub(k1_1, k1_2);
            k1_vec[i] = internal::FPTwoSub(k1_3, r_d_s);
            k2_vec[i] = internal::FPTwoAdd(k1_3, r_d_s);
            k3_vec[i] = internal::FPTwoMul(dt * one_rho_sqr, (DT)0.5);
            k4_vec[i] = k3_vec[i];

            DT A_1 = internal::FPTwoMul(k4_vec[i], (DT)0.5);
            A_vec[i] = internal::FPTwoAdd(k2_vec[i], A_1);
        }
    }

    /**
     * @brief logEvolve Log Evolve of Heston Model
     *
     * @param asset_itr Which asset to evolve.
     * @param s Current s(log price).
     * @param v Current v(volatility).
     * @param s_next Next s(log price).
     * @param v_next Next v(volatilty).
     * @param z0 Random number input for s.
     * @param z1 Random number input for v.
     */
    void logEvolve(int asset_itr, DT s, DT v, DT& s_next, DT& v_next, DT z0, DT z1) {
        DT v_old, u1;
        DT m, m2, s2, psi;
        DT bias_v, ex_bias_v, s2_pre_1, s2_pre_2, s2_pre, one_ex;
        DT td1, td2, td3, td4, td5, td6, tm1, tm2, tm3, tm4, tm5, tm6, tm7, tm8, tm9;
        DT kd1, kd2, kd3, kd4, kd5, kd6, km1, km2, km3, km4, km5, km6;
        DT Bv, one_2Aa, pre_log, log_tmp;
        DT psi_2inv, psi_2inv_1, sqrt_psi_tail, b2, b2_1, b_z1, b;
        DT p, one_u, u, u_p;
        DT kpart1, kpart2, kpart3, kpart4, kpart3_4, kpart3_4_sqrt, k_total;
        bool lt_15;

        v_old = v;

        u1 = xf::fintech::internal::CumulativeNormal(z1);
        u = u1;
        one_u = internal::FPTwoSub((DT)1.0, u);

        bias_v = internal::FPTwoSub(v, theta_vec[asset_itr]);
        ex_bias_v = ex_vec[asset_itr] * bias_v;
        m = internal::FPTwoAdd(theta_vec[asset_itr], ex_bias_v);

        one_ex = internal::FPTwoSub((DT)1.0, ex_vec[asset_itr]);
        s2_pre_1 = v * ex_vec[asset_itr];
        s2_pre_2 = divide_by_2(theta_vec[asset_itr] * one_ex);
        s2_pre = internal::FPTwoAdd(s2_pre_1, s2_pre_2);
        s2 = sigma_vec[asset_itr] * sigma_vec[asset_itr] * one_ex * s2_pre / kappa_vec[asset_itr];

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
            td1 = internal::FPTwoAdd(psi, (DT)1.0);
        }
        td3 = td2 / td1;

        psi_2inv = td3;
        psi_2inv_1 = internal::FPTwoSub(td3, (DT)1.0);

        if (lt_15) {
            tm2 = psi_2inv;
            tm1 = psi_2inv_1;
        } else {
            tm2 = td1;
            tm1 = 0.5;
        }
        tm3 = tm2 * tm1;

        sqrt_psi_tail = hls::sqrt(tm3);
        b2_1 = internal::FPTwoAdd(psi_2inv, sqrt_psi_tail);
        b2 = internal::FPTwoSub(b2_1, (DT)1.0);
        b = hls::sqrt(b2);

        if (lt_15) {
            td5 = m;
            td4 = b2_1;
        } else {
            p = internal::FPTwoSub((DT)1.0, td3);
            td5 = td3;
            td4 = one_u;
        }
        td6 = td5 / td4;

        b_z1 = internal::FPTwoAdd(b, z1);

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

        kpart1 = k1_vec[asset_itr] * v_old;
        kpart2 = k2_vec[asset_itr] * v;
        kpart3 = k3_vec[asset_itr] * v_old;
        kpart4 = k4_vec[asset_itr] * v;

        Bv = internal::FPTwoAdd(kpart1, divide_by_2(kpart3));
        kd1 = m;
        kd2 = td3;
        kd3 = kd2 / kd1;
        if (lt_15) {
            km1 = A_vec[asset_itr];
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
            kd4 = internal::FPTwoSub((DT)1, mul_by_2(km3));
            kd5 = km6;
        } else {
            kd4 = internal::FPTwoSub(kd3, A_vec[asset_itr]);
            kd5 = kd3 * td3;
        }
        kd6 = kd5 / kd4;

        if (lt_15) {
            pre_log = kd4;
        } else {
            pre_log = internal::FPTwoSub((DT)1, td3);
            pre_log = internal::FPTwoAdd(pre_log, kd6);
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
            k0_1 = internal::FPTwoSub(divide_by_2(log_tmp), Bv);
            k0_1 = internal::FPTwoSub(k0_1, kd6);
        } else {
            k0_1 = internal::FPTwoSub(-log_tmp, Bv);
        }

        kpart3_4 = internal::FPTwoAdd(kpart3, kpart4);
        kpart3_4_sqrt = hls::sqrt(kpart3_4) * z0;
        k_total = internal::FPTwoAdd(k0_1, kpart1);
        k_total = internal::FPTwoAdd(k_total, kpart2);
        k_total = internal::FPTwoAdd(k_total, kpart3_4_sqrt);
        s = internal::FPTwoAdd(s, k_total);
        s_next = s + drift_vec[asset_itr];
        v_next = v;
    }
};

} // namespace fintech
} // namespace xf
#endif //_XF_FINTECH_BSMODEL_H_
