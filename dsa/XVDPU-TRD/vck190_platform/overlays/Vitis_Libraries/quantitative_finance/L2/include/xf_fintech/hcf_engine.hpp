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

#ifndef _XF_FINTECH_HCF_ENGINE_H_
#define _XF_FINTECH_HCF_ENGINE_H_

#include "L2_utils.hpp"

namespace xf {
namespace fintech {

/// @tparam DT supported data type including double and float data type, which
/// decides the precision of result, default is float.
/// @param s0    the stock price at t=0
/// @param v0    the stock price variance at t=0
/// @param K     the strike price
/// @param rho   the correlation of the 2 Weiner processes
/// @param T     the expiration time
/// @param r     the risk free interest rate
/// @param kappa the rate of reversion
/// @param vvol  the volatility of volatility (sigma)
/// @param vbar  the long term average variance (theta)
/// @param w_max the w value to integrate up to
/// @param dw    the delta w for the integration
/// @param padding the pragma data pack requires structure size to be a power
/// of 2

template <typename DT>
struct hcfEngineInputDataType {
    DT s0;
    DT v0;
    DT K;
    DT rho;
    DT T;
    DT r;
    DT kappa;
    DT dw;
    DT vvol;
    DT vbar;
    int w_max;
    int padding[5];
};

// template specialisation in order to deal with the different padding
template <>
struct hcfEngineInputDataType<double> {
    double s0;       // stock price at t=0
    double v0;       // stock price variance at t=0
    double K;        // strike price
    double rho;      // correlation of the 2 Weiner processes
    double T;        // expiration time
    double r;        // risk free interest rate
    double kappa;    // rate of reversion
    double dw;       // delta w for the integration
    double vvol;     // volatility of volatility (sigma)
    double vbar;     // long term average variance (theta)
    int w_max;       // w value to integrate up to
    int padding[11]; // #pragma data pack requires structure size to be a power of
                     // 2
};

namespace internal {

/// @brief function to calculate the characteristic function
/// @param[in] in A structure containing the kerenl model parameters
/// @param[in] w complex representation of w
/// @return the calculated characterisic function value
template <typename DT>
struct complex_num<DT> charFunc(struct hcfEngineInputDataType<DT>* in, struct complex_num<DT> w) {
#pragma HLS PIPELINE
    DT vv = in->vvol * in->vvol;
    DT gamma = vv / 2;
    struct complex_num<DT> i = cn_init((DT)0, (DT)1);

    struct complex_num<DT> alpha = cn_scalar_mul(cn_add(cn_mul(w, w), cn_mul(w, i)), (DT)-0.5);
    struct complex_num<DT> beta = cn_sub(cn_init(in->kappa, (DT)0), cn_mul(cn_scalar_mul(w, in->rho * in->vvol), i));
    struct complex_num<DT> h = cn_sqrt(cn_sub(cn_mul(beta, beta), (cn_scalar_mul(alpha, gamma*(DT)4))));
    struct complex_num<DT> r_plus = cn_div(cn_add(beta, h), cn_init(vv, (DT)0));
    struct complex_num<DT> r_minus = cn_div(cn_sub(beta, h), cn_init(vv, (DT)0));
    struct complex_num<DT> g = cn_div(r_minus, r_plus);

    struct complex_num<DT> exp_hT = cn_exp(cn_scalar_mul(h, -in->T));
    struct complex_num<DT> tmp = cn_sub(cn_init((DT)1, (DT)0), cn_mul(g, exp_hT));
    struct complex_num<DT> D = cn_mul(r_minus, cn_div(cn_sub(cn_init((DT)1, (DT)0), exp_hT), tmp));

    struct complex_num<DT> C = cn_div(tmp, cn_sub(cn_init((DT)1, (DT)0), g));
    C = cn_scalar_mul(cn_ln(C), (DT)2);
    C = cn_div(C, cn_init(vv, (DT)0));
    C = cn_mul(cn_init(in->kappa, (DT)0), cn_sub(cn_scalar_mul(r_minus, in->T), C));

    struct complex_num<DT> cf = cn_add(cn_scalar_mul(C, in->vbar), cn_scalar_mul(D, in->v0));
    cf = cn_add(cf, cn_scalar_mul(cn_mul(i, w), LOG(in->s0* EXP(in->r * in->T))));
    cf = cn_exp(cf);

    return cf;
}

/// @brief function to calculate the integrand for pi 1
/// @param[in] in A structure containing the kerenl model parameters
/// @param[in] w the limit
/// @return the calculated integrand value
template <typename DT>
DT pi1Integrand(struct hcfEngineInputDataType<DT>* in, DT w) {
    struct complex_num<DT> ww = cn_init(w, (DT)-1);
    struct complex_num<DT> cf1 = charFunc(in, ww);

    ww = cn_init((DT)0, (DT)-1);
    struct complex_num<DT> cf2 = charFunc(in, ww);

    struct complex_num<DT> tmp = cn_scalar_mul(cn_init((DT)0, (DT)1), w);
    return cn_real(cn_mul(cn_exp(cn_scalar_mul(tmp, -LOG(in->K))), cn_div(cf1, cn_mul(tmp, cf2))));
}

/// @brief function to calculate the integrand for pi 2
/// @param[in] in A structure containing the kerenl model parameters
/// @param[in] w the limit
/// @return the calculated integrand value
template <typename DT>
DT pi2Integrand(struct hcfEngineInputDataType<DT>* in, DT w) {
    struct complex_num<DT> cf1 = charFunc(in, cn_init(w, (DT)0));

    struct complex_num<DT> tmp = cn_div(cf1, cn_scalar_mul(cn_init((DT)0, (DT)1), w));
    return cn_real(cn_mul(cn_exp(cn_scalar_mul(cn_init((DT)0, (DT)-1), w * LOG(in->K))), tmp));
}

/// @brief integration function pi 1
/// @param[in] in A structure containing the kerenl model parameters
/// @return the calculated value
template <typename DT>
DT integrateForPi1(struct hcfEngineInputDataType<DT>* in) {
#pragma HLS INLINE OFF
    DT elem;
    DT sum = 0;
    DT f_n = 0;
    DT f_n_plus_1 = 0;
    DT w = 0;
    DT max = in->w_max / in->dw;
    int n;

    f_n = pi1Integrand(in, (DT)1e-10);

pi1_loop:
    for (n = 1; n <= (int)max; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 400 max = 400 avg = 400
#pragma HLS PIPELINE
        w = n * in->dw;
        f_n_plus_1 = pi1Integrand(in, w);
        elem = in->dw * (f_n_plus_1 + f_n) / 2;
        sum += elem;
        f_n = f_n_plus_1;
    }
    return sum;
}

/// @brief integration function pi 1
/// @param[in] in A structure containing the kerenl model parameters
/// @return the calculated value
template <typename DT>
DT integrateForPi2(struct hcfEngineInputDataType<DT>* in) {
#pragma HLS INLINE OFF
    DT elem;
    DT sum = 0;
    DT f_n = 0;
    DT f_n_plus_1 = 0;
    DT w = 0;
    DT max = in->w_max / in->dw;
    int n;

    f_n = pi2Integrand(in, (DT)1e-10);

pi2_loop:
    for (n = 1; n <= (int)max; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 400 max = 400 avg = 400
#pragma HLS PIPELINE
        w = n * in->dw;
        f_n_plus_1 = pi2Integrand(in, w);
        elem = in->dw * (f_n_plus_1 + f_n) / 2;
        sum += elem;
        f_n = f_n_plus_1;
    }
    return sum;
}
} // internal

#define PI (3.14159265359f)
/// @brief Engine for Heston Closed Form Solution
/// @param[in] input_data A structure containing the kerenl model parameters
/// @return the calculated call value
template <typename DT>
DT hcfEngine(struct hcfEngineInputDataType<DT>* input_data) {
    DT pi1 = 0.5 + ((1 / PI) * internal::integrateForPi1(input_data));
    DT pi2 = 0.5 + ((1 / PI) * internal::integrateForPi2(input_data));
    return (input_data->s0 * pi1) - (internal::EXP(-(input_data->r * input_data->T)) * input_data->K * pi2);
}

#undef PI

} // namespace fintech
} // namespace xf

#endif // ifndef HLS_FINTECH_L2_UTILS_H
