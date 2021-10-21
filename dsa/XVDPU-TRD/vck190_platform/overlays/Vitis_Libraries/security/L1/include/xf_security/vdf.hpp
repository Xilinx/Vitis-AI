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
 *
 * @file vdf.hpp
 * This file is part of Vitis Security Library.
 *
 */

#ifndef _XF_SECURITY_VDF_HPP_
#define _XF_SECURITY_VDF_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include "modular.hpp"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace security {

/**
 * @brif evaluate evalute function of verifiable delay function
 * @tparam L bit width
 * @tparam N bit width
 * @param g is the base element given as input to the VDF evaluator
 * @param l is a random prime
 * @param modulus is a security parameter
 * @param r2Mod Pre-calculated Montgomery parameter
 * @param t is a time bound
 * @param y is output of evaluate
 * @param pi is a proof
 */
template <int L, int N>
void evaluate(
    ap_uint<L> g, ap_uint<L> l, ap_uint<L> modulus, ap_uint<L> r2Mod, ap_uint<N> t, ap_uint<L>& y, ap_uint<L>& pi) {
    ap_uint<L> g_temp = g;
    ap_uint<L> pi_temp = 1;
    ap_uint<L> pi_temp2 = 1;
    ap_uint<L + 1> r = 1;

#if !defined(__SYNTHESIS__) && DEBUG
    std::cout << std::hex << "g=" << g << ",modulus=" << modulus << ",t=" << t << std::endl;
#endif
    ap_uint<L> g_mon = xf::security::internal::monProduct<L>(g_temp, r2Mod, modulus);
    pi_temp = xf::security::internal::monProduct<L>(pi_temp, r2Mod, modulus);
    ap_uint<L> g_temp2 = g_mon;
    for (int i = 0; i < t; i++) {
#pragma HLS loop_tripcount max = 100 min = 100
        // evaluate
        // g_temp = (g_temp * g_temp) % m;
        g_temp2 = xf::security::internal::monProduct<L>(g_temp2, g_temp2, modulus);

        // verify Prover
        ap_uint<2> b = (r * 2) / l;
        r = (r * 2) % l;

        pi_temp = xf::security::internal::monProduct<L>(pi_temp, pi_temp, modulus);

        // pi_temp2 = (pi_temp2 * pi_temp2) % modulus;
        if (b) {
            pi_temp = xf::security::internal::monProduct<L>(pi_temp, g_mon, modulus);
            // pi_temp2 = (pi_temp2 * g) % modulus;
        }
    }
    y = xf::security::internal::monProduct<L>(1, g_temp2, modulus);
    pi = xf::security::internal::monProduct<L>(1, pi_temp, modulus);
#if !defined(__SYNTHESIS__) && DEBUG
    std::cout << "y=" << y << std::endl;
    std::cout << "pi=" << pi << std::endl;
#endif
}

/**
 * @brif verifyWesolowski verify function of verifiable delay function
 * @tparam L bit width
 * @tparam N bit width
 * @param g is the base element given as input to the VDF evaluator
 * @param l is a random prime
 * @param modulus is a security parameter
 * @param r2Mod Pre-calculated Montgomery parameter
 * @param tMod is a time bound's Pre-calculation value
 * @param t is a time bound
 * @param y is output of evaluate
 * @param pi is a proof
 */
template <int L, int N>
bool verifyWesolowski(ap_uint<L> g,
                      ap_uint<L> l,
                      ap_uint<L> modulus,
                      ap_uint<L> r2Mod,
                      ap_uint<L> tMod,
                      ap_uint<N> t,
                      ap_uint<L> y,
                      ap_uint<L>& pi) {
    // ap_uint<L> y_v = pi_temp * g_temp % modulus;
    // ap_uint<L> y_v = std::pow(pi, l) * std::pow(g, r);

    ap_uint<L> p_pi = xf::security::internal::monProduct<L>(r2Mod, pi, modulus);
    ap_uint<L> r_pi = xf::security::internal::monProduct<L>(r2Mod, 1, modulus);
    ap_uint<L> p_g = xf::security::internal::monProduct<L>(r2Mod, g, modulus);
    ap_uint<L> r_g = xf::security::internal::monProduct<L>(r2Mod, 1, modulus);
    for (int i = N - 1; i >= 0; i--) {
        r_pi = xf::security::internal::monProduct<L>(r_pi, r_pi, modulus);
        r_g = xf::security::internal::monProduct<L>(r_g, r_g, modulus);
        if (l[i] == 1) {
            r_pi = xf::security::internal::monProduct<L>(r_pi, p_pi, modulus);
        }
        if (tMod[i] == 1) {
            r_g = xf::security::internal::monProduct<L>(r_g, p_g, modulus);
        }
    }

    // ap_uint<L> pi_temp = xf::security::internal::modularExp(pi, l, modulus, r2Mod);
    // ap_uint<L> g_temp = xf::security::internal::modularExp(g, tMod, modulus, r2Mod);
    ap_uint<L> y_v = xf::security::internal::monProduct<L>(r_pi, r_g, modulus);
    y_v = xf::security::internal::monProduct<L>(1, y_v, modulus);
#if !defined(__SYNTHESIS__) && DEBUG
    std::cout << "y=" << y << ",y_v=" << y_v << std::endl;
#endif
    return (y != y_v);
}

/**
 * @brief verifyPietrzak verify function of verifiable delay function, its algorithm: if T is even, return (N, x, T/2,
 * y) = (N, x^(r+2^(T/2)), T/2, x^(r*2^(T/2)+2^T)), if T is odd, return (N,
 * x, (T+1)/2, y) = (N, x^(r+2^((T-1)/2)), T/2, x^(r*2^((T+1)/2)+2^T))
 * @tparam L bit width
 * @tparam N bit width
 * @param g is the base element given as input to the VDF evaluator
 * @param modulus is a security parameter
 * @param r2Mod Pre-calculated Montgomery parameter
 * @param T is a time bound
 * @param y is output of evaluate
 */
template <int L, int N>
bool verifyPietrzak(ap_uint<L> g, ap_uint<L> modulus, ap_uint<L> r2Mod, ap_uint<N> T, ap_uint<L> y) {
    ap_uint<L> x_tmp = xf::security::internal::monProduct<L>(g, r2Mod, modulus);
    ap_uint<L> y_tmp = xf::security::internal::monProduct<L>(y, r2Mod, modulus);
    int r = 15; // random from verifier
    while (T > 1) {
        ap_uint<N> T1 = T;
        T = T / 2;
        ap_uint<L> mu = x_tmp;
        for (int i = 0; i < T; i++) {
            mu = xf::security::internal::monProduct<L>(mu, mu, modulus); // prover
        }
        ap_uint<L> x_r = x_tmp;
        ap_uint<L> y_r = mu;
        for (int i = 0; i < r - 1; i++) {
            x_r = xf::security::internal::monProduct<L>(x_r, x_tmp, modulus);
            y_r = xf::security::internal::monProduct<L>(y_r, mu, modulus);
        }
        x_tmp = xf::security::internal::monProduct<L>(mu, x_r, modulus); // prover
        y_tmp = xf::security::internal::monProduct<L>(y_r, y_tmp, modulus);
        if (T1 % 2 == 1) {
            y_tmp = xf::security::internal::monProduct<L>(y_r, y_tmp, modulus); // verifier
            T = (T1 + 1) / 2;
        }
    }
    ap_uint<L> y2_tmp;
    if (T == 1) {
        y2_tmp = xf::security::internal::monProduct<L>(x_tmp, x_tmp, modulus); // verifier
    }
    ap_uint<L> y2 = xf::security::internal::monProduct<L>(1, y2_tmp, modulus);
    y_tmp = xf::security::internal::monProduct<L>(1, y_tmp, modulus);
#if !defined(__SYNTHESIS__) && DEBUG
    std::cout << "y=" << y_tmp << ",y2=" << y2 << ", " << (y_tmp == y2) << std::endl;
#endif
    return (y_tmp != y2);
}

} // namespace security
} // namespace xf

#endif
