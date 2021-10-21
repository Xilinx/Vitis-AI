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

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <iomanip>

using namespace std;

/* enable debug */
//#define DEBUG   (1)

#define SQRT2_RECIP 0.70710678118654752440084436210485f

template <typename DT>
DT Cdf(DT xin) {
    // Constants of approximation
    DT a1 = 0.254829592f;
    DT a2 = -0.284496736f;
    DT a3 = 1.421413741f;
    DT a4 = -1.453152027f;
    DT a5 = 1.061405429f;
    DT p = 0.3275911f;

    // Save the sign of x
    DT sign = (xin < 0.0f) ? -1.0f : 1.0f;
    DT x = SQRT2_RECIP * fabs(xin);

    // A&S formula
    DT t = 1.0f / (1.0f + p * x);
    DT y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return 0.5f * (1.0f + sign * y);
}

/*
 * Class: Cubic Spline Implementation
 */
template <typename DT>
class CubicSplineCPU {
   private:
    std::vector<DT> x; // vector of the maturities
    std::vector<DT> a; // vector of yields
    std::vector<DT> b, c, d;
    int n = 0;

   public:
    CubicSplineCPU() {}

    void init(std::map<DT, DT> data) {
        std::vector<DT> h, alpha, l, mu, z;
        n = data.size() - 1;

        // iterate over the yield curve & split into two vectors
        typename std::map<DT, DT>::iterator it;
        for (it = data.begin(); it != data.end(); it++) {
            x.push_back(it->first);
            a.push_back(it->second);
        }

        // create empty vectors
        for (int i = 0; i <= n; i++) {
            l.push_back(0.0);
            mu.push_back(0.0);
            z.push_back(0.0);
            c.push_back(0.0);

            if (i < n) {
                h.push_back(0.0);
                alpha.push_back(0.0);
                b.push_back(0.0);
                d.push_back(0.0);
            }
        }

        for (int i = 0; i < n; i++) {
            h[i] = (x[i + 1] - x[i]);
        }

        for (int i = 1; i < n; i++) {
            alpha[i] = (3 / h[i] * (a[i + 1] - a[i]) - 3 / h[i - 1] * (a[i] - a[i - 1]));
        }

        l[0] = 1.0;
        for (int i = 1; i < n; i++) {
            l[i] = (2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]);
            mu[i] = (h[i] / l[i]);
            z[i] = ((alpha[i] - h[i - 1] * z[i - 1]) / l[i]);
        }
        l[n] = 1.0;
        z[n] = 0.0;

        for (int j = n - 1; j > -1; j--) {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
            d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
        }

#ifdef DEBUG
        std::cout << "h: ";
        typename std::vector<DT>::const_iterator itdebug;
        for (itdebug = h.begin(); itdebug != h.end(); ++itdebug) {
            std::cout << *itdebug << " ";
        }
        std::cout << std::endl;

        std::cout << "alpha: ";
        for (itdebug = alpha.begin(); itdebug != alpha.end(); ++itdebug) {
            std::cout << *itdebug << " ";
        }
        std::cout << std::endl;

        std::cout << "l: ";
        for (itdebug = l.begin(); itdebug != l.end(); ++itdebug) {
            std::cout << *itdebug << " ";
        }
        std::cout << std::endl;

        std::cout << "mu: ";
        for (itdebug = mu.begin(); itdebug != mu.end(); ++itdebug) {
            std::cout << *itdebug << " ";
        }
        std::cout << std::endl;

        std::cout << "z: ";
        for (itdebug = z.begin(); itdebug != z.end(); ++itdebug) {
            std::cout << *itdebug << " ";
        }
        std::cout << std::endl;

        std::cout << "b: ";
        for (itdebug = b.begin(); itdebug != b.end(); ++itdebug) {
            std::cout << *itdebug << " ";
        }
        std::cout << std::endl;

        std::cout << "c: ";
        for (itdebug = c.begin(); itdebug != c.end(); ++itdebug) {
            std::cout << *itdebug << " ";
        }
        std::cout << std::endl;

        std::cout << "d: ";
        for (itdebug = d.begin(); itdebug != d.end(); ++itdebug) {
            std::cout << *itdebug << " ";
        }
        std::cout << std::endl;

        std::cout << "n:" << n << std::endl;
        std::cout << "--------------------------------------" << std::endl;
#endif
    }

    /* interpolated value */
    DT interValue(DT t) {
        DT value = 0;

        if (t < x.front()) t = x.front();

        if (t >= x.back()) t = x.back();

        for (int j = 0; j < n; j++) {
            if (t >= x[j] && t <= x[j + 1]) {
                value = a[j] + b[j] * (t - x[j]) + c[j] * std::pow((t - x[j]), 2) + d[j] * std::pow((t - x[j]), 3);
                break;
            }
        }

        return (value);
    }

    /* forward rate */
    DT fwdRate(DT t) {
        DT forwardRate = 0;

        if (t < x.front()) t = x.front();

        if (t >= x.back()) t = x.back();

        for (int j = 0; j < n; j++) {
            if (t >= x[j] && t <= x[j + 1]) {
                forwardRate = b[j] + 2 * c[j] * (t - x[j]) + 3 * d[j] * std::pow((t - x[j]), 2);
                break;
            }
        }

        return (forwardRate);
    }
};

/*
 * Class: Hull White Analytical Model
 *
 *
 */

template <typename DT>
class HWAnalyticalModelCPU {
   private:
    // mean reversion, sigma and risk free rate
    DT a_, sigma_;

    std::map<DT, DT> Z_;
    std::map<DT, DT> LogZ_;
    std::map<DT, DT> yieldCurve_;

    CubicSplineCPU<DT> Z_spline;
    CubicSplineCPU<DT> LogZ_spline;

   public:
    HWAnalyticalModelCPU() {}

    void init(DT a, DT sigma, std::map<DT, DT> yieldCurve) {
        // input parameters
        a_ = a;
        sigma_ = sigma;
        yieldCurve_ = yieldCurve;

        // iterate around yield curve to calculate Z/logZ & populate maps for Cubic Spline interpolation
        typename std::map<DT, DT>::iterator it;
        for (it = yieldCurve_.begin(); it != yieldCurve_.end(); it++) {
            Z_.insert(std::pair<DT, DT>(it->first, std::exp(-it->second * it->first)));
            LogZ_.insert(std::pair<DT, DT>(it->first, -it->second * it->first));
        }

        Z_spline.init(Z_);
        LogZ_spline.init(LogZ_);
    }

    DT P(DT t, DT T) {
        // P(0,T) - from spline interpolated yield curve Z at time T
        // P(0,t) - from spline interpolated yield curve Z at time t0
        // F(0,t) - from spline interpolated log yield curve Z at time t0
        DT P_T = Z_spline.interValue(T);
        DT P_t = Z_spline.interValue(t);
        DT f_t = -LogZ_spline.fwdRate(t);

        // B(t,T)
        DT B = 1 / a_ * (1 - std::exp(-a_ * (T - t)));

        // A(t,T)
        DT A = P_T / P_t *
               std::exp(B * f_t - std::pow(sigma_, 2) / (4 * a_) * (1 - std::exp(-2 * a_ * t)) * std::pow(B, 2));

        // P(t,T)
        return (A * std::exp(-f_t * B));
    }

    /* t - Settlement date
     * T - Option Maturity
     * S - Option Bond Maturity
     * K - Strike
     */
    DT ZBC(DT t, DT T, DT S, DT K) {
        // ZBC(t,T,S,K) = P(t,S)cdf(h) - KP(t,T)cdf(h-sigma_p)
        DT B_T_S = 1 / a_ * (1 - std::exp(-a_ * (S - T)));
        DT sigma_p = sigma_ * std::sqrt((1 - exp(-2 * a_ * (T - t))) / (2 * a_)) * B_T_S;
        DT h = 1 / sigma_p * std::log(P(t, S) / ((P(t, T) * K))) + sigma_p / 2;
#ifdef DEBUG
        std::cout << "B_T_S:" << B_T_S << std::endl;
        std::cout << "sigma_p:" << sigma_p << std::endl;
        std::cout << "h" << h << std::endl;
#endif
        return (P(t, S) * Cdf(h) - K * P(t, T) * Cdf(h - sigma_p));
    }

    /* t - Settlement date
     * T - Option Maturity
     * K - Strike
     * S - Option Bond Maturity
     */
    DT ZBP(DT t, DT T, DT S, DT K) {
        // ZBP(t,T,S,K) = KP(t,T)cdf(-h + sigma_p) - P(t,S)cdf(-h)
        DT B_T_S = 1 / a_ * (1 - std::exp(-a_ * (S - T)));
        DT sigma_p = sigma_ * std::sqrt((1 - std::exp(-2 * a_ * (T - t))) / (2 * a_)) * B_T_S;
        DT h = 1 / sigma_p * std::log(P(t, S) / ((P(t, T) * K))) + sigma_p / 2;
#ifdef DEBUG
        std::cout << std::setprecision(10) << std::endl;
        std::cout << "B_T_S:" << B_T_S << std::endl;
        std::cout << "sigma_p:" << sigma_p << std::endl;
        std::cout << "h:" << h << std::endl;
        std::cout << "P(t, S):" << P(t, S) << std::endl;
        std::cout << "P(t, T):" << P(t, T) << std::endl;
        std::cout << "P:" << K * P(t, T) * Cdf(-h + sigma_p) - P(t, S) * Cdf(-h) << std::endl;
#endif
        return (K * P(t, T) * Cdf(-h + sigma_p) - P(t, S) * Cdf(-h));
    }

    /* pricing floorlets and caplets
     * caplets are equivalent to zero bond puts
     */

    /* t - present time
     * T - starting and end times
     * N - Nominal value
     * X - strike interest rate
     */

    DT Cap(DT t, std::vector<DT> T, DT N, DT X) {
        DT s = 0;
        for (int i = 1; i < int(T.size()); i++) {
            DT ti = T[i] - T[i - 1];
            s += (1 + X * ti) * ZBP(t, T[i - 1], T[i], 1 / (1 + X * ti));
#ifdef DEBUG
            std::cout << "CAP: " << ti << " " << s << std::endl;
#endif
        }
        return N * s;
    }

    DT Floor(DT t, std::vector<DT> T, DT N, DT X) {
        DT s = 0;
        for (int i = 1; i < (int)(T.size()); i++) {
            DT ti = T[i] - T[i - 1];
            s += (1 + X * ti) * ZBC(t, T[i - 1], T[i], (1 / (1 + X * ti)));
#ifdef DEBUG
            std::cout << "Floor: " << ti << " " << s << std::endl;
#endif
        }
        return N * s;
    }

}; // class

typedef HWAnalyticalModelCPU<TEST_DT> Model;

void HWA_CPU_k0(TEST_DT a,
                TEST_DT sigma,
                TEST_DT* times,
                TEST_DT* rates,
                int yieldCurveLen,
                TEST_DT* t,
                TEST_DT* T,
                TEST_DT* P,
                int numToCalculate) {
    Model hwModelBond;

    std::map<TEST_DT, TEST_DT> yieldCurve;
    for (int i = 0; i < yieldCurveLen; i++) {
        yieldCurve.insert(std::pair<TEST_DT, TEST_DT>(times[i], rates[i]));
    }
    hwModelBond.init(a, sigma, yieldCurve);

    for (int i = 0; i < numToCalculate; i++) {
        P[i] = hwModelBond.P(t[i], T[i]);
    }

    return;
}

void HWA_CPU_k1(TEST_DT a,
                TEST_DT sigma,
                TEST_DT* times,
                TEST_DT* rates,
                int yieldCurveLen,
                int* callputType,
                TEST_DT* t,
                TEST_DT* T,
                TEST_DT* S,
                TEST_DT* K,
                TEST_DT* P,
                int numToCalculate) {
    Model hwModelBond;

    std::map<TEST_DT, TEST_DT> yieldCurve;
    for (int i = 0; i < yieldCurveLen; i++) {
        yieldCurve.insert(std::pair<TEST_DT, TEST_DT>(times[i], rates[i]));
    }
    hwModelBond.init(a, sigma, yieldCurve);

    for (int i = 0; i < numToCalculate; i++) {
        if (callputType[i] == 1) {
            P[i] = hwModelBond.ZBC(t[i], T[i], S[i], K[i]);
        } else {
            P[i] = hwModelBond.ZBP(t[i], T[i], S[i], K[i]);
        }
    }

    return;
}

extern void HWA_CPU_k2(TEST_DT a,
                       TEST_DT sigma,
                       TEST_DT* times,
                       TEST_DT* rates,
                       int yieldCurveLen,
                       int* capfloorType,
                       TEST_DT* startYear,
                       TEST_DT* endYear,
                       int* settlementFreq,
                       TEST_DT* N,
                       TEST_DT* X,
                       TEST_DT* P,
                       int numToCalculate) {
    Model hwModelBond;

    std::map<TEST_DT, TEST_DT> yieldCurve;
    for (int i = 0; i < yieldCurveLen; i++) {
        yieldCurve.insert(std::pair<TEST_DT, TEST_DT>(times[i], rates[i]));
    }
    hwModelBond.init(a, sigma, yieldCurve);

    for (int i = 0; i < numToCalculate; i++) {
        // build up the reset dates
        std::vector<TEST_DT> resetDates;
        TEST_DT interval = 1.0 / settlementFreq[i];
        TEST_DT numPayoffs = ((endYear[i] - startYear[i]) * settlementFreq[i]);
        TEST_DT temp = startYear[i];

        for (int i = 1; i <= numPayoffs; i++) {
            TEST_DT payOff = temp + (i * interval);
            resetDates.push_back(payOff);
        }

        if (capfloorType[i] == 0) {
            P[i] = hwModelBond.Cap(startYear[i], resetDates, N[i], X[i]);
        } else {
            P[i] = hwModelBond.Floor(startYear[i], resetDates, N[i], X[i]);
        }
    }

    return;
}
