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

#ifndef _XF_FINTECH_HESTON_MODEL_PARAMETERS_
#define _XF_FINTECH_HESTON_MODEL_PARAMETERS_

using namespace std;

namespace xf {
namespace fintech {

class HestonFDModelParameters {
   public:
    /**
     * @brief Heston FD Model Class Parameters.
     *
     *
     * @param K the strike price.
     * @param S the stock price.
     * @param V volatility of stock.
     * @param T time to maturity.
     * @param kappa mean reversion rate.
     * @param sig the volatility of volatility.
     * @param rho the correlation coefficient between price and variance.
     * @param eta long run average price.
     * @param rd risk-free domestic interest rate.
     * @param rf risk-free international interest rate.
     *
     */
    HestonFDModelParameters(
        double K, double S, double V, double T, double kappa, double sig, double rho, double eta, double rd, double rf);
    /** @brief get the strike price set */
    double Get_K() { return _K; }
    /** @brief get the stock price set */
    double Get_S() { return _S; }
    /** @brief get the volatility of stock set */
    double Get_V() { return _V; }
    /** @brief get the volatility of time to maturity set */
    double Get_T() { return _T; }
    /** @brief get the mean reversion rate set */
    double Get_kappa() { return _kappa; }
    /** @brief get the volatility of volatility set */
    double Get_sig() { return _sig; }
    /** @brief get the correlation coefficient between price and variance set */
    double Get_rho() { return _rho; }
    /** @brief get the long run average price set */
    double Get_eta() { return _eta; }
    /** @brief get the risk-free domestic interest rate set */
    double Get_rd() { return _rd; }
    /** @brief get the risk-free international interest rate set */
    double Get_rf() { return _rf; }

   private:
    double _K;
    double _S;
    double _V;
    double _T;
    double _kappa;
    double _sig;
    double _rho;
    double _eta;
    double _rd;
    double _rf;
};

} // namespace fintech
} // namespace xf

#endif // _XF_FINTECH_HESTON_MODEL_PARAMETERS_
