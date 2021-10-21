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

#ifndef _XF_FINTECH_BT_ENGINE_H_
#define _XF_FINTECH_BT_ENGINE_H_

#include "L2_utils.hpp"

namespace xf {
namespace fintech {

/**
 * @param BinomialTreeMaxNodeDepth the maximum number of intervals (0 to 1023)
 * @param BinomialTreeEuropeanPut calculate for European Put.
 * @param BinomialTreeEuropeanPut calculate for European Call.
 * @param BinomialTreeEuropeanPut calculate for American Put.
 * @param BinomialTreeEuropeanPut calculate for American Call.
 *
 */
const static int BinomialTreeMaxNodeDepth = 1024;
const static int BinomialTreeEuropeanPut = 1;
const static int BinomialTreeEuropeanCall = 2;
const static int BinomialTreeAmericanPut = 3;
const static int BinomialTreeAmericanCall = 4;

/**
 *
 * @tparam DT supported data type including double and float data type, which
 * decides the precision of result, default is float.
 * @param S the spot price.
 * @param K the strike price.
 * @param T the expiration time.
 * @param V annualized  volatility of underlying asset.
 * @param rf annualized risk-free interest rate.
 * @param q the dividend yield.
 * @param N the number of intervals.
 */
template <typename DT>
struct BinomialTreeInputDataType {
    DT S;
    DT K;
    DT T;
    DT rf;
    DT V;
    DT q;
    int N;
    int packed; // Bitwidth of (packed) data on axi master must be power of 2 (set
                // to support float)
};

template <>
struct BinomialTreeInputDataType<double> {
    double S;
    double K;
    double T;
    double rf;
    double V;
    double q;
    int N;
    int packed[3]; // Bitwidth of (packed) data on axi master must be power of 2
                   // (set to support float)
};

/// @brief BinomialTree Engine using CRR (Cox, Ross & Rubinstein)
/// @param[in] inputData A structure containing the input model parameters
/// @param[in] optionType Calculate for NPV European or American Call or Put
/// @return The calculated NPV

template <typename DT>
DT binomialTreeEngine(BinomialTreeInputDataType<DT>* inputData, int optionType) {
#pragma HLS INLINE off
#pragma HLS DATA_PACK variable = inputData

    DT options[BinomialTreeMaxNodeDepth];
#pragma HLS ARRAY_PARTITION variable = options cyclic factor = 4 dim = 1
    DT exercise[BinomialTreeMaxNodeDepth];
#pragma HLS ARRAY_PARTITION variable = exercise cyclic factor = 4 dim = 1
    DT tmpOptions[BinomialTreeMaxNodeDepth];
#pragma HLS ARRAY_PARTITION variable = tmpOptions cyclic factor = 4 dim = 1

    DT K = inputData->K;
    DT S = inputData->S;
    DT T = inputData->T;
    DT rf = inputData->rf;
    DT V = inputData->V;
    DT q = inputData->q;
    int N = inputData->N;

    DT deltaT = DT(T / N);
    DT upFactor = internal::EXP(V * internal::SQRT(deltaT));
    DT pUp =
        (upFactor * internal::EXP(-q * deltaT) - internal::EXP(-rf * deltaT)) / (internal::POW(upFactor, DT(2)) - 1);
    DT pDown = internal::EXP(-rf * deltaT) - pUp;

    // Initial values at time simulated T (Leaf nodes)
    // Range of N is 0 to 1023
    for (int i = 0; i <= N; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 1024
#pragma HLS UNROLL factor = 4

        if (optionType == BinomialTreeAmericanPut || optionType == BinomialTreeEuropeanPut) {
            // European & American Put
            options[i] = K - (S * internal::POW(upFactor, DT(2 * i - N)));
        } else {
            options[i] = (S * internal::POW(upFactor, DT(2 * i - N))) - K;
        }
    }

    for (int i = 0; i <= N; i++) {
// European & American Call
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 1024
#pragma HLS UNROLL factor = 4
        options[i] = internal::MAX(options[i], DT(0.0));
    }

    // Work back from the leaf nodes to get the NPV
    for (int j = (N - 1); j >= 0; j--) {
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 1024

        for (int i = 0; i <= j; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 1024
#pragma HLS UNROLL factor = 4

            // Binomial Value - represents the fair price of the derivative at a
            // particular point in time
            tmpOptions[i] = DT((pUp * options[i + 1]) + (pDown * options[i]));
        }

        for (int i = 0; i <= j; i++) {
#pragma HLS UNROLL factor = 4
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 1024
            if (optionType == BinomialTreeAmericanCall || optionType == BinomialTreeAmericanPut) {
                if (optionType == BinomialTreeAmericanPut) {
                    exercise[i] = K - S * internal::POW(upFactor, DT(2 * i - j));
                } else if (optionType == BinomialTreeAmericanCall) {
                    exercise[i] = S * internal::POW(upFactor, DT(2 * i - j)) - K;
                }

                options[i] = internal::MAX(tmpOptions[i], exercise[i]);
            } else {
                // European option has no early exercise option
                options[i] = tmpOptions[i];
            }
        }
    }

    // Return calculated NPV
    return (options[0]);
}

} // namespace fintech
} // namespace xf

#endif // _XF_FINTECH_BT_ENGINE_H_
