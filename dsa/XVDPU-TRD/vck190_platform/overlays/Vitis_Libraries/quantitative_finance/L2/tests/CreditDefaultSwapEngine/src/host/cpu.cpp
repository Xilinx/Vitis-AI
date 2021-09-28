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

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <cmath>

#include "utils.hpp"

// Namespaces
using namespace std;

#define IRLEN (21)
#define HAZARDLEN (6)

////////////////////////////////////////////
// Credit Default Swap (CDS)
////////////////////////////////////////////

TEST_DT divideOperation(TEST_DT a, TEST_DT b) {
    return a / b;
}

TEST_DT linearInterpolation(TEST_DT x, int len, TEST_DT* arrX, TEST_DT* arrY) {
    int i;
    for (i = 0; i < len; i++) {
        if (x < arrX[i]) break;
    }
    TEST_DT x_last = arrX[i - 1];
    TEST_DT y_last = arrY[i - 1];
    TEST_DT slope = divideOperation((arrY[i] - y_last), (arrX[i] - x_last));
    return y_last + (x - x_last) * slope;
}

static TEST_DT integrateHazard(TEST_DT T, int len, TEST_DT* arrT, TEST_DT* arrV) {
    TEST_DT retVal = 0;

    if (T > 0) {
        TEST_DT sum = 0;
        TEST_DT tLast = 0;

        for (int i = 0; i < len; i++) {
            if (arrT[i] >= T) {
                sum += arrV[i] * (T - tLast);
                break;

            } else {
                sum += arrV[i] * (arrT[i] - tLast);
                tLast = arrT[i];
            }
        }

        retVal = sum;
    }

    return retVal;
}

#define MAX_PAYOFFS (12) // 12 payoffs per year
#define MAX_YEARS (30)
#define MAX_MATURITY (MAX_YEARS * MAX_PAYOFFS)

// template <typename DT>
TEST_DT cpu_cds_kernel(std::map<TEST_DT, TEST_DT> interestRateCurve,
                       std::map<TEST_DT, TEST_DT> hazardRateCurve,
                       TEST_DT notionalValue,
                       TEST_DT recoveryRate,
                       TEST_DT maturity,
                       int frequency) {
    TEST_DT probSurvival[MAX_MATURITY];
    TEST_DT probDefault[MAX_MATURITY];
    TEST_DT expPayment[MAX_MATURITY];
    TEST_DT discountFactorA[MAX_MATURITY];
    TEST_DT discountFactorB[MAX_MATURITY];

    TEST_DT expPaymentPV[MAX_MATURITY];
    TEST_DT expPaymentPVTotal = 0;
    TEST_DT expPayoff[MAX_MATURITY];
    TEST_DT expPayoffPV[MAX_MATURITY];
    TEST_DT expPayoffPVTotal = 0;
    TEST_DT expAccrual[MAX_MATURITY];
    TEST_DT expAccrualPV[MAX_MATURITY];
    TEST_DT expAccrualPVTotal = 0;

    TEST_DT arrXInterest[IRLEN];
    TEST_DT arrYInterest[IRLEN];
    typename std::map<TEST_DT, TEST_DT>::iterator iterInterest;
    int index = 0;

    for (iterInterest = interestRateCurve.begin(); iterInterest != interestRateCurve.end(); ++iterInterest) {
        arrXInterest[index] = iterInterest->first;
        arrYInterest[index] = iterInterest->second;
        index++;
    }

    TEST_DT arrXHazardRate[HAZARDLEN] = {0};
    TEST_DT arrYHazardRate[HAZARDLEN] = {0};
    index = 0;
    typename std::map<TEST_DT, TEST_DT>::iterator iterHazard;
    for (iterHazard = hazardRateCurve.begin(); iterHazard != hazardRateCurve.end(); ++iterHazard) {
        arrXHazardRate[index] = iterHazard->first;
        arrYHazardRate[index] = iterHazard->second;
        index++;
    }

    TEST_DT period[MAX_MATURITY];

    // accurual fraction
    TEST_DT dt = (1.0 / (TEST_DT)frequency);

    // row zero used to year 0 so add on row

    int iterations = (ceil(maturity) * frequency) + 1;
    int indexStart = iterations - 1;
    period[indexStart] = maturity;
    for (int i = indexStart - 1; i >= 0; i--) {
        period[i] = period[i + 1] - dt;
        if (period[i] < 0) {
            // break at the first less than zero value
            for (int j = 0; j < (iterations - i); j++) {
                period[j] = period[j + i];
            }
            iterations = iterations - i;
            break;
        }
    }

    // calculate probability of survival/default
    probSurvival[0] = 1.0; // probability survival to time 0 is 100%
    probDefault[0] = 0.0;
    for (int i = 1; i < iterations; i++) {
        probSurvival[i] = 1 - (1 - std::exp(-integrateHazard(period[i], HAZARDLEN, arrXHazardRate, arrYHazardRate)));
        probDefault[i] = probSurvival[i - 1] - probSurvival[i];
    }

    // calculate the present value expected payments
    for (int i = 1; i < iterations; i++) {
        expPayment[i] = probSurvival[i] * 1; // unit notional
        discountFactorA[i] = std::exp(-linearInterpolation(period[i], IRLEN, arrXInterest, arrYInterest) * period[i]);
        expPaymentPV[i] = expPayment[i] * discountFactorA[i] * dt;
        expPaymentPVTotal += expPaymentPV[i];
    }

    // calculate the present value of expected payoff
    TEST_DT df = 0;
    for (int i = 1; i < iterations; i++) {
        expPayoff[i] = probDefault[i] * (1 - recoveryRate) * 1; // unit notional
        // check if the start period is less than zero
        if ((period[i] - dt) < 0) {
            df = period[i] * 0.5;
        } else {
            df = period[i] - dt * 0.5;
        }
        discountFactorB[i] = std::exp(-linearInterpolation(df, IRLEN, arrXInterest, arrYInterest) * df);
        expPayoffPV[i] = expPayoff[i] * discountFactorB[i];
        expPayoffPVTotal += expPayoffPV[i];
    }

    // calculate present value of accural interest
    for (int i = 1; i < iterations; i++) {
        expAccrual[i] = probDefault[i] * 0.5 * dt;
        expAccrualPV[i] = expAccrual[i] * discountFactorB[i];
        expAccrualPVTotal += expAccrualPV[i];
    }

    TEST_DT premiumLeg = (expPaymentPVTotal + expAccrualPVTotal);
    TEST_DT protectionLeg = expPayoffPVTotal;
    TEST_DT cdsSpread = protectionLeg / premiumLeg;

#if 0
    printf("Annunity: %.12f\n", expPaymentPVTotal);
    printf("Accurual: %.12f\n", expAccrualPVTotal);
    printf("Payoff: %.12f\n", expPayoffPVTotal);
#endif

    // return the cds spread
    return (cdsSpread);
}
