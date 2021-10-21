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

#ifndef _XF_FINTECH_CDS_ENGINE_H_
#define _XF_FINTECH_CDS_ENGINE_H_

#include "xf_fintech/linear_interpolation.hpp"
#include <stdio.h>

namespace xf {

namespace fintech {

namespace internal {

template <typename DT>
DT integrateHazard(DT T, int len, DT* arrT, DT* arrV) {
    DT retVal = 0.0;

    if (T > 0) {
        DT sum = 0.0;
        DT tLast = 0.0;

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

} // namespace internal

template <typename DT, int IRATELEN, int HAZARDLEN>
class CDSEngine {
   private:
    static const int maxPayoffs = 12;
    static const int maxYears = 30;
    static const int maxMaturity = (maxPayoffs * maxYears);

    DT arrXInterestRate_[IRATELEN];
    DT arrYInterestRate_[IRATELEN];
    DT arrXHazardRate_[HAZARDLEN];
    DT arrYHazardRate_[HAZARDLEN];

    DT probSurvival[maxMaturity];
    DT probDefault[maxMaturity];
    DT expPayment[maxMaturity];
    DT discountFactorA[maxMaturity];
    DT discountFactorB[maxMaturity];

    DT expPaymentPV[maxMaturity];
    DT expPaymentPVTotal;
    DT expPayoff[maxMaturity];
    DT expPayoffPV[maxMaturity];
    DT expPayoffPVTotal;
    DT expAccrual[maxMaturity];
    DT expAccrualPV[maxMaturity];
    DT expAccrualPVTotal;

   public:
    CDSEngine() {
#pragma HLS inline
#pragma HLS array_partition variable = arrXInterestRate_ complete dim = 1
#pragma HLS array_partition variable = arrYInterestRate_ complete dim = 1
#pragma HLS array_partition variable = arrXHazardRate_ complete dim = 1
#pragma HLS array_partition variable = arrYHazardRate_ complete dim = 1
    }

    /* init with term structure for interest rate and hazard rate */
    void init(DT* interestRateX, DT* interestRateY, DT* hazardRateX, DT* hazardRateY) {
    loop_cds_engine_0:
        for (int i = 0; i < IRATELEN; i++) {
            arrXInterestRate_[i] = interestRateX[i];
            arrYInterestRate_[i] = interestRateY[i];
        }

    loop_cds_engine_1:
        for (int i = 0; i < HAZARDLEN; i++) {
            arrXHazardRate_[i] = hazardRateX[i];
            arrYHazardRate_[i] = hazardRateY[i];
        }
    }

    DT cdsSpread(int frequency, DT maturity, DT recoveryRate) {
        DT cdsLegs[maxMaturity];

        // accurual fraction
        DT dt = (1.0 / (DT)frequency);

        int iterations = (std::ceil(maturity) * frequency) + 1;
        int indexStart = iterations - 1;

        // generate a list of cdsLegs
        cdsLegs[indexStart] = maturity;

        int zeroFound = 0;

        // reset totals
        expPaymentPVTotal = 0.0;
        expPayoffPVTotal = 0.0;
        expAccrualPVTotal = 0.0;

    loop_cds_engine_2:
        for (int i = indexStart - 1; i >= 0; i--) {
#pragma HLS loop_tripcount min = 1 max = 120 avg = 20
            cdsLegs[i] = cdsLegs[i + 1] - dt;
            if (cdsLegs[i] < 0) {
            // break at the first less than zero value
            loop_cds_engine_3:
                for (int j = 0; j < (iterations - i); j++) {
#pragma HLS loop_tripcount min = 1 max = 120 avg = 20
                    cdsLegs[j] = cdsLegs[j + i];
                }
                iterations = iterations - i;
                break;
            }
        }

        // calculate probability of survival/default
        probSurvival[0] = 1.0; // probability survival to time 0 is 100%
        probDefault[0] = 0.0;

    loop_cds_engine_4:
        for (int i = 1; i < iterations; i++) {
#pragma HLS loop_tripcount min = 1 max = 120 avg = 20
            probSurvival[i] =
                1.0 -
                (1.0 - std::exp(-internal::integrateHazard(cdsLegs[i], HAZARDLEN, arrXHazardRate_, arrYHazardRate_)));
            probDefault[i] = probSurvival[i - 1] - probSurvival[i];
        }

    // calculate the present value expected payments
    loop_cds_engine_5:
        for (int i = 1; i < iterations; i++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1 max = 120 avg = 20
            expPayment[i] = probSurvival[i] * 1; // unit notional
            discountFactorA[i] =
                std::exp(-internal::linearInterpolation(cdsLegs[i], IRATELEN, arrXInterestRate_, arrYInterestRate_) *
                         cdsLegs[i]);
            expPaymentPV[i] = expPayment[i] * discountFactorA[i] * dt;
            expPaymentPVTotal += expPaymentPV[i];
        }

        // calculate the present value of expected payoff
        DT df = 0;
        expPayoffPV[0] = 0.0;

    loop_cds_engine_6:
        for (int i = 1; i < iterations; i++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1 max = 120 avg = 20
            expPayoff[i] = probDefault[i] * (1 - recoveryRate) * 1; // unit notional
            // check if the start cdsLegs is less than zero
            if ((cdsLegs[i] - dt) < 0) {
                df = cdsLegs[i] * 0.5;
            } else {
                df = cdsLegs[i] - dt * 0.5;
            }
            discountFactorB[i] =
                std::exp(-internal::linearInterpolation(df, IRATELEN, arrXInterestRate_, arrYInterestRate_) * df);
            expPayoffPV[i] = expPayoff[i] * discountFactorB[i];
            expPayoffPVTotal += expPayoffPV[i];
        }

        // calculate present value of accural interest
        expAccrual[0] = 0.0;

    loop_cds_engine_7:
        for (int i = 1; i < iterations; i++) {
#pragma HLS loop_tripcount min = 1 max = 120 avg = 20
            expAccrual[i] = probDefault[i] * 0.5 * dt;
            expAccrualPV[i] = expAccrual[i] * discountFactorB[i];
            expAccrualPVTotal += expAccrualPV[i];
        }

#ifndef __SYNTHESIS__

        std::cout << "Payments by CDS Buyer(Buyer of Protection)" << std::endl;
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        std::cout << "Year    DiscountF    SurvivalProb    PVExpPayment" << std::endl;
        std::cout << "-------------------------------------------------------------------------" << std::endl;

        for (int i = 1; i < iterations; i++) {
            std::cout << cdsLegs[i] << "    " << discountFactorA[i] << "    " << probSurvival[i] << "    "
                      << expPaymentPV[i] << std::endl;
        }
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        std::cout << "                                                PV Exp Payment: " << expPaymentPVTotal
                  << std::endl;
        std::cout << std::endl;

        std::cout << "-------------------------------------------------------------------------" << std::endl;
        std::cout << "Year    ExpAccrual    PVExpAccrual" << std::endl;
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        for (int i = 1; i < iterations; i++) {
            std::cout << cdsLegs[i] - 0.5 << "    " << expAccrual[i] << "    " << expAccrualPV[i] << std::endl;
        }
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        std::cout << "                                                PV ExpAccrual: " << expAccrualPVTotal
                  << std::endl;
        std::cout << std::endl;

        std::cout << "Payments by CDS Seller(Protection Seller)" << std::endl;
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        std::cout << "Year    ProbDefault    DiscountF    Notional    PV" << std::endl;
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        for (int i = 1; i < iterations; i++) {
            std::cout << cdsLegs[i] - 0.5 << " " << probDefault[i] << "    " << discountFactorB[i] << "    "
                      << expPayoff[i] << "    " << expPayoffPV[i] << std::endl;
        }
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        std::cout << "                                                PV Payoff Payment: " << expPayoffPVTotal
                  << std::endl;
        std::cout << std::endl;
#endif

        DT premiumLeg = (expPaymentPVTotal + expAccrualPVTotal);
        DT protectionLeg = expPayoffPVTotal;
        DT cdsSpread = protectionLeg / premiumLeg;

        return (cdsSpread);
    }
};

} // namespace fintech
} // namespace xf

#endif // _XF_FINTECH_CDS_ENGINE_H_
