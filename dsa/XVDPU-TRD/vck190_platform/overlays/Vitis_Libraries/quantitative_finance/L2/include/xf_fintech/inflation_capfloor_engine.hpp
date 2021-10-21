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
 * @file inflation_capfloor_engine.hpp
 *
 * @brief the file include class InflationCapFloorEngine.
 */
#ifndef _XF_FINTECH_INFLATION_CAPFLOOR_ENGINE_H_
#define _XF_FINTECH_INFLATION_CAPFLOOR_ENGINE_H_

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include "hls_math.h"
#include "xf_fintech/linear_interpolation.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {

namespace fintech {

namespace internal {

template <typename DT>
DT errorFunction(DT x) {
#pragma HLS allocation operation instances = dmul limit = 1
#pragma HLS allocation operation instances = fmul limit = 1

    DT tiny = 2.2204460492503131e-16;
    DT one = 1.00000000000000000000e+00;
    DT erx = 8.45062911510467529297e-01;

    DT efx = 1.28379167095512586316e-01;
    DT efx8 = 1.02703333676410069053e+00;
    DT pp0 = 1.28379167095512558561e-01;
    DT pp1 = -3.25042107247001499370e-01;
    DT pp2 = -2.84817495755985104766e-02;
    DT pp3 = -5.77027029648944159157e-03;
    DT pp4 = -2.37630166566501626084e-05;
    DT qq1 = 3.97917223959155352819e-01;
    DT qq2 = 6.50222499887672944485e-02;
    DT qq3 = 5.08130628187576562776e-03;
    DT qq4 = 1.32494738004321644526e-04;
    DT qq5 = -3.96022827877536812320e-06;

    DT pa0 = -2.36211856075265944077e-03;
    DT pa1 = 4.14856118683748331666e-01;
    DT pa2 = -3.72207876035701323847e-01;
    DT pa3 = 3.18346619901161753674e-01;
    DT pa4 = -1.10894694282396677476e-01;
    DT pa5 = 3.54783043256182359371e-02;
    DT pa6 = -2.16637559486879084300e-03;
    DT qa1 = 1.06420880400844228286e-01;
    DT qa2 = 5.40397917702171048937e-01;
    DT qa3 = 7.18286544141962662868e-02;
    DT qa4 = 1.26171219808761642112e-01;
    DT qa5 = 1.36370839120290507362e-02;
    DT qa6 = 1.19844998467991074170e-02;

    DT ra0 = -9.86494403484714822705e-03;
    DT ra1 = -6.93858572707181764372e-01;
    DT ra2 = -1.05586262253232909814e+01;
    DT ra3 = -6.23753324503260060396e+01;
    DT ra4 = -1.62396669462573470355e+02;
    DT ra5 = -1.84605092906711035994e+02;
    DT ra6 = -8.12874355063065934246e+01;
    DT ra7 = -9.81432934416914548592e+00;
    DT sa1 = 1.96512716674392571292e+01;
    DT sa2 = 1.37657754143519042600e+02;
    DT sa3 = 4.34565877475229228821e+02;
    DT sa4 = 6.45387271733267880336e+02;
    DT sa5 = 4.29008140027567833386e+02;
    DT sa6 = 1.08635005541779435134e+02;
    DT sa7 = 6.57024977031928170135e+00;
    DT sa8 = -6.04244152148580987438e-02;

    DT rb0 = -9.86494292470009928597e-03;
    DT rb1 = -7.99283237680523006574e-01;
    DT rb2 = -1.77579549177547519889e+01;
    DT rb3 = -1.60636384855821916062e+02;
    DT rb4 = -6.37566443368389627722e+02;
    DT rb5 = -1.02509513161107724954e+03;
    DT rb6 = -4.83519191608651397019e+02;
    DT sb1 = 3.03380607434824582924e+01;
    DT sb2 = 3.25792512996573918826e+02;
    DT sb3 = 1.53672958608443695994e+03;
    DT sb4 = 3.19985821950859553908e+03;
    DT sb5 = 2.55305040643316442583e+03;
    DT sb6 = 4.74528541206955367215e+02;
    DT sb7 = -2.24409524465858183362e+01;

    x = x * 0.7071067811865475; // sqrt(0.5)

#ifdef __SYNTHESIS__
    DT ax = hls::fabs(x);
#else
    DT ax = std::fabs(x);
#endif
    DT R, S, P, Q, s, y, z, r;
    DT value;
    if (ax < 3.7252902984e-09) {
        value = x + efx * x;
    } else if (ax < 0.84375) {
        z = x * x;
        r = pp0 + z * (pp1 + z * (pp2 + z * (pp3 + z * pp4)));
        s = one + z * (qq1 + z * (qq2 + z * (qq3 + z * (qq4 + z * qq5))));
        y = r / s;
        value = x + x * y;
    } else if (ax < 1.25) {
        s = ax - one;
        P = pa0 + s * (pa1 + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
        Q = one + s * (qa1 + s * (qa2 + s * (qa3 + s * (qa4 + s * (qa5 + s * qa6)))));
        if (x >= 0)
            value = erx + P / Q;
        else
            value = -erx - P / Q;
    } else if (ax >= 6) {
        if (x >= 0)
            value = one - tiny;
        else
            value = tiny - one;
    } else if (ax < 2.85714285714285) {
        s = one / (ax * ax);
        R = ra0 + s * (ra1 + s * (ra2 + s * (ra3 + s * (ra4 + s * (ra5 + s * (ra6 + s * ra7))))));
        S = one + s * (sa1 + s * (sa2 + s * (sa3 + s * (sa4 + s * (sa5 + s * (sa6 + s * (sa7 + s * sa8)))))));
#ifdef __SYNTHESIS__
        r = hls::exp(-ax * ax - 0.5625 + R / S);
#else
        r = std::exp(-ax * ax - 0.5625 + R / S);
#endif
        if (x >= 0)
            value = one - r / ax;
        else
            value = r / ax - one;
    } else {
        s = one / (ax * ax);
        R = rb0 + s * (rb1 + s * (rb2 + s * (rb3 + s * (rb4 + s * (rb5 + s * rb6)))));
        S = one + s * (sb1 + s * (sb2 + s * (sb3 + s * (sb4 + s * (sb5 + s * (sb6 + s * sb7))))));
#ifdef __SYNTHESIS__
        r = hls::exp(-ax * ax - 0.5625 + R / S);
#else
        r = std::exp(-ax * ax - 0.5625 + R / S);
#endif
        if (x >= 0)
            value = one - r / ax;
        else
            value = r / ax - one;
    }
    return value;
}

} // internal

/**
 * @brief InflationCapFloorEngine Inflation Cap/Floor Engine
 *
 * @tparam DT data type supported include float and double.
 * @tparam LEN maximum length of array
 *
 */
template <typename DT, int LEN>
class InflationCapFloorEngine {
   private:
    int type;
    DT forwardRate;
    DT cfRate[2];
    DT nominal;
    DT gearing;
    DT accrualTime;
    DT r;
    DT len;
    DT time[LEN];
    DT rate[LEN];

    void discountFactor(int& tLen,
                        // stream out
                        hls::stream<DT>& disc_stream) {
    loop_disc:
        for (int i = 1; i < tLen + 1; i++) {
//#pragma HLS pipeline
#pragma HLS loop_tripcount max = 10 min = 10
            DT t = i;
#ifdef __SYNTHESIS__
            DT tmp = hls::exp(forwardRate * t);
#else
            DT tmp = std::exp(forwardRate * t);
#endif
            DT result = nominal * gearing * accrualTime / tmp;
            disc_stream.write(result);
        }
    }

    void totalVariance(int& tLen,
                       // stream out
                       hls::stream<DT>& v_stream) {
    loop_stdDev:
        for (int i = 1; i < tLen + 1; i++) {
//#pragma HLS pipeline
#pragma HLS loop_tripcount max = 10 min = 10
#ifdef __SYNTHESIS__
            // volatility_->totalVariance
            DT stdDev = hls::sqrt(0.0001 * i);
#else
            DT stdDev = std::sqrt(0.0001 * i);
#endif
            v_stream.write(stdDev);
        }
    }

    void yoyRateImpl(int& tLen,
                     // stream out
                     hls::stream<DT>& rate_stream) {
    loop_rate:
        for (int i = 1; i < tLen + 1; i++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount max = 10 min = 10
            DT t = time[i];
            DT value = internal::linearInterpolation(t, len, time, rate);
            rate_stream.write(value);
        }
    }

    void blackFormula(int& tLen,
                      // stream in
                      hls::stream<DT>& rate_stream,
                      hls::stream<DT>& v_stream,
                      hls::stream<DT>& disc_stream,
                      // out
                      DT& price) {
        int option = type;
        DT strike = cfRate[type];
        price = 0.0;
    loop_black:
        for (int i = 0; i < tLen; i++) {
#pragma HLS pipeline ii = 50 // reduce resource and total latency no increase
#pragma HLS loop_tripcount max = 10 min = 10
            DT forward = rate_stream.read();
            DT stdDev = v_stream.read();
            DT discount = disc_stream.read();
#ifdef __SYNTHESIS__
            DT d1 = hls::log(forward / strike) / stdDev + 0.5 * stdDev;
#else
            DT d1 = std::log(forward / strike) / stdDev + 0.5 * stdDev;
#endif
            DT d2 = d1 - stdDev;
            DT nd1, nd2;
            if (option == 1) {
                d1 = -d1;
                d2 = -d2;
            }
            nd1 = internal::errorFunction<DT>(d1);
            nd2 = internal::errorFunction<DT>(d2);
            // nd1 = 0.99999999999999989;//phi(optionType*d1), errorFunction
            // nd2 = 0.99999999999999989;
            nd1 = 0.5 * (nd1 + 1.0);
            nd2 = 0.5 * (nd2 + 1.0);

            DT result;
            if (!option)
                result = discount * (forward * nd1 - strike * nd2);
            else
                result = discount * (strike * nd2 - forward * nd1);
#ifndef __SYNTHESIS__
            cout << "blackFormula i = " << i << setprecision(16) << ", forward= " << forward << ", stdDev=" << stdDev
                 << ", discount=" << discount << ", result=" << result << endl;
#endif
            // stream_price.write(result);
            price += result;
        }
        // return result;
    }

    void streamWrapper(int& tLen,
                       // stream out
                       DT& price) {
#pragma HLS dataflow
        hls::stream<DT> rate_stream("rate_stream");
        hls::stream<DT> v_stream("v_stream");
        hls::stream<DT> disc_stream("disc_stream");
#pragma HLS stream variable = rate_stream depth = LEN
#pragma HLS stream variable = v_stream depth = LEN
#pragma HLS stream variable = disc_stream depth = LEN
#pragma HLS resource variable = rate_stream core = FIFO_LUTRAM
#pragma HLS resource variable = v_stream core = FIFO_LUTRAM
#pragma HLS resource variable = disc_stream core = FIFO_LUTRAM
        discountFactor(tLen, disc_stream);
        totalVariance(tLen, v_stream);
        yoyRateImpl(tLen, rate_stream);
        blackFormula(tLen, rate_stream, v_stream, disc_stream, price);
    }

   public:
    /**
     * @brief default constructor
     */
    InflationCapFloorEngine() {
#pragma HLS inline
#pragma HLS resource variable = time core = RAM_2P_LUTRAM
#pragma HLS resource variable = rate core = RAM_2P_LUTRAM
#pragma HLS resource variable = cfRate core = RAM_2P_LUTRAM
    }

    /**
     * @brief init initialize array and parameters
     *
     * @param typeIn 0: Cap, 1: Floor. For Collar mode, it is actually the result of Cap mode minus the result of Floor
     * mode.
     * @param forwardRateIn base of forward rate
     * @param cfRateIn cfRateIn[0]: Cap strike rate, cfRateIn[1]: Floor strike rate
     * @param nominalIn nominal principal
     * @param gearingIn gearing
     * @param accrualTimeIn accrual time
     * @param size the actual size of array timeIn
     * @param timeIn array time, the difference between the maturity date and the reference date, unit is year.
     * @param rateIn array rate
     */
    void init(int typeIn,
              DT forwardRateIn,
              DT* cfRateIn,
              DT nominalIn,
              DT gearingIn,
              DT accrualTimeIn,
              int size,
              DT* timeIn,
              DT* rateIn) {
        type = typeIn;
        forwardRate = forwardRateIn;
        cfRate[0] = cfRateIn[0];
        cfRate[1] = cfRateIn[1];
        nominal = nominalIn;
        gearing = gearingIn;
        accrualTime = accrualTimeIn;

        len = size;
    loop_init:
        for (int i = 0; i < size; i++) {
#pragma HLS loop_tripcount max = 10 min = 10
#pragma HLS pipeline ii = 1
            time[i] = timeIn[i];
            rate[i] = rateIn[i];
        }
    }

    /**
     * @brief calcuNPV calculate NPV function
     *
     * @param len length of year
     * @return return the NPV result
     */
    DT calcuNPV(int len) {
        DT value = 0.0;
        DT price;
        streamWrapper(len, price);
        return price;
    }
};

} // fintech
} // xf
#endif
