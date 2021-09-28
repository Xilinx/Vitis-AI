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
 * @file path_pricer.hpp
 * @brief This file contains the path pricer for different option
 *
 * Path pircer calculates the payoff based on the value of underling asset.
 */
#ifndef XF_FINTECH_PATH_PRICER_H
#define XF_FINTECH_PATH_PRICER_H

#include "ap_int.h"
#include "hls_stream.h"
#include "utils.hpp"
#include "lmm.hpp"
#include "xf_fintech/enums.hpp"
#include "xf_fintech/utils.hpp"
#ifndef __SYNTHESIS__
#include <assert.h>
#endif

namespace xf {
namespace fintech {
namespace internal {
using namespace fintech::enums;

template <OptionStyle style, typename DT, bool StepFirst, int SampNum, bool WithAntithetic, int MaxSteps = 1024>
class PathPricer {
   public:
    const static unsigned int InN = WithAntithetic ? 2 : 1;
    const static unsigned int OutN = InN;

    PathPricer() {
#pragma HLS inline
    }

    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> priceStrmOut[OutN]) {
#ifndef __SYNTHESIS__
        printf("Option Style is not supported now!\n");
#endif
    }
};

template <OptionStyle style, typename DT, int ASSETS, int SampleNum>
class MultiAssetPathPricer {
   public:
    const static unsigned int InN = 1;
    MultiAssetPathPricer() {
#pragma HLS inline
    }
    void Pricing(ap_uint<16> steps, ap_uint<16> paths, hls::stream<DT> pathStrmIn[InN], hls::stream<DT>& priceStrmOut) {
#ifndef __SYNTHESIS__
        printf("Option Style is not supported now!\n");
#endif
    }
};

template <typename DT, int ASSETS, int SampleNum>
class MultiAssetPathPricer<European, DT, ASSETS, SampleNum> {
   public:
    DT underlying[ASSETS];
    DT strike;
    DT discount;
    bool optionType;

    DT buffer[SampleNum];
    const static unsigned int InN = 1;
    const static unsigned int OutN = InN;
    const static bool byPassGen = false;

    MultiAssetPathPricer() {
#pragma HLS inline
    }
    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> priceStrmOut[InN]) {
        for (int i = 0; i < SampleNum; i++) {
#pragma HLS unroll
            buffer[i] = 0;
        }

        for (int t = 1; t <= steps; t++) {
            for (int i = 0; i < ASSETS; i++) {
                for (int j = 0; j < paths; j++) {
#pragma HLS PIPELINE II = 1
                    DT pIn = pathStrmIn[0].read();
                    if (t == steps) {
                        DT s1 = FPExp(pIn);
                        DT s = FPTwoMul(underlying[i], s1);
                        buffer[j] += s;
                    }
                }
            }
        }

        for (int i = 0; i < paths; i++) {
#pragma HLS PIPELINE II = 1
            DT op1 = 0;
            DT op2 = 0;
            if (optionType) {
                op1 = strike;
                op2 = buffer[i];
            } else {
                op1 = buffer[i];
                op2 = strike;
            }
            DT p1 = FPTwoSub(op1, op2);
            DT payoff = FPTwoMul(discount, MAX(p1, 0));
            priceStrmOut[0].write(payoff);
        }
    }
};

template <typename DT, bool StepFirst, int SampNum, bool WithAntithetic>
class PathPricer<European, DT, StepFirst, SampNum, WithAntithetic> {
   public:
    const static unsigned int InN = WithAntithetic ? 2 : 1;
    const static unsigned int OutN = InN;

    const static bool byPassGen = false;

    // configuration of the path pricer
    DT strike;
    DT underlying;
    DT drift;
    DT discount;

    bool optionType;

    PathPricer() {
#pragma HLS inline
    }

    void PE(ap_uint<16> steps, ap_uint<16> paths, hls::stream<DT>& pathStrmIn, hls::stream<DT>& priceStrmOut) {
#pragma HLS inline off
#ifndef __SYNTHESIS__
        static int cnt = 0;
#endif
        DT s = underlying;
        if (StepFirst) {
            for (int i = 0; i < paths; ++i) {
                for (int t = 1; t <= steps; t++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = SampNum max = SampNum
                    DT logS = pathStrmIn.read();
                    if (t == 1) s = underlying;
                    DT s1 = FPExp(logS);
                    s = FPTwoMul(s, s1);
                    if (t == steps) {
                        DT op1 = 0;
                        DT op2 = 0;
                        if (optionType) {
                            op1 = strike;
                            op2 = s;
                        } else {
                            op1 = s;
                            op2 = strike;
                        }
                        DT p1 = FPTwoSub(op1, op2);
                        DT payoff = MAX(p1, 0); // discount * MAX(s1, 0);
                        DT price = FPTwoMul(discount, payoff);
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                        if (cnt < 1024) {
                            std::cout << "pIn = " << pIn << std::endl;
                            std::cout << "payoff = " << payoff << std::endl;
                        }
                        cnt++;
#endif
#endif
                        priceStrmOut.write(price);
                    }
                }
            }
        } else {
            for (int i = 0; i < paths; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = SampNum max = SampNum
                DT logS = pathStrmIn.read();
                DT s1 = FPExp(logS);
                DT s = FPTwoMul(underlying, s1);
                DT op1 = 0;
                DT op2 = 0;
                if (optionType) {
                    op1 = strike;
                    op2 = s;
                } else {
                    op1 = s;
                    op2 = strike;
                }
                DT p1 = FPTwoSub(op1, op2);
                DT payoff = MAX(p1, 0); // discount * MAX(s1, 0);
                DT price = FPTwoMul(discount, payoff);
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                if (cnt < 1024) {
                    std::cout << "pIn = " << pIn << std::endl;
                    std::cout << "payoff = " << payoff << std::endl;
                }
                cnt++;
#endif
#endif
                priceStrmOut.write(price);
            }
        }
    } // PE()
    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> priceStrmOut[OutN]) {
        for (int i = 0; i < InN; ++i) {
#pragma HLS unroll
            PE(steps, paths, pathStrmIn[i], priceStrmOut[i]);
        }
    }
};
template <typename DT, bool StepFirst, int SampNum, bool WithAntithetic>
class PathPricer<EuropeanBypass, DT, StepFirst, SampNum, WithAntithetic> {
   public:
    const static unsigned int InN = WithAntithetic ? 2 : 1;
    const static unsigned int OutN = InN;

    const static bool byPassGen = false;

    // configuration of the path pricer
    DT strike;
    DT underlying;
    DT drift;
    DT discount;

    bool optionType;

    PathPricer() {
#pragma HLS inline
    }

    void PE(ap_uint<16> steps, ap_uint<16> paths, hls::stream<DT>& pathStrmIn, hls::stream<DT>& priceStrmOut) {
#pragma HLS inline off
#ifndef __SYNTHESIS__
        static int cnt = 0;
#endif
        DT s = underlying;
        for (int i = 0; i < paths; ++i) {
            for (int t = 1; t <= steps; ++t) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = SampNum max = SampNum
                DT logS = pathStrmIn.read();
                DT s1 = FPExp(logS);
                if (t == 1) s = underlying;
                s = FPTwoMul(s, s1);
                if (t == steps) priceStrmOut.write(s);
            }
        }
    } // PE()
    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> priceStrmOut[OutN]) {
        for (int i = 0; i < InN; ++i) {
#pragma HLS unroll
            PE(steps, paths, pathStrmIn[i], priceStrmOut[i]);
        }
    }
};

template <typename DT, bool StepFirst, int SampNum, bool WithAntithetic>
class PathPricer<Asian_AP, DT, StepFirst, SampNum, WithAntithetic> {
   public:
    const static unsigned int InN = WithAntithetic ? 2 : 1;
    const static unsigned int OutN = InN;
    const static bool byPassGen = false;

    DT strike;

    DT underlying;

    bool optionType;

    DT discount;

    DT dt;

    PathPricer() {
#pragma HLS inline
    }

    void PE(ap_uint<16> steps, ap_uint<16> paths, hls::stream<DT>& pathStrmIn, hls::stream<DT>& priceStrmOut) {
#pragma HLS inline off

        DT prelogS[SampNum];
        DT sumlogS[SampNum];
        DT sumS[SampNum];

        if (StepFirst) {
#ifndef __SYNTHESIS__
            printf("Only Samples First is supported now!\n");
#endif
        } else {
            for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 8 max = 8
                for (int j = 0; j < paths; ++j) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
#pragma HLS pipeline II = 1
                    DT dlogS = pathStrmIn.read();

                    DT tmplogS, tmpprelogS, tmpsumlogS, tmpsumS;
                    if (i == 0) {
                        tmpprelogS = 0;
                        tmpsumlogS = 0;
                        tmpsumS = 1;
                    } else {
                        tmpprelogS = prelogS[j];
                        tmpsumlogS = sumlogS[j];
                        tmpsumS = sumS[j];
                    }

                    tmplogS = tmpprelogS + dlogS;
                    prelogS[j] = tmplogS;
                    sumlogS[j] = tmpsumlogS + tmplogS;
                    sumS[j] = tmpsumS + FPExp(tmplogS);

                    if (i == steps - 1) {
                        DT sAP = sumS[j] / (steps + 1) * underlying;
                        DT sGP = FPExp(sumlogS[j] / (steps + 1)) * underlying;

                        DT op1, op2, op3, op4;
                        if (optionType) {
                            op1 = strike;
                            op2 = sAP;
                            op3 = strike;
                            op4 = sGP;
                        } else {
                            op1 = sAP;
                            op2 = strike;
                            op3 = sGP;
                            op4 = strike;
                        }
                        DT s1 = FPTwoSub(op1, op2);
                        DT s2 = FPTwoSub(op3, op4);
                        DT payoff = MAX(s1, 0);
                        DT payoff2 = MAX(s2, 0);
                        DT price1 = FPTwoMul(payoff, discount);
                        DT price2 = FPTwoMul(payoff2, discount);
                        DT price3 = price1 - price2;

                        priceStrmOut.write(price3);
                    }
                }
            }
        }
    }
    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> priceStrmOut[OutN]) {
        for (int i = 0; i < InN; ++i) {
#pragma HLS unroll
            PE(steps, paths, pathStrmIn[i], priceStrmOut[i]);
        }
    }
}; // end

template <typename DT, bool StepFirst, int SampNum, bool WithAntithetic>
class PathPricer<Asian_AS, DT, StepFirst, SampNum, WithAntithetic> {
   public:
    const static unsigned int InN = WithAntithetic ? 2 : 1;
    const static unsigned int OutN = InN;
    const static bool byPassGen = false;

    bool optionType;

    DT underlying;

    DT strike;

    DT discount;

    PathPricer() {
#pragma HLS inline
    }

    void PE(ap_uint<16> steps, ap_uint<16> paths, hls::stream<DT>& pathStrmIn, hls::stream<DT>& priceStrmOut) {
#pragma HLS inline off

        DT prelogS[SampNum];
        DT sumlogS[SampNum];
        DT sumS[SampNum];

        if (StepFirst) {
#ifndef __SYNTHESIS__
            printf("Only Samples First is supported now!\n");
#endif
        } else {
            for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 8 max = 8
                for (int j = 0; j < paths; ++j) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
#pragma HLS pipeline II = 1
                    DT dlogS = pathStrmIn.read();

                    DT tmplogS, tmpprelogS, tmpsumlogS, tmpsumS;
                    if (i == 0) {
                        tmpprelogS = 0;
                        tmpsumS = 1;
                    } else {
                        tmpprelogS = prelogS[j];
                        tmpsumS = sumS[j];
                    }

                    tmplogS = tmpprelogS + dlogS;
                    prelogS[j] = tmplogS;
                    DT tmpPrice = FPExp(tmplogS);
                    sumS[j] = tmpsumS + tmpPrice;

                    if (i == steps - 1) {
                        DT sAP = sumS[j] / (steps + 1) * underlying;
                        DT price = tmpPrice * underlying;

                        DT op1, op2;
                        if (optionType) {
                            op1 = sAP;
                            op2 = price;
                        } else {
                            op1 = price;
                            op2 = sAP;
                        }
                        DT s1 = FPTwoSub(op1, op2);
                        DT payoff = MAX(s1, 0);
                        DT price1 = FPTwoMul(payoff, discount);
                        priceStrmOut.write(price1);
                    }
                }
            }
        }
    }

    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> priceStrmOut[OutN]) {
        for (int i = 0; i < InN; ++i) {
#pragma HLS unroll
            PE(steps, paths, pathStrmIn[i], priceStrmOut[i]);
        }
    }
};

template <typename DT, bool StepFirst, int SampNum, bool WithAntithetic>
class PathPricer<Asian_GP, DT, StepFirst, SampNum, WithAntithetic> {
   public:
    const static unsigned int InN = WithAntithetic ? 2 : 1;
    const static unsigned int OutN = InN;
    const static bool byPassGen = false;

    DT strike;

    DT underlying;

    bool optionType;

    DT discount;

    DT dt;

    PathPricer() {
#pragma HLS inline
    }

    void PE(ap_uint<16> steps, ap_uint<16> paths, hls::stream<DT>& pathStrmIn, hls::stream<DT>& priceStrmOut) {
#pragma HLS inline off

        DT prelogS[SampNum];
        DT sumlogS[SampNum];

        if (StepFirst) {
#ifndef __SYNTHESIS__
            printf("Only Samples First is supported now!\n");
#endif
        } else {
            for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 8 max = 8
                for (int j = 0; j < paths; ++j) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
#pragma HLS pipeline II = 1
                    DT dlogS = pathStrmIn.read();

                    DT tmplogS, tmpprelogS, tmpsumlogS, tmpsumS;
                    if (i == 0) {
                        tmpprelogS = 0;
                        tmpsumlogS = 0;
                    } else {
                        tmpprelogS = prelogS[j];
                        tmpsumlogS = sumlogS[j];
                    }

                    tmplogS = tmpprelogS + dlogS;
                    prelogS[j] = tmplogS;
                    sumlogS[j] = tmpsumlogS + tmplogS;

                    if (i == steps - 1) {
                        DT sGP = FPExp(sumlogS[j] / (steps + 1)) * underlying;

                        // printf("GP value = %lf \n", sGP);

                        DT op1, op2, op3, op4;
                        if (optionType) {
                            op3 = strike;
                            op4 = sGP;
                        } else {
                            op3 = sGP;
                            op4 = strike;
                        }
                        DT s2 = FPTwoSub(op3, op4);
                        DT payoff2 = MAX(s2, 0);
                        DT price2 = FPTwoMul(payoff2, discount);

                        priceStrmOut.write(price2);
                    }
                }
            }
        }
    }
    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> priceStrmOut[OutN]) {
        for (int i = 0; i < InN; ++i) {
#pragma HLS unroll
            PE(steps, paths, pathStrmIn[i], priceStrmOut[i]);
        }
    }
};

template <typename DT, bool StepFirst, int SampNum>
class PathPricer<Cliquet, DT, StepFirst, SampNum, false> {
   public:
    const static unsigned int InN = 1;
    const static unsigned int OutN = 1;
    const static unsigned int MaxStep = 1024;
    const static bool byPassGen = true;

    DT strike;

    bool optionType;

    DT volSq;

    DT riskFreeRateNeg;

    DT driftRate;

    DT resetDates[MaxStep];

    PathPricer() {
#pragma HLS inline
    }

    void PE(ap_uint<16> steps, ap_uint<16> paths, hls::stream<DT> pathStrmIn[InN], hls::stream<DT>& priceStrmOut) {
#pragma HLS inline off
#ifndef __SYNTHESIS__
        static int cnt = 0;
#endif
        DT averagePayoff[SampNum];
        DT tPre = 0;
        if (!StepFirst) {
            for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 8 max = 8
                for (int j = 0; j < paths; ++j) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = SampNum max = SampNum
                    // random number
                    DT dw = pathStrmIn[0].read();

                    DT t0 = 0.0;
                    if (i > 0) t0 = tPre;
                    // calculate dt.
                    // DT tCur   = optionPara[0].t[i];
                    DT tCur = resetDates[i];
                    DT dt = 0;
                    // dt is not uniform

                    dt = FPTwoSub(tCur, t0);
                    if (j == paths - 1) tPre = tCur;
                    // calculate vol * vol * dt
                    DT var = FPTwoMul(volSq, dt);

                    DT sqrtVar = hls::sqrt(var);

                    // calculate drift = (r - q - 0.5 * vol * vol)* dt;
                    DT drift = FPTwoMul(dt, driftRate); // drift2 - varH;

                    // calculate dw*sqrt(dt)*vol
                    DT dwVar = FPTwoMul(dw, sqrtVar);
                    DT driftDw = FPTwoAdd(drift, dwVar);

                    // calcualte discount = exp(-r*t)
                    DT discount1 = FPTwoMul(riskFreeRateNeg, tCur);
                    DT discount = FPExp(discount1);
// discount  = 1.0/discount2;

#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    cnt++;
                    if (cnt < 10) {
                        std::cout << "dw=" << dw << std::endl;
                        std::cout << "dt=" << dt << std::endl;
                        std::cout << "t=" << tCur << std::endl;
                        std::cout << "discount=" << discount << std::endl;
                        std::cout << "driftDw=" << driftDw << std::endl;
                    }
#endif
#endif
                    // path[i]/path[i-1] = exp((r-q-0.5*vol*vol)*dt + dw*sqrt(dt)*vol)
                    DT s = FPExp(driftDw);
                    DT op1 = 0;
                    DT op2 = 0;
                    if (optionType) {
                        op1 = strike;
                        op2 = s;
                    } else {
                        op1 = s;
                        op2 = strike;
                    }
                    DT s1 = FPTwoSub(op1, op2);

                    DT payoff = MAX(s1, 0); // discount * MAX(s1, 0);
                    DT npv = FPTwoMul(discount, payoff);

#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    std::cout << "payoff=" << payoff << std::endl;
                    std::cout << "npv=" << npv << std::endl;
#endif
#endif

                    DT sumPayoff = 0;
                    DT newV = 0;
                    if (i > 0) {
                        sumPayoff = averagePayoff[j];
                        newV = npv;
                    } else {
                        sumPayoff = 0;
                        newV = 0;
                    }
                    sumPayoff = FPTwoAdd(sumPayoff, newV);
                    averagePayoff[j] = sumPayoff;
                    if (i == steps - 1) {
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                        std::cout << "averagePayoff=" << sumPayoff << std::endl;
#endif
#endif
                        priceStrmOut.write(sumPayoff);
                    }
                }
            }
        } else {
#ifndef __SYNTHESIS__
            printf("Step First mode is not supported!\n");
#endif
        }
    } // end PE()
    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> priceStrmOut[OutN]) {
        PE(steps, paths, pathStrmIn, priceStrmOut[0]);
    }
};

template <typename DT, bool StepFirst, int SampNum, bool WithAntithetic>
class PathPricer<BarrierNoBiased, DT, StepFirst, SampNum, WithAntithetic> {
   public:
    const static unsigned int InN = WithAntithetic ? 4 : 2;
    const static unsigned int OutN = WithAntithetic ? 2 : 1;
    const static bool byPassGen = true;

    DT barrier;
    DT strike;
    DT rebate;
    DT underlying;
    bool optionType;
    DT drift;
    DT sqrtVar;
    DT varDoub;
    DT disDt;
    ap_uint<2> barrierType;

    PathPricer() {
#pragma HLS inline
    }
    void PE(ap_uint<16> steps,
            ap_uint<16> paths,
            hls::stream<DT>& gaussRndStrmIn,
            hls::stream<DT>& unifRngStrmIn,
            hls::stream<DT>& priceStrmOut) {
#pragma HLS inline off
        DT assetPrice[SampNum];
        bool isActBuff[SampNum];
        ap_uint<16> curTBuff[SampNum];

        if (!StepFirst) {
            for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 8 max = 8
                for (int j = 0; j < paths; ++j) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
#pragma HLS pipeline II = 1
                    DT gaussD = gaussRndStrmIn.read();
                    DT uniformD = unifRngStrmIn.read();
                    // x = (riskFreeRate - dividendYield - 0.5*vol*vol)*dt + hls::sqrt(dt)
                    // * gaussD * vol;
                    DT varDw = FPTwoMul(sqrtVar, gaussD);
                    DT x = FPTwoAdd(drift, varDw);

#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    std::cout << "gaussD=" << gaussD << std::endl;
                    std::cout << "uniformD=" << uniformD << std::endl;
                    std::cout << "x=" << x << std::endl;
#endif
#endif
                    DT xSq = FPTwoMul(x, x);

                    DT logIn = 0.0;
                    DT m1LogIn = FPTwoSub((DT)1.0, uniformD);
                    if (barrierType == DownIn || barrierType == DownOut)
                        logIn = uniformD;
                    else
                        logIn = m1LogIn;
                    DT unifLogD = hls::log(logIn);
                    // cal = 2 * vol * vol * dt * unifLogD;
                    DT cal = FPTwoMul(varDoub, unifLogD);

                    DT xSqSub = FPTwoSub(xSq, cal);
                    DT sqrtXCal = hls::sqrt(xSqSub);

                    DT subOp2;
                    if (barrierType == DownIn || barrierType == DownOut)
                        subOp2 = -sqrtXCal;
                    else
                        subOp2 = sqrtXCal;
                    DT xSubSqrt = FPTwoAdd(x, subOp2);

                    // y = x + sqrt(x^2 - 2*vol*vol*dt*log(u))
                    DT y = FPTwoMul((DT)0.5, xSubSqrt);
                    DT oldAsstP = 0.0;

                    if (i == 0) {
                        oldAsstP = 0.0; // underlying;
                    } else {
                        oldAsstP = assetPrice[j];
                    }
                    DT newExpOp = FPTwoAdd(oldAsstP, y);
                    // nAsset = oldAsstP * hls::exp(y);
                    DT nAsset = underlying * FPExp(newExpOp);
                    // assetPrice = oldAsstP * hls::exp(x);
                    DT newAsstP = FPTwoAdd(oldAsstP, x);
                    assetPrice[j] = newAsstP;
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    std::cout << "nAsset=" << nAsset << std::endl;
                    std::cout << "assetPrice=" << newAsstP << std::endl;
#endif
#endif

                    bool isGrtEqual, isLessEqual;
                    if (nAsset == barrier) {
                        isLessEqual = true;
                        isGrtEqual = true;
                    } else if (nAsset < barrier) {
                        isLessEqual = true;
                        isGrtEqual = false;
                    } else {
                        isLessEqual = false;
                        isGrtEqual = true;
                    }
                    bool oldAct, newAct;
                    ap_uint<16> oldPos, newPos;
                    if (i == 0) {
                        oldPos = 0;
                        oldAct = false;
                    } else {
                        oldPos = curTBuff[j];
                        oldAct = isActBuff[j];
                    }
                    bool isEx = false;
                    if ((barrierType == DownIn && isLessEqual) || (barrierType == UpIn && isGrtEqual) ||
                        (barrierType == DownOut && isLessEqual) || (barrierType == UpOut && isGrtEqual)) {
                        isEx = true;
                    } else {
                        isEx = false;
                    }
                    bool isCurAct;
                    if (i == 0 && isEx) {
                        isCurAct = true;
                    } else if (isEx && !oldAct) {
                        isCurAct = true;
                    } else {
                        isCurAct = false;
                    }
                    if (isCurAct) {
                        newPos = i;
                        newAct = true;
                    } else {
                        newPos = oldPos;
                        newAct = oldAct;
                    }

                    curTBuff[j] = newPos;
                    isActBuff[j] = newAct;
                    if (i == steps - 1) {
                        DT p = 0.0;
                        DT mulOp0, mulOp1;
                        DT pos;
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                        std::cout << "isActive=" << newAct << std::endl;
#endif
#endif
                        if ((newAct && (barrierType == DownIn || barrierType == UpIn)) ||
                            (!newAct && (barrierType == DownOut || barrierType == UpOut))) {
                            DT s = underlying * FPExp(newAsstP);
                            DT op1 = 0;
                            DT op2 = 0;
                            if (optionType) {
                                op1 = strike;
                                op2 = s;
                            } else {
                                op1 = s;
                                op2 = strike;
                            }
                            DT s1 = FPTwoSub(op1, op2);
                            DT payoff = MAX(s1, 0); // discount * MAX(s1, 0);
                            mulOp0 = payoff;
                            pos = steps;
                        } else {
                            if (barrierType == UpIn || barrierType == DownIn) {
                                mulOp0 = rebate;
                                pos = steps;
                            } else {
                                pos = newPos + 1;
                                mulOp0 = rebate;
                            }
                        }
                        // expOp0 = -1*dt*riskFreeRate;//, expOp1;
                        DT expOp0 = disDt; //, expOp1;
                        DT expOp1 = expOp0 * pos;
                        mulOp1 = FPExp(expOp1);
                        p = FPTwoMul(mulOp0, mulOp1);
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                        std::cout << "p=" << p << std::endl;
#endif
#endif
                        priceStrmOut.write(p);
                    }
                } // for(samples)
            }     // for(steps)
        } else {
            DT assetPrice = 0.0; // underlying;
            bool isActive;
            ap_uint<16> curT;
#ifndef __SYNTHESIS__
            printf("Step First mode is not suppored!\n");
#endif

            for (int i = 0; i < SampNum; ++i) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
                for (int j = 0; j < steps; ++j) {
#pragma HLS loop_tripcount min = 8 max = 8
#pragma HLS pipeline II = 1
                    DT gaussD = gaussRndStrmIn.read();
                    DT uniformD = unifRngStrmIn.read();
                    // x = (riskFreeRate - dividendYield - 0.5*vol*vol)*dt + hls::sqrt(dt)
                    // * gaussD * vol;
                    DT varDw;
                    varDw = sqrtVar * gaussD;
                    DT x = drift + varDw;

#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    std::cout << "gaussD=" << gaussD << std::endl;
                    std::cout << "uniformD=" << uniformD << std::endl;
                    std::cout << "x=" << x << std::endl;
#endif
#endif
                    DT xSq = x * x;
                    DT logIn = 0.0;
                    if (barrierType == DownIn || barrierType == DownOut)
                        logIn = uniformD;
                    else
                        logIn = 1.0 - uniformD;
                    DT unifLogD = hls::log(logIn);
                    // cal = 2 * vol * vol * dt * unifLogD;
                    DT cal = varDoub * unifLogD;
                    DT xSqSub = xSq - cal;
                    DT sqrtXCal = hls::sqrt(xSqSub);
                    DT xSubSqrt;
                    DT subOp2;
                    if (barrierType == DownIn || barrierType == DownOut)
                        subOp2 = -sqrtXCal;
                    else
                        subOp2 = sqrtXCal;
                    xSubSqrt = x + subOp2;
                    DT y = 0.5 * xSubSqrt;
                    DT oldAsstP = 0.0;
                    if (i == 0) {
                        oldAsstP = 0.0; // underlying;
                    } else {
                        oldAsstP = assetPrice;
                    }
                    DT newExpOp = oldAsstP + y;
                    // nAsset = oldAsstP * hls::exp(y);
                    DT nAsset = FPExp(newExpOp);
                    // assetPrice = oldAsstP * hls::exp(x);
                    assetPrice = oldAsstP + x;
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    std::cout << "nAsset=" << nAsset << std::endl;
                    std::cout << "assetPrice=" << assetPrice << std::endl;
#endif
#endif
                    bool isGrtEqual, isLessEqual;
                    if (nAsset == barrier) {
                        isLessEqual = true;
                        isGrtEqual = true;
                    } else if (nAsset < barrier) {
                        isLessEqual = true;
                        isGrtEqual = false;
                    } else {
                        isLessEqual = false;
                        isGrtEqual = true;
                    }
                    bool isEx = false;
                    if ((barrierType == DownIn && isLessEqual) || (barrierType == UpIn && isGrtEqual) ||
                        (barrierType == DownOut && isLessEqual) || (barrierType == UpOut && isGrtEqual)) {
                        isEx = true;
                    }
                    if (i == 0) {
                        if (isEx)
                            isActive = true;
                        else
                            isActive = false;
                    } else {
                        if (!isActive) {
                            curT = i;
                            isActive = true;
                        }
                    }
                    if (i == steps - 1) {
                        DT p = 0.0;
                        DT mulOp0, mulOp1;
                        DT pos;
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                        std::cout << "isActive=" << isActive << std::endl;
#endif
#endif
                        if ((isActive && (barrierType == DownIn || barrierType == UpIn)) ||
                            (!isActive && (barrierType == DownOut || barrierType == UpOut))) {
                            DT s = FPExp(assetPrice);
                            DT op1 = 0;
                            DT op2 = 0;
                            if (optionType) {
                                op1 = strike;
                                op2 = s;
                            } else {
                                op1 = s;
                                op2 = strike;
                            }
                            DT s1 = 0;
#pragma HLS resource variable = s1 core = FAddSub_nodsp
                            s1 = op1 - op2;
                            DT payoff = MAX(s1, 0); // discount * MAX(s1, 0);
                            mulOp0 = payoff;
                            pos = steps;
                        } else {
                            if (barrierType == UpIn || barrierType == DownIn) {
                                mulOp0 = rebate;
                                pos = steps;
                            } else {
                                pos = curT + 1;
                                mulOp0 = rebate;
                            }
                        }
                        // expOp0 = -1*dt*riskFreeRate;//, expOp1;
                        DT expOp0 = disDt; //, expOp1;
                        DT expOp1 = expOp0 * pos;
                        mulOp1 = FPExp(expOp1);
                        p = mulOp0 * mulOp1;
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                        std::cout << "p=" << p << std::endl;
#endif
#endif
                        priceStrmOut.write(p);
                    }
                }
            } // for(int i = 0; i < SampNum; ++i)
        }     // if(!StepFirst)
    }         // PE()
    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> priceStrmOut[OutN]) {
        for (int i = 0; i < OutN; ++i) {
#pragma HLS unroll
            PE(steps, paths, pathStrmIn[i * OutN + 0], pathStrmIn[i * OutN + 1], priceStrmOut[i]);
        }
    }
};
template <typename DT, bool StepFirst, int SampNum, bool WithAntithetic>
class PathPricer<BarrierBiased, DT, StepFirst, SampNum, WithAntithetic> {
   public:
    const static unsigned int InN = WithAntithetic ? 2 : 1;
    const static unsigned int OutN = InN;
    const static bool byPassGen = false;

    DT underlying;
    DT barrier;
    DT strike;
    DT rebate;

    bool optionType;
    DT disDt;

    ap_uint<2> barrierType;

    PathPricer() {
#pragma HLS inline
    }

    void PE(ap_uint<16> steps, ap_uint<16> paths, hls::stream<DT>& pathStrmIn, hls::stream<DT>& priceStrmOut) {
        bool actBuff[SampNum];
        ap_uint<16> actPosBuff[SampNum];
        DT logSBuff[SampNum];
#pragma HLS inline off
        if (!StepFirst) {
            for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 8 max = 8
                for (int j = 0; j < SampNum; ++j) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
#pragma HLS pipeline II = 1
                    DT dLogS = pathStrmIn.read();
                    DT preLogS;
                    if (i == 0)
                        preLogS = 0.0;
                    else
                        preLogS = logSBuff[j];
                    DT curLogS = FPTwoAdd(preLogS, dLogS);
                    DT p = FPExp(curLogS);
                    DT assetPrice = FPTwoMul(underlying, p);
                    logSBuff[j] = curLogS;
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    std::cout << "assetPrice=" << assetPrice * 100 << std::endl;
#endif
#endif
                    bool isGrtEqual, isLessEqual;
                    if (assetPrice == barrier) {
                        isLessEqual = true;
                        isGrtEqual = true;
                    } else if (assetPrice < barrier) {
                        isLessEqual = true;
                        isGrtEqual = false;
                    } else {
                        isLessEqual = false;
                        isGrtEqual = true;
                    }
                    bool isEx = false;
                    if ((barrierType == DownIn && isLessEqual) || (barrierType == UpIn && isGrtEqual) ||
                        (barrierType == DownOut && isLessEqual) || (barrierType == UpOut && isGrtEqual)) {
                        isEx = true;
                    }
                    bool oldAct, curAct;
                    ap_uint<16> oldPos, newPos;
                    if (i == 0) {
                        oldAct = false;
                        oldPos = 0;
                    } else {
                        oldAct = actBuff[j];
                        oldPos = actPosBuff[j];
                    }
                    if ((i == 0 && isEx) || (isEx && !oldAct)) {
                        curAct = true;
                        newPos = i;
                    } else {
                        curAct = oldAct;
                        newPos = oldPos;
                    }
                    actBuff[j] = curAct;
                    actPosBuff[j] = newPos;
                    if (i == steps - 1) {
                        DT p = 0.0;
                        DT mulOp0, mulOp1;
                        DT pos;
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                        std::cout << "isActive=" << curAct << std::endl;
#endif
#endif
                        if ((curAct && (barrierType == DownIn || barrierType == UpIn)) ||
                            (!curAct && (barrierType == DownOut || barrierType == UpOut))) {
                            DT s = assetPrice;
                            DT op1 = 0;
                            DT op2 = 0;
                            if (optionType) {
                                op1 = strike;
                                op2 = s;
                            } else {
                                op1 = s;
                                op2 = strike;
                            }
                            DT s1 = FPTwoSub(op1, op2);
                            DT payoff = MAX(s1, 0); // discount * MAX(s1, 0);
                            mulOp0 = payoff;
                            pos = steps;
                        } else {
                            if (barrierType == UpIn || barrierType == DownIn) {
                                mulOp0 = rebate;
                                pos = steps;
                            } else {
                                pos = newPos; // curT + 1;
                                mulOp0 = rebate;
                            }
                        }
                        // expOp0 = -1*dt*riskFreeRate;//, expOp1;
                        DT expOp0 = disDt; //, expOp1;
                        DT expOp1 = expOp0 * pos;
                        mulOp1 = FPExp(expOp1);
                        p = FPTwoMul(mulOp0, mulOp1);
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                        std::cout << "j=" << i << ", payoff=" << p << std::endl;
#endif
#endif
                        priceStrmOut.write(p);
                    }
                }
            }
        } else {
#ifndef __SYNTHESIS__
            printf("Step First mode is not supported!\n");
#endif
        }
    }
    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> priceStrmOut[OutN]) {
        for (int i = 0; i < InN; ++i) {
#pragma HLS unroll
            PE(steps, paths, pathStrmIn[i], priceStrmOut[i]);
        }
    }
};
template <typename DT, bool StepFirst, int SampNum, bool WithAntithetic>
class PathPricer<Digital, DT, StepFirst, SampNum, WithAntithetic> {
   public:
    const static unsigned int InN = WithAntithetic ? 4 : 2;
    const static unsigned int OutN = WithAntithetic ? 2 : 1;
    const static bool byPassGen = true;

    bool exEarly;
    bool optionType;

    DT cashPayoff;
    DT log_strike;
    DT drift;
    DT varSqrt;
    DT log_spot;

    DT disDt;
    DT varDoub;

    PathPricer() {
#pragma HLS inline
    }

    void PE(ap_uint<16> steps,
            ap_uint<16> paths,
            hls::stream<DT>& gaussRndStrmIn,
            hls::stream<DT>& unifRndStrmIn,
            hls::stream<DT>& priceStrmOut) {
#pragma HLS inline off
        bool actBuff[SampNum];
        DT log_asset_price[SampNum];
        ap_uint<16> actPosBuff[SampNum];
        if (!StepFirst) {
            for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 8 max = 8
                for (int j = 0; j < paths; ++j) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
#pragma HLS pipeline II = 1
                    DT gaussD = gaussRndStrmIn.read();
                    DT uniformD = unifRndStrmIn.read();
                    DT varMulD = FPTwoMul(varSqrt, gaussD);
                    // x = log(S_i+1)/log(S_i)
                    DT x = FPTwoAdd(drift, varMulD);
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    std::cout << "gaussD=" << gaussD << std::endl;
                    std::cout << "uniformD=" << uniformD << std::endl;
                    std::cout << "x=" << x << std::endl;
#endif
#endif

                    DT m1UnifD = FPTwoSub((DT)1.0, uniformD);
                    DT logIn = 0.0;
                    if (optionType)
                        logIn = uniformD; // put
                    else
                        logIn = m1UnifD; // call
                    DT unifLogD = hls::log(logIn);
                    // cal = 2 * volatility * volatility * dt * unifLogD;
                    DT cal = FPTwoMul(varDoub, unifLogD);
                    DT xSq = FPTwoMul(x, x);
                    DT xSqSub = FPTwoSub(xSq, cal);
                    DT sqrtXCal = hls::sqrt(xSqSub);
                    DT AddOp1;
                    if (optionType)
                        AddOp1 = -sqrtXCal; // put;
                    else
                        AddOp1 = sqrtXCal; // call
                    DT xPlusCal = FPTwoAdd(x, AddOp1);
                    DT nLogP = 0.5 * xPlusCal;
                    DT old_log_price;
                    if (i == 0)
                        old_log_price = log_spot;
                    else
                        old_log_price = log_asset_price[j];
                    // y = log_price + 0.5*(x+sqrt(x^2-2*vol*vol*dt*log(u));
                    DT y = FPTwoAdd(old_log_price, nLogP);
                    DT nLogAsset = FPTwoAdd(old_log_price, x);
                    log_asset_price[j] = nLogAsset; // old_log_price + x;
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    std::cout << "y=" << y << std::endl;
                    std::cout << "log_asset_price=" << log_asset_price << std::endl;
#endif
#endif
                    bool isEx = false;
                    if ((y <= log_strike && optionType) || (y >= log_strike && !optionType)) {
                        isEx = true;
                    }
                    bool oldAct, newAct;
                    ap_uint<16> oldPos, newPos;
                    if (i == 0) {
                        oldPos = 0;
                        oldAct = false;
                    } else {
                        oldPos = actPosBuff[j];
                        oldAct = actBuff[j];
                    }
                    bool isCurAct;
                    if ((i == 0 && isEx) || (isEx && !oldAct)) {
                        isCurAct = true;
                        newPos = i;
                    } else {
                        isCurAct = oldAct;
                        newPos = oldPos;
                    }
                    actPosBuff[j] = newPos;
                    actBuff[j] = isCurAct;
                    assert(newPos < steps);
                    if (i == steps - 1) {
                        ap_uint<16> pos;
                        DT payoff;
                        if (isCurAct) {
                            if (exEarly)
                                pos = newPos + 1;
                            else
                                pos = steps;
                            DT disOp = disDt * pos;
                            DT discount = FPExp(disOp);
                            payoff = FPTwoMul(cashPayoff, discount);
                        } else {
                            payoff = 0.0;
                        }
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                        std::cout << "pos=" << pos << std::endl;
                        std::cout << "i=" << i << ", payoff=" << payoff << std::endl;
#endif
#endif
                        priceStrmOut.write(payoff);
                    }
                }
            }
        } else {
#ifndef __SYNTHESIS__
            printf("Step First Mode is not supported!\n");
#endif
            bool isActive;
            ap_uint<16> actPos;
            DT log_asset_price;
#pragma HLS resource variable = log_asset_price core = FAddSub_nodsp

            for (int i = 0; i < SampNum; ++i) {
#pragma HLS loop_tripcount min = SampNum max = SampNum
                for (int j = 0; j < steps; ++j) {
#pragma HLS loop_tripcount min = 8 max = 8
#pragma HLS pipeline II = 1
                    DT gaussD = gaussRndStrmIn.read();
                    DT uniformD = unifRndStrmIn.read();
                    DT varMulD = varSqrt * gaussD;
                    DT x;
#pragma HLS resource variable = x core = FAddSub_nodsp
                    x = drift + varMulD;
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    std::cout << "gaussD=" << gaussD << std::endl;
                    std::cout << "uniformD=" << uniformD << std::endl;
                    std::cout << "x=" << x << std::endl;
#endif
#endif
                    DT m1UnifD;
#pragma HLS resource variable = m1UnifD core = FAddSub_nodsp
                    m1UnifD = 1.0 - uniformD;
                    DT logIn = 0.0;
                    if (optionType)
                        logIn = uniformD; // put
                    else
                        logIn = m1UnifD; // call
                    DT unifLogD = hls::log(logIn);
                    // cal = 2 * volatility * volatility * dt * unifLogD;
                    DT cal = varDoub * unifLogD;
                    DT xSq = x * x;
                    DT xSqSub;
#pragma HLS resource variable = xSqSub core = FAddSub_nodsp
                    xSqSub = xSq - cal;
                    DT sqrtXCal = hls::sqrt(xSqSub);
                    DT AddOp1;
                    if (optionType)
                        AddOp1 = -sqrtXCal; // put;
                    else
                        AddOp1 = sqrtXCal; // call
                    DT xPlusCal;
#pragma HLS resource variable = xPlusCal core = FAddSub_nodsp
                    xPlusCal = x + AddOp1;
                    DT nLogP = 0.5 * xPlusCal;
                    DT old_log_price;
                    if (j == 0)
                        old_log_price = log_spot;
                    else
                        old_log_price = log_asset_price;
                    DT y;
#pragma HLS resource variable = y core = FAddSub_nodsp
                    y = old_log_price + nLogP;
                    log_asset_price = old_log_price + x;
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    std::cout << "y=" << y << std::endl;
                    std::cout << "log_asset_price=" << log_asset_price << std::endl;
#endif
#endif
                    bool isEx = false;
                    if ((y <= log_strike && optionType) || (y >= log_strike && !optionType)) {
                        isEx = true;
                    }
                    if (j == 0) {
                        isActive = isEx;
                        actPos = 0;
                    } else if (isEx) {
                        if (!isActive) {
                            isActive = true;
                            actPos = j;
                        }
                    }
                    assert(actPos < steps);
                    if (j == steps - 1) {
                        ap_uint<16> pos;
                        DT payoff;
                        if (isActive) {
                            if (exEarly)
                                pos = actPos + 1;
                            else
                                pos = steps;
                            DT disOp = disDt * pos;
                            DT discount = FPExp(disOp);
                            payoff = cashPayoff * discount;
                            // payoff = discount;
                        } else {
                            payoff = 0.0;
                        }
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                        std::cout << "pos=" << pos << std::endl;
                        std::cout << "i=" << i << ", payoff=" << payoff << std::endl;
#endif
#endif
                        priceStrmOut.write(payoff);
                    }
                }
            }
        } // if(!StepFirst)
    }     // PE();
    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> priceStrmOut[OutN]) {
        for (int i = 0; i < OutN; ++i) {
#pragma HLS unroll
            PE(steps, paths, pathStrmIn[i * OutN + 0], pathStrmIn[i * OutN + 1], priceStrmOut[i]);
        }
    }
};
///@brief The pathpricer that used in MCAmericanEngine Option Calibration
/// process.
template <typename DT, bool StepFirst, int SampNum, bool WithAntithetic>
class PathPricer<American, DT, StepFirst, SampNum, WithAntithetic> {
   public:
    const static unsigned int InN = WithAntithetic ? 2 : 1;
    const static unsigned int OutN = InN;
    const static bool byPassGen = false;

    DT drift;
    PathPricer() {
#pragma HLS inline
    }
    void PE(ap_uint<16> steps, hls::stream<DT>& pathStrmIn, hls::stream<DT>& priceStrmOut) {
#pragma HLS inline off
        DT preStepTable[SampNum];
#ifndef __SYNTHESIS__
        static int cnt = 0;
#endif
        if (!StepFirst) {
            for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 10 max = 10
                for (int j = 0; j < SampNum; ++j) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = SampNum max = SampNum
                    DT pIn = pathStrmIn.read();

                    DT pre = 0;
                    if (i == 0)
                        pre = 0;
                    else
                        pre = preStepTable[j];

                    // DT s1 = FPTwoAdd(pre, drift);
                    DT s2 = FPTwoAdd(pre, pIn);
                    preStepTable[j] = s2;
                    DT s = FPExp(s2);
                    priceStrmOut.write(s);
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    if (cnt < 10) {
                        std::cout << "pIn=" << pIn << std::endl;
                        std::cout << "s=" << s << std::endl;
                    }
                    cnt++;
#endif
#endif
                }
            }
        } else {
            for (int i = 0; i < SampNum; ++i) {
#pragma HLS loop_tripcount min = 8 max = 8
                DT pre = 0;
                for (int j = 0; j < steps; ++j) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1000 max = 1000
                    DT dw = pathStrmIn.read();
                    DT p = pre + drift;
                    pre = p;
                    DT s = FPExp(p);
                    priceStrmOut.write(s);
                }
            }
        }
    }

    void Pricing(ap_uint<16> steps, hls::stream<DT> pathStrmIn[InN], hls::stream<DT> priceStrmOut[OutN]) {
        for (int i = 0; i < InN; ++i) {
#pragma HLS unroll
            PE(steps, pathStrmIn[i], priceStrmOut[i]);
        }
    }
};
///@brief The pathpricer that used in MCAmericanEngine Option Pricing process.
template <typename DT, bool StepFirst, int SampNum, bool WithAntithetic, int MaxSteps>
class PathPricer<LongstaffSchwartz, DT, StepFirst, SampNum, WithAntithetic, MaxSteps> {
   public:
    const static unsigned int InN = WithAntithetic ? 2 : 1;
    const static unsigned int OutN = InN;
    const static bool byPassGen = false;
    DT strike;

    bool optionType;
    DT riskConst;
    DT riskRate;
    DT recipUnderLying;
    DT recipStrike;
    DT coefBuff_0[MaxSteps];
    DT coefBuff_1[MaxSteps];
    DT coefBuff_2[MaxSteps];
    DT coefBuff_3[MaxSteps];

    PathPricer() {
#pragma HLS inline
    }

    void PE(ap_uint<16> steps, hls::stream<DT>& pathStrmIn, hls::stream<DT>& priceStrmOut) {
#pragma HLS inline off
        int k = 0;
        static int cnt = 0;
        DT sumRead[SampNum];
        ap_uint<1> stopFlags[SampNum];
        for (int i = 0; i < steps; ++i) {
#pragma HLS loop_tripcount min = 8 max = 8
            for (int j = 0; j < SampNum; ++j) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = SampNum max = SampNum
                DT tmpRead;
                if (i == 0) {
                    tmpRead = 0;
                } else {
                    tmpRead = sumRead[j];
                }
                DT dLogS = pathStrmIn.read();

                DT tmpAdd = FPTwoAdd(tmpRead, dLogS);
                DT path = FPExp(tmpAdd);
                sumRead[j] = tmpAdd;
                DT s = path;
                DT op1 = 0;
                DT op2 = 0;
                if (optionType) {
                    op1 = strike;
                    op2 = s;
                } else {
                    op1 = s;
                    op2 = strike;
                }
                DT s1 = FPTwoSub(op1, op2);
                DT payoff = MAX(s1, 0);
                ap_uint<1> flag;
                if (i == 0) {
                    flag = ap_uint<1>(0);
                } else {
                    flag = stopFlags[j];
                }
                DT tmpPath2 = path;
                DT tmpPrice = payoff;
                if (flag == 1) {
                } else if ((i < steps - 1)) {
                    int constBuffer = steps - i - 2;
                    DT coef0 = coefBuff_0[constBuffer];
                    DT coef1 = coefBuff_1[constBuffer];
                    DT coef2 = coefBuff_2[constBuffer];
                    DT coef3 = coefBuff_3[constBuffer];
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                    if (cnt < 3) {
                        //                         std::cout<<"cnt = "<<cnt<<std::endl;
                        std::cout << "Coef=" << coef0 << ", " << coef1 << ", " << coef2 << std::endl;
                    }
                    cnt++;
#endif
#endif
                    DT exercise = tmpPrice;
                    DT path = tmpPath2;
                    if (exercise > 0.0) {
                        DT regValue = path * recipStrike;
                        DT tmpMul1 = FPTwoMul(coef1, regValue);
                        DT tmpMul4 = FPTwoMul(coef3, exercise);
                        DT tmpMul2 = FPTwoMul(coef2, regValue);
                        DT tmpMul5 = FPTwoMul(tmpMul2, regValue);
                        DT tmpAdder1 = FPTwoAdd(coef0, tmpMul1);
                        DT tmpAdder2 = FPTwoAdd(tmpAdder1, tmpMul5);
                        DT tmpMul3 = FPTwoMul(tmpAdder2, recipUnderLying);
                        DT continuousVal = FPTwoAdd(tmpMul3, tmpMul4);
                        if (continuousVal < exercise) {
                            DT price = exercise;
                            DT multiplier, tmpMul;
                            DT adder = i + 1.0;
                            tmpMul = riskConst * adder;
                            multiplier = FPExp(tmpMul);
                            price *= multiplier;
                            priceStrmOut.write(price);
                            stopFlags[j] = ap_uint<1>(1);
                        } else {
                            stopFlags[j] = ap_uint<1>(0);
                        }
                    } else {
                        stopFlags[j] = ap_uint<1>(0);
                    }
                } else if ((i == steps - 1)) {
                    DT price = FPTwoMul(tmpPrice, riskRate); // multipli
                    priceStrmOut.write(price);
                }
            }
        }
    }
    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> priceStrmOut[OutN]) {
        for (int i = 0; i < InN; ++i) {
#pragma HLS unroll
            PE(steps, pathStrmIn[i], priceStrmOut[i]);
        }
    }
};

template <typename DT, int SampNum>
class CapFloorPathPricer {
   public:
    const static unsigned int InN = 1;
    const static unsigned int OutN = InN;
    const static bool byPassGen = false;

    bool isCap;
    DT r0;
    DT strikeRate;
    DT singlePeriod;
    DT alpha;
    DT sigma;
    DT nomial;
    DT payoffBuffer[SampNum];
    DT lastRateBuffer[SampNum];

    CapFloorPathPricer() {
#pragma HLS inline
    }

    void init(
        bool cap, DT input_strike, DT input_period, DT input_r0, DT input_alpha, DT input_sigma, DT input_nomial) {
        isCap = cap;
        strikeRate = input_strike;
        singlePeriod = input_period;
        r0 = input_r0;
        alpha = input_alpha;
        sigma = input_sigma;
        nomial = input_nomial;
    }

    DT B(DT t, DT s) { return (1.0 - FPExp(FPTwoMul(-alpha, FPTwoSub(s, t)))) / alpha; }

    DT discountBond(DT t, DT s, DT r) {
        DT tmp1 = B(t, s);
        DT tmp2 = B(0, 2 * t);
        DT tmp3 = FPTwoMul(FPTwoMul(sigma, tmp1), 0.5);
        DT tmp4 = FPTwoMul(tmp3, tmp3);
        DT tmp5 = FPTwoMul(tmp4, tmp2);
        DT tmp6 = FPTwoSub(s, t);
        DT tmp7 = FPTwoSub(r, r0);
        DT tmp8 = FPTwoMul(r0, tmp6);
        DT tmp9 = FPTwoMul(tmp1, tmp7);
        DT tmp10 = FPTwoAdd(tmp8, tmp9);
        DT tmp11 = FPTwoAdd(tmp10, tmp5);
        return FPExp(tmp11);
    }

    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> priceStrmOut[OutN]) {
        DT totaltime = FPTwoMul(double(steps), singlePeriod);
        DT begintime = 0;
        DT endtime = singlePeriod;
        for (int i = 0; i < steps; i++) {
            for (int j = 0; j < paths; j++) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = payoffBuffer inter false
#pragma HLS dependence variable = lastRateBuffer inter false
                DT r1, r2;
                if (i == 0) {
                    r1 = r0;
                } else {
                    r1 = lastRateBuffer[j];
                }
                r2 = pathStrmIn[0].read();
                lastRateBuffer[j] = r2;

                DT d1 = 1.0;
                DT d2 = discountBond(begintime, endtime, r1);

                DT currentLibor = FPTwoSub(d2, 1.0) / singlePeriod;

                DT accrualFactor = discountBond(endtime, totaltime, r2);

                DT payoff = FPTwoSub(currentLibor, strikeRate);

                if (isCap) {
                    if (payoff < 0) {
                        payoff = 0;
                    }
                } else {
                    if (payoff < 0) {
                        payoff = -payoff;
                    } else {
                        payoff = 0;
                    }
                }

                DT NPV;
                if (i <= 1) { // discard first round
                    NPV = 0;
                } else {
                    NPV = payoffBuffer[j];
                }

                NPV += FPTwoMul(payoff, FPTwoMul(singlePeriod, FPTwoMul(nomial, accrualFactor)));
                payoffBuffer[j] = NPV;

                if (i == steps - 1) {
                    NPV = FPTwoMul(NPV, FPExp(-FPTwoMul(r0, totaltime)));
                    priceStrmOut[0].write(NPV);
                }
            }
        }
    }
};

} // namespace internal

/**
 * @brief Heath-Jarrow-Morton path pricer for Zero Coupon Bonds. Takes as an input an Instantaneous Forwards Rates
 * matrix
 * and calculates the price of a Zero Coupon Bond maturing at time 'T' years. For correct functionality, 'T' must be <=
 * than
 * the simulated time in the path generation.
 *
 * @tparam DT - The internal DataType of the pricer.
 * @tparam MAX_TENORS - Maximum number of tenors supported
 */
template <typename DT, unsigned int MAX_TENORS>
class hjmZeroCouponBondPricer {
    unsigned int m_noTenors;
    DT m_T;

   public:
    const static unsigned int InN = 1;
    const static unsigned int OutN = InN;
    const static bool byPassGen = false;

    hjmZeroCouponBondPricer() {
#pragma HLS inline
    }

    /**
     * @brief Sets the parameters for this ZCB pricer.
     *
     * @param noTenors - Number of tenors in the current model
     * @param T - ZCB maturing time, in years. Must be <= than the sim_years parameter on the HJM MC Engine
     */
    void init(unsigned int noTenors, const DT T) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
        assert(noTenors <= MAX_TENORS && "Provided tenors are larger than the synthetisable MAX_TENORS parameter.");
#endif
        m_noTenors = noTenors;
        m_T = T;
    }

    DT zcbPrice(ap_uint<16> steps, hls::stream<DT>& pathIn) {
        DT zcbAccum = 0.0;
        for (ap_uint<16> i = 0; i < steps; i++) {
#pragma HLS PIPELINE
            const DT zcbItem = pathIn.read();
            if (i * hjmModelParams::dt <= m_T) {
                // If it is in bounds add it to the integrator
                zcbAccum += zcbItem;
            }
            // Flush the rest of the row
            for (unsigned int j = 1; j < m_noTenors; j++) {
                pathIn.read();
            }
        }
        return hls::exp(-zcbAccum * hjmModelParams::dt);
    }

    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> priceStrmOut[OutN]) {
        for (ap_uint<16> i = 0; i < paths; i++) {
#pragma HLS PIPELINE
            priceStrmOut[0] << zcbPrice(steps, pathStrmIn[0]);
        }
    }
};

/**
 * @brief Cap pricer for a LIBOR Market Model engine.
 */
template <typename DT, unsigned MAX_TENORS>
class lmmCapPricer {
    DT m_notional;
    DT m_caprate;

   public:
    const static unsigned int InN = 1;
    const static unsigned int OutN = 1;
    const static bool byPassGen = false;

    lmmCapPricer() {
#pragma HLS inline
    }

    void init(DT notional, DT caprate) {
        m_notional = notional;
        m_caprate = caprate;
    }

    DT capletPrice(ap_uint<16> steps, hls::stream<DT>& liborRateStrm) {
        const unsigned noTenors = steps + 1;
        DT numerairesInv[MAX_TENORS + 1];
        DT capletPayoffs[MAX_TENORS - 1];

        for (ap_uint<8> i = 0; i < noTenors; i++) {
            for (ap_uint<8> j = i; j < noTenors; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = numerairesInv intra false

                const DT lRate = liborRateStrm.read();

                DT nmInv = ((j == i) ? 1.0f : numerairesInv[i]) * (1 + lmmModelParams::tau * lRate);
                numerairesInv[i] = nmInv;

                /*
                 * For caplet pricings we want to use the LIBOR rate in the diagonals starting at 1
                 */
                if ((j == i) && (i > 0)) {
                    capletPayoffs[i - 1] = hls::max<DT>(lRate - m_caprate, 0.0f) * lmmModelParams::tau * m_notional;
                }
            }
        }

        DT capPrice = 0.0f;
        numerairesInv[noTenors] = 1.0f;
        // Compute the caplet expectation under numeraire measure and add all the caplet prices
        for (ap_uint<8> i = 0; i < steps; i++) {
#pragma HLS PIPELINE II = 1
            // caplet(i) = (N(0) / N(i + 1)) * payoff(i)
            capPrice += numerairesInv[i + 2] * capletPayoffs[i];
        }
        // sum( B(0)/B(i+1)*caplet(i) ) == B(0) * sum( caplet(i)/B(i+1) )
        return capPrice / numerairesInv[0];
    }

    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> pathStrmOut[OutN]) {
        for (ap_uint<16> p = 0; p < paths; p++) {
#pragma HLS PIPELINE
            pathStrmOut[0] << capletPrice(steps, pathStrmIn[0]);
        }
    }
};

/**
 * @brief Ratchet Floater pricer for a LIBOR Market Model engine.
 */
template <typename DT, unsigned MAX_TENORS>
class lmmRatchetFloaterPricer {
    DT m_notional;
    DT m_X;
    DT m_Y;
    DT m_alpha;

   public:
    const static unsigned int InN = 1;
    const static unsigned int OutN = 1;
    const static bool byPassGen = false;

    lmmRatchetFloaterPricer() {
#pragma HLS inline
    }

    void init(DT notional, DT X, DT Y, DT alpha) {
        m_notional = notional;
        m_X = X;
        m_Y = Y;
        m_alpha = alpha;
    }

    DT ratchetFloaterPrice(ap_uint<16> steps, hls::stream<DT>& liborRateStrm) {
        const unsigned noTenors = steps + 1;
        DT numerairesInv[MAX_TENORS + 1];
        DT rfPayoffs[MAX_TENORS];
        DT prevCoupon = 0.0f;

        for (ap_uint<16> i = 0; i < noTenors; i++) {
            for (ap_uint<16> j = i; j < noTenors; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = numerairesInv intra false

                const DT lRate = liborRateStrm.read();

                DT nmInv = ((j == i) ? 1.0f : numerairesInv[i]) * (1 + lmmModelParams::tau * lRate);
                numerairesInv[i] = nmInv;

                /*
                 * For coupon pricing we want to use the LIBOR rates in the diagonals starting at 1
                 */
                if ((j == i) && (i > 0)) {
                    DT coupon;
                    if (i == 1) {
                        coupon = lmmModelParams::tau * (lRate + m_Y);
                    } else {
                        coupon =
                            prevCoupon +
                            hls::min<DT>(m_alpha, hls::max<DT>(lmmModelParams::tau * (lRate + m_Y) - prevCoupon, 0.0f));
                    }
                    prevCoupon = coupon;
                    rfPayoffs[i - 1] = m_notional * (lmmModelParams::tau * (lRate + m_X) - coupon);
                }
            }
        }

        DT rfPrice = 0.0f;
        // Last element of B is discounted by ratio with denominator 1.0
        numerairesInv[noTenors] = 1.0f;

        // Compute the ratcher floater expectation under numeraire measure and add all the payoffs
        for (ap_uint<16> i = 0; i < steps; i++) {
#pragma HLS PIPELINE II = 1
            rfPrice += numerairesInv[i + 2] * rfPayoffs[i];
        }
        // sum( B(0)/B(i+1)*caplet(i) ) == B(0) * sum( caplet(i)/B(i+1) )
        return rfPrice / numerairesInv[0];
    }

    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> pathStrmOut[OutN]) {
        for (ap_uint<16> p = 0; p < paths; p++) {
#pragma HLS PIPELINE
            pathStrmOut[0] << ratchetFloaterPrice(steps, pathStrmIn[0]);
        }
    }
};

/**
 * @brief Ratchet Cap pricer for a LIBOR Market Model engine.
 */
template <typename DT, unsigned MAX_TENORS>
class lmmRatchetCapPricer {
    DT m_notional;
    DT m_spread;
    DT m_kappa0;

   public:
    const static unsigned int InN = 1;
    const static unsigned int OutN = 1;
    const static bool byPassGen = false;

    lmmRatchetCapPricer() {
#pragma HLS inline
    }

    void init(DT notional, DT spread, DT kappa0) {
        m_notional = notional;
        m_spread = spread;
        m_kappa0 = kappa0;
    }

    DT ratchetCapPrice(ap_uint<8> steps, hls::stream<DT>& liborRateStrm) {
        const unsigned noTenors = steps + 1;
        DT numerairesInv[MAX_TENORS + 1];
        DT capletPayoffs[MAX_TENORS - 1];

        DT caprate = m_spread + m_kappa0;

        for (ap_uint<8> i = 0; i < noTenors; i++) {
            for (ap_uint<8> j = i; j < noTenors; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = numerairesInv intra false

                const DT lRate = liborRateStrm.read();

                DT nmInv = ((j == i) ? 1.0f : numerairesInv[i]) * (1 + lmmModelParams::tau * lRate);
                numerairesInv[i] = nmInv;

                /*
                 * For ratchet caplet pricings we want to use the LIBOR rate in the diagonals starting at 1
                 */
                if ((j == i) && (i > 0)) {
                    capletPayoffs[i - 1] = hls::max<DT>(lRate - caprate, 0.0f) * lmmModelParams::tau * m_notional;
                    caprate = lRate + m_spread;
                }
            }
        }

        DT ratchetCapPrice = 0.0f;
        // Last element of B is discounted by ratio with denominator 1.0
        numerairesInv[noTenors] = 1.0f;

        // Compute the ratchet caplet expectation under numeraire measures and add all the ratchet caplet prices
        for (ap_uint<8> i = 0; i < steps; i++) {
#pragma HLS PIPELINE II = 1
            // caplet(i) = (N(0) / N(i + 1)) * payoff(i) == N(0) * N(i + 1)^-1 * payoff(i)
            ratchetCapPrice += numerairesInv[i + 2] * capletPayoffs[i];
        }
        // sum( B(0)/B(i+1)*caplet(i) ) == B(0) * sum( caplet(i)/B(i+1) )
        return ratchetCapPrice / numerairesInv[0];
    }

    void Pricing(ap_uint<16> steps,
                 ap_uint<16> paths,
                 hls::stream<DT> pathStrmIn[InN],
                 hls::stream<DT> pathStrmOut[OutN]) {
        for (ap_uint<16> p = 0; p < paths; p++) {
#pragma HLS PIPELINE
            pathStrmOut[0] << ratchetCapPrice(steps, pathStrmIn[0]);
        }
    }
};

} // namespace fintech
} // namespace xf
#endif //#ifndef XF_FINTECH_PATH_PRICER_H
