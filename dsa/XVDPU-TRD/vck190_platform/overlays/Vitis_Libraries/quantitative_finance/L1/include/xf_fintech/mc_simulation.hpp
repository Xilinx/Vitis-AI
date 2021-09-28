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
 * @file mc_simulation.hpp
 * @brief This file contains the implementation of Monte Carlo Simulation
 */

#ifndef XF_FINTECH_MC_SIMULATION_H
#define XF_FINTECH_MC_SIMULATION_H
#include "ap_int.h"
#include "hls_stream.h"
#include "xf_fintech/path_generator.hpp"
#include "xf_fintech/path_pricer.hpp"
#include "xf_fintech/rng_sequence.hpp"
#ifndef __SYNTHESIS__
#include <assert.h>
#endif

namespace xf {
namespace fintech {
namespace internal {

template <typename DT>
void antithetic(ap_uint<16> paths, hls::stream<DT> priceStrmIn[2], hls::stream<DT> priceStrmOut[1]) {
    for (int i = 0; i < paths; ++i) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 1024 max = 1024
        DT in1 = priceStrmIn[0].read();
        DT in2 = priceStrmIn[1].read();
        DT s = FPTwoAdd(in1, in2);
        DT out = FPTwoMul((DT)0.5, s);
        priceStrmOut[0].write(out);
    }
}

template <typename DT>
void accumulator(ap_uint<16> paths,
                 hls::stream<DT> priceStrmIn[1],
                 hls::stream<DT>& sumStrm,
                 hls::stream<DT>& squareSumStrm) {
#pragma HLS inline off
    const unsigned int DEP = 16;
    DT sumBuffer[DEP]; // because the latency of ACC_LOOP is 14
    DT squareSumBuffer[DEP];
    DT sum = 0;
    DT squareSum = 0;
BUFF_INIT_LOOP:
    for (int i = 0; i < DEP; ++i) {
#pragma HLS pipeline II = 1
        sumBuffer[i] = 0;
        squareSumBuffer[i] = 0;
    }
    // because the latency of ACC_LOOP is 14
    ap_uint<4> cnt = 0;
ACC_LOOP:
    for (int i = 0; i < paths; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1024 max = 1024
        DT temp = priceStrmIn[0].read();
        DT sumTemp = sumBuffer[cnt];
        DT squareTemp = squareSumBuffer[cnt];
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
        std::cout << "sumBuffer[" << cnt << "]=" << sumBuffer[cnt] << std::endl;
        std::cout << "squareSumBuffer[" << cnt << "]=" << squareSumBuffer[cnt] << std::endl;
#endif
#endif
        DT mulTemp = FPTwoMul(temp, temp);
        DT squareNew = FPTwoAdd(squareTemp, mulTemp);
        DT sumNew = FPTwoAdd(sumTemp, temp);
        sumBuffer[cnt] = sumNew;
        squareSumBuffer[cnt] = squareNew;
        cnt++;
    }
POST_ACC_LOOP:
    for (int i = 0; i < DEP; ++i) {
#pragma HLS pipeline II = 8
        sum += sumBuffer[i];
        squareSum += squareSumBuffer[i];
    }
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
    std::cout << "sum=" << sum << std::endl;
    std::cout << "squareSum=" << squareSum << std::endl;
#endif
#endif
    sumStrm.write(sum);
    squareSumStrm.write(squareSum);
}

template <typename DT, typename RNG, typename PathGeneratorT, typename PathPricerT, typename RNGSeqT, int VariateNum>
void monteCarloModel(ap_uint<16> steps,
                     ap_uint<16> paths,
                     RNG rngInst[VariateNum],
                     PathGeneratorT pathGenInst[1],
                     PathPricerT pathPriInst[1],
                     RNGSeqT rngSeqInst[1],
                     hls::stream<DT>& sumStrm,
                     hls::stream<DT>& squareSumStrm) {
#pragma HLS inline off
#pragma HLS DATAFLOW
    const static unsigned int RN = RNGSeqT::OutN;
    const static unsigned int PN = PathPricerT::OutN;
    const static bool byPassGen = PathPricerT::byPassGen;

    hls::stream<DT> rdNmStrm[RN];
#pragma HLS stream variable = rdNmStrm depth = 8
    hls::stream<DT> pathStrm[PN];
#pragma HLS stream variable = pathStrm depth = 8
    hls::stream<DT> priceStrm[PN];
#pragma HLS stream variable = priceStrm depth = 8
    hls::stream<DT> avgPriStrm[1];
#pragma HLS stream variable = avgPriStrm depth = 8
    // Generate random number
    rngSeqInst[0].NextSeq(steps, paths, rngInst, rdNmStrm);
    if (!byPassGen) {
        pathGenInst[0].NextPath(steps, paths, rdNmStrm, pathStrm);
        pathPriInst[0].Pricing(steps, paths, pathStrm, priceStrm);
    } else {
        pathPriInst[0].Pricing(steps, paths, rdNmStrm, priceStrm);
    }
    if (PN == 2) {
        antithetic<DT>(paths, priceStrm, avgPriStrm);
        accumulator<DT>(paths, avgPriStrm, sumStrm, squareSumStrm);
    } else {
        accumulator<DT>(paths, priceStrm, sumStrm, squareSumStrm);
    }
}

template <typename DT,
          typename RNG,
          int UnrollNm,
          typename PathGeneratorT,
          typename PathPricerT,
          typename RNGSeqT,
          int VariateNum>
void MultipleMonteCarloModel(ap_uint<16> steps,
                             ap_uint<16> paths,
                             RNG rngInst[UnrollNm][VariateNum],
                             PathGeneratorT pathGenInst[UnrollNm][1],
                             PathPricerT pathPriInst[UnrollNm][1],
                             RNGSeqT rngSeqInst[UnrollNm][1],
                             DT& sum,
                             DT& squareSum) {
    hls::stream<DT> sumStrm[UnrollNm];
#pragma HLS stream variable = sumStrm depth = 8
#pragma HLS array_partition variable = sumStrm dim = 0
    hls::stream<DT> squareSumStrm[UnrollNm];
#pragma HLS stream variable = squareSumStrm depth = 8
#pragma HLS array_partition variable = squareSumStrm dim = 0

    for (int i = 0; i < UnrollNm; ++i) {
#pragma HLS unroll
        monteCarloModel<DT, RNG, PathGeneratorT, PathPricerT, RNGSeqT, VariateNum>(
            steps, paths, rngInst[i], pathGenInst[i], pathPriInst[i], rngSeqInst[i], sumStrm[i], squareSumStrm[i]);
    }
    for (int i = 0; i < UnrollNm; ++i) {
#pragma HLS pipeline
        DT sumTemp = sumStrm[i].read();
        DT squareTemp = squareSumStrm[i].read();
        sum = FPTwoAdd(sum, sumTemp);
        squareSum = FPTwoAdd(squareSum, squareTemp);
    }
}

template <typename DT>
inline DT SampleMean(DT sum, ap_uint<27> weightSum) {
    return sum / weightSum;
}

template <typename DT>
inline DT SampleErrorEstimate(DT mean, DT sum, DT squareSum, ap_uint<27> samplesNumbers) {
    DT squareMean = FPTwoMul(mean, mean);
    DT temp = squareSum / samplesNumbers;
    DT variance = FPTwoSub(temp, squareMean);
    return hls::sqrt(variance / samplesNumbers);
}

template <typename RNG, typename RNGSeqT, int UnrollNm, int VariateNum>
void InitWrap(RNG rngInst[UnrollNm][VariateNum], RNGSeqT rngSeqInst[UnrollNm][1]) {
    //#pragma HLS dataflow
    for (int i = 0; i < UnrollNm; ++i) {
#pragma HLS unroll
        rngSeqInst[i][0].Init(rngInst[i]);
    }
}
} // namespace internal
/**
 * @brief Monte Carlo Framework implementation
 *
 * @tparam DT supported data type including double and float data type, which
 * decides the precision of result, default double-precision data type.
 * @tparam RNG random number generator type.
 * @tparam PathGeneratorT path generator type which simulates the dynamics of
 * the asset price.
 * @tparam PathPricerT path pricer type which calcualtes the option price based
 * on asset price.
 * @tparam RNGSeqT random number sequence generator type.
 * @tparam UN number of Monte Carlo Module in parallel, which affects the
 * latency and resources utilization.
 * @tparam VariateNum number of variate.
 * @tparam SampNum the total samples are divided into several steps, SampNum is
 * the number for each step.
 * @param timeSteps number of the steps for each path.
 * @param maxSamples the maximum sample number. When reaching it, the simulation
 * will stop.
 * @param requiredTolerance the tolerance required. If requiredSamples is not
 * set, when reaching the required tolerance, simulation will stop.
 * @param requiredSamples the samples number required. When reaching the
 * required number, simulation will stop.
 * @param pathGenInst instance of path generator.
 * @param pathPriInst instance of path pricer.
 * @param rngSeqInst instance of random number sequence.
 */
template <typename DT,
          typename RNG,
          typename PathGeneratorT,
          typename PathPricerT,
          typename RNGSeqT,
          int UN,
          int VariateNum,
          int SampNum>
DT mcSimulation(ap_uint<16> timeSteps,
                ap_uint<27> maxSamples,
                ap_uint<27> requiredSamples,
                DT requiredTolerance,
                PathGeneratorT pathGenInst[UN][1],
                PathPricerT pathPriInst[UN][1],
                RNGSeqT rngSeqInst[UN][1]) {
    // total number of samples per simulation
    const static ap_uint<16> Batch = UN * SampNum;

    // RNG Instance
    RNG rngInst[UN][VariateNum];
#pragma HLS array_partition variable = rngInst dim = 1
#pragma HLS array_partition variable = rngInst dim = 2

    // Initialize RNG
    internal::InitWrap<RNG, RNGSeqT, UN, VariateNum>(rngInst, rngSeqInst);

    // record the total number of samples
    ap_uint<27> totalSamples = 0;

    // sum of all samples
    DT sum = 0;
    // square sum of all samples
    DT squareSum = 0;
    // mean of all samples
    DT mean = 0;

    // simulation times
    ap_uint<17> loopNum = 0;

    if (requiredSamples > 0) {
        loopNum = (requiredSamples + Batch - 1) / Batch;
        totalSamples = loopNum * Batch;
    } else {
        loopNum = 1;
        totalSamples = Batch;
    }

#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
    std::cout << "requiredSamples=" << requiredSamples << std::endl;
    std::cout << "requiredTolerance=" << requiredTolerance << std::endl;
#endif
#endif
Req_Samples_Loop:
    for (int i = 0; i < loopNum; ++i) {
#pragma HLS loop_tripcount min = 1 max = 1
        internal::MultipleMonteCarloModel<DT, RNG, UN, PathGeneratorT, PathPricerT, RNGSeqT, VariateNum>(
            timeSteps, SampNum, rngInst, pathGenInst, pathPriInst, rngSeqInst, sum, squareSum);
    }
    mean = internal::SampleMean(sum, totalSamples);
    DT error = internal::SampleErrorEstimate(mean, sum, squareSum, totalSamples);
    if (requiredSamples == 0) {
    Req_Tolerance_Loop:
        while ((requiredTolerance < error) && ((maxSamples > 0 && totalSamples < maxSamples) || maxSamples == 0)) {
#pragma HLS loop_tripcount min = 5 max = 5
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
            std::cout << "error = " << error << std::endl;
#endif
#endif
            totalSamples += Batch;
            // Monte Carlo Module
            internal::MultipleMonteCarloModel<DT, RNG, UN, PathGeneratorT, PathPricerT, RNGSeqT, VariateNum>(
                timeSteps, SampNum, rngInst, pathGenInst, pathPriInst, rngSeqInst, sum, squareSum);
            mean = internal::SampleMean(sum, totalSamples);
            error = internal::SampleErrorEstimate(mean, sum, squareSum, totalSamples);
        }
    }
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
    std::cout << "sum=" << sum << std::endl;
    std::cout << "totalSamples=" << totalSamples << std::endl;
#endif
#endif
    return mean; // SampleMean(sum, totalSamples);
}
} // namespace fintech
} // namespace xf
#endif
