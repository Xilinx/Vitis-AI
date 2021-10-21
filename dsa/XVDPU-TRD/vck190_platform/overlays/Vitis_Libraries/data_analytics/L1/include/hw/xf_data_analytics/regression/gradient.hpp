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
 * @file gradient.hpp
 * @brief Linear least square gradient calculate and update functions implementation
 *
 * This file is part of Vitis Data Analytics Library
 */

#ifndef _XF_DATA_ANALYTICS_L1_GRADIENT_HPP_
#define _XF_DATA_ANALYTICS_L1_GRADIENT_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include "xf_data_analytics/common/stream_local_processing.hpp"
#include "xf_data_analytics/common/table_sample.hpp"
#include "xf_data_analytics/common/enums.hpp"
#include "xf_data_analytics/common/math_helper.hpp"

namespace xf {
namespace data_analytics {
namespace regression {
namespace internal {

using namespace xf::data_analytics::common::internal;

template <typename MType, int D>
class linearLeastSquareGradient {
   public:
    void process(ap_uint<32> cols,
                 hls::stream<MType> xStrm[D],
                 hls::stream<bool>& eXStrm,
                 hls::stream<MType>& yStrm,
                 hls::stream<bool>& eYStrm,
                 hls::stream<MType>& dotMulStrm,
                 hls::stream<bool>& eDotMulStrm,
                 hls::stream<MType> thetaGradientStrm[D],
                 hls::stream<bool>& eThetaGradientStrm,
                 hls::stream<MType>& interceptGradientStrm,
                 hls::stream<bool>& eInterceptGradientStrm) {
        ap_uint<32> colCounter = 0;
        MType y;
        MType dotMul;
        MType err;
        while (!eXStrm.read()) {
#pragma HLS pipeline II = 1
            if (colCounter == 0) {
                y = yStrm.read();
                dotMul = dotMulStrm.read();
                eYStrm.read();
                eDotMulStrm.read();
                err = dotMul - y;
                interceptGradientStrm.write(err);
                eInterceptGradientStrm.write(false);
            }
            for (int i = 0; i < D; i++) {
#pragma HLS unroll
                MType x = xStrm[i].read();
                thetaGradientStrm[i].write(x * err);
            }
            eThetaGradientStrm.write(false);
            // update ptr
            colCounter += D;
            if (colCounter >= cols) {
                colCounter = 0;
            }
        }
        eDotMulStrm.read();
        eYStrm.read();
        eThetaGradientStrm.write(true);
        eInterceptGradientStrm.write(true);
    }
};

template <typename MType>
MType funcMul(MType op1, MType op2) {
    return op1 * op2;
}

template <typename MType>
void funcSum(MType& reg, MType op) {
    reg += op;
}

template <typename MType>
MType funcAssign(MType op) {
    return op;
}

template <typename MType>
MType funcExp(MType op) {
    return xf::data_analytics::internal::m::exp(op);
}

template <typename MType,
          int WAxi,
          int WData,
          int BurstLen,
          int D,
          int DDepth,
          RAMType RAMWeight,
          RAMType RAMIntercept,
          RAMType RAMAvgWeight,
          RAMType RAMAvgIntercept>
class linearLeastSquareGradientProcessor {
   public:
    linearLeastSquareGradientProcessor() {
#pragma HLS inline
    }

    static const int axi_fifo_depth = BurstLen * 2;
    static const int LatencyR = sl2<MType,
                                    D,
                                    DDepth,
                                    1,
                                    1,
                                    &funcMul<MType>,
                                    &funcSum<MType>,
                                    &funcAssign<MType>,
                                    AdditionLatency<MType>::value,
                                    RAMWeight,
                                    RAMIntercept>::LatencyR;
    static const int InputW = WAxi;
    static const int DFactor = D;
    static const int DepthFactor = DDepth;
    static const int L1 = s_aggr<MType, D, DDepth, &funcSum<MType>, AdditionLatency<MType>::value, RAMAvgWeight>::L;
    static const int L2 = s_aggr<MType, 1, 1, &funcSum<MType>, AdditionLatency<MType>::value, RAMAvgIntercept>::L;
    static const int LatencyT = sl2<MType,
                                    D,
                                    DDepth,
                                    1,
                                    1,
                                    &funcMul<MType>,
                                    &funcSum<MType>,
                                    &funcAssign<MType>,
                                    AdditionLatency<MType>::value,
                                    RAMWeight,
                                    RAMIntercept>::LatencyT;
    static const int x2_fifo_depth = axi_fifo_depth * 2 + LatencyT;
    typedef MType DataType;

    tagTableRandomLoader<WAxi, WData, BurstLen, MType, MType> scanner;

    sl2<MType,
        D,
        DDepth,
        1,
        1,
        &funcMul<MType>,
        &funcSum<MType>,
        &funcAssign<MType>,
        AdditionLatency<MType>::value,
        RAMWeight,
        RAMIntercept>
        dotMulProcessor;

    linearLeastSquareGradient<MType, D> sampleGradient;

    s_aggr<MType, D, DDepth, &funcSum<MType>, AdditionLatency<MType>::value, RAMAvgWeight> avgThetaGradientProcessor;

    s_aggr<MType, 1, 1, &funcSum<MType>, AdditionLatency<MType>::value, RAMAvgIntercept> avgInterceptGradientProcessor;

    void seedInitialization(ap_uint<32> seed) { scanner.seedInitialization(seed); }

    void initParams(ap_uint<32> cols) { dotMulProcessor.initParams(cols, 1); }

    bool L1Update(ap_uint<32> iterationIndex,
                  ap_uint<32> cols,
                  MType tolerance,
                  MType stepSize,
                  MType regVal,
                  bool withIntercept) {
        MType thisIterStepSize = stepSize / xf::data_analytics::internal::m::sqrt(MType(iterationIndex));
        MType shrinkVal = thisIterStepSize * regVal;
        MType newNorm = 0;
        MType diffNorm = 0;
        // update weight
        ap_uint<32> ptrD = 0;
        ap_uint<32> ptrDep = 0;
        ap_uint<32> ptrL = 0;
        for (int i = 0; i < cols; i++) {
#pragma HLS pipeline
            MType tmpGrad;
            MType tmpWI;
            MType tmpWI2;
            MType diff;
            MType tmpMag;
            if (i < cols - 1) {
                tmpWI = dotMulProcessor.weight[0][ptrD][ptrDep];
                tmpGrad = avgThetaGradientProcessor.sum[ptrD][ptrL];
            } else {
                tmpWI = dotMulProcessor.intercept[0][0];
                if (withIntercept) {
                    tmpGrad = avgInterceptGradientProcessor.sum[0][0];
                } else {
                    tmpGrad = 0;
                }
            }
            tmpWI2 = tmpWI;

            tmpWI -= thisIterStepSize * tmpGrad;
            if (i < cols - 1) {
                if (tmpWI > 0) {
                    tmpMag = tmpWI - shrinkVal;
                } else {
                    tmpMag = -tmpWI - shrinkVal;
                }
                if (tmpMag < 0) {
                    tmpMag = 0;
                }
                if (tmpWI > 0) {
                    tmpWI = tmpMag;
                } else {
                    tmpWI = -tmpMag;
                }
            }
            diff = tmpWI - tmpWI2;

            diffNorm += diff * diff;
            newNorm += tmpWI * tmpWI;

            if (i < cols - 1) {
                dotMulProcessor.weight[0][ptrD][ptrDep] = tmpWI;
            } else {
                if (withIntercept) {
                    dotMulProcessor.intercept[0][0] = tmpWI;
                }
            }
            // update ptr
            ptrD++;
            if (ptrD == D) {
                ptrD = 0;
                ptrDep++;
                ptrL += L1;
            }
        }
        // determine if converged: norm(new - old) < tolerance * max(1.0, norm(new))
        MType bar = newNorm > 1.0 ? newNorm : 1.0;
        bar *= tolerance;
        bar *= tolerance;
        return (diffNorm < bar);
    }

    bool L2Update(ap_uint<32> iterationIndex,
                  ap_uint<32> cols,
                  MType tolerance,
                  MType stepSize,
                  MType regVal,
                  bool withIntercept) {
        MType thisIterStepSize = stepSize / xf::data_analytics::internal::m::sqrt(MType(iterationIndex));
        MType newNorm = 0;
        MType diffNorm = 0;
        // update weight
        ap_uint<32> ptrD = 0;
        ap_uint<32> ptrDep = 0;
        ap_uint<32> ptrL = 0;
        for (int i = 0; i < cols; i++) {
#pragma HLS pipeline
            MType tmpGrad;
            MType tmpWI;
            MType tmpMag;
            MType diff;
            if (i < cols - 1) {
                tmpWI = dotMulProcessor.weight[0][ptrD][ptrDep];
                tmpGrad = avgThetaGradientProcessor.sum[ptrD][ptrL];
            } else {
                tmpWI = dotMulProcessor.intercept[0][0];
                if (withIntercept) {
                    tmpGrad = avgInterceptGradientProcessor.sum[0][0];
                } else {
                    tmpGrad = 0;
                }
            }

            tmpMag = tmpGrad;
            if (i < cols - 1) {
                tmpMag += (tmpWI * regVal);
            }

            diff = thisIterStepSize * tmpMag;
            tmpWI -= diff;
            diffNorm += diff * diff;
            newNorm += tmpWI * tmpWI;

            if (i < cols - 1) {
                dotMulProcessor.weight[0][ptrD][ptrDep] = tmpWI;
            } else {
                if (withIntercept) {
                    dotMulProcessor.intercept[0][0] = tmpWI;
                }
            }
            // update ptr
            ptrD++;
            if (ptrD == D) {
                ptrD = 0;
                ptrDep++;
                ptrL += L1;
            }
        }
        // determine if converged: norm(new - old) < tolerance * max(1.0, norm(new))
        MType bar = newNorm > 1.0 ? newNorm : 1.0;
        bar *= tolerance;
        bar *= tolerance;
        return (diffNorm < bar);
    }

    bool simpleUpdate(
        ap_uint<32> iterationIndex, ap_uint<32> cols, MType tolerance, MType stepSize, bool withIntercept) {
        MType thisIterStepSize = (-stepSize) / xf::data_analytics::internal::m::sqrt(MType(iterationIndex));
        MType newNorm = 0;
        MType diffNorm = 0;
        // update weight
        ap_uint<32> ptrD = 0;
        ap_uint<32> ptrDep = 0;
        ap_uint<32> ptrL = 0;
        for (int i = 0; i < cols; i++) {
#pragma HLS pipeline
            MType tmpGrad;
            MType tmpWI;
            MType diff;
            if (i < cols - 1) {
                tmpWI = dotMulProcessor.weight[0][ptrD][ptrDep];
                tmpGrad = avgThetaGradientProcessor.sum[ptrD][ptrL];
            } else {
                tmpWI = dotMulProcessor.intercept[0][0];
                if (withIntercept) {
                    tmpGrad = avgInterceptGradientProcessor.sum[0][0];
                } else {
                    tmpGrad = 0;
                }
            }

            diff = thisIterStepSize * tmpGrad;
            tmpWI += diff;
            diffNorm += diff * diff;
            newNorm += tmpWI * tmpWI;

            if (i < cols - 1) {
                dotMulProcessor.weight[0][ptrD][ptrDep] = tmpWI;
            } else {
                if (withIntercept) {
                    dotMulProcessor.intercept[0][0] = tmpWI;
                }
            }
            // update ptr
            ptrD++;
            if (ptrD == D) {
                ptrD = 0;
                ptrDep++;
                ptrL += L1;
            }
        }
        // determine if converged: norm(new - old) < tolerance * max(1.0, norm(new))
        MType bar = newNorm > 1.0 ? newNorm : 1.0;
        bar *= tolerance;
        bar *= tolerance;
        return (diffNorm < bar);
    }

    void processCore(ap_uint<WAxi>* ddr,
                     ap_uint<32> offset,
                     ap_uint<32> rows,
                     ap_uint<32> cols,
                     ap_uint<32> cols_1,
                     float fraction,
                     bool ifJump,
                     ap_uint<32> bucketSize) {
#pragma HLS dataflow
        hls::stream<MType> xStrm1[WAxi / WData];
#pragma HLS array_partition variable = xStrm1 dim = 1 complete
#pragma HLS stream variable = xStrm1 depth = axi_fifo_depth
        hls::stream<bool> eXStrm1;
#pragma HLS stream variable = eXStrm1 depth = axi_fifo_depth
        hls::stream<MType> xStrm2[WAxi / WData];
#pragma HLS array_partition variable = xStrm2 dim = 1 complete
#pragma HLS stream variable = xStrm2 depth = x2_fifo_depth
        hls::stream<bool> eXStrm2;
#pragma HLS stream variable = eXStrm2 depth = x2_fifo_depth
        hls::stream<MType> yStrm;
#pragma HLS stream variable = yStrm depth = x2_fifo_depth
        hls::stream<bool> eYStrm;
#pragma HLS stream variable = eYStrm depth = x2_fifo_depth
        hls::stream<MType> dotMulStrm[1];
#pragma HLS stream variable = dotMulStrm depth = axi_fifo_depth
        hls::stream<bool> eDotMulStrm;
#pragma HLS stream variable = eDotMulStrm depth = axi_fifo_depth
        hls::stream<MType> thetaGradientStrm[WAxi / WData];
#pragma HLS array_partition variable = thetaGradientStrm dim = 1 complete
#pragma HLS stream variable = thetaGradientStrm depth = axi_fifo_depth
        hls::stream<bool> eThetaGradientStrm;
#pragma HLS stream variable = eThetaGradientStrm depth = axi_fifo_depth
        hls::stream<MType> interceptGradientStrm[1];
#pragma HLS stream variable = interceptGradientStrm depth = axi_fifo_depth
        hls::stream<bool> eInterceptGradientStrm;
#pragma HLS stream variable = eInterceptGradientStrm depth = axi_fifo_depth

        scanner.sample(ddr, offset, rows, cols, fraction, ifJump, bucketSize, xStrm1, eXStrm1, xStrm2, eXStrm2, yStrm,
                       eYStrm);

        dotMulProcessor.process(xStrm1, eXStrm1, dotMulStrm, eDotMulStrm, cols_1, 1);
        sampleGradient.process(cols_1, xStrm2, eXStrm2, yStrm, eYStrm, dotMulStrm[0], eDotMulStrm, thetaGradientStrm,
                               eThetaGradientStrm, interceptGradientStrm[0], eInterceptGradientStrm);
        avgThetaGradientProcessor.processAvg(thetaGradientStrm, eThetaGradientStrm, cols_1);
        avgInterceptGradientProcessor.processAvg(interceptGradientStrm, eInterceptGradientStrm, 1);
    }

    void process(ap_uint<WAxi>* ddr,
                 ap_uint<32> offset,
                 ap_uint<32> rows,
                 ap_uint<32> cols,
                 float fraction,
                 bool ifJump,
                 ap_uint<32> bucketSize) {
        ap_uint<32> cols_1 = cols - 1;
        processCore(ddr, offset, rows, cols, cols_1, fraction, ifJump, bucketSize);
    }
};

} // namespace internal
} // namespace regression
} // namespace data_analytics
} // namespace xf

#endif
