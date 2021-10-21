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
 * @file logisticRegression.hpp
 * @brief Logistic Regression alogrithm, both predict and training.
 *
 * This file is part of Vitis Data Analytics Library.
 */

#ifndef _XF_DATA_ANALYTICS_L1_LOGISTIC_REGRESSION_HPP_
#define _XF_DATA_ANALYTICS_L1_LOGISTIC_REGRESSION_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include "xf_data_analytics/common/SGD.hpp"
#include "xf_data_analytics/common/enums.hpp"
#include "xf_data_analytics/common/stream_local_processing.hpp"
#include "xf_data_analytics/regression/gradient.hpp"

namespace xf {
namespace data_analytics {
namespace classification {

using namespace xf::data_analytics::regression::internal;
using namespace xf::data_analytics::common::internal;

/**
 * @brief linear least square regression predict
 *
 * @tparam MType datatype of regression, support double and float
 * @tparam D Number of features that processed each cycle
 * @tparam DDepth DDepth * D is max feature numbers supported.
 * @tparam K Number of weight vectors that processed each cycle
 * @tparam KDepth KDepth * K is max weight vectors supported.
 * @tparam RAMWeight Use which kind of RAM to store weight, could be LUTRAM, BRAM or URAM.
 * @tparam RAMIntercept Use which kind of RAM to store intercept, could be LUTRAM, BRAM or URAM.
 */
template <typename MType, int D, int DDepth, int K, int KDepth, RAMType RAMWeight, RAMType RAMIntercept>
class logisticRegressionPredict {
   public:
    logisticRegressionPredict() {
#pragma HLS inline
    }

    static const int marginDepth = K * 4;

    sl2<MType,
        D,
        DDepth,
        K,
        KDepth,
        &funcMul<MType>,
        &funcSum<MType>,
        &funcAssign<MType>,
        AdditionLatency<MType>::value,
        RAMWeight,
        RAMIntercept>
        marginProcessor;

    pickMaxProcess<MType, K> pickProcessor;

    /**
     * @brief pick best weight vector for classification from K vectors
     *
     * @param margin K margins generate by K weight vectors.
     * @param counter start index of this K margins in all margins.
     * @param ws number of margins
     * @param maxMargin max of K margins.
     * @param maxIndex which index does max margin sits.
     */
    void pickFromK(MType margin[K], ap_uint<32> counter, ap_uint<32> ws, MType& maxMargin, ap_uint<32>& maxIndex) {
        MType tmpMargin = margin[0];
        ap_uint<32> tmpIndex = 0;
        for (int i = 1; i < K; i++) {
            if ((counter + i) < ws) {
                if (margin[i] > tmpMargin) {
                    tmpMargin = margin[i];
                    tmpIndex = i;
                }
            }
        }
        maxMargin = tmpMargin;
        maxIndex = tmpIndex + counter;
    }

    /**
     * @brief pick best weight vector for classification
     *
     * @param marginStrm margin stream.
     * To get a vector of L margins, marginStrm will be read (L + K - 1) / D times.
     * Margin 0 to K-1 will be read from marginStrm[0] to marginStrm[D-1] at the first time.
     * Then margin D to 2*D - 1. The last round will readin fake data if L is not divisiable by K.
     * These data won't be used, just to allign K streams.
     * @param eMarginStrm Endflag of marginStrm.
     * @param retStrm result stream of classification.
     * @param eRetStrm Endflag of retStrm.
     * @param ws number of weight vectors used.
     */
    void pick(hls::stream<MType> marginStrm[K],
              hls::stream<bool>& eMarginStrm,
              hls::stream<ap_uint<32> >& retStrm,
              hls::stream<bool>& eRetStrm,
              ap_uint<32> ws) {
        ap_uint<32> counter = 0;
        MType maxMargin = 0;
        ap_uint<32> maxIndex = 0;
        while (!eMarginStrm.read()) {
#pragma HLS pipeline II = 1
            MType margins[K];
#pragma HLS array_partition variable = margins dim = 1 complete
            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                margins[i] = marginStrm[i].read();
            }

            MType tmpMaxMargin;
            ap_uint<32> tmpMaxIndex;
            pickFromK(margins, counter, ws, tmpMaxMargin, tmpMaxIndex);

            if (tmpMaxMargin > maxMargin) {
                maxMargin = tmpMaxMargin;
                maxIndex = tmpMaxIndex;
            }
            // update ptr
            counter += K;
            if (counter >= ws) {
                counter = 0;
                if (maxMargin == 0) {
                    retStrm.write(0);
                } else {
                    retStrm.write(1 + maxIndex);
                }
                maxMargin = 0;
                maxIndex = 0;
                eRetStrm.write(false);
            }
        }
        eRetStrm.write(true);
    }

    void predictCore(hls::stream<MType> opStrm[D],
                     hls::stream<bool>& eOpStrm,
                     ap_uint<32> cols,
                     ap_uint<32> classNum_1,
                     hls::stream<ap_uint<32> >& retStrm,
                     hls::stream<bool>& eRetStrm) {
#pragma HLS inline off
        hls::stream<MType> marginStrm[K];
#pragma HLS array_partition variable = marginStrm dim = 1 complete
#pragma HLS stream variable = marginStrm depth = marginDepth
        hls::stream<bool> eMarginStrm;
#pragma HLS stream variable = eMarginStrm depth = marginDepth
#pragma HLS dataflow
        marginProcessor.process(opStrm, eOpStrm, marginStrm, eMarginStrm, cols, classNum_1);
        pickProcessor.pick(marginStrm, eMarginStrm, classNum_1, retStrm, eRetStrm);
    }

    /**
     * @brief classification function of logistic regression
     *
     * @param opStrm feature input streams.
     * To get a vector of L features, opStrm will be read (L + D - 1) / D times.
     * Feature 0 to D-1 will be read from opStrm[0] to opStrm[D-1] at the first time.
     * Then feature D to 2*D - 1. The last round will readin fake data if L is not divisiable by D.
     * These data won't be used, just to allign D streams.
     * @param eOpStrm End flag of opStrm.
     * @param cols Feature numbers
     * @param classNum Number of classes.
     * @param retStrm result stream of classification.
     * @param eRetStrm Endflag of retStrm.
     */
    void predict(hls::stream<MType> opStrm[D],
                 hls::stream<bool>& eOpStrm,
                 ap_uint<32> cols,
                 ap_uint<32> classNum,
                 hls::stream<ap_uint<32> >& retStrm,
                 hls::stream<bool>& eRetStrm) {
        ap_uint<32> classNum_1 = classNum - 1;
        predictCore(opStrm, eOpStrm, cols, classNum_1, retStrm, eRetStrm);
    }

    /**
     * @brief set up weight parameters for prediction
     *
     * @param inputW weight
     * @param cols Effective weight numbers
     * @param classNum number of classes.
     */
    void setWeight(MType inputW[K][D][KDepth * DDepth], ap_uint<32> cols, ap_uint<32> classNum) {
        marginProcessor.setWeight(inputW, cols, classNum - 1);
    }

    /**
     * @brief set up intercept parameters for prediction
     *
     * @param inputI intercept, should be set to zero if don't needed.
     * @param classNum number of classes.
     */
    void setIntercept(MType inputI[K][KDepth], ap_uint<32> classNum) {
        marginProcessor.setIntercept(inputI, classNum - 1);
    }
};

} // namespace classification
} // namespace data_analytics
} // namespace xf

#endif
