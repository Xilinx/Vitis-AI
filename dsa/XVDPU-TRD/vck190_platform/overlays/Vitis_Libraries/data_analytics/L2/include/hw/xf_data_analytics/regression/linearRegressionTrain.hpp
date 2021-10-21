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

#ifndef _XF_DATA_ANALYTICS_CLASSIFICATION_LINEAR_REGRESSION_TRAIN_HPP_
#define _XF_DATA_ANALYTICS_CLASSIFICATION_LINEAR_REGRESSION_TRAIN_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include "xf_data_analytics/common/SGD.hpp"
#include "xf_data_analytics/common/enums.hpp"
#include "xf_data_analytics/common/stream_local_processing.hpp"
#include "xf_data_analytics/regression/gradient.hpp"
#include "xf_data_analytics/regression/linearRegression.hpp"
#include "xf_data_analytics/common/table_sample.hpp"

namespace xf {
namespace data_analytics {
namespace regression {

using namespace xf::data_analytics::regression::internal;
using namespace xf::data_analytics::common::internal;

/**
 * @brief linear least square regression training using SGD framework
 *
 * @tparam WAxi AXI interface width to load training data.
 * @tparam D Number of features that processed each cycle
 * @tparam DDepth DDepth * D is max feature numbers supported.
 * @tparam BurstLen, Length of burst read.
 *
 * @param input, training configs and training data
 * @param output, training result of weight and intercept
 */
template <int WAxi, int D, int Depth, int BurstLen>
void linearLeastSquareRegressionSGDTrain(ap_uint<WAxi>* input, ap_uint<WAxi>* output) {
    ap_uint<32> seed;
    double stepSize;
    double tolerance;
    bool withIntercept;
    ap_uint<32> maxIter;
    ap_uint<32> offset;
    ap_uint<32> rows;
    ap_uint<32> cols;
    ap_uint<32> bucketSize;
    float fraction;
    bool ifJump;

    ap_uint<32> ptr = 0;
    for (int i = 0; i < 11; i += D) {
#pragma HLS pipeline
        ap_uint<WAxi> tmp = input[ptr];
        for (int j = 0; j < D; j++) {
            ap_uint<64> seg = tmp.range(j * 64 + 63, j * 64);
            f_cast<double> dcast;
            f_cast<float> fcast;
            dcast.i = seg;
            fcast.i = seg.range(31, 0);
            switch (i + j) {
                case 0:
                    seed = fcast.i;
                    break;
                case 1:
                    stepSize = dcast.f;
                    break;
                case 2:
                    tolerance = dcast.f;
                    break;
                case 3:
                    withIntercept = fcast.i == 0 ? false : true;
                    break;
                case 4:
                    maxIter = fcast.i;
                    break;
                case 5:
                    offset = fcast.i;
                    break;
                case 6:
                    rows = fcast.i;
                    break;
                case 7:
                    cols = fcast.i;
                    break;
                case 8:
                    bucketSize = fcast.i;
                    break;
                case 9:
                    fraction = fcast.f;
                    break;
                case 10:
                    ifJump = fcast.i == 0 ? false : true;
                    break;
                default:
                    break;
            }
        }
        ptr++;
    }

    typedef linearLeastSquareRegressionSGDTrainer<double, WAxi, 64, BurstLen, D, Depth, BRAM, BRAM, BRAM, BRAM>
        linearLeastSquareTrainer;
    linearLeastSquareTrainer trainer;

    trainer.seedInitialization(seed);
    trainer.setTrainingConfigs(stepSize, tolerance, withIntercept, maxIter);
    trainer.setTrainingDataParams(offset, rows, cols, bucketSize, fraction, ifJump);
    trainer.train(input);

    int round = (cols - 1 + D - 1) / D;
    for (int i = 0; i <= round; i++) {
#pragma HLS pipeline II = 1
        ap_uint<WAxi> tmp;
        if (i < round) {
            for (int j = 0; j < D; j++) {
#pragma HLS unroll
                f_cast<double> dcast;
                dcast.f = trainer.gradProcessor.dotMulProcessor.weight[0][j][i];
                tmp.range(j * 64 + 63, j * 64) = dcast.i;
            }
        } else {
            f_cast<double> dcast;
            dcast.f = trainer.gradProcessor.dotMulProcessor.intercept[0][0];
            tmp.range(63, 0) = dcast.i;
        }
        output[i] = tmp;
    }
}

/**
 * @brief ridge regression training using SGD framework
 *
 * @tparam WAxi AXI interface width to load training data.
 * @tparam D Number of features that processed each cycle
 * @tparam DDepth DDepth * D is max feature numbers supported.
 * @tparam BurstLen, Length of burst read.
 *
 * @param input, training configs and training data
 * @param output, training result of weight and intercept
 */
template <int WAxi, int D, int Depth, int BurstLen>
void ridgeRegressionSGDTrain(ap_uint<WAxi>* input, ap_uint<WAxi>* output) {
    ap_uint<32> seed;
    double stepSize;
    double tolerance;
    bool withIntercept;
    ap_uint<32> maxIter;
    ap_uint<32> offset;
    ap_uint<32> rows;
    ap_uint<32> cols;
    ap_uint<32> bucketSize;
    float fraction;
    bool ifJump;
    double regVal;

    ap_uint<32> ptr = 0;
    for (int i = 0; i < 11; i += D) {
#pragma HLS pipeline
        ap_uint<WAxi> tmp = input[ptr];
        for (int j = 0; j < D; j++) {
            ap_uint<64> seg = tmp.range(j * 64 + 63, j * 64);
            f_cast<double> dcast;
            f_cast<float> fcast;
            dcast.i = seg;
            fcast.i = seg.range(31, 0);
            switch (i + j) {
                case 0:
                    seed = fcast.i;
                    break;
                case 1:
                    stepSize = dcast.f;
                    break;
                case 2:
                    tolerance = dcast.f;
                    break;
                case 3:
                    withIntercept = fcast.i == 0 ? false : true;
                    break;
                case 4:
                    maxIter = fcast.i;
                    break;
                case 5:
                    offset = fcast.i;
                    break;
                case 6:
                    rows = fcast.i;
                    break;
                case 7:
                    cols = fcast.i;
                    break;
                case 8:
                    bucketSize = fcast.i;
                    break;
                case 9:
                    fraction = fcast.f;
                    break;
                case 10:
                    ifJump = fcast.i == 0 ? false : true;
                    break;
                case 11:
                    regVal = dcast.f;
                    break;
                default:
                    break;
            }
        }
        ptr++;
    }

    typedef ridgeRegressionSGDTrainer<double, WAxi, 64, BurstLen, D, Depth, BRAM, BRAM, BRAM, BRAM>
        linearLeastSquareTrainer;
    linearLeastSquareTrainer trainer;

    trainer.seedInitialization(seed);
    trainer.setTrainingConfigs(stepSize, tolerance, regVal, withIntercept, maxIter);
    trainer.setTrainingDataParams(offset, rows, cols, bucketSize, fraction, ifJump);
    trainer.train(input);

    int round = (cols - 1 + D - 1) / D;
    for (int i = 0; i <= round; i++) {
#pragma HLS pipeline II = 1
        ap_uint<WAxi> tmp;
        if (i < round) {
            for (int j = 0; j < D; j++) {
#pragma HLS unroll
                f_cast<double> dcast;
                dcast.f = trainer.gradProcessor.dotMulProcessor.weight[0][j][i];
                tmp.range(j * 64 + 63, j * 64) = dcast.i;
            }
        } else {
            f_cast<double> dcast;
            dcast.f = trainer.gradProcessor.dotMulProcessor.intercept[0][0];
            tmp.range(63, 0) = dcast.i;
        }
        output[i] = tmp;
    }
}

/**
 * @brief lasso regression training using SGD framework
 *
 * @tparam WAxi AXI interface width to load training data.
 * @tparam D Number of features that processed each cycle
 * @tparam DDepth DDepth * D is max feature numbers supported.
 * @tparam BurstLen, Length of burst read.
 *
 * @param input, training configs and training data
 * @param output, training result of weight and intercept
 */
template <int WAxi, int D, int Depth, int BurstLen>
void LASSORegressionSGDTrain(ap_uint<WAxi>* input, ap_uint<WAxi>* output) {
    ap_uint<32> seed;
    double stepSize;
    double tolerance;
    bool withIntercept;
    ap_uint<32> maxIter;
    ap_uint<32> offset;
    ap_uint<32> rows;
    ap_uint<32> cols;
    ap_uint<32> bucketSize;
    float fraction;
    bool ifJump;
    double regVal;

    ap_uint<32> ptr = 0;
    for (int i = 0; i < 11; i += D) {
#pragma HLS pipeline
        ap_uint<WAxi> tmp = input[ptr];
        for (int j = 0; j < D; j++) {
            ap_uint<64> seg = tmp.range(j * 64 + 63, j * 64);
            f_cast<double> dcast;
            f_cast<float> fcast;
            dcast.i = seg;
            fcast.i = seg.range(31, 0);
            switch (i + j) {
                case 0:
                    seed = fcast.i;
                    break;
                case 1:
                    stepSize = dcast.f;
                    break;
                case 2:
                    tolerance = dcast.f;
                    break;
                case 3:
                    withIntercept = fcast.i == 0 ? false : true;
                    break;
                case 4:
                    maxIter = fcast.i;
                    break;
                case 5:
                    offset = fcast.i;
                    break;
                case 6:
                    rows = fcast.i;
                    break;
                case 7:
                    cols = fcast.i;
                    break;
                case 8:
                    bucketSize = fcast.i;
                    break;
                case 9:
                    fraction = fcast.f;
                    break;
                case 10:
                    ifJump = fcast.i == 0 ? false : true;
                    break;
                case 11:
                    regVal = dcast.f;
                    break;
                default:
                    break;
            }
        }
        ptr++;
    }

    typedef LASSORegressionSGDTrainer<double, WAxi, 64, BurstLen, D, Depth, LUTRAM, LUTRAM, LUTRAM, LUTRAM>
        linearLeastSquareTrainer;
    linearLeastSquareTrainer trainer;

    trainer.seedInitialization(seed);
    trainer.setTrainingConfigs(stepSize, tolerance, regVal, withIntercept, maxIter);
    trainer.setTrainingDataParams(offset, rows, cols, bucketSize, fraction, ifJump);
    trainer.train(input);

    int round = (cols - 1 + D - 1) / D;
    for (int i = 0; i <= round; i++) {
#pragma HLS pipeline II = 1
        ap_uint<WAxi> tmp;
        if (i < round) {
            for (int j = 0; j < D; j++) {
#pragma HLS unroll
                f_cast<double> dcast;
                dcast.f = trainer.gradProcessor.dotMulProcessor.weight[0][j][i];
                tmp.range(j * 64 + 63, j * 64) = dcast.i;
            }
        } else {
            f_cast<double> dcast;
            dcast.f = trainer.gradProcessor.dotMulProcessor.intercept[0][0];
            tmp.range(63, 0) = dcast.i;
        }
        output[i] = tmp;
    }
}

} // namespace regression
} // namespace data_analytics
} // namespace xf

#endif
