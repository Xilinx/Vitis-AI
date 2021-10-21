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
 * @file SGD.hpp
 * @brief Stochastic Gradient Descent Framework
 *
 * This file is part of Vitis Data Analytics Library.
 */

#ifndef _XF_DATA_ANALYTICS_L1_SGD_HPP_
#define _XF_DATA_ANALYTICS_L1_SGD_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include "xf_data_analytics/common/stream_local_processing.hpp"
#include "xf_data_analytics/common/table_sample.hpp"
#include "xf_data_analytics/common/enums.hpp"

namespace xf {
namespace data_analytics {
namespace common {

/**
 * @brief Stochasitc Gradient Descent Framework
 *
 * @tparam Gradient gradient class which suite into this framework.
 */
template <typename Gradient>
class SGDFramework {
   public:
    SGDFramework() {
#pragma HLS inline
    }

    // static const
    static const int WAxi = Gradient::InputW;
    static const int D = Gradient::DFactor;
    static const int Depth = Gradient::DepthFactor;
    typedef typename Gradient::DataType MType;

    // training data's offset in ddr
    ap_uint<32> offset;
    // number of rows of training data
    ap_uint<32> rows;
    // number of columns of training data
    ap_uint<32> cols;
    // bucketSize of jump sampling
    ap_uint<32> bucketSize;
    // sampe fraction
    float fraction;
    // if perform jump sample or not
    bool ifJump;
    // SGD iteration steps
    MType stepSize;
    // SGD convergence tolerance
    MType tolerance;
    // If training uses intercept or not
    bool withIntercept;
    // Max iteration number of SGD
    ap_uint<32> maxIter;
    // gradient
    Gradient gradProcessor;

    /**
     * @brief Initialize RNG for sampling data
     *
     * @param seed Seed for RNG
     */
    void seedInitialization(ap_uint<32> seed) { gradProcessor.seedInitialization(seed); }

    /**
     * @brief Set configs for SGD iteration
     *
     * @param inputStepSize steps size of SGD iteration.
     * @param inputTolerance convergence tolerance of SGD.
     * @param inputWithIntercept if SGD includes intercept or not.
     * @param inputMaxIter max iteration number of SGD.
     */
    void setTrainingConfigs(MType inputStepSize,
                            MType inputTolerance,
                            bool inputWithIntercept,
                            ap_uint<32> inputMaxIter) {
        stepSize = inputStepSize;
        tolerance = inputTolerance;
        withIntercept = inputWithIntercept;
        maxIter = inputMaxIter;
    }

    /**
     * @brief Set configs for loading trainging data
     *
     * @param inputOffset offset of data in ddr.
     * @param inputRows number of rows of training data
     * @param inputCols number of features of training data
     * @param inputBucketSize bucketSize of jump sampling
     * @param inputFraction sample fraction
     * @param inputIfJump perform jump scaling or not.
     */
    void setTrainingDataParams(ap_uint<32> inputOffset,
                               ap_uint<32> inputRows,
                               ap_uint<32> inputCols,
                               ap_uint<32> inputBucketSize,
                               float inputFraction,
                               bool inputIfJump) {
        offset = inputOffset;
        rows = inputRows;
        cols = inputCols;
        bucketSize = inputBucketSize;
        fraction = inputFraction;
        ifJump = inputIfJump;
    }

    /**
     * @brief Set initial weight to zeros
     *
     * @param cols feature numbers
     */
    void initGradientParams(ap_uint<32> cols) { gradProcessor.initParams(cols); }

    /**
     * @brief calculate gradient of current weight
     *
     * @param ddr Traing Data
     */
    void calcGradient(ap_uint<WAxi>* ddr) {
#pragma HLS inline off
        gradProcessor.process(ddr, offset, rows, cols, fraction, ifJump, bucketSize);
    }

    /**
     * @brief update weight and intercept based on gradient
     *
     * @param iterationIndex iteraton index.
     */
    bool updateParams(ap_uint<32> iterationIndex) {
        return gradProcessor.simpleUpdate(iterationIndex, cols, tolerance, stepSize, withIntercept);
    }

    /**
     * @brief training function
     *
     * @param ddr input Data
     */
    void train(ap_uint<WAxi>* ddr) {
        initGradientParams(cols);
        // first calculation;
        calcGradient(ddr);
        ap_uint<32> iter = 1;
        while (!this->updateParams(iter) && iter <= maxIter) {
            iter++;
            calcGradient(ddr);
        }
    }
};

} // namespace common
} // namespace data_analytics
} // namespace xf
#endif
