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
 * @file linearRegression.hpp
 * @brief Linear regression predict and train function implementation
 *
 * This file is part of Vitis Data Analytics Library.
 */

#ifndef _XF_DATA_ANALYTICS_L1_LINEAR_REGRESSION_HPP_
#define _XF_DATA_ANALYTICS_L1_LINEAR_REGRESSION_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include "xf_data_analytics/common/SGD.hpp"
#include "xf_data_analytics/common/enums.hpp"
#include "xf_data_analytics/common/stream_local_processing.hpp"
#include "xf_data_analytics/regression/gradient.hpp"

namespace xf {
namespace data_analytics {
namespace regression {

using namespace xf::data_analytics::regression::internal;
using namespace xf::data_analytics::common;
using namespace xf::data_analytics::common::internal;

/**
 * @brief linear least square regression predict
 *
 * @tparam MType datatype of regression, support double and float
 * @tparam D Number of features that processed each cycle
 * @tparam DDepth DDepth * D is max feature numbers supported.
 * @tparam RAMWeight Use which kind of RAM to store weight, could be LUTRAM, BRAM or URAM.
 * @tparam RAMIntercept Use which kind of RAM to store intercept, could be LUTRAM, BRAM or URAM.
 */
template <typename MType, int D, int DDepth, RAMType RAMWeight, RAMType RAMIntercept>
class linearLeastSquareRegressionPredict {
   public:
    linearLeastSquareRegressionPredict() {
#pragma HLS inline
    }

    // internal unroll stream processor
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

    /**
     * @brief set up weight parameters for prediction
     *
     * @param inputW weight
     * @param cols Effective weight numbers
     */
    void setWeight(MType inputW[D][DDepth], ap_uint<32> cols) { dotMulProcessor.setWeight(inputW, cols); }

    /**
     * @brief set up intercept parameters for prediction
     *
     * @param inputI intercept should be set to zero if don't needed.
     */
    void setIntercept(MType inputI) { dotMulProcessor.setIntercept(inputI); }

    /**
     * @brief predict based on input features and preset weight and intercept
     *
     * @param opStrm feature input streams.
     * To get a vector of L features, opStrm will be read (L + D - 1) / D times.
     * Feature 0 to D-1 will be read from opStrm[0] to opStrm[D-1] at the first time.
     * Then feature D to 2*D - 1. The last round will readin fake data if L is not divisiable by D.
     * These data won't be used, just to allign D streams.
     * @param eOpStrm End flag of opStrm.
     * @param retStrm Prediction result.
     * @param eRetStrm End flag of retStrm.
     * @param cols Effective feature numbers.
     */
    void predict(hls::stream<MType> opStrm[D],
                 hls::stream<bool>& eOpStrm,
                 hls::stream<MType> retStrm[1],
                 hls::stream<bool>& eRetStrm,
                 ap_uint<32> cols) {
        dotMulProcessor.process(opStrm, eOpStrm, retStrm, eRetStrm, cols, 1);
    }
};

/**
 * @brief LASSO regression predict
 *
 * @tparam MType datatype of regression, support double and float
 * @tparam D Number of features that processed each cycle
 * @tparam DDepth DDepth * D is max feature numbers supported.
 * @tparam RAMWeight Use which kind of RAM to store weight, could be LUTRAM, BRAM or URAM.
 * @tparam RAMIntercept Use which kind of RAM to store intercept, could be LUTRAM, BRAM or URAM.
 */
template <typename MType, int D, int DDepth, RAMType RAMWeight, RAMType RAMIntercept>
class LASSORegressionPredict {
   public:
    LASSORegressionPredict() {
#pragma HLS inline
    }

    // internal unroll stream processor
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

    /**
     * @brief set up weight parameters for prediction
     *
     * @param inputW weight
     * @param cols Effective weight numbers
     */
    void setWeight(MType inputW[D][DDepth], ap_uint<32> cols) { dotMulProcessor.setWeight(inputW, cols); }

    /**
     * @brief set up intercept parameters for prediction
     *
     * @param inputI intercept, should be set to zero if don't needed.
     */
    void setIntercept(MType inputI) { dotMulProcessor.setIntercept(inputI); }

    /**
     * @brief predict based on input features and preset weight and intercept
     *
     * @param opStrm feature input streams.
     * To get a vector of L features, opStrm will be read (L + D - 1) / D times.
     * Feature 0 to D-1 will be read from opStrm[0] to opStrm[D-1] at the first time.
     * Then feature D to 2*D - 1. The last round will readin fake data if L is not divisiable by D.
     * These data won't be used, just to allign D streams.
     * @param eOpStrm End flag of opStrm.
     * @param retStrm Prediction result.
     * @param eRetStrm End flag of retStrm.
     * @param cols Effective feature numbers.
     */
    void predict(hls::stream<MType> opStrm[D],
                 hls::stream<bool>& eOpStrm,
                 hls::stream<MType> retStrm[1],
                 hls::stream<bool>& eRetStrm,
                 ap_uint<32> cols) {
        dotMulProcessor.process(opStrm, eOpStrm, retStrm, eRetStrm, cols, 1);
    }
};

/**
 * @brief ridge regression predict
 *
 * @tparam MType datatype of regression, support double and float
 * @tparam D Number of features that processed each cycle
 * @tparam DDepth DDepth * D is max feature numbers supported.
 * @tparam RAMWeight Use which kind of RAM to store weight, could be LUTRAM, BRAM or URAM.
 * @tparam RAMIntercept Use which kind of RAM to store intercept, could be LUTRAM, BRAM or URAM.
 */
template <typename MType, int D, int DDepth, RAMType RAMWeight, RAMType RAMIntercept>
class ridgeRegressionPredict {
   public:
    ridgeRegressionPredict() {
#pragma HLS inline
    }

    // internal unroll stream processor
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

    /**
     * @brief set up weight parameters for prediction
     *
     * @param inputW weight
     * @param cols Effective weight numbers
     */
    void setWeight(MType inputW[D][DDepth], ap_uint<32> cols) { dotMulProcessor.setWeight(inputW, cols); }

    /**
     * @brief set up intercept parameters for prediction
     *
     * @param inputI intercept, should be set to zero if don't needed.
     */
    void setIntercept(MType inputI) { dotMulProcessor.setIntercept(inputI); }

    /**
     * @brief predict based on input features and preset weight and intercept
     *
     * @param opStrm feature input streams.
     * To get a vector of L features, opStrm will be read (L + D - 1) / D times.
     * Feature 0 to D-1 will be read from opStrm[0] to opStrm[D-1] at the first time.
     * Then feature D to 2*D - 1. The last round will readin fake data if L is not divisiable by D.
     * These data won't be used just to allign D streams.
     * @param eOpStrm End flag of opStrm.
     * @param retStrm Prediction result.
     * @param eRetStrm End flag of retStrm.
     * @param cols Effective feature numbers.
     */
    void predict(hls::stream<MType> opStrm[D],
                 hls::stream<bool>& eOpStrm,
                 hls::stream<MType> retStrm[1],
                 hls::stream<bool>& eRetStrm,
                 ap_uint<32> cols) {
        dotMulProcessor.process(opStrm, eOpStrm, retStrm, eRetStrm, cols, 1);
    }
};

namespace internal {

/**
 * @brief linear least square regression training using SGD framework
 *
 * @tparam MType datatype of regression, support double and float
 * @tparam WAxi AXI interface width to load training data.
 * @tparam WData Data width of feature data type.
 * @tparam BurstLen Length of burst read.
 * @tparam D Number of features that processed each cycle
 * @tparam DDepth DDepth * D is max feature numbers supported.
 * @tparam RAMWeight Use which kind of RAM to store weight, could be LUTRAM, BRAM or URAM.
 * @tparam RAMIntercept Use which kind of RAM to store intercept, could be LUTRAM, BRAM or URAM.
 * @tparam RAMAvgWeight Use which kind of RAM to store Avg of Weigth, could be LUTRAM, BRAM or URAM.
 * @tparam RAMAvgIntercept Use which kind of RAM to store Avg of intercept, could be LUTRAM, BRAM or URAM.
 */
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
class linearLeastSquareRegressionSGDTrainer
    : public SGDFramework<linearLeastSquareGradientProcessor<MType,
                                                             WAxi,
                                                             WData,
                                                             BurstLen,
                                                             D,
                                                             DDepth,
                                                             RAMWeight,
                                                             RAMIntercept,
                                                             RAMAvgWeight,
                                                             RAMAvgIntercept> > {
   public:
    linearLeastSquareRegressionSGDTrainer() {
#pragma HLS inline
    }
};

/**
 * @brief lasso regression training using SGD framework
 *
 * @tparam MType datatype of regression, support double and float
 * @tparam WAxi AXI interface width to load training data.
 * @tparam WData Data width of feature data type.
 * @tparam BurstLen Length of burst read.
 * @tparam D Number of features that processed each cycle
 * @tparam DDepth DDepth * D is max feature numbers supported.
 * @tparam RAMWeight Use which kind of RAM to store weight, could be LUTRAM, BRAM or URAM.
 * @tparam RAMIntercept Use which kind of RAM to store intercept, could be LUTRAM, BRAM or URAM.
 * @tparam RAMAvgWeight Use which kind of RAM to store Avg of Weigth, could be LUTRAM, BRAM or URAM.
 * @tparam RAMAvgIntercept Use which kind of RAM to store Avg of intercept, could be LUTRAM, BRAM or URAM.
 */
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
class LASSORegressionSGDTrainer : public SGDFramework<linearLeastSquareGradientProcessor<MType,
                                                                                         WAxi,
                                                                                         WData,
                                                                                         BurstLen,
                                                                                         D,
                                                                                         DDepth,
                                                                                         RAMWeight,
                                                                                         RAMIntercept,
                                                                                         RAMAvgWeight,
                                                                                         RAMAvgIntercept> > {
   public:
    MType regVal;

    LASSORegressionSGDTrainer() {
#pragma HLS inline
    }

    /**
     * @brief Set up configs related to SGD iteration.
     *
     * @param inputStepSize step size of SGD iteratin.
     * @param inputTolerance Convergence tolerance.
     * @param inputRegVal regularazation value for LASSO.
     * @param intputWithIntercept If training includes intercept or not.
     * @param inputMaxIter Max iteration number of SGD.
     */
    void setTrainingConfigs(MType inputStepSize,
                            MType inputTolerance,
                            MType inputRegVal,
                            bool inputWithIntercept,
                            ap_uint<32> inputMaxIter) {
        this->stepSize = inputStepSize;
        this->tolerance = inputTolerance;
        this->withIntercept = inputWithIntercept;
        this->regVal = inputRegVal;
        this->maxIter = inputMaxIter;
    }

    /**
     * @brief update weight and intercept based on gradient
     *
     * @param iterationIndex iteraton index.
     */
    bool updateParams(ap_uint<32> iterationIndex) {
        return this->gradProcessor.L1Update(iterationIndex, this->cols, this->tolerance, this->stepSize, this->regVal,
                                            this->withIntercept);
    }

    /**
     * @brief training function
     *
     * @param ddr input Data
     */
    void train(ap_uint<WAxi>* ddr) {
        this->initGradientParams(this->cols);
        this->calcGradient(ddr);
        ap_uint<32> iter = 1;
        while (!this->updateParams(iter) && iter <= this->maxIter) {
            iter++;
            this->calcGradient(ddr);
        }
    }
};

/**
 * @brief ridge regression training using SGD framework
 *
 * @tparam MType datatype of regression, support double and float
 * @tparam WAxi AXI interface width to load training data.
 * @tparam WData Data width of feature data type.
 * @tparam BurstLen Length of burst read.
 * @tparam D Number of features that processed each cycle
 * @tparam DDepth DDepth * D is max feature numbers supported.
 * @tparam RAMWeight Use which kind of RAM to store weight, could be LUTRAM, BRAM or URAM.
 * @tparam RAMIntercept Use which kind of RAM to store intercept, could be LUTRAM, BRAM or URAM.
 * @tparam RAMAvgWeight Use which kind of RAM to store Avg of Weigth, could be LUTRAM, BRAM or URAM.
 * @tparam RAMAvgIntercept Use which kind of RAM to store Avg of intercept, could be LUTRAM, BRAM or URAM.
 */
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
class ridgeRegressionSGDTrainer : public SGDFramework<linearLeastSquareGradientProcessor<MType,
                                                                                         WAxi,
                                                                                         WData,
                                                                                         BurstLen,
                                                                                         D,
                                                                                         DDepth,
                                                                                         RAMWeight,
                                                                                         RAMIntercept,
                                                                                         RAMAvgWeight,
                                                                                         RAMAvgIntercept> > {
   public:
    MType regVal;

    ridgeRegressionSGDTrainer() {
#pragma HLS inline
    }

    /**
     * @brief Set up configs related to SGD iteration.
     *
     * @param inputStepSize step size of SGD iteratin.
     * @param inputTolerance Convergence tolerance.
     * @param inputRegVal regularazation value for LASSO.
     * @param intputWithIntercept If training includes intercept or not.
     * @param inputMaxIter Max iteration number of SGD.
     */
    void setTrainingConfigs(MType inputStepSize,
                            MType inputTolerance,
                            MType inputRegVal,
                            bool inputWithIntercept,
                            ap_uint<32> inputMaxIter) {
        this->stepSize = inputStepSize;
        this->tolerance = inputTolerance;
        this->withIntercept = inputWithIntercept;
        this->regVal = inputRegVal;
        this->maxIter = inputMaxIter;
    }

    /**
     * @brief update weight and intercept based on gradient
     *
     * @param iterationIndex iteraton index.
     */
    bool updateParams(ap_uint<32> iterationIndex) {
        return this->gradProcessor.L2Update(iterationIndex, this->cols, this->tolerance, this->stepSize, this->regVal,
                                            this->withIntercept);
    }

    /**
     * @brief training function
     *
     * @param ddr input Data
     */
    void train(ap_uint<WAxi>* ddr) {
        this->initGradientParams(this->cols);
        this->calcGradient(ddr);
        ap_uint<32> iter = 1;
        while (!this->updateParams(iter) && iter <= this->maxIter) {
            iter++;
            this->calcGradient(ddr);
        }
    }
};

} // namespace internal
} // namespace regression
} // namespace data_analytics
} // namespace xf

#endif
