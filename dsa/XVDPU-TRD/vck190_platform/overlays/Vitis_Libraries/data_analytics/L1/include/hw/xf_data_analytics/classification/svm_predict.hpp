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
 * @file svm_predict.hpp
 * @brief svm predict function implementation.
 *
 * This file is part of Vitis Data Analytics Library.
 */
#ifndef _XF_DATA_ANALYTICS_L1_SVM_PREDICT_HPP_
#define _XF_DATA_ANALYTICS_L1_SVM_PREDICT_HPP_
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include "xf_data_analytics/common/table_sample.hpp"
#include "xf_data_analytics/common/stream_local_processing.hpp"

double funcA(double op1, double op2) {
    return (op1 * op2);
}

void funcB(double& reg, double op) {
#pragma HLS inline off
    reg += op;
}

double funcC(double op) {
    return op;
}

namespace xf {
namespace data_analytics {
namespace classification {
namespace internal {

using namespace xf::data_analytics::common::internal;

/**
 * @brief transPredictions transform predicion type to ap_uint.
 *
 * This function read prediction calculate result stream, and output prediction class tag into a stream
 *
 * @tparam MType The data type of sample
 * @tparam WD The width of data type MType, can get by sizeof(MType)
 *
 * @param retStrm Input calculate result streams of MType
 * @param eRetStrm End flag stream for input data
 * @param predictionsStrm Output class tag streams
 * @param predictionsTagStrm End flag stream for output
 */
template <typename MType, unsigned WD>
void transPredictions(hls::stream<MType> retStrm[1],
                      hls::stream<bool>& eRetStrm,
                      hls::stream<ap_uint<1> >& predictionsStrm,
                      hls::stream<bool>& predictionsTag) {
    bool e = eRetStrm.read();
    while (!e) {
#pragma HLS pipeline
        if (retStrm[0].read() > 0) {
            predictionsStrm.write(1);
            predictionsTag.write(false);
        } else {
            predictionsStrm.write(0);
            predictionsTag.write(false);
        }
        e = eRetStrm.read();
    }
    predictionsTag.write(true);
}

/**
 * @brief getPredictions implement samples predicion.
 *
 * This function read sample streams, and output prediction result into a stream
 *
 * @tparam MType The data type of sample
 * @tparam WD The width of data type MType, can get by sizeof(MType)
 * @tparam StreamN The stream number of input sample stream vector
 * @tparam SampleDepth stream depth number of one input sample
 *
 * @param cols colum number of input data sample
 * @param sample_strm Input data streams of MType
 * @param e_sample_strm End flag stream for input data
 * @param weight svm weight array
 * @param predictionsStrm Output data streams
 * @param predictionsTagStrm End flag stream for output
 */
template <typename MType, unsigned WD, unsigned StreamN, unsigned SampleDepth>
void getPredictions(const int cols,
                    hls::stream<MType> sample_strm[StreamN],
                    hls::stream<bool>& e_sample_strm,
                    MType weight[1][StreamN][SampleDepth],
                    hls::stream<ap_uint<1> >& predictionsStrm,
                    hls::stream<bool>& predictionsTag) {
    hls::stream<MType> retStrm[1];
#pragma HLS stream variable = retStrm depth = 128
    hls::stream<bool> eRetStrm;
#pragma HLS stream variable = eRetStrm depth = 128
#pragma HLS dataflow

    sl2<double, StreamN, SampleDepth, 1, 1, &funcA, &funcB, &funcC, 6, URAM, URAM> processor;
    processor.setWeight(weight, cols, 1);
    processor.process(sample_strm, e_sample_strm, retStrm, eRetStrm, cols, 1);
    transPredictions<MType, WD>(retStrm, eRetStrm, predictionsStrm, predictionsTag);
}

/**
 * @brief getWeight loads 512-bit svm weight
 *
 * @tparam MType The data type of sample
 * @tparam StreamN The stream number of input sample stream vector
 * @tparam SampleDepth stream depth number of one input sample
 *
 * @param weight_strm svm weight streams
 * @param eTag End flag stream for svm weight
 * @param weight svm array)
 */
template <typename MType, unsigned StreamN, unsigned SampleDepth>
void getWeight(hls::stream<ap_uint<512> >& weight_strm,
               hls::stream<bool>& eTag,
               MType weight[1][StreamN][SampleDepth]) {
    bool e = eTag.read();
    ap_uint<512> tmp;
    int num = 0;
    while (!e) {
#pragma HLS pipeline
        tmp = weight_strm.read();
        for (int j = 0; j < StreamN; j++) {
            f_cast<MType> w;
            w.i = tmp.range(sizeof(MType) * 8 * j + sizeof(MType) * 8 - 1, sizeof(MType) * 8 * j);
            weight[0][j][num] = w.f;
        }
        num++;
        e = eTag.read();
    }
}

} // namespace internal
} // namespace classification
} // namespace data_analytics
} // namespace xf

namespace xf {
namespace data_analytics {
namespace classification {
/**
 * @brief svmPredict, Top function of svm Predict.
 *
 * This function first loads weight (the corresponding function : getWeight) from weight_strm
 * Then, read sample from sample_strm, and output its classification id into predictionsStrm streams
 *
 * @tparam MType The data type of sample
 * @tparam WD The width of data type MType, can get by sizeof(MType)
 * @tparam StreamN The stream number of input sample stream vector
 * @tparam SampleDepth stream depth number of one input sample
 *
 * @param cols colum number of input data sample
 * @param sample_strm Input data streams of MType
 * @param e_sample_strm End flag stream for input data
 * @param weight_strm weight streams
 * @param eTag End flag stream for weight streams
 * @param predictionsStrm Output data streams
 * @param predictionsTagStrm End flag stream for output
 */
template <typename MType, unsigned WD, unsigned StreamN, unsigned SampleDepth>
void svmPredict(const int cols,
                hls::stream<MType> sample_strm[StreamN],
                hls::stream<bool>& e_sample_strm,
                hls::stream<ap_uint<512> >& weight_strm,
                hls::stream<bool>& eTag,
                hls::stream<ap_uint<1> >& predictionsStrm,
                hls::stream<bool>& predictionsTag) {
    MType weight[1][StreamN][SampleDepth];
#pragma HLS array_partition variable = weight dim = 2 complete

    internal::getWeight<MType, StreamN, SampleDepth>(weight_strm, eTag, weight);
    internal::getPredictions<MType, WD, StreamN, SampleDepth>(cols, sample_strm, e_sample_strm, weight, predictionsStrm,
                                                              predictionsTag);
}
} // namespace classification
} // namespace data_analytics
} // namespace xf
#endif
