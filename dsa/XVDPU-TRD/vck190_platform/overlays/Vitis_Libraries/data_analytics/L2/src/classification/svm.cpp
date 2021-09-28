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

#include "xf_data_analytics/classification/svm_train.hpp"
#include "xf_data_analytics/common/math_helper.hpp"

namespace xf {
namespace data_analytics {
namespace classification {
namespace svm {

using namespace xf::data_analytics::common;

template <typename MType, unsigned WD, unsigned StreamN, unsigned SampleDepth>
void gradient_processing(const ap_uint<64> feature_num,
                         hls::stream<MType> x_data_strm[StreamN],
                         hls::stream<ap_uint<WD> >& y_label_strm,
                         hls::stream<bool>& e_data_strm,
                         hls::stream<MType> retStrm[1],
                         hls::stream<bool>& eRetStrm,

                         hls::stream<MType> gradient_strm[StreamN],
                         hls::stream<bool>& e_gradient_strm) {
    int count = 0;
    eRetStrm.read();
    while (!(e_data_strm.read())) {
#pragma HLS pipeline
        MType x_data[StreamN];
#pragma HLS array_partition variable = x_data dim = 1
        MType gradient_data[StreamN];
#pragma HLS array_partition variable = gradient_data dim = 1
        MType y_data;
        MType dot_sum;
        if (count == 0) {
            y_data = MType(y_label_strm.read()) * 2.0 - 1.0;
            dot_sum = retStrm[0].read();
            eRetStrm.read();
        }
        for (int i = 0; i < StreamN; i++) {
#pragma HLS unroll
            x_data[i] = x_data_strm[i].read();
            gradient_data[i] = -y_data * x_data[i];
            if (dot_sum * y_data < 1.0) {
                gradient_strm[i].write(gradient_data[i]);
            } else {
                gradient_strm[i].write(0.0);
            }
        }
        e_gradient_strm.write(false);
        if (count < (feature_num + StreamN - 1) / StreamN - 1)
            count++;
        else
            count = 0;
    }
    e_gradient_strm.write(true);
}

void sink_stream(hls::stream<bool>& e_y_strm) {
    while (!e_y_strm.read()) {
        ;
    }
}

template <typename MType, unsigned WD, unsigned StreamN, unsigned SampleDepth>
void SVM_flow(
    xf::data_analytics::common::internal::tagTableRandomLoader<512, WD, BURST_LEN, MType, ap_uint<WD> >& loader,
    xf::data_analytics::common::internal::
        s_aggr<MType, StreamN, SampleDepth, &funcB, 7, xf::data_analytics::common::internal::URAM>& processor_agg,
    MType weight_temp[1][StreamN][SampleDepth],
    ap_uint<512>* ddr,
    const ap_uint<64> offset,
    const ap_uint<64> rows,
    const ap_uint<64> cols,
    const ap_uint<64> feature_num,
    const float fraction,
    const bool ifJump,
    const ap_uint<32> bucketSize) {
    static const int axi_fifo_depth = BURST_LEN * 2;
    static const int LatencyT =
        xf::data_analytics::common::internal::sl2<MType, StreamN, SampleDepth, 1, 1, &funcA, &funcB, &funcC, 6,
                                                  xf::data_analytics::common::internal::URAM,
                                                  xf::data_analytics::common::internal::URAM>::LatencyT;
    static const int x2_fifo_depth = axi_fifo_depth * 2 + LatencyT + 128;
    hls::stream<MType> x_data[StreamN];
#pragma HLS stream variable = x_data depth = axi_fifo_depth
    hls::stream<bool> e_x_data_strm;
#pragma HLS stream variable = e_x_data_strm depth = axi_fifo_depth
    hls::stream<MType> x_data_1[StreamN];
#pragma HLS stream variable = x_data_1 depth = x2_fifo_depth
    hls::stream<bool> e_data_strm;
#pragma HLS stream variable = e_data_strm depth = x2_fifo_depth
    hls::stream<ap_uint<WD> > y_label_1;
#pragma HLS stream variable = y_label_1 depth = x2_fifo_depth
    hls::stream<bool> e_y_strm;
#pragma HLS stream variable = e_y_strm depth = x2_fifo_depth

    hls::stream<MType> gradient_strm[StreamN];
#pragma HLS stream variable = gradient_strm depth = axi_fifo_depth
    hls::stream<bool> e_gradient_strm;
#pragma HLS stream variable = e_gradient_strm depth = axi_fifo_depth
    hls::stream<MType> retStrm[1];
#pragma HLS stream variable = retStrm depth = axi_fifo_depth
    hls::stream<bool> eRetStrm;
#pragma HLS stream variable = eRetStrm depth = axi_fifo_depth
#pragma HLS dataflow

    xf::data_analytics::common::internal::sl2<MType, StreamN, SampleDepth, 1, 1, &funcA, &funcB, &funcC, 6,
                                              xf::data_analytics::common::internal::URAM,
                                              xf::data_analytics::common::internal::URAM>
        processor;

    loader.sample(ddr, offset, rows, cols, fraction, ifJump, bucketSize, x_data, e_x_data_strm, x_data_1, e_data_strm,
                  y_label_1, e_y_strm);
    sink_stream(e_y_strm);

    processor.setWeight(weight_temp, feature_num, 1);
    processor.process(x_data, e_x_data_strm, retStrm, eRetStrm, feature_num, 1);
    gradient_processing<MType, WD, StreamN, SampleDepth>(feature_num, x_data_1, y_label_1, e_data_strm, retStrm,
                                                         eRetStrm, gradient_strm, e_gradient_strm);
    processor_agg.processAvg(gradient_strm, e_gradient_strm, feature_num);
}

} // namespace svm
} // namespace classification
} // namespace data_analytics
} // namespace xf

extern "C" void SVM(ap_uint<512>* ddr, ap_uint<512>* weight) {
#pragma HLS INTERFACE m_axi port = ddr offset = slave bundle = gmem_in1 latency = 125 num_read_outstanding = \
    32 max_read_burst_length = 32 depth = 3624
#pragma HLS INTERFACE m_axi offset = slave latency = 125 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 16 max_read_burst_length = 16 bundle = gmem_inout2 port = weight depth = 4

#pragma HLS INTERFACE s_axilite port = ddr bundle = control
#pragma HLS INTERFACE s_axilite port = weight bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    ap_uint<512> cfg = ddr[0];
    const int NUM_FEATURE = (int)cfg(63, 0);
    const int MAX_ITER = (int)cfg(127, 64);
    const ap_uint<64> rows = cfg(191, 128);
    xf::data_analytics::common::internal::f_cast<double> w;
    w.i = cfg(255, 192);
    double STEP_SIZE = w.f;
    w.i = cfg(319, 256);
    double REG_PARA = w.f;
    w.i = cfg(383, 320);
    double TOL = w.f;
    const ap_uint<64> offset = cfg(447, 384);
    const ap_uint<64> cols = cfg(511, 448);
    cfg = ddr[1];
    xf::data_analytics::common::internal::f_cast<float> ff;
    ff.i = cfg(31, 0);
    const float fraction = ff.f;
    const bool ifJump = cfg[32];
    const ap_uint<32> bucketSize = cfg(95, 64);
    const ap_uint<32> seed = cfg(127, 96);

    bool flag_tol = false;
    int i = 0;
    DATA_TYPE weight_temp[1][STREAM_NUM][SAMPLE_DEP];
#pragma HLS array_partition variable = weight_temp dim = 2 complete

    for (int i = 0; i < SAMPLE_DEP; i++) {
#pragma HLS pipeline
        ap_uint<512> temp = weight[i];
        for (ap_uint<4> j = 0; j < STREAM_NUM; j++) {
#pragma HLS unroll
            xf::data_analytics::common::internal::f_cast<DATA_TYPE> w;
            w.i = temp(j * DATA_WIDTH + DATA_WIDTH - 1, j * DATA_WIDTH);
            weight_temp[0][j][i] = w.f;
        }
    }

    xf::data_analytics::common::internal::tagTableRandomLoader<512, DATA_WIDTH, BURST_LEN, DATA_TYPE,
                                                               ap_uint<DATA_WIDTH> >
        loader;
    loader.seedInitialization(seed);
    xf::data_analytics::common::internal::s_aggr<DATA_TYPE, STREAM_NUM, SAMPLE_DEP, &funcB, 7,
                                                 xf::data_analytics::common::internal::URAM>
        processor_agg;
    int L = xf::data_analytics::common::internal::s_aggr<DATA_TYPE, STREAM_NUM, SAMPLE_DEP, &funcB, 7,
                                                         xf::data_analytics::common::internal::URAM>::L;
mainloop:
    while (i < MAX_ITER && !flag_tol) {
        DATA_TYPE sum_grediant[MAX_NUM_FEATURE];
        xf::data_analytics::classification::svm::SVM_flow<DATA_TYPE, DATA_WIDTH, STREAM_NUM, SAMPLE_DEP>(
            loader, processor_agg, weight_temp, ddr, offset, rows, cols, NUM_FEATURE, fraction, ifJump, bucketSize);
    loop_grediant:
        for (int k = 0; k < NUM_FEATURE; k++) {
#pragma HLS pipeline
            sum_grediant[k] = processor_agg.sum[k % 8][k / 8 * L];
        }
        DATA_TYPE weight_norm_diff = 0.0;
        DATA_TYPE weight_norm_new = 0.0;
        const DATA_TYPE GRA = STEP_SIZE / (xf::data_analytics::internal::m::sqrt(DATA_TYPE(i + 1.0)));
        const DATA_TYPE REG = GRA * REG_PARA;
        const DATA_TYPE REG_1 = 1.0 - REG;

    // calculate weight difference L2 norminization and update weight
    loop_update:
        for (int k = 0; k < NUM_FEATURE; k++) {
#pragma HLS pipeline
            weight_norm_diff += (weight_temp[0][k % STREAM_NUM][k / STREAM_NUM] * REG + sum_grediant[k] * GRA) *
                                (weight_temp[0][k % STREAM_NUM][k / STREAM_NUM] * REG + sum_grediant[k] * GRA);
            weight_temp[0][k % STREAM_NUM][k / STREAM_NUM] =
                weight_temp[0][k % STREAM_NUM][k / STREAM_NUM] * REG_1 - sum_grediant[k] * GRA;
            weight_norm_new +=
                weight_temp[0][k % STREAM_NUM][k / STREAM_NUM] * weight_temp[0][k % STREAM_NUM][k / STREAM_NUM];
        }
        // end condition decision

        if (weight_norm_new > 1.0) {
            if (weight_norm_diff / weight_norm_new < TOL * TOL) {
                flag_tol = true;
            }
        } else {
            if (weight_norm_diff < TOL * TOL) {
                flag_tol = true;
            }
        }

        i++;
    }
loop_writeout:
    for (int i = 0; i < SAMPLE_DEP; i++) {
#pragma HLS pipeline
        ap_uint<512> temp;
        for (ap_uint<4> j = 0; j < STREAM_NUM; j++) {
#pragma HLS unroll
            xf::data_analytics::common::internal::f_cast<DATA_TYPE> w;
            w.f = weight_temp[0][j][i];
            temp(j * DATA_WIDTH + DATA_WIDTH - 1, j * DATA_WIDTH) = w.i;
        }
        weight[i] = temp;
    }
}
