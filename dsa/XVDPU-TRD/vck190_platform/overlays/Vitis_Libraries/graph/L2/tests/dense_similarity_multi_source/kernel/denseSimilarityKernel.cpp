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

#include "denseSimilarityKernel.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

template <int Batch, int WData>
void loadSource(ap_int<32> similarity_type,
                ap_int<32> batch_nm,
                ap_int<32> edge_nm,
                ap_int<WData>* weight,
                hls::stream<ap_int<WData> > source_weight[Batch]) {
#pragma HLS INLINE off

    ap_int<32> addr = batch_nm * edge_nm;
    for (ap_int<32> j = 0; j < Batch; j++) {
        for (ap_int<32> i = 0; i < edge_nm; i++) {
#pragma HLS PIPELINE II = 1
            source_weight[j].write(weight[addr]);
            addr++;
        }
    }
}

// calculate loop number for loading data in DDR/HBM
template <int CHNM>
void loopNumGen(ap_int<32> num, ap_int<32>& loop_num) {
    ap_int<32> base = num / CHNM;
    ap_int<4> fraction = num % CHNM;
    loop_num = fraction == 0 ? base : (ap_int<32>)(base + 1);
}

// generate control parameters for loading data
template <int CHNM, int WData>
void loadControl(ap_int<32> similarity_type, ap_int<32> vertex_num, ap_int<32> edge_num, ap_int<32> data_num[4]) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    std::cout << "loading control: vertex_num=" << vertex_num << " edge_num=" << edge_num << std::endl;
#endif

    loopNumGen<CHNM>(vertex_num * edge_num, data_num[0]);
    data_num[1] = data_num[0];
    data_num[2] = data_num[0];
    data_num[3] = data_num[0];
}

template <int CHNM, int Batch, int WData>
void load(ap_int<32> num, ap_int<WData * CHNM>* data, hls::stream<ap_int<WData * CHNM> > strm[Batch][1]) {
#pragma HLS INLINE off

    ap_int<WData * CHNM> in;
    ap_int<32> base = num(31, 6);
    ap_int<32> fraction = num(5, 0);

#ifndef __SYNTHESIS__
    std::cout << "loading data: num=" << num << " base=" << base << " fraction=" << fraction << std::endl;
#endif

load_base:
    for (ap_int<32> i = 0; i < base; i++) {
        for (ap_int<32> j = 0; j < 64; j++) {
#pragma HLS PIPELINE II = 1

            ap_int<32> addr;
            addr(31, 6) = i;
            addr(5, 0) = j;

            in = data[addr];

            for (ap_int<32> k = 0; k < Batch; k++) strm[k][0].write(in);
        }
    }

load_fraction:
    for (ap_int<32> i = 0; i < fraction; i++) {
#pragma HLS PIPELINE II = 1

        ap_int<32> addr;
        addr(31, 6) = base;
        addr(5, 0) = i;

        in = data[addr];

#ifndef __SYNTHESIS__
        std::cout << "loading data: fraction=" << std::hex << in << std::dec << std::endl;
#endif

        for (ap_int<32> k = 0; k < Batch; k++) strm[k][0].write(in);
    }
}

template <int CHNM, int Batch, int WData>
void loadData(ap_int<32> data_num[4],

              ap_int<WData * CHNM>* data0,
              ap_int<WData * CHNM>* data1,
              ap_int<WData * CHNM>* data2,
              ap_int<WData * CHNM>* data3,

              hls::stream<ap_int<WData * CHNM> > strm0[Batch][1],
              hls::stream<ap_int<WData * CHNM> > strm1[Batch][1],
              hls::stream<ap_int<WData * CHNM> > strm2[Batch][1],
              hls::stream<ap_int<WData * CHNM> > strm3[Batch][1]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    load<CHNM, Batch, WData>(data_num[0], data0, strm0);

    load<CHNM, Batch, WData>(data_num[1], data1, strm1);

    load<CHNM, Batch, WData>(data_num[2], data2, strm2);

    load<CHNM, Batch, WData>(data_num[3], data3, strm3);
}

template <int Batch>
void loadConfig(ap_int<32>* config,
                ap_int<32>& k,
                ap_int<32>& source_num,
                ap_int<32>& similarity_type,
                ap_int<32>& data_type,
                ap_int<32>& start_id,
                ap_int<32>& vertex_num,
                ap_int<32>& edge_num,
                hls::stream<ap_int<32> > config_strm[Batch]) {
#pragma HLS INLINE off

    k = config[0];
    source_num = config[1];
    similarity_type = config[2];
    data_type = config[3];
    start_id = config[4];
    vertex_num = config[5];
    edge_num = config[6];

    for (ap_int<32> j = 0; j < Batch; j++)
        for (ap_int<32> i = 1; i < 7; i++) config_strm[j].write(config[i]);

#ifndef __SYNTHESIS__
    for (int i = 0; i < 8; i++) std::cout << "config" << i << " = " << config[i] << std::endl;
#endif
}

template <int CHNM, int Batch, int WData>
void loadDataBatch(ap_int<32> similarity_type,
                   ap_int<32> vertex_num,
                   ap_int<32> edge_num,

                   ap_int<WData * CHNM>* dataIn0,
                   ap_int<WData * CHNM>* dataIn1,
                   ap_int<WData * CHNM>* dataIn2,
                   ap_int<WData * CHNM>* dataIn3,

                   hls::stream<ap_int<WData * CHNM> > strm0[Batch][1],
                   hls::stream<ap_int<WData * CHNM> > strm1[Batch][1],
                   hls::stream<ap_int<WData * CHNM> > strm2[Batch][1],
                   hls::stream<ap_int<WData * CHNM> > strm3[Batch][1]) {
#pragma HLS INLINE

    ap_int<32> data_num[4];
#pragma HLS array_partition variable = data_num complete

    loadControl<CHNM, WData>(similarity_type, vertex_num, edge_num, data_num);

    loadData<CHNM, Batch, WData>(data_num, dataIn0, dataIn1, dataIn2, dataIn3, strm0, strm1, strm2, strm3);
}

template <int Batch>
void feedResult(ap_int<32> batch_nm,
                ap_int<32> topK,
                hls::stream<ap_int<32> > row_strm[Batch],
                hls::stream<float> similarity_strm[Batch],
                hls::stream<bool> end_strm[Batch],
                ap_int<32>* result_id,
                float* similarity) {
#pragma HLS INLINE off

    ap_int<32> addr = batch_nm * topK;
    for (int i = 0; i < Batch; i++) {
        bool end = end_strm[i].read();
        while (!end) {
#pragma HLS PIPELINE II = 1

            ap_int<32> row_tmp = row_strm[i].read();
            float similarity_tmp = similarity_strm[i].read();
            end = end_strm[i].read();

#ifndef __SYNTHESIS__
            std::cout << std::dec << "addr=" << addr << " row=" << row_tmp << " similarity=" << similarity_tmp
                      << std::endl;
#endif
            result_id[addr] = row_tmp;
            similarity[addr] = similarity_tmp;
            addr++;
        }
    }
}

template <int CHNM, int Batch, int WData, int RAM_SZ>
void denseSimilarityTop(ap_int<32> k,
                        ap_int<32> source_num,
                        ap_int<32> similarity_type,
                        ap_int<32> data_type,
                        ap_int<32> start_id,
                        ap_int<32> vertex_num,
                        ap_int<32> edge_num,

                        hls::stream<ap_int<32> > config_strm[Batch],
                        hls::stream<ap_int<32> > sourceWeight[Batch],

                        ap_int<WData * CHNM>* dataIn00,
                        ap_int<WData * CHNM>* dataIn01,
                        ap_int<WData * CHNM>* dataIn02,
                        ap_int<WData * CHNM>* dataIn03,

                        hls::stream<ap_int<WData> > resultID[Batch],
                        hls::stream<float> similarity[Batch],
                        hls::stream<bool> end_strm[Batch]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    const int PU = 1;

    hls::stream<ap_int<WData * CHNM> > strm_in0[Batch][PU];
#pragma HLS stream variable = strm_in0 depth = 512
#pragma HLS array_partition variable = strm_in0 complete
#pragma HLS resource variable = strm_in0 core = FIFO_BRAM
    hls::stream<ap_int<WData * CHNM> > strm_in1[Batch][PU];
#pragma HLS stream variable = strm_in1 depth = 512
#pragma HLS array_partition variable = strm_in1 complete
#pragma HLS resource variable = strm_in1 core = FIFO_BRAM
    hls::stream<ap_int<WData * CHNM> > strm_in2[Batch][PU];
#pragma HLS stream variable = strm_in2 depth = 512
#pragma HLS array_partition variable = strm_in2 complete
#pragma HLS resource variable = strm_in2 core = FIFO_BRAM
    hls::stream<ap_int<WData * CHNM> > strm_in3[Batch][PU];
#pragma HLS stream variable = strm_in3 depth = 512
#pragma HLS array_partition variable = strm_in3 complete
#pragma HLS resource variable = strm_in3 core = FIFO_BRAM

#ifndef __SYNTHESIS__
    std::cout << "loading data" << std::endl;
#endif

    loadDataBatch<CHNM, Batch, WData>(similarity_type, vertex_num, edge_num, dataIn00, dataIn01, dataIn02, dataIn03,
                                      strm_in0, strm_in1, strm_in2, strm_in3);

    hls::stream<ap_int<WData> > row_strm0[Batch];
#pragma HLS stream variable = row_strm0 depth = 8
#pragma HLS array_partition variable = row_strm0 complete
#pragma HLS resource variable = row_strm0 core = FIFO_SRL
    hls::stream<float> similarity_strm0[Batch];
#pragma HLS stream variable = similarity_strm0 depth = 8
#pragma HLS array_partition variable = similarity_strm0 complete
#pragma HLS resource variable = similarity_strm0 core = FIFO_SRL
    hls::stream<bool> end_strm0[Batch];
#pragma HLS stream variable = end_strm0 depth = 8
#pragma HLS array_partition variable = end_strm0 complete
#pragma HLS resource variable = end_strm0 core = FIFO_SRL

    for (ap_int<32> i = 0; i < Batch; i++) {
#pragma HLS UNROLL

#ifndef __SYNTHESIS__
        std::cout << "processing similarity batch" << i << std::endl;
#endif

        xf::graph::denseSimilarity<CHNM, PU, WData, RAM_SZ>(config_strm[i], sourceWeight[i], strm_in0[i], strm_in1[i],
                                                            strm_in2[i], strm_in3[i], row_strm0[i], similarity_strm0[i],
                                                            end_strm0[i]);

#ifndef __SYNTHESIS__
        std::cout << "sorting for topK result" << std::endl;
#endif

        xf::graph::sortTopK<float, ap_int<32>, MAX_K>(row_strm0[i], similarity_strm0[i], end_strm0[i], resultID[i],
                                                      similarity[i], end_strm[i], k, true);
    }
}

extern "C" void denseSimilarityKernel(ap_int<32>* config,
                                      ap_int<32>* sourceWeight,

                                      ap_int<32 * CHANNEL_NUMBER>* dataIn0,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn1,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn2,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn3,

                                      ap_int<32>* resultID,
                                      float* similarity) {
    const int ext_mem_size = EXT_MEM_SZ;

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_0 port = dataIn0 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn0 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_1 port = dataIn1 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn1 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_2 port = dataIn2 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn2 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_3 port = dataIn3 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn3 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 8 max_read_burst_length = 8 bundle = gmem1_0 port = config depth = 64
#pragma HLS INTERFACE s_axilite port = config bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 8 max_read_burst_length = 8 bundle = gmem1_0 port = sourceWeight depth = 65536
#pragma HLS INTERFACE s_axilite port = sourceWeight bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 8 max_read_burst_length = 8 bundle = gmem1_0 port = resultID depth = 65536
#pragma HLS INTERFACE s_axilite port = resultID bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 8 max_read_burst_length = 8 bundle = gmem1_0 port = similarity depth = 65536
#pragma HLS INTERFACE s_axilite port = similarity bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    std::cout << "kernel call success" << std::endl;
#endif

#pragma HLS INLINE off

    const int PU = 1;
    const int BATCH = BATCH_NUMBER;

    ap_int<32> k;
    ap_int<32> source_num;
    ap_int<32> similarity_type;
    ap_int<32> data_type;
    ap_int<32> start_id;
    ap_int<32> vertex_num;
    ap_int<32> edge_num;

    ap_int<32> batch_num = config[7];
    ap_int<32> batch_cnt = 0;

batch_loop:
    while (batch_cnt < batch_num) {
        hls::stream<ap_int<32> > config_strm[BATCH];
#pragma HLS stream variable = config_strm depth = 8
#pragma HLS array_partition variable = config_strm complete
#pragma HLS resource variable = config_strm core = FIFO_SRL

#ifndef __SYNTHESIS__
        std::cout << "loading config" << std::endl;
#endif

        loadConfig<BATCH>(config, k, source_num, similarity_type, data_type, start_id, vertex_num, edge_num,
                          config_strm);

        hls::stream<ap_int<W_DATA> > source_weight[BATCH];
#pragma HLS stream variable = source_weight depth = 1024
#pragma HLS array_partition variable = source_weight complete
#pragma HLS resource variable = source_weight core = FIFO_BRAM

#ifndef __SYNTHESIS__
        std::cout << "loading source" << std::endl;
#endif

        loadSource<BATCH, W_DATA>(similarity_type, batch_cnt, edge_num, sourceWeight, source_weight);

        hls::stream<ap_int<W_DATA> > row_strm[BATCH];
#pragma HLS stream variable = row_strm depth = 512
#pragma HLS resource variable = row_strm core = FIFO_BRAM
        hls::stream<float> similarity_strm[BATCH];
#pragma HLS stream variable = similarity_strm depth = 512
#pragma HLS resource variable = similarity_strm core = FIFO_BRAM
        hls::stream<bool> end_strm[BATCH];
#pragma HLS stream variable = end_strm depth = 512
#pragma HLS resource variable = end_strm core = FIFO_SRL

        denseSimilarityTop<CHANNEL_NUMBER, BATCH, W_DATA, RAM_SIZE>(
            k, source_num, similarity_type, data_type, start_id, vertex_num, edge_num, config_strm, source_weight,
            dataIn0, dataIn1, dataIn2, dataIn3, row_strm, similarity_strm, end_strm);

#ifndef __SYNTHESIS__
        std::cout << "returning results" << std::endl;
#endif
        feedResult<BATCH>(batch_cnt, k, row_strm, similarity_strm, end_strm, resultID, similarity);

        batch_cnt += BATCH;
    }
}
