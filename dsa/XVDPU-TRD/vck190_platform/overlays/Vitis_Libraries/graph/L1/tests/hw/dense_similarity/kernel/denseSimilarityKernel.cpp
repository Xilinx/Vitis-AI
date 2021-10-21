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

template <int CHNM, int WData>
void loadSource(ap_uint<32> similarity_type,
                ap_uint<32> source_nm,

                ap_uint<WData>* weight,
                hls::stream<ap_uint<WData> >& source_weight) {
#pragma HLS INLINE off

    for (ap_uint<32> i = 0; i < source_nm; i++) {
#pragma HLS PIPELINE II = 1
        source_weight.write(weight[i]);
    }
}

// calculate loop number for loading data in DDR/HBM
template <int CHNM>
void loopNumGen(ap_uint<32> num, ap_uint<32>& loop_num) {
    ap_uint<32> base = num / CHNM;
    ap_uint<4> fraction = num % CHNM;
    loop_num = fraction == 0 ? base : (ap_uint<32>)(base + 1);
}

// generate control parameters for loading data
template <int CHNM, int WData>
void loadControl(ap_uint<32> similarity_type, ap_uint<32> vertex_num, ap_uint<32> edge_num, ap_uint<32> data_num[4]) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    std::cout << "loading control: vertex_num=" << vertex_num << " edge_num=" << edge_num << std::endl;
#endif

    loopNumGen<CHNM>(vertex_num * edge_num, data_num[0]);
    data_num[1] = data_num[0];
    data_num[2] = data_num[0];
    data_num[3] = data_num[0];
}

template <int CHNM, int WData>
void load(ap_uint<32> num, ap_uint<WData * CHNM>* data, hls::stream<ap_uint<WData * CHNM> >& strm) {
#pragma HLS INLINE off

    ap_uint<WData * CHNM> in;
    ap_uint<32> base = num(31, 6);
    ap_uint<32> fraction = num(5, 0);

#ifndef __SYNTHESIS__
    std::cout << "loading data: num=" << num << " base=" << base << " fraction=" << fraction << std::endl;
#endif

load_base:
    for (ap_uint<32> i = 0; i < base; i++) {
        for (ap_uint<32> j = 0; j < 64; j++) {
#pragma HLS PIPELINE II = 1

            ap_uint<32> addr;
            addr(31, 6) = i;
            addr(5, 0) = j;

            in = data[addr];
            strm.write(in);
        }
    }

load_fraction:
    for (ap_uint<32> i = 0; i < fraction; i++) {
#pragma HLS PIPELINE II = 1

        ap_uint<32> addr;
        addr(31, 6) = base;
        addr(5, 0) = i;

        in = data[addr];

#ifndef __SYNTHESIS__
        std::cout << "loading data: fraction=" << std::hex << in << std::dec << std::endl;
#endif

        strm.write(in);
    }
}

template <int CHNM, int WData>
void loadData(ap_uint<32> data_num[4],

              ap_uint<WData * CHNM>* data0,
              ap_uint<WData * CHNM>* data1,
              ap_uint<WData * CHNM>* data2,
              ap_uint<WData * CHNM>* data3,

              hls::stream<ap_uint<WData * CHNM> >& strm0,
              hls::stream<ap_uint<WData * CHNM> >& strm1,
              hls::stream<ap_uint<WData * CHNM> >& strm2,
              hls::stream<ap_uint<WData * CHNM> >& strm3) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    load<CHNM, WData>(data_num[0], data0, strm0);

    load<CHNM, WData>(data_num[1], data1, strm1);

    load<CHNM, WData>(data_num[2], data2, strm2);

    load<CHNM, WData>(data_num[3], data3, strm3);
}

template <int CHNM, int WData>
void loadPU(ap_uint<32> similarity_type,
            ap_uint<32> vertex_num,
            ap_uint<32> edge_num,

            ap_uint<WData * CHNM>* data0,
            ap_uint<WData * CHNM>* data1,
            ap_uint<WData * CHNM>* data2,
            ap_uint<WData * CHNM>* data3,

            hls::stream<ap_uint<WData * CHNM> >& strm0,
            hls::stream<ap_uint<WData * CHNM> >& strm1,
            hls::stream<ap_uint<WData * CHNM> >& strm2,
            hls::stream<ap_uint<WData * CHNM> >& strm3) {
#pragma HLS INLINE off

    ap_uint<32> data_num[4];
#pragma HLS array_partition variable = data_num complete

    loadControl<CHNM, WData>(similarity_type, vertex_num, edge_num, data_num);

    loadData<CHNM, WData>(data_num, data0, data1, data2, data3, strm0, strm1, strm2, strm3);
}

template <int PU>
void loadConfig(ap_uint<32>* config,
                ap_uint<32>& k,
                ap_uint<32>& source_num,
                ap_uint<32>& similarity_type,
                ap_uint<32>& data_type,
                ap_uint<32> start_id[PU],
                ap_uint<32> vertex_num[PU],
                ap_uint<32> edge_num[PU],
                hls::stream<ap_uint<32> >& config_strm) {
#pragma HLS INLINE off

    k = config[0];
    source_num = config[1];
    similarity_type = config[2];
    data_type = config[3];

    for (ap_uint<8> i = 0; i < PU; i++) {
        start_id[i] = config[4 + i];
    }

    for (ap_uint<8> i = 0; i < PU; i++) {
        vertex_num[i] = config[4 + PU + i];
    }

    for (ap_uint<8> i = 0; i < PU; i++) {
        edge_num[i] = config[4 + 2 * PU + i];
    }

    for (ap_uint<8> i = 1; i < 3 * PU + 4; i++) {
        config_strm.write(config[i]);
    }

#ifndef __SYNTHESIS__
    for (int i = 0; i < 3 * PU + 4; i++) std::cout << "config" << i << " = " << config[i] << std::endl;
#endif
}

template <int CHNM, int WData>
void loadData4PU(ap_uint<32> similarity_type,
                 ap_uint<32> vertex_num[8],
                 ap_uint<32> edge_num[8],

                 ap_uint<WData * CHNM>* dataIn00,
                 ap_uint<WData * CHNM>* dataIn01,
                 ap_uint<WData * CHNM>* dataIn02,
                 ap_uint<WData * CHNM>* dataIn03,

                 ap_uint<WData * CHNM>* dataIn10,
                 ap_uint<WData * CHNM>* dataIn11,
                 ap_uint<WData * CHNM>* dataIn12,
                 ap_uint<WData * CHNM>* dataIn13,

                 ap_uint<WData * CHNM>* dataIn20,
                 ap_uint<WData * CHNM>* dataIn21,
                 ap_uint<WData * CHNM>* dataIn22,
                 ap_uint<WData * CHNM>* dataIn23,

                 ap_uint<WData * CHNM>* dataIn30,
                 ap_uint<WData * CHNM>* dataIn31,
                 ap_uint<WData * CHNM>* dataIn32,
                 ap_uint<WData * CHNM>* dataIn33,

                 hls::stream<ap_uint<WData * CHNM> > strm0[4],
                 hls::stream<ap_uint<WData * CHNM> > strm1[4],
                 hls::stream<ap_uint<WData * CHNM> > strm2[4],
                 hls::stream<ap_uint<WData * CHNM> > strm3[4]) {
#pragma HLS INLINE

    loadPU<CHNM, WData>(similarity_type, vertex_num[0], edge_num[0], dataIn00, dataIn01, dataIn02, dataIn03, strm0[0],
                        strm1[0], strm2[0], strm3[0]);

    loadPU<CHNM, WData>(similarity_type, vertex_num[1], edge_num[1], dataIn10, dataIn11, dataIn12, dataIn13, strm0[1],
                        strm1[1], strm2[1], strm3[1]);

    loadPU<CHNM, WData>(similarity_type, vertex_num[2], edge_num[2], dataIn20, dataIn21, dataIn22, dataIn23, strm0[2],
                        strm1[2], strm2[2], strm3[2]);

    loadPU<CHNM, WData>(similarity_type, vertex_num[3], edge_num[3], dataIn30, dataIn31, dataIn32, dataIn33, strm0[3],
                        strm1[3], strm2[3], strm3[3]);
}

void feedResult(hls::stream<ap_uint<32> >& row_strm,
                hls::stream<float>& similarity_strm,
                hls::stream<bool>& end_strm,
                ap_uint<32>* result_id,
                float* similarity) {
    ap_uint<32> addr = 0;
    bool end = end_strm.read();
    while (!end) {
        ap_uint<32> row_tmp = row_strm.read();
        float similarity_tmp = similarity_strm.read();
        end = end_strm.read();

#ifndef __SYNTHESIS__
        std::cout << "addr=" << addr << " row=" << row_tmp << " similarity=" << similarity_tmp << std::endl;
#endif
        result_id[addr] = row_tmp;
        similarity[addr] = similarity_tmp;
        addr++;
    }
}

template <int CHNM, int WData, int RAM_SZ, int MAXK>
void denseSimilarityTop4PU(ap_uint<32> k,
                           ap_uint<32> source_num,
                           ap_uint<32> similarity_type,
                           ap_uint<32> data_type,

                           ap_uint<32> start_id[8],
                           ap_uint<32> vertex_num[8],
                           ap_uint<32> edge_num[8],

                           hls::stream<ap_uint<32> >& config_strm,
                           hls::stream<ap_uint<32> >& sourceWeight,

                           ap_uint<WData * CHNM>* dataIn00,
                           ap_uint<WData * CHNM>* dataIn01,
                           ap_uint<WData * CHNM>* dataIn02,
                           ap_uint<WData * CHNM>* dataIn03,

                           ap_uint<WData * CHNM>* dataIn10,
                           ap_uint<WData * CHNM>* dataIn11,
                           ap_uint<WData * CHNM>* dataIn12,
                           ap_uint<WData * CHNM>* dataIn13,

                           ap_uint<WData * CHNM>* dataIn20,
                           ap_uint<WData * CHNM>* dataIn21,
                           ap_uint<WData * CHNM>* dataIn22,
                           ap_uint<WData * CHNM>* dataIn23,

                           ap_uint<WData * CHNM>* dataIn30,
                           ap_uint<WData * CHNM>* dataIn31,
                           ap_uint<WData * CHNM>* dataIn32,
                           ap_uint<WData * CHNM>* dataIn33,

                           hls::stream<ap_uint<WData> >& resultID,
                           hls::stream<float>& similarity,
                           hls::stream<bool>& end_strm) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    const int PU = 4;

    hls::stream<ap_uint<WData * CHNM> > strm_in0[PU];
#pragma HLS stream variable = strm_in0 depth = 512
#pragma HLS array_partition variable = strm_in0 complete
#pragma HLS resource variable = strm_in0 core = FIFO_BRAM
    hls::stream<ap_uint<WData * CHNM> > strm_in1[PU];
#pragma HLS stream variable = strm_in1 depth = 512
#pragma HLS array_partition variable = strm_in1 complete
#pragma HLS resource variable = strm_in1 core = FIFO_BRAM
    hls::stream<ap_uint<WData * CHNM> > strm_in2[PU];
#pragma HLS stream variable = strm_in2 depth = 512
#pragma HLS array_partition variable = strm_in2 complete
#pragma HLS resource variable = strm_in2 core = FIFO_BRAM
    hls::stream<ap_uint<WData * CHNM> > strm_in3[PU];
#pragma HLS stream variable = strm_in3 depth = 512
#pragma HLS array_partition variable = strm_in3 complete
#pragma HLS resource variable = strm_in3 core = FIFO_BRAM

#ifndef __SYNTHESIS__
    std::cout << "loading data" << std::endl;
#endif

    loadData4PU<CHNM, WData>(similarity_type, vertex_num, edge_num, dataIn00, dataIn01, dataIn02, dataIn03, dataIn10,
                             dataIn11, dataIn12, dataIn13, dataIn20, dataIn21, dataIn22, dataIn23, dataIn30, dataIn31,
                             dataIn32, dataIn33, strm_in0, strm_in1, strm_in2, strm_in3);

    hls::stream<ap_uint<WData> > row_strm0;
#pragma HLS stream variable = row_strm0 depth = 8
#pragma HLS resource variable = row_strm0 core = FIFO_SRL
    hls::stream<float> similarity_strm0;
#pragma HLS stream variable = similarity_strm0 depth = 8
#pragma HLS resource variable = similarity_strm0 core = FIFO_SRL
    hls::stream<bool> end_strm0;
#pragma HLS stream variable = end_strm0 depth = 8
#pragma HLS resource variable = end_strm0 core = FIFO_SRL

#ifndef __SYNTHESIS__
    std::cout << "processing similarity" << std::endl;
#endif

    xf::graph::denseSimilarity<CHNM, PU, WData, RAM_SZ, true>(config_strm, sourceWeight, strm_in0, strm_in1, strm_in2,
                                                              strm_in3, row_strm0, similarity_strm0, end_strm0);

#ifndef __SYNTHESIS__
    std::cout << "sorting for topK result" << std::endl;
#endif

    xf::graph::sortTopK<float, ap_uint<32>, MAX_K>(row_strm0, similarity_strm0, end_strm0, resultID, similarity,
                                                   end_strm, k, true);
}

void denseSimilarityKernel(ap_uint<32>* config,
                           ap_uint<32>* sourceWeight,

                           ap_uint<32 * CHANNEL_NUMBER>* dataIn00,
                           ap_uint<32 * CHANNEL_NUMBER>* dataIn01,
                           ap_uint<32 * CHANNEL_NUMBER>* dataIn02,
                           ap_uint<32 * CHANNEL_NUMBER>* dataIn03,

                           ap_uint<32 * CHANNEL_NUMBER>* dataIn10,
                           ap_uint<32 * CHANNEL_NUMBER>* dataIn11,
                           ap_uint<32 * CHANNEL_NUMBER>* dataIn12,
                           ap_uint<32 * CHANNEL_NUMBER>* dataIn13,

                           ap_uint<32 * CHANNEL_NUMBER>* dataIn20,
                           ap_uint<32 * CHANNEL_NUMBER>* dataIn21,
                           ap_uint<32 * CHANNEL_NUMBER>* dataIn22,
                           ap_uint<32 * CHANNEL_NUMBER>* dataIn23,

                           ap_uint<32 * CHANNEL_NUMBER>* dataIn30,
                           ap_uint<32 * CHANNEL_NUMBER>* dataIn31,
                           ap_uint<32 * CHANNEL_NUMBER>* dataIn32,
                           ap_uint<32 * CHANNEL_NUMBER>* dataIn33,

                           ap_uint<32>* resultID,
                           float* similarity) {
    const int ext_mem_size = 4096;

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_0 port = dataIn00 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn00 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_1 port = dataIn01 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn01 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_2 port = dataIn02 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn02 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_3 port = dataIn03 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn03 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem1_0 port = dataIn10 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn10 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem1_1 port = dataIn11 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn11 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem1_2 port = dataIn12 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn12 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem1_3 port = dataIn13 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn13 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem2_0 port = dataIn20 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn20 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem2_1 port = dataIn21 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn21 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem2_2 port = dataIn22 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn22 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem2_3 port = dataIn23 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn23 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem3_0 port = dataIn30 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn30 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem3_1 port = dataIn31 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn31 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem3_2 port = dataIn32 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn32 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem3_3 port = dataIn33 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = dataIn33 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 8 max_read_burst_length = 8 bundle = gmem4_0 port = config depth = 64
#pragma HLS INTERFACE s_axilite port = config bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 8 max_read_burst_length = 8 bundle = gmem4_0 port = sourceWeight depth = 4096
#pragma HLS INTERFACE s_axilite port = sourceWeight bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 8 max_read_burst_length = 8 bundle = gmem4_0 port = resultID depth = 32
#pragma HLS INTERFACE s_axilite port = resultID bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 8 max_read_burst_length = 8 bundle = gmem4_0 port = similarity depth = 32
#pragma HLS INTERFACE s_axilite port = similarity bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    std::cout << "kernel call success" << std::endl;
#endif

#pragma HLS INLINE off

    const int PU = 4;

    ap_uint<32> k;
    ap_uint<32> source_num;
    ap_uint<32> similarity_type;
    ap_uint<32> data_type;

    ap_uint<32> start_id[PU];
#pragma HLS ARRAY_PARTITION variable = start_id complete
    ap_uint<32> vertex_num[PU];
#pragma HLS ARRAY_PARTITION variable = vertex_num complete
    ap_uint<32> edge_num[PU];
#pragma HLS ARRAY_PARTITION variable = edge_num complete
    hls::stream<ap_uint<32> > config_strm;
#pragma HLS stream variable = config_strm depth = 512
#pragma HLS resource variable = config_strm core = FIFO_BRAM

#ifndef __SYNTHESIS__
    std::cout << "loading config" << std::endl;
#endif

    loadConfig<PU>(config, k, source_num, similarity_type, data_type, start_id, vertex_num, edge_num, config_strm);

    hls::stream<ap_uint<W_DATA> > source_weight;
#pragma HLS stream variable = source_weight depth = 512
#pragma HLS resource variable = source_weight core = FIFO_BRAM

#ifndef __SYNTHESIS__
    std::cout << "loading source" << std::endl;
#endif

    loadSource<CHANNEL_NUMBER, W_DATA>(similarity_type, source_num, sourceWeight, source_weight);

    hls::stream<ap_uint<W_DATA> > row_strm;
#pragma HLS stream variable = row_strm depth = 512
#pragma HLS resource variable = row_strm core = FIFO_BRAM
    hls::stream<float> similarity_strm;
#pragma HLS stream variable = similarity_strm depth = 512
#pragma HLS resource variable = similarity_strm core = FIFO_BRAM
    hls::stream<bool> end_strm;
#pragma HLS stream variable = end_strm depth = 512
#pragma HLS resource variable = end_strm core = FIFO_SRL

    denseSimilarityTop4PU<CHANNEL_NUMBER, W_DATA, RAM_SIZE, 64>(
        k, source_num, similarity_type, data_type, start_id, vertex_num, edge_num, config_strm, source_weight,

        dataIn00, dataIn01, dataIn02, dataIn03, dataIn10, dataIn11, dataIn12, dataIn13, dataIn20, dataIn21, dataIn22,
        dataIn23, dataIn30, dataIn31, dataIn32, dataIn33,

        row_strm, similarity_strm, end_strm);

#ifndef __SYNTHESIS__
    std::cout << "returning results" << std::endl;
#endif

    feedResult(row_strm, similarity_strm, end_strm, resultID, similarity);
}
