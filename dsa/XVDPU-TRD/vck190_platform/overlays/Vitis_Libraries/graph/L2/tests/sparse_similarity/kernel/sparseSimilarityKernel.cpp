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

#include "sparseSimilarityKernel.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

template <int CHNM, int WData>
void loadSource(ap_uint<32> similarity_type,
                ap_uint<32> source_nm,

                ap_uint<WData>* indice,
                ap_uint<WData>* weight,

                hls::stream<ap_uint<WData> >& source_indice,
                hls::stream<ap_uint<WData> >& source_weight) {
#pragma HLS INLINE off

    for (ap_uint<32> i = 0; i < source_nm; i++) {
#pragma HLS PIPELINE II = 1
        source_indice.write(indice[i]);
        if (!(similarity_type == xf::graph::enums::JACCARD_SIMILARITY)) source_weight.write(weight[i]);
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
void loadControl(ap_uint<32> similarity_type, ap_uint<32> vertex_num, ap_uint<32> edge_num, ap_uint<32> data_num[3]) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    std::cout << "loading control: vertex_num=" << vertex_num << " edge_num=" << edge_num << std::endl;
#endif

    loopNumGen<CHNM>(vertex_num + 1, data_num[0]);
    loopNumGen<CHNM>(edge_num, data_num[1]);
    if (similarity_type == xf::graph::enums::JACCARD_SIMILARITY)
        data_num[2] = 0;
    else
        data_num[2] = data_num[1];
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
void loadData(ap_uint<32> data_num[3],

              ap_uint<WData * CHNM>* data0,
              ap_uint<WData * CHNM>* data1,
              ap_uint<WData * CHNM>* data2,

              hls::stream<ap_uint<WData * CHNM> >& strm0,
              hls::stream<ap_uint<WData * CHNM> >& strm1,
              hls::stream<ap_uint<WData * CHNM> >& strm2) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    load<CHNM, WData>(data_num[0], data0, strm0);

    load<CHNM, WData>(data_num[1], data1, strm1);

    load<CHNM, WData>(data_num[2], data2, strm2);
}

template <int CHNM, int WData>
void loadPU(ap_uint<32> similarity_type,
            ap_uint<32> vertex_num,
            ap_uint<32> edge_num,

            ap_uint<WData * CHNM>* data0,
            ap_uint<WData * CHNM>* data1,
            ap_uint<WData * CHNM>* data2,

            hls::stream<ap_uint<WData * CHNM> >& strm0,
            hls::stream<ap_uint<WData * CHNM> >& strm1,
            hls::stream<ap_uint<WData * CHNM> >& strm2) {
#pragma HLS INLINE off

    ap_uint<32> data_num[3];
#pragma HLS array_partition variable = data_num complete

    loadControl<CHNM, WData>(similarity_type, vertex_num, edge_num, data_num);

    loadData<CHNM, WData>(data_num, data0, data1, data2, strm0, strm1, strm2);
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
void loadData8PU(ap_uint<32> similarity_type,
                 ap_uint<32> vertex_num[8],
                 ap_uint<32> edge_num[8],

                 ap_uint<WData * CHNM>* offset0,
                 ap_uint<WData * CHNM>* indice0,
                 ap_uint<WData * CHNM>* weight0,

                 ap_uint<WData * CHNM>* offset1,
                 ap_uint<WData * CHNM>* indice1,
                 ap_uint<WData * CHNM>* weight1,

                 ap_uint<WData * CHNM>* offset2,
                 ap_uint<WData * CHNM>* indice2,
                 ap_uint<WData * CHNM>* weight2,

                 ap_uint<WData * CHNM>* offset3,
                 ap_uint<WData * CHNM>* indice3,
                 ap_uint<WData * CHNM>* weight3,

                 ap_uint<WData * CHNM>* offset4,
                 ap_uint<WData * CHNM>* indice4,
                 ap_uint<WData * CHNM>* weight4,

                 ap_uint<WData * CHNM>* offset5,
                 ap_uint<WData * CHNM>* indice5,
                 ap_uint<WData * CHNM>* weight5,

                 ap_uint<WData * CHNM>* offset6,
                 ap_uint<WData * CHNM>* indice6,
                 ap_uint<WData * CHNM>* weight6,

                 ap_uint<WData * CHNM>* offset7,
                 ap_uint<WData * CHNM>* indice7,
                 ap_uint<WData * CHNM>* weight7,

                 hls::stream<ap_uint<WData * CHNM> > offset_strm[8],
                 hls::stream<ap_uint<WData * CHNM> > indice_strm[8],
                 hls::stream<ap_uint<WData * CHNM> > weight_strm[8]) {
#pragma HLS INLINE

    loadPU<CHNM, WData>(similarity_type, vertex_num[0], edge_num[0], offset0, indice0, weight0, offset_strm[0],
                        indice_strm[0], weight_strm[0]);

    loadPU<CHNM, WData>(similarity_type, vertex_num[1], edge_num[1], offset1, indice1, weight1, offset_strm[1],
                        indice_strm[1], weight_strm[1]);

    loadPU<CHNM, WData>(similarity_type, vertex_num[2], edge_num[2], offset2, indice2, weight2, offset_strm[2],
                        indice_strm[2], weight_strm[2]);

    loadPU<CHNM, WData>(similarity_type, vertex_num[3], edge_num[3], offset3, indice3, weight3, offset_strm[3],
                        indice_strm[3], weight_strm[3]);

    loadPU<CHNM, WData>(similarity_type, vertex_num[4], edge_num[4], offset4, indice4, weight4, offset_strm[4],
                        indice_strm[4], weight_strm[4]);

    loadPU<CHNM, WData>(similarity_type, vertex_num[5], edge_num[5], offset5, indice5, weight5, offset_strm[5],
                        indice_strm[5], weight_strm[5]);

    loadPU<CHNM, WData>(similarity_type, vertex_num[6], edge_num[6], offset6, indice6, weight6, offset_strm[6],
                        indice_strm[6], weight_strm[6]);

    loadPU<CHNM, WData>(similarity_type, vertex_num[7], edge_num[7], offset7, indice7, weight7, offset_strm[7],
                        indice_strm[7], weight_strm[7]);
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
void sparseSimilarityTop8PU(ap_uint<32> k,
                            ap_uint<32> source_num,
                            ap_uint<32> similarity_type,
                            ap_uint<32> data_type,

                            ap_uint<32> start_id[8],
                            ap_uint<32> vertex_num[8],
                            ap_uint<32> edge_num[8],

                            hls::stream<ap_uint<32> >& config_strm,
                            hls::stream<ap_uint<32> >& sourceIndice,
                            hls::stream<ap_uint<32> >& sourceWeight,

                            ap_uint<WData * CHNM>* offset0,
                            ap_uint<WData * CHNM>* indice0,
                            ap_uint<WData * CHNM>* weight0,

                            ap_uint<WData * CHNM>* offset1,
                            ap_uint<WData * CHNM>* indice1,
                            ap_uint<WData * CHNM>* weight1,

                            ap_uint<WData * CHNM>* offset2,
                            ap_uint<WData * CHNM>* indice2,
                            ap_uint<WData * CHNM>* weight2,

                            ap_uint<WData * CHNM>* offset3,
                            ap_uint<WData * CHNM>* indice3,
                            ap_uint<WData * CHNM>* weight3,

                            ap_uint<WData * CHNM>* offset4,
                            ap_uint<WData * CHNM>* indice4,
                            ap_uint<WData * CHNM>* weight4,

                            ap_uint<WData * CHNM>* offset5,
                            ap_uint<WData * CHNM>* indice5,
                            ap_uint<WData * CHNM>* weight5,

                            ap_uint<WData * CHNM>* offset6,
                            ap_uint<WData * CHNM>* indice6,
                            ap_uint<WData * CHNM>* weight6,

                            ap_uint<WData * CHNM>* offset7,
                            ap_uint<WData * CHNM>* indice7,
                            ap_uint<WData * CHNM>* weight7,

                            hls::stream<ap_uint<WData> >& resultID,
                            hls::stream<float>& similarity,
                            hls::stream<bool>& end_strm) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    const int PU = 8;

    hls::stream<ap_uint<WData * CHNM> > offset_strm1[PU];
#pragma HLS stream variable = offset_strm1 depth = 512
#pragma HLS array_partition variable = offset_strm1 complete
#pragma HLS resource variable = offset_strm1 core = FIFO_BRAM
    hls::stream<ap_uint<WData * CHNM> > indice_strm1[PU];
#pragma HLS stream variable = indice_strm1 depth = 512
#pragma HLS array_partition variable = indice_strm1 complete
#pragma HLS resource variable = indice_strm1 core = FIFO_BRAM
    hls::stream<ap_uint<WData * CHNM> > weight_strm1[PU];
#pragma HLS stream variable = weight_strm1 depth = 512
#pragma HLS array_partition variable = weight_strm1 complete
#pragma HLS resource variable = weight_strm1 core = FIFO_BRAM

#ifndef __SYNTHESIS__
    std::cout << "loading data" << std::endl;
#endif

    loadData8PU<CHNM, WData>(similarity_type, vertex_num, edge_num, offset0, indice0, weight0, offset1, indice1,
                             weight1, offset2, indice2, weight2, offset3, indice3, weight3, offset4, indice4, weight4,
                             offset5, indice5, weight5, offset6, indice6, weight6, offset7, indice7, weight7,
                             offset_strm1, indice_strm1, weight_strm1);

    hls::stream<ap_uint<WData> > row_strm2;
#pragma HLS stream variable = row_strm2 depth = 8
#pragma HLS resource variable = row_strm2 core = FIFO_SRL
    hls::stream<float> similarity_strm2;
#pragma HLS stream variable = similarity_strm2 depth = 8
#pragma HLS resource variable = similarity_strm2 core = FIFO_SRL
    hls::stream<bool> end_strm2;
#pragma HLS stream variable = end_strm2 depth = 8
#pragma HLS resource variable = end_strm2 core = FIFO_SRL

#ifndef __SYNTHESIS__
    std::cout << "processing similarity" << std::endl;
#endif

    xf::graph::sparseSimilarity<CHNM, PU, WData, RAM_SZ, true>(config_strm, sourceIndice, sourceWeight, offset_strm1,
                                                               indice_strm1, weight_strm1, row_strm2, similarity_strm2,
                                                               end_strm2);

#ifndef __SYNTHESIS__
    std::cout << "sorting for topK result" << std::endl;
#endif

    xf::graph::sortTopK<float, ap_uint<32>, MAX_K>(row_strm2, similarity_strm2, end_strm2, resultID, similarity,
                                                   end_strm, k, true);
}

extern "C" void sparseSimilarityKernel(ap_uint<32>* config,
                                       ap_uint<32>* sourceIndice,
                                       ap_uint<32>* sourceWeight,

                                       ap_uint<32 * CHANNEL_NUMBER>* offsetCSR0,
                                       ap_uint<32 * CHANNEL_NUMBER>* indiceCSR0,
                                       ap_uint<32 * CHANNEL_NUMBER>* weight0,

                                       ap_uint<32 * CHANNEL_NUMBER>* offsetCSR1,
                                       ap_uint<32 * CHANNEL_NUMBER>* indiceCSR1,
                                       ap_uint<32 * CHANNEL_NUMBER>* weight1,

                                       ap_uint<32 * CHANNEL_NUMBER>* offsetCSR2,
                                       ap_uint<32 * CHANNEL_NUMBER>* indiceCSR2,
                                       ap_uint<32 * CHANNEL_NUMBER>* weight2,

                                       ap_uint<32 * CHANNEL_NUMBER>* offsetCSR3,
                                       ap_uint<32 * CHANNEL_NUMBER>* indiceCSR3,
                                       ap_uint<32 * CHANNEL_NUMBER>* weight3,

                                       ap_uint<32 * CHANNEL_NUMBER>* offsetCSR4,
                                       ap_uint<32 * CHANNEL_NUMBER>* indiceCSR4,
                                       ap_uint<32 * CHANNEL_NUMBER>* weight4,

                                       ap_uint<32 * CHANNEL_NUMBER>* offsetCSR5,
                                       ap_uint<32 * CHANNEL_NUMBER>* indiceCSR5,
                                       ap_uint<32 * CHANNEL_NUMBER>* weight5,

                                       ap_uint<32 * CHANNEL_NUMBER>* offsetCSR6,
                                       ap_uint<32 * CHANNEL_NUMBER>* indiceCSR6,
                                       ap_uint<32 * CHANNEL_NUMBER>* weight6,

                                       ap_uint<32 * CHANNEL_NUMBER>* offsetCSR7,
                                       ap_uint<32 * CHANNEL_NUMBER>* indiceCSR7,
                                       ap_uint<32 * CHANNEL_NUMBER>* weight7,

                                       ap_uint<32>* resultID,
                                       float* similarity) {
    const int ext_mem_size = EXT_MEM_SZ;

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_0 port = offsetCSR0 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = offsetCSR0 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_1 port = indiceCSR0 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = indiceCSR0 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_2 port = weight0 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = weight0 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem1_0 port = offsetCSR1 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = offsetCSR1 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem1_1 port = indiceCSR1 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = indiceCSR1 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem1_2 port = weight1 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = weight1 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem2_0 port = offsetCSR2 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = offsetCSR2 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem2_1 port = indiceCSR2 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = indiceCSR2 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem2_2 port = weight2 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = weight2 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem3_0 port = offsetCSR3 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = offsetCSR3 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem3_1 port = indiceCSR3 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = indiceCSR3 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem3_2 port = weight3 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = weight3 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem4_0 port = offsetCSR4 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = offsetCSR4 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem4_1 port = indiceCSR4 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = indiceCSR4 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem4_2 port = weight4 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = weight4 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem5_0 port = offsetCSR5 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = offsetCSR5 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem5_1 port = indiceCSR5 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = indiceCSR5 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem5_2 port = weight5 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = weight5 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem6_0 port = offsetCSR6 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = offsetCSR6 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem6_1 port = indiceCSR6 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = indiceCSR6 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem6_2 port = weight6 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = weight6 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem7_0 port = offsetCSR7 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = offsetCSR7 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem7_1 port = indiceCSR7 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = indiceCSR7 bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem7_2 port = weight7 depth = ext_mem_size
#pragma HLS INTERFACE s_axilite port = weight7 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 8 max_read_burst_length = 8 bundle = gmem8_0 port = config depth = 64
#pragma HLS INTERFACE s_axilite port = config bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 8 max_read_burst_length = 8 bundle = gmem8_0 port = sourceIndice depth = 65536
#pragma HLS INTERFACE s_axilite port = sourceIndice bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 8 max_read_burst_length = 8 bundle = gmem8_0 port = sourceWeight depth = 65536
#pragma HLS INTERFACE s_axilite port = sourceWeight bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 8 max_read_burst_length = 8 bundle = gmem8_0 port = resultID depth = 128
#pragma HLS INTERFACE s_axilite port = resultID bundle = control
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 8 max_read_burst_length = 8 bundle = gmem8_0 port = similarity depth = 128
#pragma HLS INTERFACE s_axilite port = similarity bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    std::cout << "kernel call success" << std::endl;
#endif

#pragma HLS INLINE off

    const int PU = 8;

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

    hls::stream<ap_uint<W_DATA> > source_indice;
#pragma HLS stream variable = source_indice depth = 512
#pragma HLS resource variable = source_indice core = FIFO_BRAM
    hls::stream<ap_uint<W_DATA> > source_weight;
#pragma HLS stream variable = source_weight depth = 512
#pragma HLS resource variable = source_weight core = FIFO_BRAM

#ifndef __SYNTHESIS__
    std::cout << "loading source" << std::endl;
#endif

    loadSource<CHANNEL_NUMBER, W_DATA>(similarity_type, source_num, sourceIndice, sourceWeight, source_indice,
                                       source_weight);

    hls::stream<ap_uint<W_DATA> > row_strm;
#pragma HLS stream variable = row_strm depth = 512
#pragma HLS resource variable = row_strm core = FIFO_BRAM
    hls::stream<float> similarity_strm;
#pragma HLS stream variable = similarity_strm depth = 512
#pragma HLS resource variable = similarity_strm core = FIFO_BRAM
    hls::stream<bool> end_strm;
#pragma HLS stream variable = end_strm depth = 512
#pragma HLS resource variable = end_strm core = FIFO_SRL

    sparseSimilarityTop8PU<CHANNEL_NUMBER, W_DATA, RAM_SIZE, 64>(
        k, source_num, similarity_type, data_type, start_id, vertex_num, edge_num, config_strm, source_indice,
        source_weight,

        offsetCSR0, indiceCSR0, weight0, offsetCSR1, indiceCSR1, weight1, offsetCSR2, indiceCSR2, weight2, offsetCSR3,
        indiceCSR3, weight3, offsetCSR4, indiceCSR4, weight4, offsetCSR5, indiceCSR5, weight5, offsetCSR6, indiceCSR6,
        weight6, offsetCSR7, indiceCSR7, weight7,

        row_strm, similarity_strm, end_strm);

#ifndef __SYNTHESIS__
    std::cout << "returning results" << std::endl;
#endif

    feedResult(row_strm, similarity_strm, end_strm, resultID, similarity);
}
