
/*
 * Copyright 2020 Xilinx, Inc.
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
 * @file sparse_similarity.hpp
 *
 */

#ifndef __XF_GRAPH_SPARSE_SIMILARITY_HPP_
#define __XF_GRAPH_SPARSE_SIMILARITY_HPP_

#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"

#include "similarity/types.hpp"
#include "similarity/enums.hpp"
#include "similarity/general_similarity.hpp"

#ifndef __SYNTHESIS__

#ifdef DEBUG_SIMILARITY

#define DEBUG_CSR_TO_COO true

#endif
#endif

namespace xf {

namespace graph {

namespace internal {

namespace sparse_similarity {

template <int PU>
void load_config(hls::stream<ap_uint<32> >& config,

                 ap_uint<32>& source_num,
                 ap_uint<32>& similarity_type,
                 ap_uint<32>& data_type,

                 ap_uint<32> start_id[PU],
                 ap_uint<32> vertex_nm[PU],
                 ap_uint<32> edge_nm[PU]) {
#pragma HLS INLINE off

    source_num = config.read();
    similarity_type = config.read();
    data_type = config.read();

    for (ap_uint<8> i = 0; i < PU; i++) start_id[i] = config.read();

    for (ap_uint<8> i = 0; i < PU; i++) vertex_nm[i] = config.read();

    for (ap_uint<8> i = 0; i < PU; i++) edge_nm[i] = config.read();
}

template <int PU, int CHNM, int RAM_SZ, int WData, bool EN_FLOAT>
void load_source_vertex32(ap_uint<WData> num,
                          ap_uint<WData> similarity_type,
                          ap_uint<WData> data_type,

                          hls::stream<ap_uint<WData> >& source_col_id,
                          hls::stream<ap_uint<WData> >& source_weight,
#ifndef __SYNTHESIS__
                          ap_uint<WData>* col_vector[PU],
                          ap_uint<WData>* sparse_weight_vector[PU],
#else
                          ap_uint<WData> col_vector[PU][1 << RAM_SZ],
                          ap_uint<WData> sparse_weight_vector[PU][1 << RAM_SZ],
#endif
                          ap_uint<WData>& norm,
                          ap_uint<WData>& max_col) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    // define URAM structure
    const int RAM_Size = 1 << RAM_SZ;

    uint32_t square_uint32 = 0;
    float square_float = 0;

    uint32_t accum_uint32 = 0;
    float accum_float = 0;

    uint32_t norm_uint32 = 0;
    float norm_float = 0;

    ap_uint<WData> max = 0;
    ap_uint<WData> cnt = 0;

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "load source indice" << std::endl;
#endif
#endif

Load_indices:
    for (int j = 0; j < num; j++) {
#pragma HLS PIPELINE

        ap_uint<WData> col = source_col_id.read();

        if (col > max) max = col;
        for (ap_uint<8> i = 0; i < PU; i++) {
#pragma HLS unroll
            col_vector[i][j] = col;
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "source col[" << j << "]=" << col << std::endl;
#endif
#endif
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "load source weight" << std::endl;
#endif
#endif

    ap_uint<4> i = 0;
    ap_uint<32 * CHNM> weight_tmp = 0;
    ap_uint<WData> addr = 0;
Load_weight:
    for (int j = 0; j < num; j++) {
#pragma HLS PIPELINE

        ap_uint<32> weight;
        float weight_float;
        if (i == 0) weight_tmp = 0;
        if (!(similarity_type == enums::JACCARD_SIMILARITY)) {
            weight = source_weight.read();

            if (EN_FLOAT) {
                if (data_type == enums::FLOAT_TYPE)
                    weight_float = bitsToFloat<uint32_t, float>((uint32_t)weight);
                else
                    weight_float = (float)weight;

                weight_tmp((i + 1) * 32 - 1, i * 32) = floatToBits<float, uint32_t>(weight_float);
                square_float = weight_float * weight_float;
                accum_float += square_float;
            } else {
                weight_tmp((i + 1) * 32 - 1, i * 32) = weight;
                square_uint32 = weight * weight;
                accum_uint32 += square_uint32;
            }
        }

        for (ap_uint<8> k = 0; k < PU; k++) {
#pragma HLS unroll
            if (EN_FLOAT) {
                if (similarity_type != enums::JACCARD_SIMILARITY) {
                    sparse_weight_vector[k][j] = floatToBits<float, uint32_t>(weight_float);
                } else {
                    // jaccard sparse
                    sparse_weight_vector[k][j] = floatToBits<float, uint32_t>(1.0);

#ifndef __SYNTHESIS__
#ifdef DEBUG
                    if (k == 0) std::cout << "set weight to 1.0" << std::endl;
#endif
#endif
                }
            } else {
                if (similarity_type != enums::JACCARD_SIMILARITY) {
                    sparse_weight_vector[k][j] = weight;
                } else {
                    sparse_weight_vector[k][j] = 1;

#ifndef __SYNTHESIS__
#ifdef DEBUG
                    if (k == 0) std::cout << "set weight to 1" << std::endl;
#endif
#endif
                }
            }
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG
        if (EN_FLOAT)
            std::cout << std::dec << "source sparse weight[" << j
                      << "]=" << bitsToFloat<uint32_t, float>((uint32_t)weight) << std::endl;
        else
            std::cout << std::dec << "source sparse weight[" << j << "]=" << weight << std::endl;
#endif
#endif
    }

    // get normalization of source vertex
    if (similarity_type == enums::JACCARD_SIMILARITY) {
        if (EN_FLOAT) {
            norm = floatToBits<float, uint32_t>((float)num);
        } else {
            norm = num;
        }
    } else {
        if (EN_FLOAT) {
            norm_float = hls::sqrt(accum_float);
        } else {
            norm_float = hls::sqrt((float)accum_uint32);
        }
        norm = floatToBits<float, uint32_t>(norm_float);
    }

    max_col = max;

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "square_float=" << square_float << std::endl;
    std::cout << "square_uint=" << square_uint32 << std::endl;
    std::cout << "accum_float=" << accum_float << std::endl;
    std::cout << "accum_uint=" << accum_uint32 << std::endl;
    std::cout << "norm_float=" << norm_float << std::endl;
    std::cout << "norm_uint=" << norm_uint32 << std::endl;
    std::cout << "source_norm=" << norm << std::endl;
#endif
#endif
}

template <int PU, int CHNM, int RAM_SZ, int WData, bool EN_DOUBLE>
void load_source_vertex64(ap_uint<WData> num,
                          ap_uint<WData> similarity_type,
                          ap_uint<WData> data_type,

                          hls::stream<ap_uint<WData> >& source_col_id,
                          hls::stream<ap_uint<WData> >& source_weight,
#ifndef __SYNTHESIS__
                          ap_uint<WData>* col_vector[PU],
                          ap_uint<WData>* sparse_weight_vector[PU],
#else
                          ap_uint<WData> col_vector[PU][1 << RAM_SZ],
                          ap_uint<WData> sparse_weight_vector[PU][1 << RAM_SZ],
#endif
                          ap_uint<WData>& norm,
                          ap_uint<WData>& max_col) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    // define URAM structure
    const int RAM_Size = 1 << RAM_SZ;

    uint64_t square_uint64 = 0;
    double square_double = 0;

    uint32_t accum_uint32 = 0;
    double accum_double = 0;

    uint64_t norm_uint64 = 0;
    double norm_double = 0;

    ap_uint<WData> max = 0;
    ap_uint<WData> cnt = 0;

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "load source indice" << std::endl;
#endif
#endif

Load_indices:
    for (int j = 0; j < num; j++) {
#pragma HLS PIPELINE

        ap_uint<WData> col = source_col_id.read();

        if (col > max) max = col;
        for (ap_uint<8> i = 0; i < PU; i++) {
#pragma HLS unroll
            col_vector[i][j] = col;
        }
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "load source weight" << std::endl;
#endif
#endif

    ap_uint<4> i = 0;
    ap_uint<64 * CHNM> weight_tmp = 0;
    ap_uint<WData> addr = 0;
Load_weight:
    for (int j = 0; j < num; j++) {
#pragma HLS PIPELINE

        ap_uint<64> weight;
        double weight_double;
        if (i == 0) weight_tmp = 0;

        if (!(similarity_type == enums::JACCARD_SIMILARITY)) {
            weight = source_weight.read();

            if (weight != 0) {
                cnt++;
            }

            if (EN_DOUBLE) {
                if (data_type == enums::DOUBLE_TYPE)
                    weight_double = bitsToFloat<uint64_t, double>((uint64_t)weight);
                else
                    weight_double = weight;

                weight_tmp((i + 1) * 64 - 1, i * 64) = floatToBits<double, uint64_t>(weight_double);
                square_double = weight_double * weight_double;
                accum_double += square_double;
            } else {
                weight_tmp((i + 1) * 64 - 1, i * 64) = weight;
                square_uint64 = weight * weight;
                accum_uint32 += square_uint64;
            }
        }

        for (ap_uint<8> k = 0; k < PU; k++) {
            if (EN_DOUBLE) {
                if (similarity_type != enums::JACCARD_SIMILARITY) {
                    // cosine sparse
                    sparse_weight_vector[k][j] = floatToBits<double, uint64_t>(weight_double);
                } else {
                    // jaccard sparse
                    sparse_weight_vector[k][j] = floatToBits<double, uint64_t>(1.0);
                }
            } else {
                if (similarity_type != enums::JACCARD_SIMILARITY) {
                    sparse_weight_vector[k][j] = weight;
                } else {
                    sparse_weight_vector[k][j] = 1;
                }
            }
        }
    }

    // get normalization of base vertex
    if (similarity_type == enums::JACCARD_SIMILARITY) {
        if (EN_DOUBLE) {
            norm = floatToBits<double, uint64_t>((double)num);
        } else {
            norm = num;
        }
    } else {
        if (EN_DOUBLE) {
            norm_double = hls::sqrt(accum_double); // sqrt
        } else {
            norm_double = hls::sqrt((double)accum_uint32); // no uint64 sqrt
        }
        norm = floatToBits<double, uint64_t>(norm_double);
    }
    max_col = max;
}

template <int CHNM, int WData, bool EN_FLOAT_POINT>
void findCorrelationSparse(hls::stream<ap_uint<WData> >& row_id,
                           hls::stream<ap_uint<WData> > col_id[CHNM],
                           hls::stream<ap_uint<WData> > weight_in[CHNM],
                           hls::stream<ap_uint<CHNM> >& compute_enable_in,
                           hls::stream<bool>& strm_in_end,

                           ap_uint<WData> similarity_type,
                           ap_uint<WData> data_type,
                           ap_uint<WData> source_num,
                           ap_uint<WData> max_col,
                           ap_uint<WData>* col_vector,
                           ap_uint<WData>* weight_vector,

                           hls::stream<ap_uint<WData> >& row_id_out,
                           hls::stream<ap_uint<WData> > weight_out[CHNM],
                           hls::stream<ap_uint<CHNM> >& compute_enable_out,
                           hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
#ifdef DEBUG
    ap_uint<WData> rcnt = 0;
#endif
#endif

    ap_uint<CHNM> enable;
    ap_uint<CHNM> terminate = 0;
    ap_uint<CHNM> next_terminate = 0;
    bool search_end = true;

    ap_uint<WData> base = source_num / CHNM;
    ap_uint<16> fraction = source_num % CHNM;
    ap_uint<WData> range = fraction == 0 ? (ap_uint<WData>)(base - 1) : base;

    ap_uint<WData> row;
    ap_uint<WData> col[CHNM];
#pragma HLS ARRAY_PARTITION variable = col complete
    ap_int<WData> current_weight[CHNM];
#pragma HLS ARRAY_PARTITION variable = current_weight complete

    ap_uint<WData> source_col;
    ap_int<WData> source_weight;

    ap_int<WData> result_weight[CHNM];
#pragma HLS ARRAY_PARTITION variable = result_weight complete

    bool strm_end = strm_in_end.read();
    ap_uint<WData> search_idx = 0;
    while (!strm_end || !search_end) {
#pragma HLS PIPELINE

        if (search_end) {
            row = row_id.read();
            enable = compute_enable_in.read();

            for (ap_uint<8> j = 0; j < CHNM; j++) {
#pragma HLS UNROLL
                if (enable[j] == 1) {
                    col[j] = col_id[j].read();
                    current_weight[j] = weight_in[j].read();
                    terminate[j] = 0;

#ifndef __SYNTHESIS__
#ifdef DEBUG
                    std::cout << "in cnt=" << rcnt << " row=" << row << " col[" << j << "]=" << col[j] << " weight["
                              << j << "]=" << current_weight[j] << std::endl;
                    rcnt++;
#endif
#endif
                } else
                    terminate[j] = 1;
            }
            strm_end = strm_in_end.read();
            search_idx = 0;
        } else {
            terminate = next_terminate;
            search_idx++;
        }

        source_col = col_vector[search_idx];
        source_weight = weight_vector[search_idx];

        bool idx_overflow = search_idx >= source_num;
        for (ap_uint<8> j = 0; j < CHNM; j++) {
#pragma HLS UNROLL
            if (terminate[j] == 0) {
                if (idx_overflow) {
                    next_terminate[j] = 1;
                    result_weight[j] = 0;
                } else if (col[j] == source_col) {
                    next_terminate[j] = 1;

                    if (EN_FLOAT_POINT && WData == 32) {
                        float float_tmp, in1, in2;
                        in1 = bitsToFloat<uint32_t, float>((uint32_t)source_weight);
                        in2 = bitsToFloat<uint32_t, float>((uint32_t)current_weight[j]);
                        float_tmp = in1 * in2;
                        result_weight[j] = floatToBits<float, uint32_t>(float_tmp);
                    } else if (EN_FLOAT_POINT && WData == 64) {
                        double double_tmp, in1, in2;
                        in1 = bitsToFloat<uint64_t, double>((uint64_t)source_weight);
                        in2 = bitsToFloat<uint64_t, double>((uint64_t)current_weight[j]);
                        double_tmp = in1 * in2;
                        result_weight[j] = floatToBits<double, uint64_t>(double_tmp);
                    } else {
                        result_weight[j] = source_weight * current_weight[j];
                    }

                } else if (col[j] < source_col) {
                    next_terminate[j] = 1;
                    result_weight[j] = 0;
                } else if (col[j] > max_col) {
                    next_terminate[j] = 1;
                    result_weight[j] = 0;
                } else {
                    next_terminate[j] = 0;
                    result_weight[j] = 0;
                }
            } else {
                next_terminate[j] = 1;
            }

#ifndef __SYNTHESIS__
#ifdef DEBUG
            if (EN_FLOAT_POINT)
                std::cout << "result[" << j << "]=" << bitsToFloat<uint32_t, float>(result_weight[j]) << std::endl;
            else
                std::cout << "result[" << j << "]=" << result_weight[j] << std::endl;
#endif
#endif
        }

        if (CHNM == 16) {
            search_end = next_terminate == 65535;
        } else if (CHNM == 8) {
            search_end = next_terminate == 255;
        } else if (CHNM == 4) {
            search_end = next_terminate == 15;
        } else if (CHNM == 2) {
            search_end = next_terminate == 3;
        } else {
            search_end = next_terminate == 1;
        }

        if (search_end) {
            row_id_out.write(row);
            compute_enable_out.write(enable);
            strm_out_end.write(false);

            for (ap_uint<8> j = 0; j < CHNM; j++) {
#pragma HLS UNROLL
                if (enable[j] == 1) {
                    weight_out[j].write(result_weight[j]);
#ifndef __SYNTHESIS__
#ifdef DEBUG
                    if (EN_FLOAT_POINT)
                        std::cout << "out row=" << row << " enable=" << enable << " weight[" << j
                                  << "]=" << bitsToFloat<uint32_t, float>(result_weight[j]) << std::endl;
                    else
                        std::cout << "out row=" << row << " enable=" << enable << " weight[" << j
                                  << "]=" << result_weight[j] << std::endl;
#endif
#endif
                }
            }
        }
    }
    strm_out_end.write(true);
}

template <int CHNM, int WData, int RAM_SZ, bool EN_FLOAT_POINT>
void similarity_processing_unit(
    // input
    hls::stream<ap_uint<WData * CHNM> >& offset_in,
    hls::stream<ap_uint<WData * CHNM> >& column_in,
    hls::stream<ap_uint<WData * CHNM> >& weight_in,

    // config
    ap_uint<32> similarity_type,
    ap_uint<32> data_type,
    ap_uint<32> row_nm,
    ap_uint<32> col_nm,
    ap_uint<32> start_id,
    ap_uint<32> source_num,
    ap_uint<WData> source_norm,
    ap_uint<32> max_col,

    // ram
    ap_uint<WData>* col_vector,
    ap_uint<WData>* sparse_weight_vector,

    // output
    hls::stream<ap_uint<WData> >& rowID,
    hls::stream<float>& similarity,
    hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "-----------------CSR_decode-------------" << std::endl;
#endif
#endif

    hls::stream<ap_uint<WData> > row0;
#pragma HLS stream variable = row0 depth = 512
#pragma HLS resource variable = row0 core = FIFO_BRAM
    hls::stream<ap_uint<WData> > col0[CHNM];
#pragma HLS stream variable = col0 depth = 512
#pragma HLS array_partition variable = col0 complete
#pragma HLS resource variable = col0 core = FIFO_BRAM
    hls::stream<ap_uint<WData> > weight0[CHNM];
#pragma HLS stream variable = weight0 depth = 512
#pragma HLS array_partition variable = weight0 complete
#pragma HLS resource variable = weight0 core = FIFO_BRAM
    hls::stream<ap_uint<WData> > weight1[CHNM];
#pragma HLS stream variable = weight1 depth = 512
#pragma HLS array_partition variable = weight1 complete
#pragma HLS resource variable = weight1 core = FIFO_BRAM
    hls::stream<ap_uint<CHNM> > compute_enable0;
#pragma HLS stream variable = compute_enable0 depth = 512
#pragma HLS resource variable = compute_enable0 core = FIFO_BRAM
    hls::stream<bool> strm_end0;
#pragma HLS stream variable = strm_end0 depth = 512
#pragma HLS resource variable = strm_end0 core = FIFO_SRL

    internal::general_similarity::sparseDecode<CHNM, WData, EN_FLOAT_POINT>(
        similarity_type, data_type, row_nm, col_nm, start_id, offset_in, column_in, weight_in, row0, col0, weight0,
        weight1, compute_enable0, strm_end0);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "-------------findCorrelation------------" << std::endl;
#endif
#endif

    hls::stream<ap_uint<WData> > row2;
#pragma HLS stream variable = row2 depth = 512
#pragma HLS resource variable = row2 core = FIFO_BRAM
    hls::stream<ap_uint<WData> > weight2[CHNM];
#pragma HLS stream variable = weight2 depth = 8
#pragma HLS array_partition variable = weight2 complete
#pragma HLS resource variable = weight2 core = FIFO_SRL
    hls::stream<ap_uint<CHNM> > compute_enable2;
#pragma HLS stream variable = compute_enable2 depth = 512
#pragma HLS resource variable = compute_enable2 core = FIFO_BRAM
    hls::stream<bool> strm_end2;
#pragma HLS stream variable = strm_end2 depth = 512
#pragma HLS resource variable = strm_end2 core = FIFO_SRL

    findCorrelationSparse<CHNM, WData, EN_FLOAT_POINT>(row0, col0, weight0, compute_enable0, strm_end0, similarity_type,
                                                       data_type, source_num, max_col, col_vector, sparse_weight_vector,
                                                       row2, weight2, compute_enable2, strm_end2);

    hls::stream<ap_uint<WData> > row3;
#pragma HLS stream variable = row3 depth = 8
#pragma HLS resource variable = row3 core = FIFO_SRL
    hls::stream<ap_uint<WData> > weight3;
#pragma HLS stream variable = weight3 depth = 8
#pragma HLS resource variable = weight3 core = FIFO_SRL
    hls::stream<ap_uint<WData> > weight4;
#pragma HLS stream variable = weight4 depth = 8
#pragma HLS resource variable = weight4 core = FIFO_SRL
    hls::stream<bool> strm_end3;
#pragma HLS stream variable = strm_end3 depth = 8
#pragma HLS resource variable = strm_end3 core = FIFO_SRL

    internal::general_similarity::accumAddTree<CHNM, WData, 1, EN_FLOAT_POINT, false>(
        row2, weight1, weight2, compute_enable2, strm_end2, similarity_type, source_norm, row3, weight3, weight4,
        strm_end3);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "----------ALU(sqrt and divide)----------" << std::endl;
#endif
#endif

    internal::general_similarity::ALU<WData, EN_FLOAT_POINT>(similarity_type, source_norm, row3, weight3, weight4,
                                                             strm_end3, rowID, similarity, strm_out_end);
}

template <int CHNM, int PU, int WData, int RAM_SZ, bool EN_FLOAT_POINT>
void similarity_processing_unit_wrapper(
    // input
    hls::stream<ap_uint<WData * CHNM> > offset_in[PU],
    hls::stream<ap_uint<WData * CHNM> > column_in[PU],
    hls::stream<ap_uint<WData * CHNM> > weight_in[PU],

    // config
    ap_uint<32> similarity_type,
    ap_uint<32> data_type,
    ap_uint<32> row_nm[PU],
    ap_uint<32> col_nm[PU],
    ap_uint<32> start_id[PU],
    ap_uint<32> source_num,
    ap_uint<WData> source_norm,
    ap_uint<32> max_col,
// ram
#ifndef __SYNTHESIS__
    ap_uint<WData>* col_vector[PU],
    ap_uint<WData>* sparse_weight_vector[PU],
#else
    ap_uint<WData> col_vector[PU][1 << RAM_SZ],
    ap_uint<WData> sparse_weight_vector[PU][1 << RAM_SZ],
#endif

    // output
    hls::stream<ap_uint<WData> > rowID[PU],
    hls::stream<float> similarity[PU],
    hls::stream<bool> strm_out_end[PU]) {
loop_pu:
    for (ap_uint<8> i = 0; i < PU; i++) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << std::endl;
        std::cout << "--------------------pu" << i << "-----------------" << std::endl;
#endif
#endif

#pragma HLS UNROLL

        similarity_processing_unit<CHNM, WData, RAM_SZ, EN_FLOAT_POINT>(
            offset_in[i], column_in[i], weight_in[i], similarity_type, data_type, row_nm[i], col_nm[i], start_id[i],
            source_num, source_norm, max_col, col_vector[i], sparse_weight_vector[i], rowID[i], similarity[i],
            strm_out_end[i]);
    }
}

template <int CHNM, int PU, int WData, int RAM_SZ, bool EN_FLOAT_POINT>
void similarityTop(
    // input
    hls::stream<ap_uint<WData * CHNM> > offset_in[PU],
    hls::stream<ap_uint<WData * CHNM> > column_in[PU],
    hls::stream<ap_uint<WData * CHNM> > weight_in[PU],

    // config
    ap_uint<32> similarity_type,
    ap_uint<32> data_type,
    ap_uint<32> row_nm[PU],
    ap_uint<32> col_nm[PU],
    ap_uint<32> start_id[PU],
    ap_uint<32> source_num,
    ap_uint<WData> source_norm,
    ap_uint<32> max_col,
// ram
#ifndef __SYNTHESIS__
    ap_uint<WData>* col_vector[PU],
    ap_uint<WData>* sparse_weight_vector[PU],
#else
    ap_uint<WData> col_vector[PU][1 << RAM_SZ],
    ap_uint<WData> sparse_weight_vector[PU][1 << RAM_SZ],
#endif

    // output
    hls::stream<ap_uint<WData> >& rowID,
    hls::stream<float>& similarity,
    hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<32> > rowID1[16];
#pragma HLS stream variable = rowID1 depth = 512
#pragma HLS array_partition variable = rowID1 complete
#pragma HLS resource variable = rowID1 core = FIFO_BRAM
    hls::stream<float> similarity_strm1[16];
#pragma HLS stream variable = similarity_strm1 depth = 512
#pragma HLS array_partition variable = similarity_strm1 complete
#pragma HLS resource variable = similarity_strm1 core = FIFO_BRAM
    hls::stream<bool> strm_end1[16];
#pragma HLS stream variable = strm_end1 depth = 512
#pragma HLS array_partition variable = strm_end1 complete
#pragma HLS resource variable = strm_end1 core = FIFO_BRAM

    similarity_processing_unit_wrapper<CHNM, PU, WData, RAM_SZ, EN_FLOAT_POINT>(
        offset_in, column_in, weight_in, similarity_type, data_type, row_nm, col_nm, start_id, source_num, source_norm,
        max_col, col_vector, sparse_weight_vector, rowID1, similarity_strm1, strm_end1);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "-------------------------collect-----------------------" << std::endl;
#endif
#endif

    if (PU == 1) {
        internal::general_similarity::collect1_1<32, float, false>(rowID1[0], similarity_strm1[0], strm_end1[0], rowID,
                                                                   similarity, strm_out_end);
    } else if (PU == 2) {
        internal::general_similarity::collect2_1<32, float, false>(rowID1[0], similarity_strm1[0], strm_end1[0],
                                                                   rowID1[1], similarity_strm1[1], strm_end1[1], rowID,
                                                                   similarity, strm_out_end);
    } else if (PU == 4) {
        internal::general_similarity::collect4_1<32, float, false>(
            rowID1[0], similarity_strm1[0], strm_end1[0], rowID1[1], similarity_strm1[1], strm_end1[1], rowID1[2],
            similarity_strm1[2], strm_end1[2], rowID1[3], similarity_strm1[3], strm_end1[3], rowID, similarity,
            strm_out_end);
    } else if (PU == 8) {
        internal::general_similarity::collect8_1<32, float, false>(
            rowID1[0], similarity_strm1[0], strm_end1[0], rowID1[1], similarity_strm1[1], strm_end1[1], rowID1[2],
            similarity_strm1[2], strm_end1[2], rowID1[3], similarity_strm1[3], strm_end1[3], rowID1[4],
            similarity_strm1[4], strm_end1[4], rowID1[5], similarity_strm1[5], strm_end1[5], rowID1[6],
            similarity_strm1[6], strm_end1[6], rowID1[7], similarity_strm1[7], strm_end1[7], rowID, similarity,
            strm_out_end);
    } else if (PU == 16) {
        hls::stream<ap_uint<WData> > row_id_tmp[4];
#pragma HLS stream variable = row_id_tmp depth = 8
#pragma HLS array_partition variable = row_id_tmp complete
#pragma HLS resource variable = row_id_tmp core = FIFO_SRL
        hls::stream<float> similarity_tmp[4];
#pragma HLS stream variable = similarity_tmp depth = 8
#pragma HLS array_partition variable = similarity_tmp complete
#pragma HLS resource variable = similarity_tmp core = FIFO_SRL
        hls::stream<bool> tmp_out_end[4];
#pragma HLS stream variable = tmp_out_end depth = 8
#pragma HLS array_partition variable = tmp_out_end complete
#pragma HLS resource variable = tmp_out_end core = FIFO_SRL

        internal::general_similarity::collect4_1<32, float, true>(
            rowID1[0], similarity_strm1[0], strm_end1[0], rowID1[1], similarity_strm1[1], strm_end1[1], rowID1[2],
            similarity_strm1[2], strm_end1[2], rowID1[3], similarity_strm1[3], strm_end1[3], row_id_tmp[0],
            similarity_tmp[0], tmp_out_end[0]);

        internal::general_similarity::collect4_1<32, float, true>(
            rowID1[4], similarity_strm1[4], strm_end1[4], rowID1[5], similarity_strm1[5], strm_end1[5], rowID1[6],
            similarity_strm1[6], strm_end1[6], rowID1[7], similarity_strm1[7], strm_end1[7], row_id_tmp[1],
            similarity_tmp[1], tmp_out_end[1]);

        internal::general_similarity::collect4_1<32, float, true>(
            rowID1[8], similarity_strm1[8], strm_end1[8], rowID1[9], similarity_strm1[9], strm_end1[9], rowID1[10],
            similarity_strm1[10], strm_end1[10], rowID1[11], similarity_strm1[11], strm_end1[11], row_id_tmp[2],
            similarity_tmp[2], tmp_out_end[2]);

        internal::general_similarity::collect4_1<32, float, true>(
            rowID1[12], similarity_strm1[12], strm_end1[12], rowID1[13], similarity_strm1[13], strm_end1[13],
            rowID1[14], similarity_strm1[14], strm_end1[14], rowID1[15], similarity_strm1[15], strm_end1[15],
            row_id_tmp[3], similarity_tmp[3], tmp_out_end[3]);

        internal::general_similarity::collect4_1<32, float, false>(
            row_id_tmp[0], similarity_tmp[0], tmp_out_end[0], row_id_tmp[1], similarity_tmp[1], tmp_out_end[1],
            row_id_tmp[2], similarity_tmp[2], tmp_out_end[2], row_id_tmp[3], similarity_tmp[3], tmp_out_end[3], rowID,
            similarity, strm_out_end);
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "-------------------------finish------------------------" << std::endl;
#endif
#endif
}
} // sparse_similarity
} // internal

/**
 * @brief similarity function for sparse graph. It support both Jaccard and Cosine Similarity.
 *
 * @tparam CHNM the channel number of input data
 * @tparam PU the number of processing unit
 * @tparam WData the width of input data
 * @tparam RAM_SZ the log size of internal URAM
 * @tparam EN_FLOAT_POINT if it is true, the primitive will support both float and int type of input. Otherwise, it only
 * support int. Multiple channel of float input should be compacted as type of ap_uint.
 *
 * @param config the control parameter of the primitive which contains: sourceNUM, similarityType, dataType,
 * startID, rowNUM and colNUM of each processing unit(PU)
 * @param sourceIndice input indice as source vertex for computing similarity
 * @param sourceWeight input weight as source vertex for computing similarity
 * @param offsetCSR input muti-channel offset stream
 * @param indiceCSR input muti-channel indice stream
 * @param weight input muti-channel weight stream
 * @param rowID output result ID stream
 * @param similarity output similarity value corresponding to its ID
 * @param strmOutEnd end flag stream for output
 */
template <int CHNM, int PU, int WData, int RAM_SZ, bool EN_FLOAT_POINT>
void sparseSimilarity(hls::stream<ap_uint<32> >& config,

                      hls::stream<ap_uint<WData> >& sourceIndice,
                      hls::stream<ap_uint<WData> >& sourceWeight,

                      hls::stream<ap_uint<WData * CHNM> > offsetCSR[PU],
                      hls::stream<ap_uint<WData * CHNM> > indiceCSR[PU],
                      hls::stream<ap_uint<WData * CHNM> > weight[PU],

                      hls::stream<ap_uint<WData> >& rowID,
                      hls::stream<float>& similarity,
                      hls::stream<bool>& strmOutEnd) {
#pragma HLS INLINE off

#pragma HLS array_partition variable = offsetCSR complete
#pragma HLS array_partition variable = indiceCSR complete
#pragma HLS array_partition variable = weight complete

    // define URAM structure
    const int RAM_Size = 1 << RAM_SZ;

#ifndef __SYNTHESIS__

    ap_uint<WData>* col_vector[PU];
    ap_uint<WData>* sparse_weight_vector[PU];

    for (int i = 0; i < PU; i++) {
        col_vector[i] = (ap_uint<WData>*)malloc(RAM_Size * sizeof(ap_uint<WData>));
        sparse_weight_vector[i] = (ap_uint<WData>*)malloc(RAM_Size * sizeof(ap_uint<WData>));
    }

#else

    ap_uint<WData> col_vector[PU][RAM_Size];
#pragma HLS resource variable = col_vector core = RAM_2P_URAM
#pragma HLS ARRAY_PARTITION variable = col_vector dim = 1

    ap_uint<WData> sparse_weight_vector[PU][RAM_Size];
#pragma HLS resource variable = sparse_weight_vector core = RAM_2P_URAM
#pragma HLS ARRAY_PARTITION variable = sparse_weight_vector dim = 1

#endif

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "------------------------load_config---------------------" << std::endl;
#endif
#endif

    ap_uint<32> source_num;
    ap_uint<32> similarity_type;
    ap_uint<32> data_type;
    ap_uint<32> start_id[PU];
#pragma HLS array_partition variable = start_id complete
    ap_uint<32> row_nm[PU];
#pragma HLS array_partition variable = row_nm complete
    ap_uint<32> col_nm[PU];
#pragma HLS array_partition variable = col_nm complete

    internal::sparse_similarity::load_config<PU>(config, source_num, similarity_type, data_type, start_id, row_nm,
                                                 col_nm);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "--------------------load_source_vertex------------------" << std::endl;
#endif
#endif

    ap_uint<WData> source_norm;
    ap_uint<WData> max_col;

    if (WData == 64) {
        internal::sparse_similarity::load_source_vertex64<PU, CHNM, RAM_SZ, WData, EN_FLOAT_POINT>(
            source_num, similarity_type, data_type, sourceIndice, sourceWeight, col_vector, sparse_weight_vector,
            source_norm, max_col);
    } else {
        internal::sparse_similarity::load_source_vertex32<PU, CHNM, RAM_SZ, WData, EN_FLOAT_POINT>(
            source_num, similarity_type, data_type, sourceIndice, sourceWeight, col_vector, sparse_weight_vector,
            source_norm, max_col);
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "----------------------similarity_top--------------------" << std::endl;
#endif
#endif

    internal::sparse_similarity::similarityTop<CHNM, PU, WData, RAM_SZ, EN_FLOAT_POINT>(
        offsetCSR, indiceCSR, weight, similarity_type, data_type, row_nm, col_nm, start_id, source_num, source_norm,
        max_col, col_vector, sparse_weight_vector, rowID, similarity, strmOutEnd);

#ifndef __SYNTHESIS__
    for (int i = 0; i < PU; i++) {
        free(col_vector[i]);
        free(sparse_weight_vector[i]);
    }
#endif
}
} // graph
} // xf

#endif
