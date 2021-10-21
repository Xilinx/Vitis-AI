
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
 * @file general_similarity.hpp
 *
 */

#ifndef __XF_GRAPH_GENERAL_SIMILARITY_HPP_
#define __XF_GRAPH_GENERAL_SIMILARITY_HPP_

#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"

#include "similarity/types.hpp"
#include "similarity/enums.hpp"

#ifndef __SYNTHESIS__
#define DEBUG_SIMILARITY true
#ifdef DEBUG_SIMILARITY
#define DEBUG_INPUT_MUX true
#endif
#endif

namespace xf {

namespace graph {

namespace internal {

namespace general_similarity {

template <int CHNM, int WData>
void loadOffset(ap_uint<32> vertex_num,
                hls::stream<ap_uint<WData * CHNM> >& offset_in,
                hls::stream<ap_uint<WData> >& offset_out) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    std::cout << "loading offset: vertex_num=" << vertex_num << std::endl;
#endif

    ap_uint<WData * CHNM> offset;
    ap_uint<WData> offset_tmp[CHNM];
#pragma HLS array_partition variable = offset_tmp dim = 1
    ap_uint<WData> out;

    for (ap_uint<32> i = 0; i < vertex_num + 1; i++) {
#pragma HLS PIPELINE II = 1

        bool read_sign;
        if (CHNM == 16) {
            if (i(3, 0) == 0)
                read_sign = true;
            else
                read_sign = false;
        } else if (CHNM == 8) {
            if (i(2, 0) == 0)
                read_sign = true;
            else
                read_sign = false;
        } else if (CHNM == 4) {
            if (i(1, 0) == 0)
                read_sign = true;
            else
                read_sign = false;
        } else if (CHNM == 2) {
            if (i[0] == 0)
                read_sign = true;
            else
                read_sign = false;
        } else {
            read_sign = true;
        }

        if (read_sign) offset = offset_in.read();

        for (ap_uint<32> j = 0; j < CHNM; j++) {
#pragma HLS unroll
            offset_tmp[j] = offset((j + 1) * WData - 1, j * WData);
        }

        if (CHNM == 16) {
            out = offset_tmp[i(3, 0)];
        } else if (CHNM == 8) {
            out = offset_tmp[i(2, 0)];
        } else if (CHNM == 4) {
            out = offset_tmp[i(1, 0)];
        } else if (CHNM == 2) {
            out = offset_tmp[i[0]];
        } else {
            out = offset_tmp[0];
        }

        offset_out.write(out);
    }
}

template <int WData>
void offsetToNumber(ap_uint<32> vertex_num,
                    ap_uint<32> edge_num,
                    ap_uint<32> start_id,
                    hls::stream<ap_uint<WData> >& offset_in,
                    hls::stream<ap_uint<WData> >& row_id,
                    hls::stream<ap_uint<WData> >& number,
                    hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE off

    ap_uint<32> start, current_offset, previous_offset;

    start = start_id;
    previous_offset = offset_in.read();
    for (ap_uint<32> i = 0; i < vertex_num; i++) {
#pragma HLS PIPELINE II = 1

        current_offset = offset_in.read();
        ap_uint<WData> num = current_offset - previous_offset;

        if (num > 0) {
            row_id.write(start);
            number.write(num);
            strm_out_end.write(false);
        }

#ifdef DEBUG_INPUT_MUX
        std::cout << "offsetToNumber i=" << i << " row_id=" << start << " current=" << current_offset
                  << " previous=" << previous_offset << " num=" << num << std::endl;
#endif

        start++;
        previous_offset = current_offset;
    }
    strm_out_end.write(true);
}

// generate control signal for dense input
template <int CHNM, int WData>
void rowGen(ap_uint<32> vertex_num,
            ap_uint<32> edge_num,
            ap_uint<32> start_id,

            hls::stream<ap_uint<WData> >& row_out,
            hls::stream<ap_uint<CHNM> >& compute_enable,
            hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE off

    ap_uint<32> residual = edge_num;
    ap_uint<WData> current_row = start_id;
    ap_uint<CHNM> enable;
    ap_uint<32> cnt = 0;

generate_row:
    while (cnt < vertex_num) {
#pragma HLS PIPELINE II = 1

        for (ap_uint<8> i = 0; i < CHNM; i++) {
#pragma HLS UNROLL
            if (residual >= CHNM)
                enable[i] = 1;
            else {
                if (i < residual)
                    enable[i] = 1;
                else
                    enable[i] = 0;
            }
        }

        row_out.write(current_row);
        compute_enable.write(enable);
        strm_out_end.write(false);

#ifdef DEBUG_INPUT_MUX
        std::cout << "rowGen cnt=" << cnt << " current_row=" << current_row << " enable=" << enable << std::endl;
#endif

        if (residual > CHNM) {
            residual = residual - CHNM;
        } else {
            residual = edge_num;
            current_row++;
            cnt++;
        }
    }
    strm_out_end.write(true);
}

// generate control signal for sparse input
template <int CHNM, int WData>
void rowGen(hls::stream<ap_uint<WData> >& row_id,
            hls::stream<ap_uint<WData> >& number,
            hls::stream<bool>& strm_in_end,

            hls::stream<ap_uint<WData> >& row_out,
            hls::stream<ap_uint<CHNM> >& compute_enable,
            hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE off

    ap_uint<32> residual = 0;
    ap_uint<8> previous_idx = 0;
    ap_uint<8> current_idx = 0;
    ap_uint<8> low_idx = 0;
    ap_uint<8> high_idx = 0;

    bool initial = true;
    bool update_row = false;

    ap_uint<WData> current_row;
    ap_uint<WData> num;
    ap_uint<CHNM> enable;

    ap_uint<32> cnt = 0;
    bool end = strm_in_end.read();
generate_row:
    while (!end || residual > 0) {
#pragma HLS PIPELINE II = 1

        if (initial || residual == 0) {
            current_row = row_id.read();
            num = number.read();
            end = strm_in_end.read();

            residual += num;
            current_idx = previous_idx + num;
        }

        if (CHNM == 16) {
            high_idx = current_idx(3, 0);
            low_idx = previous_idx(3, 0);
        } else if (CHNM == 8) {
            high_idx = current_idx(2, 0);
            low_idx = previous_idx(2, 0);
        } else if (CHNM == 4) {
            high_idx = current_idx(1, 0);
            low_idx = previous_idx(1, 0);
        } else if (CHNM == 2) {
            high_idx = current_idx[0];
            low_idx = previous_idx[0];
        } else {
            high_idx = 0;
            low_idx = 0;
        }

        if (residual >= CHNM) {
            for (ap_uint<8> i = 0; i < CHNM; i++) {
#pragma HLS UNROLL
                enable[i] = 1;
            }
        } else {
            if (high_idx >= low_idx) {
                for (ap_uint<8> i = 0; i < CHNM; i++) {
#pragma HLS UNROLL
                    if (i >= low_idx && i < high_idx) {
                        enable[i] = 1;
                    } else {
                        enable[i] = 0;
                    }
                }
            } else {
                for (ap_uint<8> i = 0; i < CHNM; i++) {
#pragma HLS UNROLL
                    if ((i >= 0 && i < high_idx) || (i >= low_idx && i < CHNM)) {
                        enable[i] = 1;
                    } else {
                        enable[i] = 0;
                    }
                }
            }
        }

        row_out.write(current_row);
        compute_enable.write(enable);
        strm_out_end.write(false);

#ifdef DEBUG_INPUT_MUX
        std::cout << "rowGen i=" << cnt << " current_idx=" << current_idx << " previous_idx=" << previous_idx
                  << " current_row=" << current_row << " num=" << num << std::endl;
#endif
        if (residual > CHNM) {
            residual = residual - CHNM;
            update_row = false;
        } else {
            residual = 0;
            update_row = true;
            previous_idx = high_idx;
        }
        initial = false;
        cnt++;
    }
    strm_out_end.write(true);
}

// feed dense data
template <int CHNM, int WData, bool EN_FLOAT_POINT>
void feedData(ap_uint<32> similarity_type,
              ap_uint<32> data_type,
              ap_uint<32> vertex_num,
              ap_uint<32> edge_num,

              hls::stream<ap_uint<WData * CHNM> >& weight_in0,
              hls::stream<ap_uint<WData * CHNM> >& weight_in1,
              hls::stream<ap_uint<WData * CHNM> >& weight_in2,
              hls::stream<ap_uint<WData * CHNM> >& weight_in3,

              hls::stream<ap_uint<WData> > weight_out0[4][CHNM],
              hls::stream<ap_uint<WData> > weight_out1[4][CHNM]) {
#pragma HLS INLINE off

    ap_uint<WData> count = 0;
    ap_uint<WData> residual;

#ifdef DEBUG_INPUT_MUX
    ap_uint<WData> cnt = 0;
#endif

    ap_uint<WData> num = edge_num * vertex_num;

generate_data:
    while (count < num) {
#pragma HLS PIPELINE II = 1

        ap_uint<WData * CHNM> weight_tmp[4];
        weight_tmp[0] = weight_in0.read();
        weight_tmp[1] = weight_in1.read();
        weight_tmp[2] = weight_in2.read();
        weight_tmp[3] = weight_in3.read();
        residual = num - count;

        for (ap_uint<8> i = 0; i < CHNM; i++) {
#pragma HLS UNROLL

            ap_uint<WData> weight[4];
            ap_uint<WData> weight_tmp1[4];
            for (ap_uint<8> j = 0; j < 4; j++) {
#pragma HLS UNROLL

                weight_tmp1[j] = weight_tmp[j](WData * (i + 1) - 1, WData * i);
            }

            for (ap_uint<8> j = 0; j < 4; j++) {
#pragma HLS UNROLL

                if (EN_FLOAT_POINT) {
                    if (WData == 32) {
                        float weight_float;
                        if (data_type == enums::FLOAT_TYPE)
                            weight_float = bitsToFloat<uint32_t, float>((uint32_t)weight_tmp1[j]);
                        else
                            weight_float = (float)weight_tmp1[j];
                        weight[j] = floatToBits<float, uint32_t>(weight_float);
                    } else {
                        double weight_double;
                        if (data_type == enums::DOUBLE_TYPE)
                            weight_double = bitsToFloat<uint64_t, double>((uint64_t)weight_tmp1[j]);
                        else
                            weight_double = (double)weight_tmp1[j];
                        weight[j] = floatToBits<double, uint64_t>(weight_double);
                    }
                } else {
                    weight[j] = weight_tmp1[j];
                }
            }

            if (i < residual) {
                for (ap_uint<8> j = 0; j < 4; j++) {
#pragma HLS UNROLL

                    weight_out0[j][i].write(weight[j]);
                    weight_out1[j][i].write(weight[j]);

#ifdef DEBUG_INPUT_MUX
                    if (EN_FLOAT_POINT)
                        std::cout << "feedDenseData weight" << j << "=" << bitsToFloat<uint32_t, float>(weight[j])
                                  << std::endl;
                    else
                        std::cout << "feedDenseData weight" << j << "=" << weight[j] << std::endl;
#endif
                }
            }
        }
#ifdef DEBUG_INPUT_MUX
        std::cout << "feedDenseData i=" << cnt << " residual=" << residual << " eldge_num=" << num << " count=" << count
                  << std::endl;
        cnt++;
#endif
        count += CHNM;
    }
}

// feed sparse data
template <int CHNM, int WData, bool EN_FLOAT_POINT>
void feedData(ap_uint<32> similarity_type,
              ap_uint<32> data_type,
              ap_uint<32> vertex_num,
              ap_uint<32> edge_num,

              hls::stream<ap_uint<WData * CHNM> >& column_in,
              hls::stream<ap_uint<WData * CHNM> >& weight_in,

              hls::stream<ap_uint<WData> > column_out[CHNM],
              hls::stream<ap_uint<WData> > weight_out0[CHNM],
              hls::stream<ap_uint<WData> > weight_out1[CHNM]) {
#pragma HLS INLINE off

    ap_uint<WData> count = 0;
    ap_uint<WData> residual;

#ifdef DEBUG_INPUT_MUX
    ap_uint<WData> cnt = 0;
#endif

    ap_uint<WData> num = edge_num;

generate_data:
    while (count < num) {
#pragma HLS PIPELINE II = 1

        ap_uint<WData* CHNM> column_tmp = column_in.read();

        ap_uint<WData * CHNM> weight_tmp;
        if (similarity_type == enums::JACCARD_SIMILARITY) {
            weight_tmp = 0;
        } else {
            weight_tmp = weight_in.read();
        }

        residual = num - count;

        for (ap_uint<8> i = 0; i < CHNM; i++) {
#pragma HLS UNROLL

            ap_uint<WData> col = column_tmp(WData * (i + 1) - 1, WData * i);
            ap_uint<WData> weight;
            if (similarity_type == enums::JACCARD_SIMILARITY) {
                if (EN_FLOAT_POINT) {
                    if (WData == 32)
                        weight = floatToBits<float, uint32_t>(1.0);
                    else
                        weight = floatToBits<double, uint64_t>(1.0);
                } else
                    weight = 1;
            } else {
                ap_uint<WData> weight_tmp1 = weight_tmp(WData * (i + 1) - 1, WData * i);
                if (EN_FLOAT_POINT) {
                    if (WData == 32) {
                        float weight_float;
                        if (data_type == enums::FLOAT_TYPE)
                            weight_float = bitsToFloat<uint32_t, float>((uint32_t)weight_tmp1);
                        else
                            weight_float = (float)weight_tmp1;
                        weight = floatToBits<float, uint32_t>(weight_float);
                    } else {
                        double weight_double;
                        if (data_type == enums::DOUBLE_TYPE)
                            weight_double = bitsToFloat<uint64_t, double>((uint64_t)weight_tmp1);
                        else
                            weight_double = (double)weight_tmp1;
                        weight = floatToBits<double, uint64_t>(weight_double);
                    }
                } else {
                    weight = weight_tmp1;
                }
            }

            if (i < residual) {
                column_out[i].write(col);
                weight_out0[i].write(weight);
                weight_out1[i].write(weight);

#ifdef DEBUG_INPUT_MUX
                if (EN_FLOAT_POINT)
                    std::cout << "feedCOO col=" << col << " weight=" << bitsToFloat<uint32_t, float>(weight)
                              << std::endl;
                else
                    std::cout << "feedCOO col=" << col << " weight=" << weight << std::endl;
#endif
            }
        }

#ifdef DEBUG_INPUT_MUX
        std::cout << "feedCOO i=" << cnt << " residual=" << residual << " eldge_num=" << num << " count=" << count
                  << std::endl;
        cnt++;
#endif

        count += CHNM;
    }
}

template <int CHNM, int WData, bool EN_FLOAT_POINT>
void sparseDecode(ap_uint<32> similarity_type,
                  ap_uint<32> data_type,
                  ap_uint<32> vertex_num,
                  ap_uint<32> edge_num,
                  ap_uint<32> start_id,

                  hls::stream<ap_uint<WData * CHNM> >& offset_in,
                  hls::stream<ap_uint<WData * CHNM> >& column_in,
                  hls::stream<ap_uint<WData * CHNM> >& weight_in,

                  hls::stream<ap_uint<WData> >& row_out,
                  hls::stream<ap_uint<WData> > column_out[CHNM],
                  hls::stream<ap_uint<WData> > weight_out0[CHNM],
                  hls::stream<ap_uint<WData> > weight_out1[CHNM],
                  hls::stream<ap_uint<CHNM> >& compute_enable,
                  hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<WData> > offset;
#pragma HLS stream variable = offset depth = 8
#pragma HLS resource variable = offset core = FIFO_SRL
    hls::stream<ap_uint<WData> > row_id;
#pragma HLS stream variable = row_id depth = 8
#pragma HLS resource variable = row_id core = FIFO_SRL
    hls::stream<ap_uint<WData> > number;
#pragma HLS stream variable = number depth = 8
#pragma HLS resource variable = number core = FIFO_SRL
    hls::stream<bool> strm_end0;
#pragma HLS stream variable = strm_end0 depth = 8
#pragma HLS resource variable = strm_end0 core = FIFO_SRL

    loadOffset<CHNM, WData>(vertex_num, offset_in, offset);

    offsetToNumber<WData>(vertex_num, edge_num, start_id, offset, row_id, number, strm_end0);

    rowGen<CHNM, WData>(row_id, number, strm_end0, row_out, compute_enable, strm_out_end);

    feedData<CHNM, WData, EN_FLOAT_POINT>(similarity_type, data_type, vertex_num, edge_num, column_in, weight_in,
                                          column_out, weight_out0, weight_out1);
}

template <int CHNM, int WData, bool EN_FLOAT_POINT>
void denseDecode(ap_uint<32> similarity_type,
                 ap_uint<32> data_type,
                 ap_uint<32> vertex_num,
                 ap_uint<32> edge_num,
                 ap_uint<32> start_id,

                 hls::stream<ap_uint<WData * CHNM> >& strm_in0,
                 hls::stream<ap_uint<WData * CHNM> >& strm_in1,
                 hls::stream<ap_uint<WData * CHNM> >& strm_in2,
                 hls::stream<ap_uint<WData * CHNM> >& strm_in3,

                 hls::stream<ap_uint<WData> >& row_out,
                 hls::stream<ap_uint<WData> > weight_out0[4][CHNM],
                 hls::stream<ap_uint<WData> > weight_out1[4][CHNM],
                 hls::stream<ap_uint<CHNM> >& compute_enable,
                 hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    rowGen<CHNM, WData>(vertex_num, edge_num, start_id, row_out, compute_enable, strm_out_end);

    feedData<CHNM, WData, EN_FLOAT_POINT>(similarity_type, data_type, vertex_num, edge_num, strm_in0, strm_in1,
                                          strm_in2, strm_in3, weight_out0, weight_out1);
}

// inputmux to decode dense or sparse input
template <int CHNM, int WData, bool EN_FLOAT_POINT>
void inputMux(ap_uint<32> similarity_type,
              ap_uint<32> data_type,
              ap_uint<32> graph_type,
              ap_uint<32> vertex_num,
              ap_uint<32> edge_num,
              ap_uint<32> start_id,

              hls::stream<ap_uint<WData * CHNM> >& strm_in0,
              hls::stream<ap_uint<WData * CHNM> >& strm_in1,
              hls::stream<ap_uint<WData * CHNM> >& strm_in2,
              hls::stream<ap_uint<WData * CHNM> >& strm_in3,

              hls::stream<ap_uint<WData> >& row_out,
              hls::stream<ap_uint<WData> > column_out[CHNM],
              hls::stream<ap_uint<WData> > weight_out0[4][CHNM],
              hls::stream<ap_uint<WData> > weight_out1[4][CHNM],
              hls::stream<ap_uint<CHNM> >& compute_enable,
              hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE off

    if (graph_type == enums::DENSE) {
        denseDecode<CHNM, WData, EN_FLOAT_POINT>(similarity_type, data_type, vertex_num, edge_num, start_id, strm_in0,
                                                 strm_in1, strm_in2, strm_in3, row_out, weight_out0, weight_out1,
                                                 compute_enable, strm_out_end);
    } else {
        sparseDecode<CHNM, WData, EN_FLOAT_POINT>(similarity_type, data_type, vertex_num, edge_num, start_id, strm_in0,
                                                  strm_in1, strm_in2, row_out, column_out, weight_out0[0],
                                                  weight_out1[0], compute_enable, strm_out_end);
    }
}

// Only support mux8_1, mux4_1, mux2_1
template <int IN_NM>
ap_uint<3> mux(ap_uint<IN_NM> rd) {
#pragma HLS inline
    ap_uint<3> o = 0;
    if (IN_NM == 8) {
        o[0] = rd[1] | rd[3] | rd[5] | rd[7];
        o[1] = rd[2] | rd[3] | rd[6] | rd[7];
        o[2] = rd[4] | rd[5] | rd[6] | rd[7];
    } else if (IN_NM == 4) {
        o[0] = rd[1] | rd[3];
        o[1] = rd[2] | rd[3];
    } else if (IN_NM == 2) {
        o[0] = rd[1];
    } else {
        o = 0;
    }
    return o;
}

template <int CH_NM>
ap_uint<CH_NM> mul_ch_read(ap_uint<CH_NM> empty) {
    ap_uint<CH_NM> rd = 0;
#pragma HLS inline
    for (int i = 0; i < CH_NM; i++) {
#pragma HLS unroll
        ap_uint<CH_NM> t_e = 0;
        if (i > 0) t_e = empty(i - 1, 0);
        rd[i] = t_e > 0 ? (bool)0 : (bool)empty[i];
    }
    return rd;
}

template <int KEYW, typename HASHW, bool PADD>
void collect1_1(hls::stream<ap_uint<KEYW> >& i_key_strm,
                hls::stream<HASHW>& i_hash_strm,
                hls::stream<bool>& i_e_strm,
                hls::stream<ap_uint<KEYW> >& o_key_strm,
                hls::stream<HASHW>& o_hash_strm,
                hls::stream<bool>& o_e_strm) {
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
    bool last = 0; // i_e_strm.read();
LOOP_MERGE1_1:
    do {
#pragma HLS pipeline II = 1
        ap_uint<KEYW> key = i_key_strm.read();
        HASHW hash_val = i_hash_strm.read();
        last = i_e_strm.read();
        if (!last) {
            o_key_strm.write(key);
            o_hash_strm.write(hash_val);
            o_e_strm.write(false);
#ifndef __SYNTHESIS__
            cnt++;
#endif
        }
    } while (!last);
    if (PADD) {
        o_key_strm.write(0);
        o_hash_strm.write(0);
    }
    o_e_strm.write(true);
#ifndef __SYNTHESIS__
    std::cout << "collect number=" << cnt << std::endl;
#endif
}

template <int KEYW, typename HASHW, bool PADD>
void collect2_1(hls::stream<ap_uint<KEYW> >& i0_key_strm,
                hls::stream<HASHW>& i0_hash_strm,
                hls::stream<bool>& i0_e_strm,
                hls::stream<ap_uint<KEYW> >& i1_key_strm,
                hls::stream<HASHW>& i1_hash_strm,
                hls::stream<bool>& i1_e_strm,

                hls::stream<ap_uint<KEYW> >& o_key_strm,
                hls::stream<HASHW>& o_hash_strm,
                hls::stream<bool>& o_e_strm) {
    ap_uint<KEYW> key_arry[2];
#pragma HLS array_partition variable = key_arry dim = 1
    HASHW hash_val_arry[2];
#pragma HLS array_partition variable = hash_val_arry dim = 1
    ap_uint<2> empty_e = 0;
    ;
    ap_uint<2> rd_e = 0;
    ;
    ap_uint<2> last = 0;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
LOOP_MERGE2_1:
    do {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS PIPELINE II = 1
        empty_e[0] = !i0_e_strm.empty() && !last[0];
        empty_e[1] = !i1_e_strm.empty() && !last[1];
        rd_e = mul_ch_read(empty_e);
        if (rd_e[0]) {
            key_arry[0] = i0_key_strm.read();
            hash_val_arry[0] = i0_hash_strm.read();
            last[0] = i0_e_strm.read();
        }
        if (rd_e[1]) {
            key_arry[1] = i1_key_strm.read();
            hash_val_arry[1] = i1_hash_strm.read();
            last[1] = i1_e_strm.read();
        }
        // only support 8 channels, 4 channels and 2 channels
        ap_uint<3> id = mux<2>(rd_e);
        ap_uint<KEYW> key = key_arry[id];
        HASHW hash_val = hash_val_arry[id];
        bool valid_n = last[id];
        if (!valid_n && rd_e != 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_key_strm.write(key);
            o_hash_strm.write(hash_val);
            o_e_strm.write(false);
        }
    } while (last != 3);
    if (PADD) {
        o_key_strm.write(0);
        o_hash_strm.write(0);
    }
    o_e_strm.write(true);
#ifndef __SYNTHESIS__
    std::cout << "collect number=" << cnt << std::endl;
#endif
}

template <int KEYW, typename HASHW, bool PADD>
void collect4_1(hls::stream<ap_uint<KEYW> >& i0_key_strm,
                hls::stream<HASHW>& i0_hash_strm,
                hls::stream<bool>& i0_e_strm,
                hls::stream<ap_uint<KEYW> >& i1_key_strm,
                hls::stream<HASHW>& i1_hash_strm,
                hls::stream<bool>& i1_e_strm,
                hls::stream<ap_uint<KEYW> >& i2_key_strm,
                hls::stream<HASHW>& i2_hash_strm,
                hls::stream<bool>& i2_e_strm,
                hls::stream<ap_uint<KEYW> >& i3_key_strm,
                hls::stream<HASHW>& i3_hash_strm,
                hls::stream<bool>& i3_e_strm,

                hls::stream<ap_uint<KEYW> >& o_key_strm,
                hls::stream<HASHW>& o_hash_strm,
                hls::stream<bool>& o_e_strm) {
    ap_uint<KEYW> key_arry[4];
#pragma HLS array_partition variable = key_arry dim = 1
    HASHW hash_val_arry[4];
#pragma HLS array_partition variable = hash_val_arry dim = 1
    ap_uint<4> empty_e = 0;
    ;
    ap_uint<4> rd_e = 0;
    ;
    ap_uint<4> last = 0;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
LOOP_MERGE4_1:
    do {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS PIPELINE II = 1
        empty_e[0] = !i0_e_strm.empty() && !last[0];
        empty_e[1] = !i1_e_strm.empty() && !last[1];
        empty_e[2] = !i2_e_strm.empty() && !last[2];
        empty_e[3] = !i3_e_strm.empty() && !last[3];
        rd_e = mul_ch_read(empty_e);
        if (rd_e[0]) {
            key_arry[0] = i0_key_strm.read();
            hash_val_arry[0] = i0_hash_strm.read();
            last[0] = i0_e_strm.read();
        }
        if (rd_e[1]) {
            key_arry[1] = i1_key_strm.read();
            hash_val_arry[1] = i1_hash_strm.read();
            last[1] = i1_e_strm.read();
        }
        if (rd_e[2]) {
            key_arry[2] = i2_key_strm.read();
            hash_val_arry[2] = i2_hash_strm.read();
            last[2] = i2_e_strm.read();
        }
        if (rd_e[3]) {
            key_arry[3] = i3_key_strm.read();
            hash_val_arry[3] = i3_hash_strm.read();
            last[3] = i3_e_strm.read();
        }
        // only support 8 channels, 4 channels and 2 channels
        ap_uint<3> id = mux<4>(rd_e);
        ap_uint<KEYW> key = key_arry[id];
        HASHW hash_val = hash_val_arry[id];
        bool valid_n = last[id];
        if (!valid_n && rd_e != 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_key_strm.write(key);
            o_hash_strm.write(hash_val);
            o_e_strm.write(false);
        }
    } while (last != 15);
    if (PADD) {
        o_key_strm.write(0);
        o_hash_strm.write(0);
    }
    o_e_strm.write(true);
#ifndef __SYNTHESIS__
    std::cout << "collect number=" << cnt << std::endl;
#endif
}

template <int KEYW, typename HASHW, bool PADD>
void collect8_1(hls::stream<ap_uint<KEYW> >& i0_key_strm,
                hls::stream<HASHW>& i0_hash_strm,
                hls::stream<bool>& i0_e_strm,
                hls::stream<ap_uint<KEYW> >& i1_key_strm,
                hls::stream<HASHW>& i1_hash_strm,
                hls::stream<bool>& i1_e_strm,
                hls::stream<ap_uint<KEYW> >& i2_key_strm,
                hls::stream<HASHW>& i2_hash_strm,
                hls::stream<bool>& i2_e_strm,
                hls::stream<ap_uint<KEYW> >& i3_key_strm,
                hls::stream<HASHW>& i3_hash_strm,
                hls::stream<bool>& i3_e_strm,
                hls::stream<ap_uint<KEYW> >& i4_key_strm,
                hls::stream<HASHW>& i4_hash_strm,
                hls::stream<bool>& i4_e_strm,
                hls::stream<ap_uint<KEYW> >& i5_key_strm,
                hls::stream<HASHW>& i5_hash_strm,
                hls::stream<bool>& i5_e_strm,
                hls::stream<ap_uint<KEYW> >& i6_key_strm,
                hls::stream<HASHW>& i6_hash_strm,
                hls::stream<bool>& i6_e_strm,
                hls::stream<ap_uint<KEYW> >& i7_key_strm,
                hls::stream<HASHW>& i7_hash_strm,
                hls::stream<bool>& i7_e_strm,

                hls::stream<ap_uint<KEYW> >& o_key_strm,
                hls::stream<HASHW>& o_hash_strm,
                hls::stream<bool>& o_e_strm) {
    ap_uint<KEYW> key_arry[8];
#pragma HLS array_partition variable = key_arry dim = 1
    HASHW hash_val_arry[8];
#pragma HLS array_partition variable = hash_val_arry dim = 1
    ap_uint<8> empty_e = 0;
    ;
    ap_uint<8> rd_e = 0;
    ;
    ap_uint<8> last = 0;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
LOOP_MERGE8_1:
    do {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS PIPELINE II = 1
        empty_e[0] = !i0_e_strm.empty() && !last[0];
        empty_e[1] = !i1_e_strm.empty() && !last[1];
        empty_e[2] = !i2_e_strm.empty() && !last[2];
        empty_e[3] = !i3_e_strm.empty() && !last[3];
        empty_e[4] = !i4_e_strm.empty() && !last[4];
        empty_e[5] = !i5_e_strm.empty() && !last[5];
        empty_e[6] = !i6_e_strm.empty() && !last[6];
        empty_e[7] = !i7_e_strm.empty() && !last[7];
        rd_e = mul_ch_read(empty_e);
        if (rd_e[0]) {
            key_arry[0] = i0_key_strm.read();
            hash_val_arry[0] = i0_hash_strm.read();
            last[0] = i0_e_strm.read();
        }
        if (rd_e[1]) {
            key_arry[1] = i1_key_strm.read();
            hash_val_arry[1] = i1_hash_strm.read();
            last[1] = i1_e_strm.read();
        }
        if (rd_e[2]) {
            key_arry[2] = i2_key_strm.read();
            hash_val_arry[2] = i2_hash_strm.read();
            last[2] = i2_e_strm.read();
        }
        if (rd_e[3]) {
            key_arry[3] = i3_key_strm.read();
            hash_val_arry[3] = i3_hash_strm.read();
            last[3] = i3_e_strm.read();
        }
        if (rd_e[4]) {
            key_arry[4] = i4_key_strm.read();
            hash_val_arry[4] = i4_hash_strm.read();
            last[4] = i4_e_strm.read();
        }
        if (rd_e[5]) {
            key_arry[5] = i5_key_strm.read();
            hash_val_arry[5] = i5_hash_strm.read();
            last[5] = i5_e_strm.read();
        }
        if (rd_e[6]) {
            key_arry[6] = i6_key_strm.read();
            hash_val_arry[6] = i6_hash_strm.read();
            last[6] = i6_e_strm.read();
        }
        if (rd_e[7]) {
            key_arry[7] = i7_key_strm.read();
            hash_val_arry[7] = i7_hash_strm.read();
            last[7] = i7_e_strm.read();
        }
        // only support 8 channels, 4 channels and 2 channels
        ap_uint<3> id = mux<8>(rd_e);
        ap_uint<KEYW> key = key_arry[id];
        HASHW hash_val = hash_val_arry[id];
        bool valid_n = last[id];
        if (!valid_n && rd_e != 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_key_strm.write(key);
            o_hash_strm.write(hash_val);
            o_e_strm.write(false);
        }
    } while (last != 255);
    if (PADD) {
        o_key_strm.write(0);
        o_hash_strm.write(0);
    }
    o_e_strm.write(true);
#ifndef __SYNTHESIS__
    std::cout << "collect number=" << cnt << std::endl;
#endif
}

template <int KEYW, int PW, typename HASHW, bool PADD>
void merge1_1(hls::stream<ap_uint<KEYW> >& i_key_strm,
              hls::stream<ap_uint<PW> >& i_pld_strm,
              hls::stream<HASHW>& i_hash_strm,
              hls::stream<bool>& i_e_strm,
              hls::stream<ap_uint<KEYW> >& o_key_strm,
              hls::stream<ap_uint<PW> >& o_pld_strm,
              hls::stream<HASHW>& o_hash_strm,
              hls::stream<bool>& o_e_strm) {
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
    bool last = 0; // i_e_strm.read();
LOOP_MERGE1_1:
    do {
#pragma HLS pipeline II = 1
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        HASHW hash_val = i_hash_strm.read();
        last = i_e_strm.read();
        if (!last) {
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_hash_strm.write(hash_val);
            o_e_strm.write(false);
        }
    } while (!last);
    if (PADD) {
        o_key_strm.write(0);
        o_pld_strm.write(0);
        o_hash_strm.write(0);
    }
    o_e_strm.write(true);
#ifndef __SYNTHESIS__
    std::cout << "merged number=" << cnt << std::endl;
#endif
}

template <int KEYW, int PW, typename HASHW, bool PADD>
void merge2_1(hls::stream<ap_uint<KEYW> >& i0_key_strm,
              hls::stream<ap_uint<PW> >& i0_pld_strm,
              hls::stream<HASHW>& i0_hash_strm,
              hls::stream<bool>& i0_e_strm,
              hls::stream<ap_uint<KEYW> >& i1_key_strm,
              hls::stream<ap_uint<PW> >& i1_pld_strm,
              hls::stream<HASHW>& i1_hash_strm,
              hls::stream<bool>& i1_e_strm,

              hls::stream<ap_uint<KEYW> >& o_key_strm,
              hls::stream<ap_uint<PW> >& o_pld_strm,
              hls::stream<HASHW>& o_hash_strm,
              hls::stream<bool>& o_e_strm) {
    ap_uint<KEYW> key_arry[2];
#pragma HLS array_partition variable = key_arry dim = 1
    ap_uint<PW> pld_arry[2];
#pragma HLS array_partition variable = pld_arry dim = 1
    HASHW hash_val_arry[2];
#pragma HLS array_partition variable = hash_val_arry dim = 1
    ap_uint<2> empty_e = 0;
    ;
    ap_uint<2> rd_e = 0;
    ;
    ap_uint<2> last = 0;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
LOOP_MERGE2_1:
    do {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS PIPELINE II = 1
        empty_e[0] = !i0_e_strm.empty() && !last[0];
        empty_e[1] = !i1_e_strm.empty() && !last[1];
        rd_e = mul_ch_read(empty_e);
        if (rd_e[0]) {
            key_arry[0] = i0_key_strm.read();
            pld_arry[0] = i0_pld_strm.read();
            hash_val_arry[0] = i0_hash_strm.read();
            last[0] = i0_e_strm.read();
        }
        if (rd_e[1]) {
            key_arry[1] = i1_key_strm.read();
            pld_arry[1] = i1_pld_strm.read();
            hash_val_arry[1] = i1_hash_strm.read();
            last[1] = i1_e_strm.read();
        }
        // only support 8 channels, 4 channels and 2 channels
        ap_uint<3> id = mux<2>(rd_e);
        ap_uint<KEYW> key = key_arry[id];
        ap_uint<PW> pld = pld_arry[id];
        HASHW hash_val = hash_val_arry[id];
        bool valid_n = last[id];
        if (!valid_n && rd_e != 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_hash_strm.write(hash_val);
            o_e_strm.write(false);
        }
    } while (last != 3);
    if (PADD) {
        o_key_strm.write(0);
        o_pld_strm.write(0);
        o_hash_strm.write(0);
    }
    o_e_strm.write(true);
#ifndef __SYNTHESIS__
    std::cout << "merged number=" << cnt << std::endl;
#endif
}

template <int KEYW, int PW, typename HASHW, bool PADD>
void merge4_1(hls::stream<ap_uint<KEYW> >& i0_key_strm,
              hls::stream<ap_uint<PW> >& i0_pld_strm,
              hls::stream<HASHW>& i0_hash_strm,
              hls::stream<bool>& i0_e_strm,
              hls::stream<ap_uint<KEYW> >& i1_key_strm,
              hls::stream<ap_uint<PW> >& i1_pld_strm,
              hls::stream<HASHW>& i1_hash_strm,
              hls::stream<bool>& i1_e_strm,
              hls::stream<ap_uint<KEYW> >& i2_key_strm,
              hls::stream<ap_uint<PW> >& i2_pld_strm,
              hls::stream<HASHW>& i2_hash_strm,
              hls::stream<bool>& i2_e_strm,
              hls::stream<ap_uint<KEYW> >& i3_key_strm,
              hls::stream<ap_uint<PW> >& i3_pld_strm,
              hls::stream<HASHW>& i3_hash_strm,
              hls::stream<bool>& i3_e_strm,

              hls::stream<ap_uint<KEYW> >& o_key_strm,
              hls::stream<ap_uint<PW> >& o_pld_strm,
              hls::stream<HASHW>& o_hash_strm,
              hls::stream<bool>& o_e_strm) {
    ap_uint<KEYW> key_arry[4];
#pragma HLS array_partition variable = key_arry dim = 1
    ap_uint<PW> pld_arry[4];
#pragma HLS array_partition variable = pld_arry dim = 1
    HASHW hash_val_arry[4];
#pragma HLS array_partition variable = hash_val_arry dim = 1
    ap_uint<4> empty_e = 0;
    ;
    ap_uint<4> rd_e = 0;
    ;
    ap_uint<4> last = 0;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
LOOP_MERGE4_1:
    do {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS PIPELINE II = 1
        empty_e[0] = !i0_e_strm.empty() && !last[0];
        empty_e[1] = !i1_e_strm.empty() && !last[1];
        empty_e[2] = !i2_e_strm.empty() && !last[2];
        empty_e[3] = !i3_e_strm.empty() && !last[3];
        rd_e = mul_ch_read(empty_e);
        if (rd_e[0]) {
            key_arry[0] = i0_key_strm.read();
            pld_arry[0] = i0_pld_strm.read();
            hash_val_arry[0] = i0_hash_strm.read();
            last[0] = i0_e_strm.read();
        }
        if (rd_e[1]) {
            key_arry[1] = i1_key_strm.read();
            pld_arry[1] = i1_pld_strm.read();
            hash_val_arry[1] = i1_hash_strm.read();
            last[1] = i1_e_strm.read();
        }
        if (rd_e[2]) {
            key_arry[2] = i2_key_strm.read();
            pld_arry[2] = i2_pld_strm.read();
            hash_val_arry[2] = i2_hash_strm.read();
            last[2] = i2_e_strm.read();
        }
        if (rd_e[3]) {
            key_arry[3] = i3_key_strm.read();
            pld_arry[3] = i3_pld_strm.read();
            hash_val_arry[3] = i3_hash_strm.read();
            last[3] = i3_e_strm.read();
        }
        // only support 8 channels, 4 channels and 2 channels
        ap_uint<3> id = mux<4>(rd_e);
        ap_uint<KEYW> key = key_arry[id];
        ap_uint<PW> pld = pld_arry[id];
        HASHW hash_val = hash_val_arry[id];
        bool valid_n = last[id];
        if (!valid_n && rd_e != 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_hash_strm.write(hash_val);
            o_e_strm.write(false);
        }
    } while (last != 15);
    if (PADD) {
        o_key_strm.write(0);
        o_pld_strm.write(0);
        o_hash_strm.write(0);
    }
    o_e_strm.write(true);
#ifndef __SYNTHESIS__
    std::cout << "merged number=" << cnt << std::endl;
#endif
}

template <int KEYW, int PW, typename HASHW, bool PADD>
void merge8_1(hls::stream<ap_uint<KEYW> >& i0_key_strm,
              hls::stream<ap_uint<PW> >& i0_pld_strm,
              hls::stream<HASHW>& i0_hash_strm,
              hls::stream<bool>& i0_e_strm,
              hls::stream<ap_uint<KEYW> >& i1_key_strm,
              hls::stream<ap_uint<PW> >& i1_pld_strm,
              hls::stream<HASHW>& i1_hash_strm,
              hls::stream<bool>& i1_e_strm,
              hls::stream<ap_uint<KEYW> >& i2_key_strm,
              hls::stream<ap_uint<PW> >& i2_pld_strm,
              hls::stream<HASHW>& i2_hash_strm,
              hls::stream<bool>& i2_e_strm,
              hls::stream<ap_uint<KEYW> >& i3_key_strm,
              hls::stream<ap_uint<PW> >& i3_pld_strm,
              hls::stream<HASHW>& i3_hash_strm,
              hls::stream<bool>& i3_e_strm,
              hls::stream<ap_uint<KEYW> >& i4_key_strm,
              hls::stream<ap_uint<PW> >& i4_pld_strm,
              hls::stream<HASHW>& i4_hash_strm,
              hls::stream<bool>& i4_e_strm,
              hls::stream<ap_uint<KEYW> >& i5_key_strm,
              hls::stream<ap_uint<PW> >& i5_pld_strm,
              hls::stream<HASHW>& i5_hash_strm,
              hls::stream<bool>& i5_e_strm,
              hls::stream<ap_uint<KEYW> >& i6_key_strm,
              hls::stream<ap_uint<PW> >& i6_pld_strm,
              hls::stream<HASHW>& i6_hash_strm,
              hls::stream<bool>& i6_e_strm,
              hls::stream<ap_uint<KEYW> >& i7_key_strm,
              hls::stream<ap_uint<PW> >& i7_pld_strm,
              hls::stream<HASHW>& i7_hash_strm,
              hls::stream<bool>& i7_e_strm,

              hls::stream<ap_uint<KEYW> >& o_key_strm,
              hls::stream<ap_uint<PW> >& o_pld_strm,
              hls::stream<HASHW>& o_hash_strm,
              hls::stream<bool>& o_e_strm) {
    ap_uint<KEYW> key_arry[8];
#pragma HLS array_partition variable = key_arry dim = 1
    ap_uint<PW> pld_arry[8];
#pragma HLS array_partition variable = pld_arry dim = 1
    HASHW hash_val_arry[8];
#pragma HLS array_partition variable = hash_val_arry dim = 1
    ap_uint<8> empty_e = 0;
    ;
    ap_uint<8> rd_e = 0;
    ;
    ap_uint<8> last = 0;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
LOOP_MERGE8_1:
    do {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS PIPELINE II = 1
        empty_e[0] = !i0_e_strm.empty() && !last[0];
        empty_e[1] = !i1_e_strm.empty() && !last[1];
        empty_e[2] = !i2_e_strm.empty() && !last[2];
        empty_e[3] = !i3_e_strm.empty() && !last[3];
        empty_e[4] = !i4_e_strm.empty() && !last[4];
        empty_e[5] = !i5_e_strm.empty() && !last[5];
        empty_e[6] = !i6_e_strm.empty() && !last[6];
        empty_e[7] = !i7_e_strm.empty() && !last[7];
        rd_e = mul_ch_read(empty_e);
        if (rd_e[0]) {
            key_arry[0] = i0_key_strm.read();
            pld_arry[0] = i0_pld_strm.read();
            hash_val_arry[0] = i0_hash_strm.read();
            last[0] = i0_e_strm.read();
        }
        if (rd_e[1]) {
            key_arry[1] = i1_key_strm.read();
            pld_arry[1] = i1_pld_strm.read();
            hash_val_arry[1] = i1_hash_strm.read();
            last[1] = i1_e_strm.read();
        }
        if (rd_e[2]) {
            key_arry[2] = i2_key_strm.read();
            pld_arry[2] = i2_pld_strm.read();
            hash_val_arry[2] = i2_hash_strm.read();
            last[2] = i2_e_strm.read();
        }
        if (rd_e[3]) {
            key_arry[3] = i3_key_strm.read();
            pld_arry[3] = i3_pld_strm.read();
            hash_val_arry[3] = i3_hash_strm.read();
            last[3] = i3_e_strm.read();
        }
        if (rd_e[4]) {
            key_arry[4] = i4_key_strm.read();
            pld_arry[4] = i4_pld_strm.read();
            hash_val_arry[4] = i4_hash_strm.read();
            last[4] = i4_e_strm.read();
        }
        if (rd_e[5]) {
            key_arry[5] = i5_key_strm.read();
            pld_arry[5] = i5_pld_strm.read();
            hash_val_arry[5] = i5_hash_strm.read();
            last[5] = i5_e_strm.read();
        }
        if (rd_e[6]) {
            key_arry[6] = i6_key_strm.read();
            pld_arry[6] = i6_pld_strm.read();
            hash_val_arry[6] = i6_hash_strm.read();
            last[6] = i6_e_strm.read();
        }
        if (rd_e[7]) {
            key_arry[7] = i7_key_strm.read();
            pld_arry[7] = i7_pld_strm.read();
            hash_val_arry[7] = i7_hash_strm.read();
            last[7] = i7_e_strm.read();
        }
        // only support 8 channels, 4 channels and 2 channels
        ap_uint<3> id = mux<8>(rd_e);
        ap_uint<KEYW> key = key_arry[id];
        ap_uint<PW> pld = pld_arry[id];
        HASHW hash_val = hash_val_arry[id];
        bool valid_n = last[id];
        if (!valid_n && rd_e != 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif
            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_hash_strm.write(hash_val);
            o_e_strm.write(false);
        }
    } while (last != 255);
    if (PADD) {
        o_key_strm.write(0);
        o_pld_strm.write(0);
        o_hash_strm.write(0);
    }
    o_e_strm.write(true);
#ifndef __SYNTHESIS__
    std::cout << "merged number=" << cnt << std::endl;
#endif
}

template <int PU>
void load_config(hls::stream<ap_uint<32> >& config,

                 ap_uint<32>& source_num,
                 ap_uint<32>& similarity_type,
                 ap_uint<32>& data_type,
                 ap_uint<32>& graph_type,

                 ap_uint<32> start_id[PU],
                 ap_uint<32> vertex_nm[PU],
                 ap_uint<32> edge_nm[PU]) {
#pragma HLS INLINE off

    source_num = config.read();
    similarity_type = config.read();
    data_type = config.read();
    graph_type = config.read();

    for (ap_uint<8> i = 0; i < PU; i++) start_id[i] = config.read();

    for (ap_uint<8> i = 0; i < PU; i++) vertex_nm[i] = config.read();

    for (ap_uint<8> i = 0; i < PU; i++) edge_nm[i] = config.read();
}

template <int PU, int CHNM, int RAM_SZ, int WData, bool EN_FLOAT>
void load_source_vertex32(ap_uint<WData> num,
                          ap_uint<WData> similarity_type,
                          ap_uint<WData> data_type,
                          ap_uint<WData> graph_type,

                          hls::stream<ap_uint<WData> >& source_col_id,
                          hls::stream<ap_uint<WData> >& source_weight,
#ifndef __SYNTHESIS__
                          ap_uint<WData>* col_vector[PU],
                          ap_uint<WData>* sparse_weight_vector[PU],
                          ap_uint<WData * CHNM>* dense_weight_vector[PU],
#else
                          ap_uint<WData> col_vector[PU][1 << RAM_SZ],
                          ap_uint<WData> sparse_weight_vector[PU][1 << RAM_SZ],
                          ap_uint<WData * CHNM> dense_weight_vector[PU][(1 << RAM_SZ) / CHNM],
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

        if (graph_type == enums::SPARSE) {
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
        if (!((similarity_type == enums::JACCARD_SIMILARITY) && (graph_type == enums::SPARSE))) {
            // read ource eight
            weight = source_weight.read();

            // count non zero for jaccard dense
            if (weight != 0) cnt++;

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
                } else if (graph_type == enums::SPARSE) {
                    // jaccard sparse
                    sparse_weight_vector[k][j] = floatToBits<float, uint32_t>(1.0);

#ifndef __SYNTHESIS__
#ifdef DEBUG
                    if (k == 0) std::cout << "set weight to 1.0" << std::endl;
#endif
#endif
                } else {
                    // jaccard dense
                    sparse_weight_vector[k][j] = weight;
                }
            } else {
                if (similarity_type != enums::JACCARD_SIMILARITY) {
                    sparse_weight_vector[k][j] = weight;
                } else if (graph_type == enums::SPARSE) {
                    sparse_weight_vector[k][j] = 1;

#ifndef __SYNTHESIS__
#ifdef DEBUG
                    if (k == 0) std::cout << "set weight to 1" << std::endl;
#endif
#endif
                } else {
                    sparse_weight_vector[k][j] = weight;
                }
            }

            dense_weight_vector[k][addr] = weight_tmp;
        }

        if (i == CHNM - 1) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
            std::cout << std::dec << "source dense weight[" << addr << "]=";
            std::cout << std::hex << weight_tmp << std::endl;
#endif
#endif
            addr++;
            i = 0;
        } else {
            i++;
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << std::dec << "source sparse weight[" << j << "]=" << weight << std::endl;
#endif
#endif
    }

    // get normalization of source vertex
    if ((similarity_type == enums::JACCARD_SIMILARITY) && (graph_type == enums::DENSE)) {
        if (EN_FLOAT) {
            norm = floatToBits<float, uint32_t>((float)cnt);
        } else {
            norm = cnt;
        }
    } else if ((similarity_type == enums::JACCARD_SIMILARITY) && (graph_type == enums::SPARSE)) {
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
    std::cout << "norm=" << norm_float << std::endl;
    std::cout << "source_norm=" << norm << std::endl;
#endif
#endif
}

template <int PU, int CHNM, int RAM_SZ, int WData, bool EN_DOUBLE>
void load_source_vertex64(ap_uint<WData> num,
                          ap_uint<WData> similarity_type,
                          ap_uint<WData> data_type,
                          ap_uint<WData> graph_type,

                          hls::stream<ap_uint<WData> >& source_col_id,
                          hls::stream<ap_uint<WData> >& source_weight,
#ifndef __SYNTHESIS__
                          ap_uint<WData>* col_vector[PU],
                          ap_uint<WData>* sparse_weight_vector[PU],
                          ap_uint<WData * CHNM>* dense_weight_vector[PU],
#else
                          ap_uint<WData> col_vector[PU][1 << RAM_SZ],
                          ap_uint<WData> sparse_weight_vector[PU][1 << RAM_SZ],
                          ap_uint<WData * CHNM> dense_weight_vector[PU][(1 << RAM_SZ) / CHNM],
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

        if (graph_type == enums::SPARSE) {
            ap_uint<WData> col = source_col_id.read();

            if (col > max) max = col;
            for (ap_uint<8> i = 0; i < PU; i++) {
#pragma HLS unroll
                col_vector[i][j] = col;
            }
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

        if (!((similarity_type == enums::JACCARD_SIMILARITY) && (graph_type == enums::SPARSE))) {
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
                    sparse_weight_vector[k][j] = floatToBits<double, uint64_t>(weight_double);
                } else if (graph_type == enums::SPARSE) {
                    // jaccard sparse
                    sparse_weight_vector[k][j] = floatToBits<double, uint64_t>(1.0);
                } else {
                    // jaccard dense
                    sparse_weight_vector[k][j] = weight;
                }
            } else {
                if (similarity_type != enums::JACCARD_SIMILARITY) {
                    sparse_weight_vector[k][j] = weight;
                } else if (graph_type == enums::SPARSE) {
                    sparse_weight_vector[k][j] = 1;
                } else {
                    sparse_weight_vector[k][j] = weight;
                }
            }

            dense_weight_vector[k][addr] = weight_tmp;
        }

        if (i == CHNM - 1) {
            addr++;
            i = 0;
        }
        i++;
    }

    // get normalization of base vertex
    if ((similarity_type == enums::JACCARD_SIMILARITY) && (graph_type == enums::DENSE)) {
        if (EN_DOUBLE) {
            norm = floatToBits<double, uint64_t>((double)cnt);
        } else {
            norm = cnt;
        }
    } else if ((similarity_type == enums::JACCARD_SIMILARITY) && (graph_type == enums::SPARSE)) {
        if (EN_DOUBLE) {
            norm = floatToBits<double, uint64_t>((double)num);
        } else {
            norm = num;
        }
    } else {
        if (EN_DOUBLE) {
            norm_double = hls::sqrt(accum_double); // sqrt
            norm = floatToBits<double, uint64_t>(norm_double);
        } else {
            norm_uint64 = hls::sqrt(accum_uint32); // no uint64 sqrt
            norm = norm_uint64;
        }
    }
    max_col = max;
}

template <int CHNM, int WData, bool EN_FLOAT_POINT>
void findCorrelationSparse(hls::stream<ap_uint<WData> >& row_id,
                           hls::stream<ap_uint<WData> > col_id[CHNM],
                           hls::stream<ap_uint<WData> > weight_in[4][CHNM],
                           hls::stream<ap_uint<CHNM> >& compute_enable_in,
                           hls::stream<bool>& strm_in_end,

                           ap_uint<WData> similarity_type,
                           ap_uint<WData> data_type,
                           ap_uint<WData> source_num,
                           ap_uint<WData> max_col,
                           ap_uint<WData>* col_vector,
                           ap_uint<WData>* weight_vector,

                           hls::stream<ap_uint<WData> > row_id_out[4],
                           hls::stream<ap_uint<WData> > weight_out[4][CHNM],
                           hls::stream<ap_uint<CHNM> > compute_enable_out[4],
                           hls::stream<bool> strm_out_end[4]) {
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
                    current_weight[j] = weight_in[0][j].read();
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
            row_id_out[0].write(row);
            compute_enable_out[0].write(enable);
            strm_out_end[0].write(false);

            for (ap_uint<8> j = 0; j < CHNM; j++) {
#pragma HLS UNROLL
                if (enable[j] == 1) {
                    weight_out[0][j].write(result_weight[j]);
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
    for (ap_uint<8> j = 0; j < 4; j++) strm_out_end[j].write(true);
}

template <int CHNM, int WData, bool EN_FLOAT_POINT>
void findCorrelationDense(hls::stream<ap_uint<WData> >& row_id,
                          hls::stream<ap_uint<WData> > weight_in[4][CHNM],
                          hls::stream<ap_uint<CHNM> >& compute_enable_in,
                          hls::stream<bool>& strm_in_end,

                          ap_uint<32> vertex_num,
                          ap_uint<WData> similarity_type,
                          ap_uint<WData> data_type,
                          ap_uint<WData> source_num,
                          ap_uint<WData * CHNM>* weight_vector,

                          hls::stream<ap_uint<WData> > row_id_out[4],
                          hls::stream<ap_uint<WData> > weight_out[4][CHNM],
                          hls::stream<ap_uint<CHNM> > compute_enable_out[4],
                          hls::stream<bool> strm_out_end[4]) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
#ifdef DEBUG
    ap_uint<WData> rcnt = 0;
#endif
#endif

    ap_uint<CHNM> enable;
    ap_uint<WData> row[4];
    ap_int<WData> current_weight[4][CHNM];
#pragma HLS ARRAY_PARTITION variable = current_weight complete
    ap_int<WData * CHNM> source_weight;
    ap_int<WData> source_weight_tmp[CHNM];
#pragma HLS ARRAY_PARTITION variable = source_weight_tmp complete
    ap_int<WData> result_weight[4][CHNM];
#pragma HLS ARRAY_PARTITION variable = result_weight complete

    ap_uint<WData> base = source_num / CHNM;
    ap_uint<16> fraction = source_num % CHNM;
    ap_uint<WData> range = fraction == 0 ? (ap_uint<WData>)(base - 1) : base;

    bool strm_end = strm_in_end.read();
    ap_uint<WData> search_idx = 0;
    while (!strm_end) {
#pragma HLS PIPELINE II = 1

        row[0] = row_id.read();
        row[1] = row[0] + vertex_num;
        row[2] = row[0] + 2 * vertex_num;
        row[3] = row[0] + 3 * vertex_num;

        enable = compute_enable_in.read();
        strm_end = strm_in_end.read();

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "in cnt=" << rcnt << " row=" << row[0] << " source_idx=" << search_idx << std::endl;
        rcnt++;
#endif
#endif

        source_weight = weight_vector[search_idx];
        for (ap_uint<8> j = 0; j < CHNM; j++) {
#pragma HLS UNROLL
            source_weight_tmp[j] = source_weight((j + 1) * WData - 1, j * WData);

#ifndef __SYNTHESIS__
#ifdef DEBUG
            std::cout << "source weight[" << j << "]=" << source_weight_tmp[j] << std::endl;
#endif
#endif
        }

        for (ap_uint<8> i = 0; i < 4; i++) {
#pragma HLS UNROLL
            for (ap_uint<8> j = 0; j < CHNM; j++) {
#pragma HLS UNROLL
                if (enable[j] == 1) {
                    current_weight[i][j] = weight_in[i][j].read();

#ifndef __SYNTHESIS__
#ifdef DEBUG
                    std::cout << "current weight[" << i << "][" << j << "]=" << current_weight[i][j] << std::endl;
#endif
#endif
                }
            }
        }

        bool idx_overflow = search_idx >= source_num;
        for (ap_uint<8> i = 0; i < 4; i++) {
#pragma HLS UNROLL
            for (ap_uint<8> j = 0; j < CHNM; j++) {
#pragma HLS UNROLL
                if (EN_FLOAT_POINT && WData == 32) {
                    float float_tmp, in1, in2;
                    in1 = bitsToFloat<uint32_t, float>((uint32_t)source_weight_tmp[j]);
                    in2 = bitsToFloat<uint32_t, float>((uint32_t)current_weight[i][j]);

                    float_tmp = in1 * in2;

                    result_weight[i][j] = floatToBits<float, uint32_t>(float_tmp);
                } else if (EN_FLOAT_POINT && WData == 64 && data_type == enums::DOUBLE_TYPE) {
                    double double_tmp, in1, in2;
                    in1 = bitsToFloat<uint64_t, double>((uint64_t)source_weight_tmp[j]);
                    in2 = bitsToFloat<uint64_t, double>((uint64_t)current_weight[i][j]);

                    double_tmp = in1 * in2;

                    result_weight[i][j] = floatToBits<double, uint64_t>(double_tmp);
                } else {
                    result_weight[i][j] = source_weight_tmp[j] * current_weight[i][j];
                }
            }
        }

        if (search_idx < range) {
            search_idx++;
        } else {
            search_idx = 0;
        }

        for (ap_uint<8> i = 0; i < 4; i++) {
#pragma HLS UNROLL
            row_id_out[i].write(row[i]);
            compute_enable_out[i].write(enable);

            for (ap_uint<8> j = 0; j < CHNM; j++) {
#pragma HLS UNROLL
                if (enable[j] == 1) {
                    weight_out[i][j].write(result_weight[i][j]);

#ifndef __SYNTHESIS__
#ifdef DEBUG
                    std::cout << "correlation[" << i << "][" << j << "]=" << result_weight[i][j] << std::endl;
#endif
#endif
                }
            }
            strm_out_end[i].write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG
            std::cout << "out row[" << i << "]=" << row[i] << " enable=" << enable << std::endl;
#endif
#endif
        }
    }

    for (ap_uint<8> j = 0; j < 4; j++) strm_out_end[j].write(true);
}

template <int CHNM, int WData, bool EN_FLOAT_POINT>
void findCorrelation(hls::stream<ap_uint<WData> >& row_id,
                     hls::stream<ap_uint<WData> > col_id[CHNM],
                     hls::stream<ap_uint<WData> > weight_in[4][CHNM],
                     hls::stream<ap_uint<CHNM> >& compute_enable_in,
                     hls::stream<bool>& strm_in_end,

                     ap_uint<32> vertex_num,
                     ap_uint<WData> similarity_type,
                     ap_uint<WData> data_type,
                     ap_uint<WData> graph_type,
                     ap_uint<WData> source_num,
                     ap_uint<WData> max_col,

                     ap_uint<WData>* col_vector,
                     ap_uint<WData>* sparse_weight_vector,
                     ap_uint<WData * CHNM>* dense_weight_vector,

                     hls::stream<ap_uint<WData> > row_id_out[4],
                     hls::stream<ap_uint<WData> > weight_out[4][CHNM],
                     hls::stream<ap_uint<CHNM> > compute_enable_out[4],
                     hls::stream<bool> strm_out_end[4]) {
#pragma HLS INLINE off

    if (graph_type == enums::SPARSE) {
        findCorrelationSparse<CHNM, WData, EN_FLOAT_POINT>(
            row_id, col_id, weight_in, compute_enable_in, strm_in_end, similarity_type, data_type, source_num, max_col,
            col_vector, sparse_weight_vector, row_id_out, weight_out, compute_enable_out, strm_out_end);
    } else {
        findCorrelationDense<CHNM, WData, EN_FLOAT_POINT>(row_id, weight_in, compute_enable_in, strm_in_end, vertex_num,
                                                          similarity_type, data_type, source_num, dense_weight_vector,
                                                          row_id_out, weight_out, compute_enable_out, strm_out_end);
    }
}

template <typename DT>
DT adder(bool enable0, bool enable1, DT a, DT b) {
#pragma HLS inline

    DT result, in0, in1;
    if (enable0)
        in0 = a;
    else
        in0 = 0;

    if (enable1)
        in1 = b;
    else
        in1 = 0;

    result = in0 + in1;
    return result;
}

template <typename DT>
DT adder(DT a, DT b) {
#pragma HLS inline

    DT result;
    result = a + b;
    return result;
}

template <int CHNM, typename DT>
DT adder_tree_top(ap_uint<CHNM> enable, DT in[CHNM]) {
#pragma HLS inline

    DT level1[CHNM / 2];
#pragma HLS ARRAY_PARTITION variable = level1 complete dim = 1
    DT level2[CHNM / 4];
#pragma HLS ARRAY_PARTITION variable = level2 complete dim = 1
    DT level3[CHNM / 8];
#pragma HLS ARRAY_PARTITION variable = level3 complete dim = 1
    DT result;

    if (CHNM >= 2) {
        for (ap_uint<8> i = 0; i < CHNM / 2; i++)
            level1[i] = adder<DT>(enable[2 * i], enable[2 * i + 1], in[2 * i], in[2 * i + 1]);
    }

    if (CHNM >= 4) {
        for (ap_uint<8> i = 0; i < CHNM / 4; i++) level2[i] = adder<DT>(level1[2 * i], level1[2 * i + 1]);
    }

    if (CHNM >= 8) {
        for (ap_uint<8> i = 0; i < CHNM / 8; i++) level3[i] = adder<DT>(level2[2 * i], level2[2 * i + 1]);
    }

    if (CHNM == 16) {
        result = adder<DT>(level3[0], level3[1]);
    } else if (CHNM == 8)
        result = level3[0];
    else if (CHNM == 4)
        result = level2[0];
    else if (CHNM == 2)
        result = level1[0];

    return result;
}

template <int CHNM, int WData, int DispatchNM, bool EN_FLOAT_POINT>
void adderTree(hls::stream<ap_uint<WData> >& row_id_in,
               hls::stream<ap_uint<WData> > weight_in[CHNM],
               hls::stream<ap_uint<WData> > correlation_in[CHNM],
               hls::stream<ap_uint<CHNM> >& compute_enable,
               hls::stream<bool>& i_end,

               hls::stream<ap_uint<WData> > row_id_out[DispatchNM],
               hls::stream<ap_uint<WData> > weight_out[DispatchNM],
               hls::stream<ap_uint<WData> > correlation_out[DispatchNM],
               hls::stream<bool> o_end[DispatchNM]) {
#pragma HLS inline off

    // for weight
    ap_uint<WData> weight[CHNM];
    ap_uint<WData> weight_square[CHNM];
    ap_uint<WData> weight_result_uint;

    float weight_float[CHNM];
    float weight_square_float[CHNM];
    float weight_result_float;

    double weight_double[CHNM];
    double weight_square_double[CHNM];
    double weight_result_double;

    ap_uint<WData> weight_result;

    // for correlation
    ap_uint<WData> correlation[CHNM];
    ap_uint<WData> correlation_result_uint;

    float correlation_float[CHNM];
    float correlation_result_float;

    double correlation_double[CHNM];
    double correlation_result_double;

    ap_uint<WData> correlation_result;

    // loop
    bool end = i_end.read();
    while (!end) {
#pragma HLS PIPELINE II = 1

        ap_uint<WData> row_id = row_id_in.read();
        ap_uint<CHNM> enable = compute_enable.read();

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "in row=" << row_id << " enable=" << enable << std::endl;
#endif
#endif

        for (ap_uint<8> j = 0; j < CHNM; j++) {
#pragma HLS UNROLL
            if (enable[j] == 1) {
                weight[j] = weight_in[j].read();
                correlation[j] = correlation_in[j].read();
            } else {
                weight[j] = 0;
                correlation[j] = 0;
            }

#ifndef __SYNTHESIS__
#ifdef DEBUG
            std::cout << "weight[" << j << "]=" << weight[j] << " correlation[" << j << "]=" << correlation[j]
                      << std::endl;
#endif
#endif
        }
        end = i_end.read();

        for (ap_uint<8> i = 0; i < CHNM; i++) {
#pragma HLS unroll

            if (EN_FLOAT_POINT) {
                if (WData == 32) {
                    weight_float[i] = bitsToFloat<uint32_t, float>((uint32_t)weight[i]);
                    correlation_float[i] = bitsToFloat<uint32_t, float>((uint32_t)correlation[i]);
                    weight_square_float[i] = weight_float[i] * weight_float[i];
                } else if (WData == 64) {
                    weight_double[i] = bitsToFloat<uint64_t, double>((uint64_t)weight);
                    correlation_double[i] = bitsToFloat<uint64_t, double>((uint64_t)correlation);
                    weight_square_double[i] = weight_double[i] * weight_double[i];
                }
            } else {
                weight_square[i] = weight[i] * weight[i];
            }
        }

        if (WData == 32 && EN_FLOAT_POINT) {
            weight_result_float = adder_tree_top<CHNM, float>(enable, weight_square_float);
            correlation_result_float = adder_tree_top<CHNM, float>(enable, correlation_float);
        } else if (WData == 64 && EN_FLOAT_POINT) {
            weight_result_double = adder_tree_top<CHNM, double>(enable, weight_square_double);
            correlation_result_double = adder_tree_top<CHNM, double>(enable, correlation_double);
        } else {
            weight_result_uint = adder_tree_top<CHNM, ap_uint<WData> >(enable, weight_square);
            correlation_result_uint = adder_tree_top<CHNM, ap_uint<WData> >(enable, correlation);
        }

        if (WData == 64 && EN_FLOAT_POINT) {
            weight_result = floatToBits<double, uint64_t>(weight_result_double);
            correlation_result = floatToBits<double, uint64_t>(correlation_result_double);
        } else if (WData == 32 && EN_FLOAT_POINT) {
            weight_result = floatToBits<float, uint32_t>(weight_result_float);
            correlation_result = floatToBits<float, uint32_t>(correlation_result_float);
        } else {
            weight_result = weight_result_uint;
            correlation_result = correlation_result_uint;
        }

        if ((weight_result != 0) || (correlation_result != 0) || ((weight_result == 0) && (correlation_result == 0))) {
            ap_uint<4> idx;
            if (DispatchNM == 8)
                idx = row_id(2, 0);
            else if (DispatchNM == 4)
                idx = row_id(1, 0);
            else if (DispatchNM == 2)
                idx = row_id[0];
            else
                idx = 0;

            row_id_out[idx].write(row_id);
            weight_out[idx].write(weight_result);
            correlation_out[idx].write(correlation_result);
            o_end[idx].write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG
            std::cout << "out row=" << row_id << " enable=" << enable << " weight=" << weight_result
                      << " correlation=" << correlation_result << std::endl;
#endif
#endif
        }
    }

    for (ap_uint<4> i = 0; i < DispatchNM; i++) {
#pragma HLS UNROLL
        o_end[i].write(true);
    }
}

template <int WData, bool EN_FLOAT_POINT>
void accumulator(hls::stream<ap_uint<WData> >& row_id_in,
                 hls::stream<ap_uint<WData> >& weight_in,
                 hls::stream<ap_uint<WData> >& correlation_in,
                 hls::stream<bool>& strm_in_end,

                 hls::stream<ap_uint<WData> >& row_id_out,
                 hls::stream<ap_uint<WData> >& weight_out,
                 hls::stream<ap_uint<WData> >& correlation_out,
                 hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE off

    // for weight
    ap_uint<WData> weight;
    ap_uint<WData> weight_accum = 0;

    ap_uint<WData> weight_accum_uint = 0;

    float weight_float;
    float weight_accum_float = 0;

    double weight_double;
    double weight_accum_double = 0;

    // for correlation
    ap_uint<WData> correlation;
    ap_uint<WData> correlation_accum = 0;

    ap_uint<WData> correlation_accum_uint = 0;

    float correlation_float;
    float correlation_accum_float = 0;

    double correlation_double;
    double correlation_accum_double = 0;

    // loop control
    ap_uint<WData> row;
    ap_uint<WData> previous_row;
    ap_uint<WData> loop_cnt = 0;

    bool end = strm_in_end.read();
    while (!end) {
#pragma HLS PIPELINE II = 1

        row = row_id_in.read();
        weight = weight_in.read();
        correlation = correlation_in.read();
        end = strm_in_end.read();

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "in row=" << row << " weight=" << weight << " correlation=" << correlation << std::endl;
#endif
#endif

        if (loop_cnt == 0) {
            previous_row = row;
        }

        if (row != previous_row) {
            // output current value and reset accum
            if (EN_FLOAT_POINT && WData == 32) {
                weight_accum = floatToBits<float, uint32_t>(weight_accum_float);
                correlation_accum = floatToBits<float, uint32_t>(correlation_accum_float);
            } else if (EN_FLOAT_POINT && WData == 64) {
                weight_accum = floatToBits<double, uint64_t>(weight_accum_double);
                correlation_accum = floatToBits<double, uint64_t>(correlation_accum_double);
            } else {
                weight_accum = weight_accum_uint;
                correlation_accum = correlation_accum_uint;
            }

            row_id_out.write(previous_row);
            weight_out.write(weight_accum);
            correlation_out.write(correlation_accum);
            strm_out_end.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG
            std::cout << "out row=" << previous_row << " weight=" << weight_accum
                      << " correlation=" << correlation_accum << std::endl;
#endif
#endif

            weight_accum_uint = 0;
            weight_accum_float = 0;
            weight_accum_double = 0;

            correlation_accum_uint = 0;
            correlation_accum_float = 0;
            correlation_accum_double = 0;
        }

        // accumulator
        if (EN_FLOAT_POINT && WData == 32) {
            weight_accum_float += bitsToFloat<uint32_t, float>((uint32_t)weight);
            correlation_accum_float += bitsToFloat<uint32_t, float>((uint32_t)correlation);
        } else if (EN_FLOAT_POINT && WData == 64) {
            weight_accum_double += bitsToFloat<uint64_t, double>((uint64_t)weight);
            correlation_accum_double += bitsToFloat<uint64_t, double>((uint64_t)correlation);
        } else {
            weight_accum_uint += weight;
            correlation_accum_uint += correlation;
        }

        previous_row = row;
        loop_cnt++;
    }

    // output the last element
    if (loop_cnt != 0) {
        if (EN_FLOAT_POINT && WData == 32) {
            weight_accum = floatToBits<float, uint32_t>(weight_accum_float);
            correlation_accum = floatToBits<float, uint32_t>(correlation_accum_float);
        } else if (EN_FLOAT_POINT && WData == 64) {
            weight_accum = floatToBits<double, uint64_t>(weight_accum_double);
            correlation_accum = floatToBits<double, uint64_t>(correlation_accum_double);
        } else {
            weight_accum = weight_accum_uint;
            correlation_accum = correlation_accum_uint;
        }

        row_id_out.write(previous_row);
        weight_out.write(weight_accum);
        correlation_out.write(correlation_accum);
        strm_out_end.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "out row=" << previous_row << " weight=" << weight_accum << " correlation=" << correlation_accum
                  << std::endl;
#endif
#endif
    }
    // last flag && padd
    row_id_out.write(0);
    weight_out.write(0);
    correlation_out.write(0);
    strm_out_end.write(true);
}

template <int WData, bool EN_FLOAT_POINT>
void ALU(ap_uint<WData> similarity_type,
         ap_uint<WData> source_norm,

         hls::stream<ap_uint<WData> >& row_id_in,
         hls::stream<ap_uint<WData> >& square,
         hls::stream<ap_uint<WData> >& correlation,
         hls::stream<bool>& strm_in_end,

         hls::stream<ap_uint<WData> >& row_id_out,
         hls::stream<float>& similarity_out,
         hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE off

    bool end = strm_in_end.read();
    while (!end) {
#pragma HLS PIPELINE
        ap_uint<WData> row_id = row_id_in.read();
        ap_uint<WData> current_correlation = correlation.read();
        ap_uint<WData> current_square = square.read();
        end = strm_in_end.read();

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "in row=" << row_id << " correlation=" << current_correlation << " square=" << current_square
                  << " source_norm=" << source_norm << std::endl;
#endif
#endif
        float similarity;
        if (EN_FLOAT_POINT && (WData == 32)) {
            if (bitsToFloat<uint32_t, float>(current_square) == 0 && bitsToFloat<uint32_t, float>(source_norm) == 0) {
                similarity = 1.0;
            } else if (bitsToFloat<uint32_t, float>(current_square) == 0 ||
                       bitsToFloat<uint32_t, float>(source_norm) == 0) {
                similarity = 0;
            } else {
                float square_tmp = bitsToFloat<uint32_t, float>(current_square);
                float current_norm = hls::sqrt(square_tmp);
                float a, b, c, d;
                a = bitsToFloat<uint32_t, float>(current_correlation);
                if (similarity_type == enums::JACCARD_SIMILARITY) {
                    b = bitsToFloat<uint32_t, float>(current_square);
                    c = bitsToFloat<uint32_t, float>(source_norm);
                    d = b + c - a;
                } else {
                    b = current_norm;
                    c = bitsToFloat<uint32_t, float>(source_norm);
                    d = b * c;
                }

                similarity = a / d;
#ifndef __SYNTHESIS__
#ifdef DEBUG
                std::cout << "out row=" << row_id << " correlation=" << a << " current=" << b << " source=" << c
                          << " divider=" << d << " similarity1=" << similarity << std::endl;
#endif
#endif
            }
        } else if (EN_FLOAT_POINT && (WData == 64)) {
            if (bitsToFloat<uint64_t, double>(current_square) == 0 && bitsToFloat<uint64_t, double>(source_norm) == 0) {
                similarity = 1.0;
            } else if (bitsToFloat<uint64_t, double>(current_square) == 0 ||
                       bitsToFloat<uint64_t, double>(source_norm) == 0) {
                similarity = 0;
            } else {
                float current_norm = hls::sqrt((float)bitsToFloat<uint64_t, double>(current_square));
                float a, b, c, d;
                a = bitsToFloat<uint64_t, double>(current_correlation);
                b = current_norm;
                c = bitsToFloat<uint64_t, double>(source_norm);
                if (similarity_type == enums::JACCARD_SIMILARITY) {
                    b = bitsToFloat<uint32_t, float>(current_square);
                    c = bitsToFloat<uint64_t, double>(source_norm);
                    d = b + c - a;
                } else {
                    b = current_norm;
                    c = bitsToFloat<uint64_t, double>(source_norm);
                    d = b * c;
                }
                similarity = a / d;
#ifndef __SYNTHESIS__
#ifdef DEBUG
                std::cout << "out row=" << row_id << " correlation=" << a << " current=" << b << " source=" << c
                          << " divider=" << d << " similarity1=" << similarity << std::endl;
#endif
#endif
            }
        } else {
            if (current_square == 0 && source_norm == 0) {
                similarity = 1.0;
            } else if (current_square == 0 || source_norm == 0) {
                similarity = 0;
            } else {
                float current_norm = hls::sqrt((float)current_square);
                float a, b, c, d;
                a = current_correlation;
                if (similarity_type == enums::JACCARD_SIMILARITY) {
                    b = current_square;
                    c = source_norm;
                    d = current_square + source_norm - current_correlation;
                } else {
                    b = current_norm;
                    c = bitsToFloat<uint32_t, float>(source_norm);
                    d = b * c;
                }

                similarity = a / d;

#ifndef __SYNTHESIS__
#ifdef DEBUG
                std::cout << "out row=" << row_id << " correlation=" << a << " current=" << b << " source=" << c
                          << " divider=" << d << " similarity0=" << similarity << std::endl;
#endif
#endif
            }
        }

        if (similarity != 0) {
            row_id_out.write(row_id);
            similarity_out.write(similarity);
            strm_out_end.write(false);
        }
    }

    // padd
    row_id_out.write(0);
    similarity_out.write(0);
    strm_out_end.write(true);
}

template <int CHNM, int WData, int ACCU_NUM, bool EN_FLOAT_POINT, bool PADD>
void accumAddTree(hls::stream<ap_uint<WData> >& row_in,
                  hls::stream<ap_uint<WData> > weight_in[CHNM],
                  hls::stream<ap_uint<WData> > correlation_in[CHNM],
                  hls::stream<ap_uint<CHNM> >& compute_enable,
                  hls::stream<bool>& strm_in_end,

                  ap_uint<WData> similarity_type,
                  ap_uint<WData> source_norm,

                  hls::stream<ap_uint<WData> >& row_out,
                  hls::stream<ap_uint<WData> >& weight_out,
                  hls::stream<ap_uint<WData> >& correlation_out,
                  hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "----------------add_tree----------------" << std::endl;
#endif
#endif

    hls::stream<ap_uint<WData> > row0[ACCU_NUM];
    hls::stream<ap_uint<WData> > weight0[ACCU_NUM];
    hls::stream<ap_uint<WData> > correlation0[ACCU_NUM];
    hls::stream<bool> strm_end0[ACCU_NUM];

    if (PADD) {
#pragma HLS stream variable = row0 depth = 512
#pragma HLS array_partition variable = row0 complete
#pragma HLS resource variable = row0 core = FIFO_BRAM

#pragma HLS stream variable = weight0 depth = 8
#pragma HLS array_partition variable = weight0 complete
#pragma HLS resource variable = weight0 core = FIFO_SRL

#pragma HLS stream variable = correlation0 depth = 8
#pragma HLS array_partition variable = correlation0 complete
#pragma HLS resource variable = correlation0 core = FIFO_SRL

#pragma HLS stream variable = strm_end0 depth = 512
#pragma HLS array_partition variable = strm_end0 complete
#pragma HLS resource variable = strm_end0 core = FIFO_SRL
    } else {
#pragma HLS stream variable = row0 depth = 32
#pragma HLS array_partition variable = row0 complete
#pragma HLS resource variable = row0 core = FIFO_SRL

#pragma HLS stream variable = weight0 depth = 8
#pragma HLS array_partition variable = weight0 complete
#pragma HLS resource variable = weight0 core = FIFO_SRL

#pragma HLS stream variable = correlation0 depth = 8
#pragma HLS array_partition variable = correlation0 complete
#pragma HLS resource variable = correlation0 core = FIFO_SRL

#pragma HLS stream variable = strm_end0 depth = 32
#pragma HLS array_partition variable = strm_end0 complete
#pragma HLS resource variable = strm_end0 core = FIFO_SRL
    }

    adderTree<CHNM, WData, ACCU_NUM, EN_FLOAT_POINT>(row_in, weight_in, correlation_in, compute_enable, strm_in_end,
                                                     row0, weight0, correlation0, strm_end0);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "----------------accumulator-------------" << std::endl;
#endif
#endif

    hls::stream<ap_uint<WData> > row1[ACCU_NUM];
#pragma HLS stream variable = row1 depth = 8
#pragma HLS array_partition variable = row1 complete
#pragma HLS resource variable = row1 core = FIFO_SRL
    hls::stream<ap_uint<WData> > weight1[ACCU_NUM];
#pragma HLS stream variable = weight1 depth = 8
#pragma HLS array_partition variable = weight1 complete
#pragma HLS resource variable = weight1 core = FIFO_SRL
    hls::stream<ap_uint<WData> > correlation1[ACCU_NUM];
#pragma HLS stream variable = correlation1 depth = 8
#pragma HLS array_partition variable = correlation1 complete
#pragma HLS resource variable = correlation1 core = FIFO_SRL
    hls::stream<bool> strm_end1[ACCU_NUM];
#pragma HLS stream variable = strm_end1 depth = 8
#pragma HLS array_partition variable = strm_end1 complete
#pragma HLS resource variable = strm_end1 core = FIFO_SRL

    for (ap_uint<8> i = 0; i < ACCU_NUM; i++) {
#pragma HLS UNROLL
        accumulator<WData, EN_FLOAT_POINT>(row0[i], weight0[i], correlation0[i], strm_end0[i], row1[i], weight1[i],
                                           correlation1[i], strm_end1[i]);
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "------------mergeAccumulator-----------" << std::endl;
#endif
#endif

    if (ACCU_NUM == 8) {
        merge8_1<WData, WData, ap_uint<WData>, PADD>(
            row1[0], weight1[0], correlation1[0], strm_end1[0], row1[1], weight1[1], correlation1[1], strm_end1[1],
            row1[2], weight1[2], correlation1[2], strm_end1[2], row1[3], weight1[3], correlation1[3], strm_end1[3],
            row1[4], weight1[4], correlation1[4], strm_end1[4], row1[5], weight1[5], correlation1[5], strm_end1[5],
            row1[6], weight1[6], correlation1[6], strm_end1[6], row1[7], weight1[7], correlation1[7], strm_end1[7],
            row_out, weight_out, correlation_out, strm_out_end);
    } else if (ACCU_NUM == 4) {
        merge4_1<WData, WData, ap_uint<WData>, PADD>(
            row1[0], weight1[0], correlation1[0], strm_end1[0], row1[1], weight1[1], correlation1[1], strm_end1[1],
            row1[2], weight1[2], correlation1[2], strm_end1[2], row1[3], weight1[3], correlation1[3], strm_end1[3],
            row_out, weight_out, correlation_out, strm_out_end);
    } else if (ACCU_NUM == 2) {
        merge2_1<WData, WData, ap_uint<WData>, PADD>(row1[0], weight1[0], correlation1[0], strm_end1[0], row1[1],
                                                     weight1[1], correlation1[1], strm_end1[1], row_out, weight_out,
                                                     correlation_out, strm_out_end);
    } else {
        merge1_1<WData, WData, ap_uint<WData>, PADD>(row1[0], weight1[0], correlation1[0], strm_end1[0], row_out,
                                                     weight_out, correlation_out, strm_out_end);
    }
}

template <int CHNM, int WData, int TreeNM, bool EN_FLOAT_POINT>
void accumAddTreeTop(hls::stream<ap_uint<WData> > row_in[TreeNM],
                     hls::stream<ap_uint<WData> > weight_in[TreeNM][CHNM],
                     hls::stream<ap_uint<WData> > correlation_in[TreeNM][CHNM],
                     hls::stream<ap_uint<CHNM> > compute_enable[TreeNM],
                     hls::stream<bool> strm_in_end[TreeNM],

                     ap_uint<WData> similarity_type,
                     ap_uint<WData> source_norm,

                     hls::stream<ap_uint<WData> > row_out[TreeNM],
                     hls::stream<ap_uint<WData> > weight_out[TreeNM],
                     hls::stream<ap_uint<WData> > correlation_out[TreeNM],
                     hls::stream<bool> strm_out_end[TreeNM]) {
#pragma HLS INLINE

    for (ap_uint<8> i = 0; i < TreeNM; i++) {
#pragma HLS UNROLL

        accumAddTree<CHNM, WData, (EN_FLOAT_POINT ? 2 : 1), EN_FLOAT_POINT, true>(
            row_in[i], weight_in[i], correlation_in[i], compute_enable[i], strm_in_end[i], similarity_type, source_norm,
            row_out[i], weight_out[i], correlation_out[i], strm_out_end[i]);
    }
}

template <int CHNM, int WData, int RAM_SZ, bool EN_FLOAT_POINT>
void similarity_processing_unit(
    // input
    hls::stream<ap_uint<WData * CHNM> >& strm_in0,
    hls::stream<ap_uint<WData * CHNM> >& strm_in1,
    hls::stream<ap_uint<WData * CHNM> >& strm_in2,
    hls::stream<ap_uint<WData * CHNM> >& strm_in3,

    // config
    ap_uint<32> similarity_type,
    ap_uint<32> data_type,
    ap_uint<32> graph_type,
    ap_uint<32> row_nm,
    ap_uint<32> col_nm,
    ap_uint<32> start_id,
    ap_uint<32> source_num,
    ap_uint<WData> source_norm,
    ap_uint<32> max_col,

    // ram
    ap_uint<WData>* col_vector,
    ap_uint<WData>* sparse_weight_vector,
    ap_uint<WData * CHNM>* dense_weight_vector,

    // output
    hls::stream<ap_uint<WData> >& rowID,
    hls::stream<float>& similarity,
    hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "-----------------InputMux---------------" << std::endl;
#endif
#endif

    hls::stream<ap_uint<WData> > row0;
#pragma HLS stream variable = row0 depth = 512
#pragma HLS resource variable = row0 core = FIFO_BRAM
    hls::stream<ap_uint<WData> > col0[CHNM];
#pragma HLS stream variable = col0 depth = 8
#pragma HLS array_partition variable = col0 complete
#pragma HLS resource variable = col0 core = FIFO_SRL
    hls::stream<ap_uint<WData> > weight0[4][CHNM];
#pragma HLS stream variable = weight0 depth = 8
#pragma HLS array_partition variable = weight0 complete
#pragma HLS resource variable = weight0 core = FIFO_SRL
    hls::stream<ap_uint<WData> > weight1[4][CHNM];
#pragma HLS stream variable = weight1 depth = 512
#pragma HLS array_partition variable = weight1 complete
#pragma HLS resource variable = weight1 core = FIFO_BRAM
    hls::stream<ap_uint<CHNM> > compute_enable0;
#pragma HLS stream variable = compute_enable0 depth = 512
#pragma HLS resource variable = compute_enable0 core = FIFO_BRAM
    hls::stream<bool> strm_end0;
#pragma HLS stream variable = strm_end0 depth = 512
#pragma HLS resource variable = strm_end0 core = FIFO_SRL

    inputMux<CHNM, WData, EN_FLOAT_POINT>(similarity_type, data_type, graph_type, row_nm, col_nm, start_id, strm_in0,
                                          strm_in1, strm_in2, strm_in3, row0, col0, weight0, weight1, compute_enable0,
                                          strm_end0);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "--------------findCorrelation--------------" << std::endl;
#endif
#endif

    hls::stream<ap_uint<WData> > row1[4];
#pragma HLS stream variable = row1 depth = 512
#pragma HLS array_partition variable = row1 complete
#pragma HLS resource variable = row1 core = FIFO_BRAM
    hls::stream<ap_uint<WData> > correlation1[4][CHNM];
#pragma HLS stream variable = correlation1 depth = 8
#pragma HLS array_partition variable = correlation1 complete
#pragma HLS resource variable = correlation1 core = FIFO_SRL
    hls::stream<ap_uint<CHNM> > compute_enable1[4];
#pragma HLS stream variable = compute_enable1 depth = 512
#pragma HLS array_partition variable = compute_enable1 complete
#pragma HLS resource variable = compute_enable1 core = FIFO_BRAM
    hls::stream<bool> strm_end1[4];
#pragma HLS stream variable = strm_end1 depth = 512
#pragma HLS array_partition variable = strm_end1 complete
#pragma HLS resource variable = strm_end1 core = FIFO_SRL

    findCorrelation<CHNM, WData, EN_FLOAT_POINT>(
        row0, col0, weight0, compute_enable0, strm_end0, row_nm, similarity_type, data_type, graph_type, source_num,
        max_col, col_vector, sparse_weight_vector, dense_weight_vector, row1, correlation1, compute_enable1, strm_end1);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "----------------accumAddTree----------------" << std::endl;
#endif
#endif

    hls::stream<ap_uint<WData> > row2[4];
#pragma HLS stream variable = row2 depth = 8
#pragma HLS array_partition variable = row2 complete
#pragma HLS resource variable = row2 core = FIFO_SRL
    hls::stream<ap_uint<WData> > weight2[4];
#pragma HLS stream variable = weight2 depth = 8
#pragma HLS array_partition variable = weight2 complete
#pragma HLS resource variable = weight2 core = FIFO_SRL
    hls::stream<ap_uint<WData> > correlation2[4];
#pragma HLS stream variable = correlation2 depth = 8
#pragma HLS array_partition variable = correlation2 complete
#pragma HLS resource variable = correlation2 core = FIFO_SRL
    hls::stream<bool> strm_end2[4];
#pragma HLS stream variable = strm_end2 depth = 8
#pragma HLS array_partition variable = strm_end2 complete
#pragma HLS resource variable = strm_end2 core = FIFO_SRL

    accumAddTreeTop<CHNM, WData, 4, EN_FLOAT_POINT>(row1, weight1, correlation1, compute_enable1, strm_end1,
                                                    similarity_type, source_norm, row2, weight2, correlation2,
                                                    strm_end2);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "---------------mergeAddTree---------------" << std::endl;
#endif
#endif

    hls::stream<ap_uint<WData> > row3;
#pragma HLS stream variable = row3 depth = 8
#pragma HLS resource variable = row3 core = FIFO_SRL
    hls::stream<ap_uint<WData> > weight3;
#pragma HLS stream variable = weight3 depth = 8
#pragma HLS resource variable = weight3 core = FIFO_SRL
    hls::stream<ap_uint<WData> > correlation3;
#pragma HLS stream variable = correlation3 depth = 8
#pragma HLS resource variable = correlation3 core = FIFO_SRL
    hls::stream<bool> strm_end3;
#pragma HLS stream variable = strm_end3 depth = 8
#pragma HLS resource variable = strm_end3 core = FIFO_SRL

    merge4_1<WData, WData, ap_uint<WData>, false>(row2[0], weight2[0], correlation2[0], strm_end2[0], row2[1],
                                                  weight2[1], correlation2[1], strm_end2[1], row2[2], weight2[2],
                                                  correlation2[2], strm_end2[2], row2[3], weight2[3], correlation2[3],
                                                  strm_end2[3], row3, weight3, correlation3, strm_end3);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "------------ALU(sqrt and divide)------------" << std::endl;
#endif
#endif

    ALU<WData, EN_FLOAT_POINT>(similarity_type, source_norm, row3, weight3, correlation3, strm_end3, rowID, similarity,
                               strm_out_end);
}

template <int CHNM, int PU, int WData, int RAM_SZ, bool EN_FLOAT_POINT>
void similarity_processing_unit_wrapper(
    // input
    hls::stream<ap_uint<WData * CHNM> > strm_in0[PU],
    hls::stream<ap_uint<WData * CHNM> > strm_in1[PU],
    hls::stream<ap_uint<WData * CHNM> > strm_in2[PU],
    hls::stream<ap_uint<WData * CHNM> > strm_in3[PU],

    // config
    ap_uint<32> similarity_type,
    ap_uint<32> data_type,
    ap_uint<32> graph_type,
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
    ap_uint<WData * CHNM>* dense_weight_vector[PU],
#else
    ap_uint<WData> col_vector[PU][1 << RAM_SZ],
    ap_uint<WData> sparse_weight_vector[PU][1 << RAM_SZ],
    ap_uint<WData * CHNM> dense_weight_vector[PU][(1 << RAM_SZ) / CHNM],
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
            strm_in0[i], strm_in1[i], strm_in2[i], strm_in3[i], similarity_type, data_type, graph_type, row_nm[i],
            col_nm[i], start_id[i], source_num, source_norm, max_col, col_vector[i], sparse_weight_vector[i],
            dense_weight_vector[i], rowID[i], similarity[i], strm_out_end[i]);
    }
}

template <int CHNM, int PU, int WData, int RAM_SZ, bool EN_FLOAT_POINT>
void similarityTop(
    // input
    hls::stream<ap_uint<WData * CHNM> > strm_in0[PU],
    hls::stream<ap_uint<WData * CHNM> > strm_in1[PU],
    hls::stream<ap_uint<WData * CHNM> > strm_in2[PU],
    hls::stream<ap_uint<WData * CHNM> > strm_in3[PU],

    // config
    ap_uint<32> similarity_type,
    ap_uint<32> data_type,
    ap_uint<32> graph_type,
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
    ap_uint<WData * CHNM>* dense_weight_vector[PU],
#else
    ap_uint<WData> col_vector[PU][1 << RAM_SZ],
    ap_uint<WData> sparse_weight_vector[PU][1 << RAM_SZ],
    ap_uint<WData * CHNM> dense_weight_vector[PU][(1 << RAM_SZ) / CHNM],
#endif

    // output
    hls::stream<ap_uint<WData> >& rowID,
    hls::stream<float>& similarity,
    hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<32> > rowID1[16];
#pragma HLS stream variable = rowID1 depth = 8
#pragma HLS array_partition variable = rowID1 complete
#pragma HLS resource variable = rowID1 core = FIFO_SRL
    hls::stream<float> similarity_strm1[16];
#pragma HLS stream variable = similarity_strm1 depth = 8
#pragma HLS array_partition variable = similarity_strm1 complete
#pragma HLS resource variable = similarity_strm1 core = FIFO_SRL
    hls::stream<bool> strm_end1[16];
#pragma HLS stream variable = strm_end1 depth = 8
#pragma HLS array_partition variable = strm_end1 complete
#pragma HLS resource variable = strm_end1 core = FIFO_SRL

    similarity_processing_unit_wrapper<CHNM, PU, WData, RAM_SZ, EN_FLOAT_POINT>(
        strm_in0, strm_in1, strm_in2, strm_in3, similarity_type, data_type, graph_type, row_nm, col_nm, start_id,
        source_num, source_norm, max_col, col_vector, sparse_weight_vector, dense_weight_vector, rowID1,
        similarity_strm1, strm_end1);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "-------------------------collectPU-----------------------" << std::endl;
#endif
#endif

    if (PU == 1) {
        collect1_1<32, float, false>(rowID1[0], similarity_strm1[0], strm_end1[0], rowID, similarity, strm_out_end);
    } else if (PU == 2) {
        collect2_1<32, float, false>(rowID1[0], similarity_strm1[0], strm_end1[0], rowID1[1], similarity_strm1[1],
                                     strm_end1[1], rowID, similarity, strm_out_end);
    } else if (PU == 4) {
        collect4_1<32, float, false>(rowID1[0], similarity_strm1[0], strm_end1[0], rowID1[1], similarity_strm1[1],
                                     strm_end1[1], rowID1[2], similarity_strm1[2], strm_end1[2], rowID1[3],
                                     similarity_strm1[3], strm_end1[3], rowID, similarity, strm_out_end);
    } else if (PU == 8) {
        collect8_1<32, float, false>(rowID1[0], similarity_strm1[0], strm_end1[0], rowID1[1], similarity_strm1[1],
                                     strm_end1[1], rowID1[2], similarity_strm1[2], strm_end1[2], rowID1[3],
                                     similarity_strm1[3], strm_end1[3], rowID1[4], similarity_strm1[4], strm_end1[4],
                                     rowID1[5], similarity_strm1[5], strm_end1[5], rowID1[6], similarity_strm1[6],
                                     strm_end1[6], rowID1[7], similarity_strm1[7], strm_end1[7], rowID, similarity,
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

        collect4_1<32, float, true>(rowID1[0], similarity_strm1[0], strm_end1[0], rowID1[1], similarity_strm1[1],
                                    strm_end1[1], rowID1[2], similarity_strm1[2], strm_end1[2], rowID1[3],
                                    similarity_strm1[3], strm_end1[3], row_id_tmp[0], similarity_tmp[0],
                                    tmp_out_end[0]);

        collect4_1<32, float, true>(rowID1[4], similarity_strm1[4], strm_end1[4], rowID1[5], similarity_strm1[5],
                                    strm_end1[5], rowID1[6], similarity_strm1[6], strm_end1[6], rowID1[7],
                                    similarity_strm1[7], strm_end1[7], row_id_tmp[1], similarity_tmp[1],
                                    tmp_out_end[1]);

        collect4_1<32, float, true>(rowID1[8], similarity_strm1[8], strm_end1[8], rowID1[9], similarity_strm1[9],
                                    strm_end1[9], rowID1[10], similarity_strm1[10], strm_end1[10], rowID1[11],
                                    similarity_strm1[11], strm_end1[11], row_id_tmp[2], similarity_tmp[2],
                                    tmp_out_end[2]);

        collect4_1<32, float, true>(rowID1[12], similarity_strm1[12], strm_end1[12], rowID1[13], similarity_strm1[13],
                                    strm_end1[13], rowID1[14], similarity_strm1[14], strm_end1[14], rowID1[15],
                                    similarity_strm1[15], strm_end1[15], row_id_tmp[3], similarity_tmp[3],
                                    tmp_out_end[3]);

        collect4_1<32, float, false>(row_id_tmp[0], similarity_tmp[0], tmp_out_end[0], row_id_tmp[1], similarity_tmp[1],
                                     tmp_out_end[1], row_id_tmp[2], similarity_tmp[2], tmp_out_end[2], row_id_tmp[3],
                                     similarity_tmp[3], tmp_out_end[3], rowID, similarity, strm_out_end);
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "-------------------------finish------------------------" << std::endl;
#endif
#endif
}
} // general_similarity
} // internal

/**
 * @brief similarity function which support both dense and sparse graph. It also support both Jaccard and Cosine
 * Similarity.
 *
 * @tparam CHNM the channel number of input data
 * @tparam PU the number of processing unit
 * @tparam WData the width of input data
 * @tparam RAM_SZ the log size of internal URAM
 * @tparam EN_FLOAT_POINT if it is true, the primitive will support both float and int type of input. Otherwise, it only
 * support int. Multiple channel of float input should be compacted as type of ap_uint.
 *
 * @param config the control parameter of the primitive which contains: sourceNUM, similarityType, graphType, dataType,
 * startID, rowNUM and colNUM of each processing unit(PU)
 * @param sourceIndice input indice as source for computing similarity
 * @param sourceWeight input weight as source for computing similarity
 * @param strmIn0 input muti-channel data stream for PU0
 * @param strmIn1 input muti-channel data stream for PU1
 * @param strmIn2 input muti-channel data stream for PU2
 * @param strmIn3 input muti-channel data stream for PU3
 * @param rowID output result ID stream
 * @param similarity output similarity value corresponding to its ID
 * @param strmOutEnd end flag stream for output
 */
template <int CHNM, int PU, int WData, int RAM_SZ, bool EN_FLOAT_POINT>
void generalSimilarity(hls::stream<ap_uint<32> >& config,

                       hls::stream<ap_uint<WData> >& sourceIndice,
                       hls::stream<ap_uint<WData> >& sourceWeight,

                       hls::stream<ap_uint<WData * CHNM> > strmIn0[PU],
                       hls::stream<ap_uint<WData * CHNM> > strmIn1[PU],
                       hls::stream<ap_uint<WData * CHNM> > strmIn2[PU],
                       hls::stream<ap_uint<WData * CHNM> > strmIn3[PU],

                       hls::stream<ap_uint<WData> >& rowID,
                       hls::stream<float>& similarity,
                       hls::stream<bool>& strmOutEnd) {
#pragma HLS INLINE off

#pragma HLS array_partition variable = strmIn0 complete
#pragma HLS array_partition variable = strmIn1 complete
#pragma HLS array_partition variable = strmIn2 complete
#pragma HLS array_partition variable = strmIn3 complete

    // define URAM structure
    const int RAM_Size = 1 << RAM_SZ;

#ifndef __SYNTHESIS__

    ap_uint<WData>* col_vector[PU];
    ap_uint<WData>* sparse_weight_vector[PU];
    ap_uint<WData * CHNM>* dense_weight_vector[PU];

    for (int i = 0; i < PU; i++) {
        col_vector[i] = (ap_uint<WData>*)malloc(RAM_Size * sizeof(ap_uint<WData>));
        sparse_weight_vector[i] = (ap_uint<WData>*)malloc(RAM_Size * sizeof(ap_uint<WData>));
        dense_weight_vector[i] = (ap_uint<WData * CHNM>*)malloc((RAM_Size / CHNM) * sizeof(ap_uint<WData * CHNM>));
    }

#else

    ap_uint<WData> col_vector[PU][RAM_Size];
#pragma HLS resource variable = col_vector core = RAM_2P_URAM
#pragma HLS ARRAY_PARTITION variable = col_vector dim = 1

    ap_uint<WData> sparse_weight_vector[PU][RAM_Size];
#pragma HLS resource variable = sparse_weight_vector core = RAM_2P_URAM
#pragma HLS ARRAY_PARTITION variable = sparse_weight_vector dim = 1

    ap_uint<WData * CHNM> dense_weight_vector[PU][RAM_Size / CHNM];
#pragma HLS resource variable = dense_weight_vector core = RAM_2P_URAM
#pragma HLS ARRAY_PARTITION variable = dense_weight_vector dim = 1

#endif

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "------------------------load_config---------------------" << std::endl;
#endif
#endif

    ap_uint<32> source_num;
    ap_uint<32> similarity_type;
    ap_uint<32> data_type;
    ap_uint<32> graph_type;

    ap_uint<32> start_id[PU];
#pragma HLS array_partition variable = start_id complete
    ap_uint<32> row_nm[PU];
#pragma HLS array_partition variable = row_nm complete
    ap_uint<32> col_nm[PU];
#pragma HLS array_partition variable = col_nm complete

    internal::general_similarity::load_config<PU>(config, source_num, similarity_type, data_type, graph_type, start_id,
                                                  row_nm, col_nm);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "--------------------load_source_vertex------------------" << std::endl;
#endif
#endif

    ap_uint<WData> source_norm;
    ap_uint<WData> max_col;

    if (WData == 64) {
        internal::general_similarity::load_source_vertex64<PU, CHNM, RAM_SZ, WData, EN_FLOAT_POINT>(
            source_num, similarity_type, data_type, graph_type, sourceIndice, sourceWeight, col_vector,
            sparse_weight_vector, dense_weight_vector, source_norm, max_col);
    } else {
        internal::general_similarity::load_source_vertex32<PU, CHNM, RAM_SZ, WData, EN_FLOAT_POINT>(
            source_num, similarity_type, data_type, graph_type, sourceIndice, sourceWeight, col_vector,
            sparse_weight_vector, dense_weight_vector, source_norm, max_col);
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "----------------------similarity_top--------------------" << std::endl;
#endif
#endif

    internal::general_similarity::similarityTop<CHNM, PU, WData, RAM_SZ, EN_FLOAT_POINT>(
        strmIn0, strmIn1, strmIn2, strmIn3, similarity_type, data_type, graph_type, row_nm, col_nm, start_id,
        source_num, source_norm, max_col, col_vector, sparse_weight_vector, dense_weight_vector, rowID, similarity,
        strmOutEnd);

#ifndef __SYNTHESIS__
    for (int i = 0; i < PU; i++) {
        free(col_vector[i]);
        free(dense_weight_vector[i]);
        free(sparse_weight_vector[i]);
    }
#endif
}
} // graph

} // xf

#endif
