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

// top header
#include <hls_stream.h>
#include "hash_aggr_kernel.hpp"

// used modules
#include "xf_database/scan_col.hpp"
#include "xf_database/aggregate.hpp"
#include "xf_database/hash_group_aggregate.hpp"
#include "xf_utils_hw/stream_to_axi.hpp"

#define DEBUG

// ------------------------------------------------------------
void read_config(hls::stream<ap_uint<32> >& config_strm, ap_uint<32> config[PU_STATUS_DEPTH]) {
    for (int i = 0; i < PU_STATUS_DEPTH; i++) {
#pragma HLS pipeline II = 1
        config_strm.write(config[i]);

#ifndef __SYNTHESIS__
        std::cout << std::hex << "read_config: config[" << i << "]=" << config[i] << std::endl;
#endif
    }
}

void write_info(hls::stream<ap_uint<32> >& result_info_strm, ap_uint<32> result_info[PU_STATUS_DEPTH]) {
    for (int i = 0; i < PU_STATUS_DEPTH; i++) {
#pragma HLS pipeline II = 1
        result_info[i] = result_info_strm.read();

#ifndef __SYNTHESIS__
        std::cout << std::hex << "write_info: result_info[" << i << "]=" << result_info[i] << std::endl;
#endif
    }
}

// ------------------------------------------------------------

template <int CH_NM, int _WKey, int _WPay, int _KeyCol, int _PayCol>
void stream_converter(hls::stream<ap_uint<_WKey> >& i_key_strm,
                      hls::stream<ap_uint<_WPay> >& i_pld_strm,
                      hls::stream<bool>& i_e_strm,

                      hls::stream<ap_uint<_WKey> > o_key_strm[_KeyCol],
                      hls::stream<ap_uint<_WPay> > o_pld_strm[_PayCol],
                      hls::stream<bool>& o_e_strm) {
#pragma HLS inline off

    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS pipeline II = 1

        ap_uint<_WKey> i_key = i_key_strm.read();
        ap_uint<_WPay> i_pld = i_pld_strm.read();
        last = i_e_strm.read();

        for (int i = 0; i < _KeyCol; i++) {
#pragma HLS unroll
            o_key_strm[i].write(i_key);
        }

        for (int i = 0; i < _PayCol; i++) {
#pragma HLS unroll
            o_pld_strm[i].write(i_pld);
        }
        o_e_strm.write(false);
    }
    o_e_strm.write(true);
}

/// @brief convert stream to mutil-column
template <int CH_NM, int _WKey, int _WPay, int _KeyCol, int _PayCol>
void stream_converter_wrapper(hls::stream<ap_uint<_WKey> > i_key_strm[CH_NM],
                              hls::stream<ap_uint<_WPay> > i_pld_strm[CH_NM],
                              hls::stream<bool> i_e_strm[CH_NM],

                              hls::stream<ap_uint<_WKey> > o_key_strm[CH_NM][_KeyCol],
                              hls::stream<ap_uint<_WPay> > o_pld_strm[CH_NM][_PayCol],
                              hls::stream<bool> o_e_strm[CH_NM]) {
#pragma HLS inline off

    for (int i = 0; i < CH_NM; i++) {
#pragma HLS unroll

        stream_converter<CH_NM, _WKey, _WPay, _KeyCol, _PayCol>(i_key_strm[i], i_pld_strm[i], i_e_strm[i],
                                                                o_key_strm[i], o_pld_strm[i], o_e_strm[i]);
    }
}

// ------------------------------------------------------------

template <int _WKey, int _KeyNM, int _WPay, int _PayNM, int _WBuffer>
void mergeColumn(hls::stream<ap_uint<_WKey> > i_key_strm[_KeyNM],
                 hls::stream<ap_uint<_WPay> > i_pld_strm[3][_PayNM],
                 hls::stream<bool>& i_e_strm,

                 hls::stream<ap_uint<_WBuffer> >& o_merge_strm,
                 hls::stream<bool>& o_e_strm) {
#pragma HLS inline off

    // ap_uint<64> addr = 0;
    bool e = i_e_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1

        ap_uint<_WKey> key[_KeyNM];
        ap_uint<_WPay> pld[3][_PayNM];
        ap_uint<_WKey * _KeyNM + 3 * _WPay * _PayNM> aggr_out;

        for (int i = 0; i < _KeyNM; i++) {
#pragma HLS unroll
            key[i] = i_key_strm[i].read();
            aggr_out((i + 1) * _WKey + 3 * _WPay * _PayNM - 1, i * _WKey + 3 * _WPay * _PayNM) = key[i];
        }

        for (int i = 0; i < _PayNM; i++) {
#pragma HLS unroll
            pld[0][i] = i_pld_strm[0][i].read();
            pld[1][i] = i_pld_strm[1][i].read();
            pld[2][i] = i_pld_strm[2][i].read();
            aggr_out(1 * _WPay + 3 * i * _WPay - 1, 3 * i * _WPay) = pld[0][i];
            aggr_out(2 * _WPay + 3 * i * _WPay - 1, 1 * _WPay + 3 * i * _WPay) = pld[1][i];
            aggr_out(3 * _WPay + 3 * i * _WPay - 1, 2 * _WPay + 3 * i * _WPay) = pld[2][i];
        }

        // vec_buf[addr++] = aggr_out;
        o_merge_strm.write(aggr_out);
        o_e_strm.write(false);
        e = i_e_strm.read();
    }

    o_e_strm.write(true);

#ifndef __SYNTHESIS__
#ifdef DEBUG
// std::cout << std::dec << "write out " << addr << " rows" << std::endl;
#endif
#endif
}

template <int _WKey, int _KeyNM, int _WPay, int _PayNM, int _WBuffer>
void write_out(hls::stream<ap_uint<_WKey> > i_key_strm[_KeyNM],
               hls::stream<ap_uint<_WPay> > i_pld_strm[3][_PayNM],
               hls::stream<bool>& i_e_strm,
               ap_uint<_WBuffer>* vec_buf) {
#pragma HLS inline off
#pragma HLS dataflow

    hls::stream<ap_uint<_WBuffer> > merge_data_strm;
#pragma HLS stream variable = merge_data_strm depth = 8
    hls::stream<bool> merge_e_strm;
#pragma HLS stream variable = merge_e_strm depth = 8

    mergeColumn<_WKey, _KeyNM, _WPay, _PayNM, _WBuffer>(i_key_strm, i_pld_strm, i_e_strm, merge_data_strm,
                                                        merge_e_strm);

    xf::common::utils_hw::streamToAxi<8, _WBuffer, _WBuffer>(vec_buf, merge_data_strm, merge_e_strm);
}
// ------------------------------------------------------------

/// @brief top function of aggr kernel
void aggr_kernel_internal(ap_uint<8 * KEY_SZ * VEC_LEN> buf_l_orderkey[L_DEPTH],
                          ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_extendedprice[L_DEPTH],
                          ap_uint<32> l_nrow,

                          ap_uint<32> config[PU_STATUS_DEPTH],
                          ap_uint<32> result_info[PU_STATUS_DEPTH],

                          ap_uint<512> ping_buf0[L_DEPTH],
                          ap_uint<512> ping_buf1[L_DEPTH],
                          ap_uint<512> ping_buf2[L_DEPTH],
                          ap_uint<512> ping_buf3[L_DEPTH],
                          ap_uint<512> pong_buf0[L_DEPTH],
                          ap_uint<512> pong_buf1[L_DEPTH],
                          ap_uint<512> pong_buf2[L_DEPTH],
                          ap_uint<512> pong_buf3[L_DEPTH],

                          ap_uint<1024> result[L_DEPTH] // result k1
                          ) {
#pragma HLS dataflow

    hls::stream<ap_uint<8 * KEY_SZ> > k0_strm_arry[VEC_LEN];
    hls::stream<ap_uint<8 * MONEY_SZ> > p0_strm_arry[VEC_LEN];
    hls::stream<bool> e0_strm_arry[VEC_LEN];
#pragma HLS bind_storage variable = k0_strm_arry type = fifo impl = lutram
#pragma HLS STREAM variable = k0_strm_arry depth = 64
#pragma HLS bind_storage variable = p0_strm_arry type = fifo impl = lutram
#pragma HLS STREAM variable = p0_strm_arry depth = 64
#pragma HLS bind_storage variable = e0_strm_arry type = fifo impl = lutram
#pragma HLS STREAM variable = e0_strm_arry depth = 64

    hls::stream<ap_uint<8 * KEY_SZ> > k1_strm_arry[VEC_LEN][KEY_COL];
    hls::stream<ap_uint<8 * MONEY_SZ> > p1_strm_arry[VEC_LEN][PLD_COL];
    hls::stream<bool> e1_strm_arry[VEC_LEN];
#pragma HLS bind_storage variable = k1_strm_arry type = fifo impl = lutram
#pragma HLS STREAM variable = k1_strm_arry depth = 64
#pragma HLS bind_storage variable = p1_strm_arry type = fifo impl = lutram
#pragma HLS STREAM variable = p1_strm_arry depth = 64
#pragma HLS bind_storage variable = e1_strm_arry type = fifo impl = lutram
#pragma HLS STREAM variable = e1_strm_arry depth = 64

    hls::stream<ap_uint<32> > config_strm;
    hls::stream<ap_uint<32> > result_info_strm;
#pragma HLS bind_storage variable = config_strm type = fifo impl = lutram
#pragma HLS STREAM variable = config_strm depth = 8
#pragma HLS bind_storage variable = result_info_strm type = fifo impl = lutram
#pragma HLS STREAM variable = result_info_strm depth = 8

    hls::stream<ap_uint<KEY_SZ * 8> > aggr_key_out[KEY_COL];
    hls::stream<ap_uint<MONEY_SZ * 8> > aggr_pld_out[3][PLD_COL];
    hls::stream<bool> strm_e_out;
#pragma HLS bind_storage variable = aggr_key_out type = fifo impl = bram
#pragma HLS STREAM variable = aggr_key_out depth = 512
#pragma HLS bind_storage variable = aggr_pld_out type = fifo impl = bram
#pragma HLS STREAM variable = aggr_pld_out depth = 512
#pragma HLS bind_storage variable = strm_e_out type = fifo impl = lutram
#pragma HLS STREAM variable = strm_e_out depth = 512

    //--------------------------- scan in -----------------------------//
    xf::database::scanCol<32, VEC_LEN, 4, KEY_SZ, MONEY_SZ>(buf_l_orderkey, buf_l_extendedprice, l_nrow, k0_strm_arry,
                                                            p0_strm_arry, e0_strm_arry);

    stream_converter_wrapper<VEC_LEN, 8 * KEY_SZ, 8 * MONEY_SZ, KEY_COL, PLD_COL>(
        k0_strm_arry, p0_strm_arry, e0_strm_arry, k1_strm_arry, p1_strm_arry, e1_strm_arry);

    //---------------------------- hash-aggr --------------------------//
    read_config(config_strm, config);

    xf::database::hashGroupAggregate<8 * KEY_SZ, KEY_COL, 8 * MONEY_SZ, PLD_COL, 1, PU_HASH, WHASH, VEC_LEN, WCNT, 512,
                                     32, 32>( //
        k1_strm_arry, p1_strm_arry, e1_strm_arry, config_strm, result_info_strm, ping_buf0, ping_buf1, ping_buf2,
        ping_buf3, pong_buf0, pong_buf1, pong_buf2, pong_buf3, aggr_key_out, aggr_pld_out, strm_e_out);

    write_info(result_info_strm, result_info);

    //--------------------------- write out ---------------------------//
    write_out<8 * KEY_SZ, KEY_COL, 8 * MONEY_SZ, PLD_COL, 1024>(aggr_key_out, aggr_pld_out, strm_e_out, result);
}

extern "C" void hash_aggr_kernel(ap_uint<8 * KEY_SZ * VEC_LEN> buf_l_orderkey[L_DEPTH],
                                 ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_extendedprice[L_DEPTH],
                                 const int l_nrow,

                                 ap_uint<32> config[PU_STATUS_DEPTH],
                                 ap_uint<32> result_info[PU_STATUS_DEPTH],

                                 ap_uint<512> ping_buf0[L_DEPTH],
                                 ap_uint<512> ping_buf1[L_DEPTH],
                                 ap_uint<512> ping_buf2[L_DEPTH],
                                 ap_uint<512> ping_buf3[L_DEPTH],
                                 ap_uint<512> pong_buf0[L_DEPTH],
                                 ap_uint<512> pong_buf1[L_DEPTH],
                                 ap_uint<512> pong_buf2[L_DEPTH],
                                 ap_uint<512> pong_buf3[L_DEPTH],

                                 ap_uint<1024> result[R_DEPTH] // result k1
                                 ) {
    // clang-format off
    ;
#pragma HLS INTERFACE m_axi offset=slave latency=64 num_write_outstanding=16 num_read_outstanding=16 \
  max_write_burst_length = 8 max_read_burst_length = 8 bundle=gmem0_0 port=buf_l_orderkey
#pragma HLS INTERFACE s_axilite port = buf_l_orderkey bundle = control

#pragma HLS INTERFACE m_axi offset=slave latency=64 num_write_outstanding=16 num_read_outstanding=16 \
  max_write_burst_length = 8 max_read_burst_length = 8 bundle=gmem0_1 port=buf_l_extendedprice
#pragma HLS INTERFACE s_axilite port = buf_l_extendedprice bundle = control

#pragma HLS INTERFACE s_axilite port=l_nrow bundle=control

#pragma HLS INTERFACE m_axi offset=slave latency=64 num_write_outstanding=16 num_read_outstanding=16 \
  max_write_burst_length = 8 max_read_burst_length = 8 bundle=gmem0_2 port=config
#pragma HLS INTERFACE s_axilite port = config bundle = control

#pragma HLS INTERFACE m_axi offset=slave latency=64 num_write_outstanding=16 num_read_outstanding=16 \
  max_write_burst_length = 8 max_read_burst_length = 8 bundle=gmem0_3 port=result_info
#pragma HLS INTERFACE s_axilite port = result_info bundle = control

#pragma HLS INTERFACE m_axi port=ping_buf0 bundle=gmem1_0 num_write_outstanding=16 num_read_outstanding=16 \
  max_write_burst_length = 8 max_read_burst_length = 8 latency=125
#pragma HLS INTERFACE s_axilite port = ping_buf0 bundle = control

#pragma HLS INTERFACE m_axi port=ping_buf1 bundle=gmem1_1 num_write_outstanding=16 num_read_outstanding=16 \
  max_write_burst_length = 8 max_read_burst_length = 8 latency=125
#pragma HLS INTERFACE s_axilite port = ping_buf1 bundle = control

#pragma HLS INTERFACE m_axi port=ping_buf2 bundle=gmem1_2 num_write_outstanding=16 num_read_outstanding=16 \
  max_write_burst_length = 8 max_read_burst_length = 8 latency=125
#pragma HLS INTERFACE s_axilite port = ping_buf2 bundle = control

#pragma HLS INTERFACE m_axi port=ping_buf3 bundle=gmem1_3 num_write_outstanding=16 num_read_outstanding=16 \
  max_write_burst_length = 8 max_read_burst_length = 8 latency=125
#pragma HLS INTERFACE s_axilite port = ping_buf3 bundle = control


#pragma HLS INTERFACE m_axi port=pong_buf0 bundle=gmem2_0 num_write_outstanding=16 num_read_outstanding=16 \
  max_write_burst_length = 8 max_read_burst_length = 8 latency=125
#pragma HLS INTERFACE s_axilite port = pong_buf0 bundle = control

#pragma HLS INTERFACE m_axi port=pong_buf1 bundle=gmem2_1 num_write_outstanding=16 num_read_outstanding=16 \
  max_write_burst_length = 8 max_read_burst_length = 8 latency=125
#pragma HLS INTERFACE s_axilite port = pong_buf1 bundle = control

#pragma HLS INTERFACE m_axi port=pong_buf2 bundle=gmem2_2 num_write_outstanding=16 num_read_outstanding=16 \
  max_write_burst_length = 8 max_read_burst_length = 8 latency=125
#pragma HLS INTERFACE s_axilite port = pong_buf2 bundle = control

#pragma HLS INTERFACE m_axi port=pong_buf3 bundle=gmem2_3 num_write_outstanding=16 num_read_outstanding=16 \
  max_write_burst_length = 8 max_read_burst_length = 8 latency=125
#pragma HLS INTERFACE s_axilite port = pong_buf3 bundle = control


#pragma HLS INTERFACE m_axi port=result bundle=gmem3_0 num_write_outstanding=16 num_read_outstanding=16 \
  max_write_burst_length = 8 max_read_burst_length=8 latency=125
#pragma HLS INTERFACE s_axilite port = result bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

    // clang-format on
    ;

    aggr_kernel_internal(buf_l_orderkey, buf_l_extendedprice, l_nrow, config, result_info, ping_buf0, ping_buf1,
                         ping_buf2, ping_buf3, pong_buf0, pong_buf1, pong_buf2, pong_buf3, result);

} // end aggr_kernel
