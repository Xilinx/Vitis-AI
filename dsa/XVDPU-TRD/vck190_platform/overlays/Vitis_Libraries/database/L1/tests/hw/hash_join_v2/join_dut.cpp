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

// used modules
#include <hls_stream.h>

#define URAM_SPLITTING 1
#include "xf_database/hash_join_v2.hpp"

// top header
#include "join_dut.hpp"

void join_dut(hls::stream<ap_uint<WKEY> > k_strms[HJ_CH_NM],
              hls::stream<ap_uint<WPAY> > p_strms[HJ_CH_NM],
              hls::stream<bool> e_strms[HJ_CH_NM],
              // out
              hls::stream<ap_uint<WPAY * 2> >& j_strm,
              hls::stream<bool>& e_strm,
              // temp PU = 8
              ap_uint<WKEY + WPAY> buf0[BUFF_DEPTH],
              ap_uint<WKEY + WPAY> buf1[BUFF_DEPTH],
              ap_uint<WKEY + WPAY> buf2[BUFF_DEPTH],
              ap_uint<WKEY + WPAY> buf3[BUFF_DEPTH],
              ap_uint<WKEY + WPAY> buf4[BUFF_DEPTH],
              ap_uint<WKEY + WPAY> buf5[BUFF_DEPTH],
              ap_uint<WKEY + WPAY> buf6[BUFF_DEPTH],
              ap_uint<WKEY + WPAY> buf7[BUFF_DEPTH]) {
    // clang-format off
    ;
#pragma HLS INTERFACE m_axi port=buf0 bundle=gmem1_0 num_write_outstanding=32 num_read_outstanding=32 \
  max_read_burst_length=8 latency=125
#pragma HLS INTERFACE m_axi port=buf1 bundle=gmem1_1 num_write_outstanding=32 num_read_outstanding=32 \
  max_read_burst_length=8 latency=125
#pragma HLS INTERFACE m_axi port=buf2 bundle=gmem1_2 num_write_outstanding=32 num_read_outstanding=32 \
  max_read_burst_length=8 latency=125
#pragma HLS INTERFACE m_axi port=buf3 bundle=gmem1_3 num_write_outstanding=32 num_read_outstanding=32 \
  max_read_burst_length=8 latency=125
#pragma HLS INTERFACE m_axi port=buf4 bundle=gmem1_4 num_write_outstanding=32 num_read_outstanding=32 \
  max_read_burst_length=8 latency=125
#pragma HLS INTERFACE m_axi port=buf5 bundle=gmem1_5 num_write_outstanding=32 num_read_outstanding=32 \
  max_read_burst_length=8 latency=125
#pragma HLS INTERFACE m_axi port=buf6 bundle=gmem1_6 num_write_outstanding=32 num_read_outstanding=32 \
  max_read_burst_length=8 latency=125
#pragma HLS INTERFACE m_axi port=buf7 bundle=gmem1_7 num_write_outstanding=32 num_read_outstanding=32 \
  max_read_burst_length=8 latency=125
    // clang-format on
    ;

    xf::database::hashJoinMPU<HJ_MODE,                   // hash algorithm
                              WKEY,                      // key width
                              WPAY,                      // payload max width
                              WPAY,                      // s payload width
                              WPAY,                      // t payload width
                              HJ_HW_P,                   // log2(number of PU)
                              HJ_HW_J,                   // hash width for join
                              HJ_AW,                     // address width
                              WKEY + WPAY,               // buffer width
                              HJ_CH_NM,                  // channel number
                              24, 0>                     // bloom-filter not enabled
        (k_strms, p_strms, e_strms,                      //
         buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, //
         j_strm, e_strm);
}
