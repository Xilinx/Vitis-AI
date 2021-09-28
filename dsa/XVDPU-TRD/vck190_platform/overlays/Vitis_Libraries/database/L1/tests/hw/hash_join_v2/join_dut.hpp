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

#ifndef JOIN_DUT_HPP
#define JOIN_DUT_HPP

#define HJ_MODE 1
#define HJ_HW_P 3
// XXX smaller for faster co-sim
#define HJ_HW_J 10
#define HJ_AW 19
#define HJ_CH_NM 4

// XXX smaller than normal but more than enough
#define BUFF_DEPTH (1 << 20)

#define WKEY 32
#define WPAY 32

// used modules
#include <ap_int.h>
#include <hls_stream.h>

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
              ap_uint<WKEY + WPAY> buf7[BUFF_DEPTH]);

#endif
