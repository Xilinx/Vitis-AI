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
#ifndef _XF_DB_GQE_JOIN_FILTER_H_
#define _XF_DB_GQE_JOIN_FILTER_H_

/**
 * @file gqe_kernel_join_filter.hpp
 * @brief interface of GQE Join/Filter kernel.
 */

#include <ap_int.h>
#include <hls_burst_maxi.h>
#include "xf_database/gqe_blocks_v3/gqe_types.hpp"

/**
 * @breif GQE Join/Filter kernel (64-bit key version)
 *
 * @param _build_probe_flag build/probe flag, 0 for build, 1 for probe
 *
 * @param din_col input table columns
 * @param din_val validation bits column
 *
 * @param din_krn_cfg input kernel configurations
 *
 * @param din_meta input meta info
 * @param dout_meta output meta info
 *
 * @param dout_col output table columns
 *
 * @param htb_buf HBM buffers used to save build table key/payload
 * @param stb_buf HBM buffers used to save overflowed build table key/payload for Join flow, to save hash-table of
 * bloom-filter for Bloom-filter probe only flow
 *
 */
extern "C" void gqeJoin(size_t _build_probe_flag, // build/probe flag

                        // input data columns
                        hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col0,
                        hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col1,
                        hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col2,

                        // validation buffer
                        hls::burst_maxi<ap_uint<64> > din_val,

                        // kernel config
                        ap_uint<512> din_krn_cfg[14],

                        // meta input buffer
                        ap_uint<512> din_meta[24],
                        // meta output buffer
                        ap_uint<512> dout_meta[24],

                        //  output data columns
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col0,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col1,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col2,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col3,

                        // hbm buffers used to save build table key/payload
                        ap_uint<256>* htb_buf0,
                        ap_uint<256>* htb_buf1,
                        ap_uint<256>* htb_buf2,
                        ap_uint<256>* htb_buf3,
                        ap_uint<256>* htb_buf4,
                        ap_uint<256>* htb_buf5,
                        ap_uint<256>* htb_buf6,
                        ap_uint<256>* htb_buf7,

                        // overflow buffers
                        ap_uint<256>* stb_buf0,
                        ap_uint<256>* stb_buf1,
                        ap_uint<256>* stb_buf2,
                        ap_uint<256>* stb_buf3,
                        ap_uint<256>* stb_buf4,
                        ap_uint<256>* stb_buf5,
                        ap_uint<256>* stb_buf6,
                        ap_uint<256>* stb_buf7);

#endif // _XF_DB_GQE_JOIN_FILTER_H_
