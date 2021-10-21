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
#ifndef _XF_DB_ISV_GQE_AGGR_H_
#define _XF_DB_ISV_GQE_AGGR_H_

/**
 * @file gqe_kernel_aggr_v2.hpp
 * @brief interface of GQE group-by aggregation kernel.
 */

#include "xf_utils_hw/types.hpp"
#include "xf_database/gqe_blocks/gqe_types.hpp"

#include <stddef.h>

#ifndef __SYNTHESIS__
#include <iostream>
#endif

#define TEST_BUF_DEPTH 4 << 20 //(512 bit x 4M = 256 MB)
#define VEC_SCAN 8

/**
 * @brief GQE Aggr Kernel
 * \rst
 * For detailed document, see :ref:`gqe_kernel_design`.
 * \endrst
 * @param buf_in0 input table buffer.
 * @param buf_in1 input table buffer.
 * @param buf_in2 input table buffer.
 * @param buf_in3 input table buffer.
 * @param buf_in4 input table buffer.
 * @param buf_in5 input table buffer.
 * @param buf_in6 input table buffer.
 * @param buf_in7 input table buffer.
 * @param nrow input row number.
 * @param buf_out output table buffer.
 * @param buf_cfg input configuration buffer.
 * @param buf_result_info output information buffer.
 *
 * @param ping_buf0 gqeAggr's temporal buffer for storing overflow.
 * @param ping_buf1 gqeAggr's temporal buffer for storing overflow.
 * @param ping_buf2 gqeAggr's temporal buffer for storing overflow.
 * @param ping_buf3 gqeAggr's temporal buffer for storing overflow.
 *
 * @param pong_buf0 gqeAggr's temporal buffer for storing overflow.
 * @param pong_buf1 gqeAggr's temporal buffer for storing overflow.
 * @param pong_buf2 gqeAggr's temporal buffer for storing overflow.
 * @param pong_buf3 gqeAggr's temporal buffer for storing overflow.
 *
 */
extern "C" void gqeAggr(ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in0[],
                        ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in1[],
                        ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in2[],
                        ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in3[],
                        ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in4[],
                        ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in5[],
                        ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in6[],
                        ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in7[],
                        ap_uint<512> buf_metain[],
                        ap_uint<512> buf_metaout[],
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out0[],  // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out1[],  // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out2[],  // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out3[],  // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out4[],  // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out5[],  // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out6[],  // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out7[],  // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out8[],  // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out9[],  // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out10[], // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out11[], // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out12[], // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out13[], // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out14[], // output data
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out15[], // output data

                        ap_uint<8 * TPCH_INT_SZ> buf_cfg[],
                        ap_uint<8 * TPCH_INT_SZ> buf_result_info[],

                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> ping_buf0[],
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> ping_buf1[],
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> ping_buf2[],
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> ping_buf3[],

                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> pong_buf0[],
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> pong_buf1[],
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> pong_buf2[],
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN> pong_buf3[]);

#endif
