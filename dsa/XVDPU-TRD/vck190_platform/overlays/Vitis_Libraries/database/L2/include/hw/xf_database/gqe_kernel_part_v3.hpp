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
#ifndef _XF_DB_GQE_PART_V3_H_
#define _XF_DB_GQE_PART_V3_H_

/**
 * @file gqe_kernel_part_v3.hpp
 * @brief interface of GQE partition kernel.
 */

#include <ap_int.h>

/**
 * @breif GQE partition kernel (64-bit key version)
 *
 * @param bucket_depth bucket depth
 * @param table_index table index indicating build table or join table
 * @param log_part log of number of partitions
 *
 * @param din_col input table columns
 * @param din_val validation bits column
 * @param din_krn_cfg input kernel configurations
 *
 * @param din_meta input meta info
 * @param dout_meta output meta info
 *
 * @param dout_col output table columns
 *
 */
extern "C" void gqePart(const int bucket_depth, // bucket depth

                        // table index indicate build table or join table
                        const int tab_index,

                        // the log partition number
                        const int log_part,

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

                        // output data columns
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col0,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col1,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col2);

#endif // _XF_DB_GQE_PART_V3_H_
