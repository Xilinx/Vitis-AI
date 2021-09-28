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

/**
 * @file bufTransNnzColKernel.hpp
 * @brief bufTransNnzColKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "cscKernel.hpp"

/**
 * @brief bufTransNnzColKernel is used to buffer and dispatch the column pointers across multiple CUs of the
 * xBarColKernel
 * @param in0 input column pointer entries stream
 * @param out0 output column pointer entries stream for CU0 of xBarColKernel
 * @param out1 output column pointer entries stream for CU1 of xBarColKernel
 * @param out2 output column pointer entries stream for CU2 of xBarColKernel
 * @param out3 output column pointer entries stream for CU3 of xBarColKernel
 * @param out4 output column pointer entries stream for CU4 of xBarColKernel
 * @param out5 output column pointer entries stream for CU5 of xBarColKernel
 * @param out6 output column pointer entries stream for CU6 of xBarColKernel
 * @param out7 output column pointer entries stream for CU7 of xBarColKernel
 * @param out8 output column pointer entries stream for CU8 of xBarColKernel
 * @param out9 output column pointer entries stream for CU9 of xBarColKernel
 * @param out10 output column pointer entries stream for CU10 of xBarColKernel
 * @param out11 output column pointer entries stream for CU11 of xBarColKernel
 * @param out12 output column pointer entries stream for CU12 of xBarColKernel
 * @param out13 output column pointer entries stream for CU13 of xBarColKernel
 * @param out14 output column pointer entries stream for CU14 of xBarColKernel
 * @param out15 output column pointer entries stream for CU15 of xBarColKernel
 */
extern "C" void bufTransNnzColKernel(hls::stream<ap_uint<SPARSE_ddrMemBits> >& in0,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out0,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out1,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out2,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out3,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out4,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out5,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out6,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out7,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out8,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out9,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out10,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out11,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out12,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out13,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out14,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out15);
