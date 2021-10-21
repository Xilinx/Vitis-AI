
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
#ifndef XF_SPARSE_READWRITEHBMKERNEL_HPP
#define XF_SPARSE_READWRITEHBMKERNEL_HPP
/**
 * @file readWriteHbmKernel.hpp
 * @brief readWriteHbmKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "cscKernel.hpp"

/**
 * @brief readWriteHbmKernel is used to read NNZ values and row indices from HBM and write result row vector to HBM
 * @param p_rd0 the device memory pointer, which is mapped to HBM channel 0, for reading NNZs' values and row indices
 * @param p_wr0 the device memory pointer, which is mapped to HBM channel 0, for storing result row vector
 * @param p_rd1 the device memory pointer, which is mapped to HBM channel 1, for reading NNZs' values and row indices
 * @param p_wr1 the device memory pointer, which is mapped to HBM channel 1, for storing result row vector
 * @param p_rd2 the device memory pointer, which is mapped to HBM channel 2, for reading NNZs' values and row indices
 * @param p_wr2 the device memory pointer, which is mapped to HBM channel 2, for storing result row vector
 * @param p_rd3 the device memory pointer, which is mapped to HBM channel 3, for reading NNZs' values and row indices
 * @param p_wr3 the device memory pointer, which is mapped to HBM channel 3, for storing result row vector
 * @param p_rd4 the device memory pointer, which is mapped to HBM channel 4, for reading NNZs' values and row indices
 * @param p_wr4 the device memory pointer, which is mapped to HBM channel 4, for storing result row vector
 * @param p_rd5 the device memory pointer, which is mapped to HBM channel 5, for reading NNZs' values and row indices
 * @param p_wr5 the device memory pointer, which is mapped to HBM channel 5, for storing result row vector
 * @param p_rd6 the device memory pointer, which is mapped to HBM channel 6, for reading NNZs' values and row indices
 * @param p_wr6 the device memory pointer, which is mapped to HBM channel 6, for storing result row vector
 * @param p_rd7 the device memory pointer, which is mapped to HBM channel 7, for reading NNZs' values and row indices
 * @param p_wr7 the device memory pointer, which is mapped to HBM channel 7, for storing result row vector
 * @param out0 the output NNZ values and row indices axis stream to CU0 of cscRowKernel
 * @param in0 the input result row vector axis stream from CU0 of cscRowKernel
 * @param out1 the output NNZ values and row indices axis stream to CU1 of cscRowKernel
 * @param in1 the input result row vector axis stream from CU1 of cscRowKernel
 * @param out2 the output NNZ values and row indices axis stream to CU2 of cscRowKernel
 * @param in2 the input result row vector axis stream from CU2 of cscRowKernel
 * @param out3 the output NNZ values and row indices axis stream to CU3 of cscRowKernel
 * @param in3 the input result row vector axis stream from CU3 of cscRowKernel
 * @param out4 the output NNZ values and row indices axis stream to CU4 of cscRowKernel
 * @param in4 the input result row vector axis stream from CU4 of cscRowKernel
 * @param out5 the output NNZ values and row indices axis stream to CU5 of cscRowKernel
 * @param in5 the input result row vector axis stream from CU5 of cscRowKernel
 * @param out6 the output NNZ values and row indices axis stream to CU6 of cscRowKernel
 * @param in6 the input result row vector axis stream from CU6 of cscRowKernel
 * @param out7 the output NNZ values and row indices axis stream to CU7 of cscRowKernel
 * @param in7 the input result row vector axis stream from CU7 of cscRowKernel
 */
extern "C"

    void
    readWriteHbmKernel(ap_uint<SPARSE_hbmMemBits>* p_rd0,
                       ap_uint<SPARSE_hbmMemBits>* p_wr0,
                       ap_uint<SPARSE_hbmMemBits>* p_rd1,
                       ap_uint<SPARSE_hbmMemBits>* p_wr1,
                       ap_uint<SPARSE_hbmMemBits>* p_rd2,
                       ap_uint<SPARSE_hbmMemBits>* p_wr2,
                       ap_uint<SPARSE_hbmMemBits>* p_rd3,
                       ap_uint<SPARSE_hbmMemBits>* p_wr3,
                       ap_uint<SPARSE_hbmMemBits>* p_rd4,
                       ap_uint<SPARSE_hbmMemBits>* p_wr4,
                       ap_uint<SPARSE_hbmMemBits>* p_rd5,
                       ap_uint<SPARSE_hbmMemBits>* p_wr5,
                       ap_uint<SPARSE_hbmMemBits>* p_rd6,
                       ap_uint<SPARSE_hbmMemBits>* p_wr6,
                       ap_uint<SPARSE_hbmMemBits>* p_rd7,
                       ap_uint<SPARSE_hbmMemBits>* p_wr7,
                       hls::stream<ap_uint<SPARSE_hbmMemBits> >& out0,
                       hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in0,
                       hls::stream<ap_uint<SPARSE_hbmMemBits> >& out1,
                       hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in1,
                       hls::stream<ap_uint<SPARSE_hbmMemBits> >& out2,
                       hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in2,
                       hls::stream<ap_uint<SPARSE_hbmMemBits> >& out3,
                       hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in3,
                       hls::stream<ap_uint<SPARSE_hbmMemBits> >& out4,
                       hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in4,
                       hls::stream<ap_uint<SPARSE_hbmMemBits> >& out5,
                       hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in5,
                       hls::stream<ap_uint<SPARSE_hbmMemBits> >& out6,
                       hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in6,
                       hls::stream<ap_uint<SPARSE_hbmMemBits> >& out7,
                       hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in7);
#endif
