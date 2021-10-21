
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
 * @file loadNnz.hpp
 * @brief loadNnz definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_LOADNNZ_HPP
#define XF_SPARSE_LOADNNZ_HPP

#include "xf_sparse_fp64.hpp"

namespace xf {
namespace sparse {

/**
 */
template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_DataBits, unsigned int t_MemBits>
void loadNnz(ap_uint<t_MemBits>* p_nnzPtr0,
             ap_uint<t_MemBits>* p_nnzPtr1,
             ap_uint<t_MemBits>* p_nnzPtr2,
             ap_uint<t_MemBits>* p_nnzPtr3,
             ap_uint<t_MemBits>* p_nnzPtr4,
             ap_uint<t_MemBits>* p_nnzPtr5,
             ap_uint<t_MemBits>* p_nnzPtr6,
             ap_uint<t_MemBits>* p_nnzPtr7,
             ap_uint<t_MemBits>* p_nnzPtr8,
             ap_uint<t_MemBits>* p_nnzPtr9,
             ap_uint<t_MemBits>* p_nnzPtr10,
             ap_uint<t_MemBits>* p_nnzPtr11,
             ap_uint<t_MemBits>* p_nnzPtr12,
             ap_uint<t_MemBits>* p_nnzPtr13,
             ap_uint<t_MemBits>* p_nnzPtr14,
             ap_uint<t_MemBits>* p_nnzPtr15,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr0,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr1,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr2,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr3,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr4,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr5,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr6,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr7,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr8,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr9,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr10,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr11,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr12,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr13,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr14,
             hls::stream<ap_uint<t_MemBits> >& p_nnzStr15) {
#pragma HLS DATAFLOW
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr0, p_nnzStr0);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr1, p_nnzStr1);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr2, p_nnzStr2);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr3, p_nnzStr3);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr4, p_nnzStr4);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr5, p_nnzStr5);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr6, p_nnzStr6);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr7, p_nnzStr7);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr8, p_nnzStr8);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr9, p_nnzStr9);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr10, p_nnzStr10);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr11, p_nnzStr11);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr12, p_nnzStr12);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr13, p_nnzStr13);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr14, p_nnzStr14);
    loadNnz<t_DataType, t_ParEntries, t_DataBits, t_MemBits>(p_nnzPtr15, p_nnzStr15);
}

} // end namespace sparse
} // end namespace xf
#endif
