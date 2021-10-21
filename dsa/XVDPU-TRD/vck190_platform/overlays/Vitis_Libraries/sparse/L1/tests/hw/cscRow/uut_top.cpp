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
#include "uut_top.hpp"

void uut_top(unsigned int p_nnzBlocks,
             unsigned int p_rowBlocks,
             hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& p_nnzValStr,
             hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& p_nnzColValStr,
             hls::stream<ap_uint<SPARSE_indexBits * SPARSE_parEntries> >& p_rowIndexStr,
             hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& p_rowAggStr) {
    xf::sparse::cscRow<SPARSE_maxRowBlocks, SPARSE_logParEntries, SPARSE_logParGroups, SPARSE_dataType,
                       SPARSE_indexType, SPARSE_dataBits, SPARSE_indexBits>(p_nnzBlocks, p_rowBlocks, p_nnzValStr,
                                                                            p_nnzColValStr, p_rowIndexStr, p_rowAggStr);
}
