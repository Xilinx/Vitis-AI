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

void uut_top(unsigned int p_colPtrBlocks,
             unsigned int p_nnzBlocks,
             hls::stream<ap_uint<(SPARSE_indexBits * (1 << SPARSE_logParEntries))> >& p_colPtrStr,
             hls::stream<ap_uint<(SPARSE_dataBits * (1 << SPARSE_logParEntries))> >& p_colValStr,
             hls::stream<ap_uint<(SPARSE_dataBits * (1 << SPARSE_logParEntries))> >& p_nnzColValStr) {
    xBarCol<SPARSE_logParEntries, SPARSE_dataType, SPARSE_indexType, SPARSE_dataBits, SPARSE_indexBits>(
        p_colPtrBlocks, p_nnzBlocks, p_colPtrStr, p_colValStr, p_nnzColValStr);
}
