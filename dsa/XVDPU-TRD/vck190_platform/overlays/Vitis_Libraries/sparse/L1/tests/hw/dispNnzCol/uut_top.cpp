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

void uut_top(hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& p_datStr,
             hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > p_datOutStr[SPARSE_hbmChannels]) {
    static const unsigned int t_IntsPerColParam = 2 + SPARSE_hbmChannels * 2;
    static const unsigned int t_ParamsPerPar = SPARSE_dataBits * SPARSE_parEntries / 32;
    static const unsigned int t_ParBlocks4Param = (t_IntsPerColParam + t_ParamsPerPar - 1) / t_ParamsPerPar;

    xf::sparse::dispNnzCol<SPARSE_maxColParBlocks, t_ParBlocks4Param, SPARSE_hbmChannels, SPARSE_parEntries,
                           SPARSE_dataBits>(p_datStr, p_datOutStr);
}
