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

#include "inflation_capfloor_engine_kernel.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
using namespace std;
#endif

extern "C" void INFLATION_k0(int type,
                             DT forwardRate,
                             DT cfRate[2],
                             DT nomial,
                             DT gearing,
                             DT accrualTime,
                             int size,
                             DT time[LEN],
                             DT rate[LEN],
                             int optionlets,
                             DT NPV[N]) {
#pragma HLS INTERFACE m_axi port = NPV bundle = gmem0 offset = slave num_read_outstanding = 16 num_write_outstanding = \
    16 max_read_burst_length = 32 max_write_burst_length = 32
#pragma HLS INTERFACE m_axi port = cfRate bundle = gmem1 offset = slave num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 32 max_write_burst_length = 32
#pragma HLS INTERFACE m_axi port = time bundle = gmem2 offset = slave num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 32 max_write_burst_length = 32
#pragma HLS INTERFACE m_axi port = rate bundle = gmem3 offset = slave num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 32 max_write_burst_length = 32

#pragma HLS INTERFACE s_axilite port = type bundle = control
#pragma HLS INTERFACE s_axilite port = forwardRate bundle = control
#pragma HLS INTERFACE s_axilite port = cfRate bundle = control
#pragma HLS INTERFACE s_axilite port = nomial bundle = control
#pragma HLS INTERFACE s_axilite port = gearing bundle = control
#pragma HLS INTERFACE s_axilite port = accrualTime bundle = control
#pragma HLS INTERFACE s_axilite port = size bundle = control
#pragma HLS INTERFACE s_axilite port = time bundle = control
#pragma HLS INTERFACE s_axilite port = rate bundle = control
#pragma HLS INTERFACE s_axilite port = optionlets bundle = control
#pragma HLS INTERFACE s_axilite port = NPV bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    InflationCapFloorEngine<DT, LEN> cf;
    cf.init(type, forwardRate, cfRate, nomial, gearing, accrualTime, size, time, rate);
    NPV[0] = cf.calcuNPV(optionlets);
}
