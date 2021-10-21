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

#include "cpi_capfloor_engine_kernel.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
using namespace std;
#endif

extern "C" void CPI_k0(int xSize, int ySize, DT time[LEN], DT strike[LEN], DT price[LEN * LEN], DT t, DT r, DT NPV[N]) {
#pragma HLS INTERFACE m_axi port = NPV bundle = gmem0 offset = slave
#pragma HLS INTERFACE m_axi port = time bundle = gmem1 offset = slave
#pragma HLS INTERFACE m_axi port = strike bundle = gmem2 offset = slave
#pragma HLS INTERFACE m_axi port = price bundle = gmem3 offset = slave

#pragma HLS INTERFACE s_axilite port = xSize bundle = control
#pragma HLS INTERFACE s_axilite port = ySize bundle = control
#pragma HLS INTERFACE s_axilite port = time bundle = control
#pragma HLS INTERFACE s_axilite port = strike bundle = control
#pragma HLS INTERFACE s_axilite port = price bundle = control
#pragma HLS INTERFACE s_axilite port = t bundle = control
#pragma HLS INTERFACE s_axilite port = r bundle = control
#pragma HLS INTERFACE s_axilite port = NPV bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    cout << "t=" << t << endl;
#endif

    CPICapFloorEngine<DT, LEN> cpi;
    cpi.init(xSize, ySize, time, strike, price);
    NPV[0] = cpi.calcuNPV(t, r);
}
