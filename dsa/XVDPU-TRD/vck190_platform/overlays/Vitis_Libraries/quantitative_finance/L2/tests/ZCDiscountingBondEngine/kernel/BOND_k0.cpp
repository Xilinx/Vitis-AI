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

#include "discounting_bond_engine_kernel.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
using namespace std;
#endif

extern "C" void BOND_k0(int size, DT time[LEN], DT disc[LEN], DT amount, DT t, DT NPV[N]) {
#pragma HLS INTERFACE m_axi port = NPV bundle = gmem0 offset = slave
#pragma HLS INTERFACE m_axi port = time bundle = gmem1 offset = slave
#pragma HLS INTERFACE m_axi port = disc bundle = gmem2 offset = slave

#pragma HLS INTERFACE s_axilite port = size bundle = control
#pragma HLS INTERFACE s_axilite port = time bundle = control
#pragma HLS INTERFACE s_axilite port = disc bundle = control
#pragma HLS INTERFACE s_axilite port = amount bundle = control
#pragma HLS INTERFACE s_axilite port = t bundle = control
#pragma HLS INTERFACE s_axilite port = NPV bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    cout << "t=" << t << endl;
#endif

    DiscountingBondEngine<DT, LEN> bond;
    bond.init(size, time, disc);
    NPV[0] = bond.calcuNPV(t, amount);
}
