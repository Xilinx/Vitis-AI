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
#include <iostream>
#include <vector>
#include <cstdlib>
#include "utils.hpp"
#include "top.hpp"
#include "token.hpp"

using namespace std;

int main() {
    CG_dataType beta = 0.3;
    vector<CG_dataType> l_rk, l_pk, l_pk_golden;
    for (int i = 0; i < CG_dataSize; i++) {
        CG_dataType r = rand() / 931.4;
        CG_dataType p = rand() / 413.9;
        CG_dataType p1 = r + beta * p;
        l_rk.push_back(r);
        l_pk.push_back(p);
        l_pk_golden.push_back(p1);
    }

    hls::stream<ap_uint<CG_tkWidth> > l_strIn, l_strOut;

    xf::hpc::cg::Token<CG_dataType> l_token;
    xf::hpc::StreamInstr<sizeof(l_token)> l_cs;
    l_token.setID(1);
    l_token.setVecSize(CG_dataSize);
    l_token.setBeta(beta);
    l_token.encode_write(l_strIn, l_cs);

    l_token.setExit();
    l_token.encode_write(l_strIn, l_cs);

    top((ap_uint<CG_memBits>*)l_rk.data(), (ap_uint<CG_memBits>*)l_pk.data(), (ap_uint<CG_memBits>*)l_pk.data(),
        l_strIn, l_strOut);

    l_token.read_decode(l_strOut, l_cs);

    int err = 0;

    compare(l_pk.size(), l_pk.data(), l_pk_golden.data(), err);

    if (err == 0)
        return 0;
    else {
        cout << "ERROR: There are in total " << err << " errors." << endl;
        return -1;
    }
}
