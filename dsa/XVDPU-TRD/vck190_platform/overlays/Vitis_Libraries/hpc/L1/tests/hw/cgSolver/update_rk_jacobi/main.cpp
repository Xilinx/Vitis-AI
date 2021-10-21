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
    CG_dataType alpha = 0.3, beta = 0, l_rz = 5.7;
    vector<CG_dataType> l_zk(CG_dataSize), l_rk, l_jacobi, l_Apk, l_zk_golden, l_rk_golden;
    CG_dataType l_sum = 0;
    for (int i = 0; i < CG_dataSize; i++) {
        CG_dataType r = rand() / (CG_dataType)rand();
        l_rk.push_back(r);

        CG_dataType Ap = rand() / (CG_dataType)rand();
        l_Apk.push_back(Ap);

        CG_dataType jacobi = rand() / (CG_dataType)rand();
        l_jacobi.push_back(jacobi);

        CG_dataType rk = r - alpha * Ap, zk = jacobi * rk;
        l_rk_golden.push_back(rk);
        l_zk_golden.push_back(zk);
        l_sum += rk * zk;
    }

    hls::stream<ap_uint<CG_tkWidth> > l_strIn, l_strOut;

    xf::hpc::cg::Token<CG_dataType> l_token;
    xf::hpc::StreamInstr<sizeof(l_token)> l_cs;
    l_token.setID(1);
    l_token.setVecSize(CG_dataSize);

    l_token.setAlpha(alpha);
    l_token.setRZ(l_rz);
    l_token.encode_write(l_strIn, l_cs);

    l_token.setExit();
    l_token.encode_write(l_strIn, l_cs);

    top((ap_uint<CG_memBits>*)l_rk.data(), (ap_uint<CG_memBits>*)l_rk.data(), (ap_uint<CG_memBits>*)l_zk.data(),
        (ap_uint<CG_memBits>*)l_jacobi.data(), (ap_uint<CG_memBits>*)l_Apk.data(), l_strIn, l_strOut);

    l_token.read_decode(l_strOut, l_cs);

    int err = 0;

    compare(l_zk.size(), l_zk.data(), l_zk_golden.data(), err);
    compare(l_rk.size(), l_rk.data(), l_rk_golden.data(), err);
    if (!compare(l_token.getRZ(), l_sum)) err++;
    if (!compare(l_token.getBeta(), l_sum / l_rz)) err++;

    l_token.read_decode(l_strOut, l_cs);

    if (err == 0)
        return 0;
    else {
        cout << "ERROR: There are in total " << err << " errors." << endl;
        return -1;
    }
}
