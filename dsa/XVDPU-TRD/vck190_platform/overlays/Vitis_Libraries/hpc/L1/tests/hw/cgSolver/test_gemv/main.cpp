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
#include "xf_blas.hpp"

using namespace std;

int main() {
    srand(CG_dataSize);
    CG_dataType dot = 0, l_RZ = 0.3;
    vector<CG_dataType> l_Ak, l_A[CG_numChannels], l_pk, l_Apk(CG_dataSize), l_Apk_golden;

    hls::stream<ap_uint<CG_memBits> > p_A[CG_numChannels];
    hls::stream<ap_uint<CG_memBits> > p_pk;
    hls::stream<ap_uint<sizeof(CG_dataType) * 8> > p_dot;
    hls::stream<ap_uint<sizeof(CG_dataType) * 8> > p_pkc;
    hls::stream<ap_uint<sizeof(CG_dataType) * 8> > p_Apk;

    for (int i = 0; i < CG_dataSize; i++) {
        CG_dataType p = rand() / (CG_dataType)rand();

        l_pk.push_back(p);
    }

    for (int i = 0; i < CG_dataSize / CG_numChannels; i++)
        xf::blas::mem2stream(CG_dataSize / CG_parEntries, (ap_uint<CG_memBits>*)l_pk.data(), p_pk);
    xf::blas::mem2stream(CG_dataSize, (ap_uint<sizeof(CG_dataType) * 8>*)l_pk.data(), p_pkc);

    for (int i = 0; i < CG_dataSize; i++) {
        CG_dataType Ap = 0;
        for (int j = 0; j < CG_dataSize; j++) {
            CG_dataType a = rand() / (CG_dataType)rand();
            l_Ak.push_back(a);
            Ap += a * l_pk[j];
            l_A[i % CG_numChannels].push_back(a);
        }
        l_Apk_golden.push_back(Ap);
        dot += l_pk[i] * Ap;
    }
    for (int i = 0; i < CG_numChannels; i++)
        xf::blas::mem2stream(CG_dataSize * CG_dataSize / CG_numChannels / CG_parEntries,
                             (ap_uint<CG_memBits>*)l_A[i].data(), p_A[i]);

    hls::stream<ap_uint<CG_tkWidth> > l_strIn, l_strOut;

    top(CG_dataSize, p_dot, p_A, p_pk, p_pkc, p_Apk);

    xf::blas::stream2mem(CG_dataSize, p_Apk, (ap_uint<sizeof(CG_dataType) * 8>*)l_Apk.data());

    int err = 0;
    compare(l_Apk.size(), l_Apk.data(), l_Apk_golden.data(), err);
    CG_dataType l_dot = p_dot.read();
    if (!compare(dot, l_dot)) err++;

    if (err == 0)
        return 0;
    else {
        cout << "ERROR: There are in total " << err << " errors." << endl;
        return -1;
    }
}
