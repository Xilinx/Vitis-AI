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
#include <string>
#include <fstream>
#include "binFiles.hpp"
#include "utils.hpp"
#include "forward.hpp"
using namespace std;

void converter(vector<DATATYPE>& pin, vector<WIDE_TYPE>& out) {
    int index = 0;
    for (int i = 0; i < RTM_x / nPE; i++)
        for (int j = 0; j < RTM_y; j++)
            for (int k = 0; k < RTM_z / nPE; k++) {
                WIDE_TYPE l_p;
                for (int pe = 0; pe < nPE; pe++) {
                    TYPEX l_px;
                    for (int px = 0; px < nPE; px++) {
                        l_px[px] = pin[(i * nPE + px) * RTM_y * RTM_z + j * RTM_z + k * nPE + pe];
                    }
                    l_p[pe] = l_px;
                }
                out[index++] = l_p;
            }
}
void converter(vector<IN_TYPE>& out, vector<DATATYPE>& p0) {
    int index = 0;
    for (int i = 0; i < RTM_x / nPE; i++)
        for (int j = 0; j < RTM_y; j++)
            for (int k = 0; k < RTM_z / nPE; k++) {
                WIDE_TYPE l_p = out[index++];
                for (int pe = 0; pe < nPE; pe++) {
                    TYPEX l_px = l_p[pe];
                    for (int px = 0; px < nPE; px++) {
                        p0[(i * nPE + px) * RTM_y * RTM_z + j * RTM_z + k * nPE + pe] = l_px[px];
                    }
                }
            }
}

int main(int argc, char** argv) {
    vector<DATATYPE> p, pp, snap0, snap1, coefx, coefy, coefz, taperx, tapery, taperz, srcwavelet, ref;
    vector<DATATYPE> l_vt2;
    vector<WIDE_TYPE> v2dt2;
    v2dt2.resize(RTM_x * RTM_y * RTM_z / nPE / nPE);

    p.resize(RTM_x * RTM_y * RTM_z);
    pp.resize(RTM_x * RTM_y * RTM_z);

    string filePath = string(argv[1]);

    readBin(filePath + "snap0.bin", sizeof(float) * RTM_x * RTM_y * RTM_z, snap0);
    readBin(filePath + "snap1.bin", sizeof(float) * RTM_x * RTM_y * RTM_z, snap1);

    readBin(filePath + "v2dt2.bin", sizeof(float) * RTM_x * RTM_y * RTM_z, l_vt2);
    converter(l_vt2, v2dt2);
    readBin(filePath + "src_s0.bin", sizeof(float) * NTime, srcwavelet);

    readBin(filePath + "coefx.bin", sizeof(float) * (ORDER + 1), coefx);
    readBin(filePath + "coefy.bin", sizeof(float) * (ORDER + 1), coefy);
    readBin(filePath + "coefz.bin", sizeof(float) * (ORDER + 1), coefz);

    vector<IN_TYPE> l_pi, l_po;
    l_po.resize(RTM_x * RTM_y * RTM_z / nPE);
    l_pi.resize(RTM_x * RTM_y * RTM_z / nPE);
    vector<IN_TYPE> l_ppi, l_ppo;
    l_ppo.resize(RTM_x * RTM_y * RTM_z / nPE);
    l_ppi.resize(RTM_x * RTM_y * RTM_z / nPE);
    assert(NTime % NUM_INST == 0);
    assert(RTM_z % nPE == 0);

    for (int t = 0; t < NTime / NUM_INST; t++)
        top(t & 0x01, RTM_z, RTM_y, RTM_x, t, MaxB, RTM_y / 2, RTM_x / 2, srcwavelet.data(), coefz.data(), coefy.data(),
            coefx.data(), (IN_TYPE*)v2dt2.data(), l_pi.data(), l_po.data(), l_pi.data(), l_po.data(), l_ppi.data(),
            l_ppo.data(), l_ppi.data(), l_ppo.data());

    vector<IN_TYPE>& out_p = (NTime / NUM_INST & 0x01) ? l_po : l_pi;
    vector<IN_TYPE>& out_pp = (NTime / NUM_INST & 0x01) ? l_ppo : l_ppi;
    converter(out_pp, pp);
    converter(out_p, p);

    int err0 = 0, err1 = 0;
    bool pass0 = compare<DATATYPE>(RTM_x * RTM_y * RTM_z, pp.data(), snap0.data(), err0);
    cout << "There are in total " << err0 << " errors in pp v.s. snap0" << endl;
    writeBin(filePath + "snap_c_0.bin", sizeof(float) * RTM_x * RTM_y * RTM_z, pp.data());

    bool pass1 = compare<DATATYPE>(RTM_x * RTM_y * RTM_z, p.data(), snap1.data(), err1);
    cout << "There are in total " << err1 << " errors in p v.s. snap1" << endl;
    writeBin(filePath + "snap_c_1.bin", sizeof(float) * RTM_x * RTM_y * RTM_z, p.data());

    if (pass0 && pass1) {
        cout << "Test passed!" << endl;
        return 0;
    } else {
        cout << "Test failed, there are in total " << err0 + err1 << " errors!" << endl;
        return -1;
    }
}
