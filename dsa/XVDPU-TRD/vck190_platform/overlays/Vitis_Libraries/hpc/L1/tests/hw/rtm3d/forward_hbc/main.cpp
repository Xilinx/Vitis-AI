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
#include "forward.hpp"
#include "utils.hpp"
using namespace std;

void converter_upb(vector<DATATYPE>& pin, vector<DATATYPE>& out) {
    int index = 0;
    for (int i = 0; i < NTime; i++)
        for (int k = 0; k < RTM_x / nPE; k++)
            for (int j = 0; j < RTM_y; j++)
                for (int po = 0; po < ORDER / 2; po++)
                    for (int pe = 0; pe < nPE; pe++)
                        out[i * RTM_x * RTM_y * ORDER / 2 + (k * nPE + pe) * RTM_y * ORDER / 2 + j * ORDER / 2 + po] =
                            pin[index++];
}
void converter_vt(vector<DATATYPE>& pin, vector<WIDE_TYPE>& out) {
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
void converter_p(vector<IN_TYPE>& out, vector<DATATYPE>& p0) {
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
    vector<DATATYPE> p, pp, snap0, snap1, coefx, coefy, coefz, taperx, tapery, taperz, srcwavelet, ref, upbConv, upb;
    vector<DATATYPE> l_vt2;
    vector<WIDE_TYPE> v2dt2;
    v2dt2.resize(RTM_x * RTM_y * RTM_z / nPE / nPE);
    upbConv.resize(RTM_x * RTM_y * ORDER * NTime / 2);
    upb.resize(RTM_x * RTM_y * ORDER * NTime / 2);

    p.resize(RTM_x * RTM_y * RTM_z);
    pp.resize(RTM_x * RTM_y * RTM_z);

    string filePath = string(argv[1]);

    readBin(filePath + "snap0.bin", sizeof(float) * RTM_x * RTM_y * RTM_z, snap0);
    readBin(filePath + "snap1.bin", sizeof(float) * RTM_x * RTM_y * RTM_z, snap1);

    readBin(filePath + "v2dt2.bin", sizeof(float) * RTM_x * RTM_y * RTM_z, l_vt2);
    converter_vt(l_vt2, v2dt2);
    readBin(filePath + "src_s0.bin", sizeof(float) * NTime, srcwavelet);

    readBin(filePath + "coefx.bin", sizeof(float) * (ORDER + 1), coefx);
    readBin(filePath + "coefy.bin", sizeof(float) * (ORDER + 1), coefy);
    readBin(filePath + "coefz.bin", sizeof(float) * (ORDER + 1), coefz);
    readBin(filePath + "taperx.bin", sizeof(float) * (MaxB), taperx);
    readBin(filePath + "tapery.bin", sizeof(float) * (MaxB), tapery);
    readBin(filePath + "taperz.bin", sizeof(float) * (MaxB), taperz);

    readBin(filePath + "upb.bin", sizeof(float) * RTM_x * RTM_y * ORDER * NTime / 2, ref);

    vector<IN_TYPE> l_p0, l_p1;
    vector<IN_TYPE> l_pp0, l_pp1;
    l_p1.resize(RTM_x * RTM_y * RTM_z / nPE);
    l_p0.resize(RTM_x * RTM_y * RTM_z / nPE);
    l_pp1.resize(RTM_x * RTM_y * RTM_z / nPE);
    l_pp0.resize(RTM_x * RTM_y * RTM_z / nPE);
    assert(NTime % NUM_INST == 0);
    assert(RTM_z % nPE == 0);

    for (int t = 0; t < NTime / NUM_INST; t++)
        top(t & 0x01, RTM_z, RTM_y, RTM_x, t, MaxB, RTM_y / 2, RTM_x / 2, srcwavelet.data(), coefz.data(), coefy.data(),
            coefx.data(), taperz.data(), tapery.data(), taperx.data(), (IN_TYPE*)v2dt2.data(), l_p0.data(), l_p1.data(),
            l_p0.data(), l_p1.data(), l_pp0.data(), l_pp1.data(), l_pp0.data(), l_pp1.data(), (UPB_TYPE*)upb.data());

    vector<IN_TYPE>& out_p = (NTime / NUM_INST & 0x01) ? l_p1 : l_p0;
    vector<IN_TYPE>& out_pp = (NTime / NUM_INST & 0x01) ? l_pp1 : l_pp0;
    converter_p(out_p, p);
    converter_p(out_pp, pp);
    converter_upb(upb, upbConv);

    int err0 = 0, err1 = 0, err2 = 0;
    bool pass0 = compare<DATATYPE>(RTM_x * RTM_y * RTM_z, pp.data(), snap0.data(), err0);
    cout << "There are in total " << err0 << " errors in pp v.s. snap0" << endl;
    // writeBin(filePath + "snap_c_0.bin", sizeof(float) * RTM_x * RTM_y * RTM_z, pp.data());

    bool pass1 = compare<DATATYPE>(RTM_x * RTM_y * RTM_z, p.data(), snap1.data(), err1);
    cout << "There are in total " << err1 << " errors in p v.s. snap1" << endl;
    // writeBin(filePath + "snap_c_1.bin", sizeof(float) * RTM_x * RTM_y * RTM_z, p.data());

    bool pass2 = compare<DATATYPE>(RTM_x * RTM_y * ORDER * NTime / 2, upbConv.data(), ref.data(), err2);
    cout << "There are in total " << err2 << " errors in upb v.s. ref" << endl;

    if (pass0 && pass1 && pass2) {
        cout << "Test passed!" << endl;
        return 0;
    } else {
        cout << "Test failed, there are in total " << err0 + err1 + err2 << " errors!" << endl;
        return -1;
    }
}
