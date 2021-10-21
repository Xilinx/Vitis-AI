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

int main(int argc, char** argv) {
    vector<DATATYPE> p, pp, snap0, snap1, coefx, coefz, taperx, taperz, srcwavelet, upb, ref;
    vector<IN_TYPE> v2dt2;

    p.resize(WIDTH * HEIGHT);
    pp.resize(WIDTH * HEIGHT);
    upb.resize(WIDTH * ORDER * NTime / 2);

    string filePath = string(argv[1]);

    readBin(filePath + "snap0.bin", sizeof(float) * WIDTH * HEIGHT, snap0);
    readBin(filePath + "snap1.bin", sizeof(float) * WIDTH * HEIGHT, snap1);

    readBin(filePath + "v2dt2.bin", sizeof(float) * WIDTH * HEIGHT, v2dt2);
    readBin(filePath + "src.bin", sizeof(float) * NTime, srcwavelet);

    readBin(filePath + "taperx.bin", sizeof(float) * (NXB), taperx);
    readBin(filePath + "taperz.bin", sizeof(float) * (NZB), taperz);

    readBin(filePath + "coefx.bin", sizeof(float) * (ORDER + 1), coefx);
    readBin(filePath + "coefz.bin", sizeof(float) * (ORDER + 1), coefz);

    readBin(filePath + "upb.bin", sizeof(float) * WIDTH * ORDER * NTime / 2, ref);

    vector<PAIRIN_TYPE> po, ppo;
    ppo.resize(WIDTH * HEIGHT / nPE);
    po.resize(WIDTH * HEIGHT / nPE);
    assert(NTime % NUM_INST == 0);
    assert(HEIGHT % nPE == 0);

    for (int t = 0; t < NTime / NUM_INST; t++)
        top(t & 0x01, HEIGHT, WIDTH, t, NZB, WIDTH / 2, srcwavelet.data(), coefz.data(), coefx.data(), taperz.data(),
            taperx.data(), v2dt2.data(), po.data(), ppo.data(), po.data(), ppo.data(), (UPB_TYPE*)upb.data());

    vector<PAIRIN_TYPE>& out = (NTime / NUM_INST & 0x01) ? ppo : po;
    for (int i = 0; i < WIDTH * HEIGHT / nPE; i++) {
        PAIR_TYPE l_pair = out[i];
        WIDE_TYPE l_pp = l_pair[0];
        WIDE_TYPE l_p = l_pair[1];
        for (int pe = 0; pe < nPE; pe++) {
            pp[i * nPE + pe] = l_pp[pe];
            p[i * nPE + pe] = l_p[pe];
        }
    }

    fstream pfile("p.txt", ios::out);
    fstream ppfile("pp.txt", ios::out);
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            pfile << scientific << p[i * HEIGHT + j] << ',';
            ppfile << scientific << pp[i * HEIGHT + j] << ',';
        }
        pfile << endl;
        ppfile << endl;
    }
    ppfile.close();
    pfile.close();

    int err0 = 0, err1 = 0, err2 = 0;
    bool pass0 = compare<DATATYPE>(WIDTH * HEIGHT, pp.data(), snap0.data(), err0);
    cout << "There are in total " << err0 << " errors in pp v.s. snap0" << endl;

    bool pass1 = compare<DATATYPE>(WIDTH * HEIGHT, p.data(), snap1.data(), err1);
    cout << "There are in total " << err1 << " errors in p v.s. snap1" << endl;

    bool pass2 = compare<DATATYPE>(WIDTH * ORDER * NTime / 2, upb.data(), ref.data(), err2);
    cout << "There are in total " << err2 << " errors in upb v.s. ref" << endl;

    if (pass0 && pass1 && pass2) {
        cout << "Test passed!" << endl;
        return 0;
    } else {
        cout << "Test failed, there are in total " << err0 + err1 + err2 << " errors!" << endl;
        return -1;
    }
}
