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
#include "backward.hpp"
using namespace std;

int main(int argc, char** argv) {
    vector<DATATYPE> v2dt2, snap0, snap1, coefx, coefz, taperx, taperz, dobs, upb, pr, rr, p, pp, imloc, pref, ppref,
        prref, pprref, imlocref;

    p.resize(WIDTH * HEIGHT);
    pp.resize(WIDTH * HEIGHT);
    pr.resize(WIDTH * HEIGHT);
    rr.resize(WIDTH * HEIGHT);
    imloc.resize(NX * NZ);

    string filePath = "./";
    if (argc == 2) filePath = string(argv[1]);

    readBin(filePath + "snap0.bin", sizeof(DATATYPE) * WIDTH * HEIGHT, snap0);
    readBin(filePath + "snap1.bin", sizeof(DATATYPE) * WIDTH * HEIGHT, snap1);

    readBin(filePath + "v2dt2.bin", sizeof(DATATYPE) * WIDTH * HEIGHT, v2dt2);

    readBin(filePath + "taperx.bin", sizeof(DATATYPE) * (NXB), taperx);
    readBin(filePath + "taperz.bin", sizeof(DATATYPE) * (NZB), taperz);

    readBin(filePath + "coefx.bin", sizeof(DATATYPE) * (ORDER + 1), coefx);
    readBin(filePath + "coefz.bin", sizeof(DATATYPE) * (ORDER + 1), coefz);

    readBin(filePath + "upb.bin", sizeof(DATATYPE) * WIDTH * ORDER * NTime / 2, upb);

    readBin(filePath + "imloc.bin", sizeof(DATATYPE) * NX * NZ, imlocref);

    readBin(filePath + "p0.bin", sizeof(DATATYPE) * WIDTH * HEIGHT, ppref);
    readBin(filePath + "p1.bin", sizeof(DATATYPE) * WIDTH * HEIGHT, pref);

    readBin(filePath + "r0.bin", sizeof(DATATYPE) * WIDTH * HEIGHT, pprref);
    readBin(filePath + "r1.bin", sizeof(DATATYPE) * WIDTH * HEIGHT, prref);

    int err0 = 0, err1 = 0, err2 = 0, err3 = 0, err4 = 0;

    readBin(filePath + "sensor.bin", sizeof(DATATYPE) * NTime * NX, dobs);
    vector<PAIRIN_TYPE> po, ppo;
    ppo.resize(WIDTH * HEIGHT / nPE);
    po.resize(WIDTH * HEIGHT / nPE);

    vector<PAIRIN_TYPE> ro, rro;
    rro.resize(WIDTH * HEIGHT / nPE);
    ro.resize(WIDTH * HEIGHT / nPE);

    vector<IN_TYPE> io, iio;
    iio.resize(NX * NZ / nPE);
    io.resize(NX * NZ / nPE);

    assert(NTime % NUM_INST == 0);
    for (int i = 0; i < WIDTH * HEIGHT / nPE; i++) {
        PAIR_TYPE l_po;
        WIDE_TYPE l_w0, l_w1;
        for (int pe = 0; pe < nPE; pe++) {
            l_w1[pe] = snap0[i * nPE + pe];
            l_w0[pe] = snap1[i * nPE + pe];
        }
        l_po[1] = l_w1;
        l_po[0] = l_w0;
        po[i] = l_po;
    }

    for (int t = 0; t < NTime / NUM_INST; t++)
        top(t & 0x01, HEIGHT, WIDTH, NTime / NUM_INST - 1 - t, NTime, NZB, dobs.data(), coefz.data(), coefx.data(),
            taperz.data(), taperx.data(), (IN_TYPE*)v2dt2.data(), po.data(), ppo.data(), po.data(), ppo.data(),
            ro.data(), rro.data(), ro.data(), rro.data(), io.data(), iio.data(), io.data(), iio.data(),
            (UPB_TYPE*)upb.data());

    vector<PAIRIN_TYPE>& pout = (NTime / NUM_INST & 0x01) ? ppo : po;
    vector<PAIRIN_TYPE>& rout = (NTime / NUM_INST & 0x01) ? rro : ro;
    vector<IN_TYPE>& iout = (NTime / NUM_INST & 0x01) ? iio : io;

    for (int i = 0; i < NX * NZ / nPE; i++) {
        for (int pe = 0; pe < nPE; pe++) {
            imloc[i * nPE + pe] = ((WIDE_TYPE)iout[i])[pe];
        }
    }
    for (int i = 0; i < WIDTH * HEIGHT / nPE; i++) {
        for (int pe = 0; pe < nPE; pe++) {
            pp[i * nPE + pe] = ((WIDE_TYPE)((PAIR_TYPE)pout[i])[0])[pe];
            p[i * nPE + pe] = ((WIDE_TYPE)((PAIR_TYPE)pout[i])[1])[pe];
            rr[i * nPE + pe] = ((WIDE_TYPE)((PAIR_TYPE)rout[i])[0])[pe];
            pr[i * nPE + pe] = ((WIDE_TYPE)((PAIR_TYPE)rout[i])[1])[pe];
        }
    }

    fstream ofile;
    ofile.open("pp.txt", ios::out);
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            ofile << scientific << pp[i * HEIGHT + j] << ',';
        }
        ofile << endl;
    }
    ofile.close();
    ofile.open("p.txt", ios::out);
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            ofile << scientific << p[i * HEIGHT + j] << ',';
        }
        ofile << endl;
    }
    ofile.close();

    bool pass0 = compare<DATATYPE>(NX * NZ, imloc.data(), imlocref.data(), err0);
    cout << "There are in total " << err0 << " errors in imloc v.s. imlocref" << endl;

    bool pass1 = compare<DATATYPE>(WIDTH * HEIGHT, p.data(), pref.data(), err1);
    cout << "There are in total " << err1 << " errors in p v.s. pref" << endl;

    bool pass2 = compare<DATATYPE>(WIDTH * HEIGHT, pp.data(), ppref.data(), err2);
    cout << "There are in total " << err2 << " errors in pp v.s. ppref" << endl;

    bool pass3 = compare<DATATYPE>(WIDTH * HEIGHT, pr.data(), prref.data(), err3);
    cout << "There are in total " << err3 << " errors in pr v.s. prref" << endl;

    bool pass4 = compare<DATATYPE>(WIDTH * HEIGHT, rr.data(), pprref.data(), err4);
    cout << "There are in total " << err4 << " errors in rr v.s. pprref" << endl;

    if (pass0 && pass1 && pass2 && pass3 && pass4) {
        cout << "Test passed!" << endl;
        return 0;
    } else {
        cout << "Test failed, there are in total " << err0 + err1 + err2 + err3 + err4 << " errors!" << endl;
        return -1;
    }
}
