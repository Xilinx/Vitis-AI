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
#include <fstream>
#include <cmath>
#include <cstring>
#include "propagate.hpp"
#include "utils.hpp"
#include "binFiles.hpp"

using namespace std;

int main(int argc, char** argv) {
    t_InType* v2dt2 = new t_InType[M_z * M_y * M_x / nPE];
    t_PairInType* img = new t_PairInType[M_z * M_y * M_x / nPE];
    t_PairInType* out = new t_PairInType[M_z * M_y * M_x / nPE];
    float* ref0 = new float[M_z * M_y * M_x];
    float* ref1 = new float[M_z * M_y * M_x];
    float* p0 = new float[M_z * M_y * M_x];
    float* p1 = new float[M_z * M_y * M_x];
    float coefx[ORDER + 1];
    float coefy[ORDER + 1];
    float coefz[ORDER + 1];

    string filePath = string(argv[1]);
    readBin(filePath + "ip0.bin", sizeof(float) * M_x * M_y * M_z, p0);
    readBin(filePath + "ip1.bin", sizeof(float) * M_x * M_y * M_z, p1);
    readBin(filePath + "sp0.bin", sizeof(float) * M_x * M_y * M_z, ref0);
    readBin(filePath + "sp1.bin", sizeof(float) * M_x * M_y * M_z, ref1);
    readBin(filePath + "v2dt2.bin", sizeof(t_InType) * M_x * M_y * M_z / nPE, v2dt2);
    readBin(filePath + "coefx.bin", sizeof(float) * (ORDER + 1), coefx);
    readBin(filePath + "coefy.bin", sizeof(float) * (ORDER + 1), coefy);
    readBin(filePath + "coefz.bin", sizeof(float) * (ORDER + 1), coefz);

    for (int i = 0; i < M_x * M_y * M_z / nPE; i++) {
        t_PairType s;
        t_WideType s0, s1;
        for (int pe = 0; pe < nPE; pe++) {
            t_DataTypeX x0, x1;
            x0[0] = p0[i * nPE + pe];
            s0[pe] = x0;
            x1[0] = p1[i * nPE + pe];
            s1[pe] = x1;
        }
        s[0] = s0;
        s[1] = s1;
        img[i] = s;
    }

    top(M_z, M_y, M_x, coefz, coefy, coefx, v2dt2, img, out);

    int err = 0;
    float atol = 1e-5, rtol = 1e-3;
    for (int i = 0; i < M_x * M_y * M_z / nPE; i++) {
        t_PairType s = out[i];
        for (int pe = 0; pe < nPE; pe++) {
            float outV = ((t_DataTypeX)((t_WideType)s[0])[pe])[0];
            float refV = ref0[i * nPE + pe];
            if (fabs(refV - outV) <= atol + rtol * fabs(refV))
                continue;
            else {
                err++;
            }
        }
        for (int pe = 0; pe < nPE; pe++) {
            float outV = ((t_DataTypeX)((t_WideType)s[1])[pe])[0];
            float refV = ref1[i * nPE + pe];
            if (fabs(refV - outV) <= atol + rtol * fabs(refV))
                continue;
            else {
                err++;
            }
        }
    }
    delete[] img;
    delete[] out;
    delete[] ref0;
    delete[] ref1;
    delete[] p0;
    delete[] p1;
    delete[] v2dt2;

    if (err == 0)

        return 0;
    else {
        cout << "There are in total " << err << " errors." << endl;
        return -1;
    }
}
