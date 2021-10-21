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
#include "laplacian.hpp"
#include "binFiles.hpp"

using namespace std;

int main(int argc, char** argv) {
    t_InType* img = new t_InType[M_z * M_y * M_x / nPE];
    t_InType* out = new t_InType[M_z * M_y * M_x / nPE];
    float* ref = new float[M_z * M_y * M_x];
    float coefx[ORDER + 1];
    float coefy[ORDER + 1];
    float coefz[ORDER + 1];

    string filePath = string(argv[1]);
    readBin(filePath + "image.bin", sizeof(float) * M_x * M_y * M_z, img);
    readBin(filePath + "result.bin", sizeof(float) * M_x * M_y * M_z, ref);
    readBin(filePath + "coefx.bin", sizeof(float) * (ORDER + 1), coefx);
    readBin(filePath + "coefy.bin", sizeof(float) * (ORDER + 1), coefy);
    readBin(filePath + "coefz.bin", sizeof(float) * (ORDER + 1), coefz);

    top(M_z, M_y, M_x, coefz, coefy, coefx, img, out);

    int err = 0;
    float atol = 1e-4, rtol = 1e-3;
    for (int i = 0; i < M_z; i++)
        for (int j = 0; j < M_y; j++) {
            for (int k = 0; k < M_x; k++) {
                int index = i * M_y * M_x + j * M_x + k;
                float outV = ((t_DataTypeX)((t_WideType)out[index / nPE])[index % nPE])[0];
                float refV = ref[index];

                if (fabs(refV - outV) <= atol + rtol * fabs(refV))
                    continue;
                else {
                    err++;
                    cout << "ref = " << refV << '\t' << "val = " << outV << '\t' << "x = " << k << ", y=" << j
                         << ", z =" << i << endl;
                }
            }
        }
    delete[] img;
    delete[] out;
    delete[] ref;

    if (err == 0)

        return 0;
    else {
        cout << "There are in total " << err << " errors." << endl;
        return -1;
    }
}
