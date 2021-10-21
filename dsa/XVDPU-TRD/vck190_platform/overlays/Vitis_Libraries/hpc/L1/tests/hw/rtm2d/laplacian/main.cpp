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
#include "binFiles.hpp"
#include "laplacian.hpp"

using namespace std;

int main(int argc, char** argv) {
    t_InType img[HEIGHT * WIDTH / nPE];
    t_InType out[HEIGHT * WIDTH / nPE];
    float ref[HEIGHT * WIDTH];
    float coefx[ORDER + 1];
    float coefz[ORDER + 1];

    char filename[100];

    string filePath = string(argv[1]);
    readBin(filePath + "image.bin", sizeof(float) * WIDTH * HEIGHT, img);
    readBin(filePath + "result.bin", sizeof(float) * WIDTH * HEIGHT, ref);
    readBin(filePath + "coefx.bin", sizeof(float) * (ORDER + 1), coefx);
    readBin(filePath + "coefz.bin", sizeof(float) * (ORDER + 1), coefz);

    top(HEIGHT, WIDTH, coefz, coefx, img, out);

    int err = 0;
    for (int i = 0; i < HEIGHT; i++)
        for (int j = 0; j < WIDTH; j++) {
            int index = i * WIDTH + j;
            float outV = ((t_WideType)out[index / nPE])[index % nPE];
            if (fabs(ref[index] - outV) < 1e-3 || fabs(outV / ref[index] - 1.0) < 1e-3)
                continue;
            else {
                err++;
            }
        }

    if (err == 0)

        return 0;
    else {
        cout << "There are in total " << err << " errors." << endl;
        return -1;
    }
}
