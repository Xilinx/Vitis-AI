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

#include <math.h>
#include <iostream>
#include "covariance_regularization.hpp"

#define DT double

void dut(int n, DT threshold, hls::stream<DT>& inStrm, hls::stream<DT>& outStrm);

int main() {
    int nerr = 0;
    DT err = 1e-12;
    const int N = 6;
    DT inMat[N][N] = {{8.27159, -1.07314, 0.773794, 0.878236, 0.816598, -0.10948},
                      {-1.07314, 8.12296, 0.694735, -0.719803, 1.4851, 0.326867},
                      {0.773794, 0.694735, 8.79739, -0.994484, 0.248038, -0.39592},
                      {0.878236, -0.719803, -0.994484, 8.34666, 0.318372, -0.353867},
                      {0.816598, 1.4851, 0.248038, 0.318372, 7.87127, 0.676107},
                      {-0.10948, 0.326867, -0.39592, -0.353867, 0.676107, 8.80956}};
    DT goldenMat[N][N] = {{8.271590, -1.073140, 0.773794, 0.878236, 0.816598, 0.000000},
                          {-1.073140, 8.122960, 0.694735, -0.719803, 1.485100, 0.000000},
                          {0.773794, 0.694735, 8.797390, -0.994484, 0.000000, 0.000000},
                          {0.878236, -0.719803, -0.994484, 8.346660, 0.000000, 0.000000},
                          {0.816598, 1.485100, 0.000000, 0.000000, 7.871270, 0.676107},
                          {0.000000, 0.000000, 0.000000, 0.000000, 0.676107, 8.809560}};
    hls::stream<DT> inStrm;
    hls::stream<DT> outStrm;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            inStrm.write(inMat[i][j]);
        }
    }
    dut(N, 0.5, inStrm, outStrm);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            DT out = outStrm.read();
            if (std::abs(out - goldenMat[i][j]) > err) nerr++;
        }
    }
    return nerr;
}
