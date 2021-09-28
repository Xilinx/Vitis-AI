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
#include "dut.hpp"

int main() {
    int nerr = 0;
    DT err = 1e-12;
    DT golden = 0.73187894736842107;
    DT x[N], y[N], f[N][N];
    int n = 6;

    for (int i = 0; i < n; ++i) {
        x[i] = y[i] = i * 0.2;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            f[i][j] = x[j] * (x[j] + y[i]);
        }
    }

    DT res = dut(6, x, y, f, x[2] + 0.13, y[4] + 0.05);
    if (std::abs(res - golden) / golden > err) {
        nerr++;
        std::cout << "dut output=" << res << ", diff=" << std::abs(res - golden) / golden << std::endl;
    }
    return nerr;
}
