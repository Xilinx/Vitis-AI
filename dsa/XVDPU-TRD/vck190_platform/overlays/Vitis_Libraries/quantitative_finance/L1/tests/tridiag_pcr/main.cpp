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
#include "trsv.hpp"

#define N 64
#define logN 6
#define NCU 2

void top_trsv(double inlow[N], double indiag[N], double inup[N], double inrhs[N]) {
    xf::fintech::trsvCore<double, N, logN, NCU>(inlow, indiag, inup, inrhs);
};

int main() {
    double inlow[N];
    double indiag[N];
    double inup[N];
    double inrhs[N];
    double sol[N];
    double result[N];
    bool fail = false;

    for (int i = 0; i < N; i++) {
        inlow[i] = -1.0;
        indiag[i] = 4.0 + i;
        inup[i] = -2.0;
        inrhs[i] = 0.0;
        sol[i] = i + 1;
    };
    inlow[0] = 0.0;
    inup[N - 1] = 0.0;

    inrhs[0] = indiag[0] * sol[0] + inup[0] * sol[1];
    inrhs[N - 1] = inlow[N - 1] * sol[N - 2] + indiag[N - 1] * sol[N - 1];
    for (int i = 1; i < N - 1; i++) {
        inrhs[i] = inlow[i] * sol[i - 1] + indiag[i] * sol[i] + inup[i] * sol[i + 1];
    };
    std::cout << " reference solution:\n";
    for (int k = 0; k < N; k++) {
        std::cout << sol[k] << " ";
    };
    std::cout << std::endl;

    // solve
    top_trsv(inlow, indiag, inup, inrhs);

    std::cout << " Calculated solution (output):\n";
    for (int k = 0; k < N; k++) {
        std::cout << inrhs[k] / indiag[k] << " ";
        result[k] = inrhs[k] / indiag[k];
        if (result[k] - sol[k] > 0.1) {
            fail = true;
            std::cout << fail << " ";
        }
    }
    std::cout << std::endl;
    if (fail == true) {
        std::cout << " TEST FAILED: calculated solution is different than reference\n";
        return 1;
    } else {
        std::cout << " TEST PASSED\n";
        return 0;
    }
};
