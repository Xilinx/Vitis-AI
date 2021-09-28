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
/** @file main.cpp
* @brief Testbench application file.
*/

#include <stdio.h>
#include <iostream>
#include "pentadiag_top.hpp"

int main() {
    // pentadiag
    TEST_DT d[P_SIZE];
    TEST_DT a[P_SIZE];
    TEST_DT b[P_SIZE];
    TEST_DT e[P_SIZE];
    TEST_DT c[P_SIZE];
    TEST_DT f[P_SIZE];
    TEST_DT r[P_SIZE];
    TEST_DT u[P_SIZE];
    TEST_DT inrhs[P_SIZE];
    TEST_DT sol[P_SIZE];
    unsigned int N = P_SIZE;
    bool fail;
    /*
    * Generate diagonals
    */
    for (int i = 0; i < N; i++) {
        e[i] = -1.0 - 0.37 * i;
        d[i] = 1.0;
        a[i] = 4.0 + i;
        b[i] = -2.0;
        c[i] = -3.0;
        sol[i] = i + 1;
    }
    d[0] = 0.0;
    b[N - 1] = 0.0;
    /*
    * Compute right hand side vector based on a generated diagonals
    */
    inrhs[0] = a[0] * sol[0] + b[0] * sol[1] + c[0] * sol[2];
    inrhs[1] = d[1] * sol[0] + a[1] * sol[1] + b[1] * sol[2] + c[1] * sol[3];
    inrhs[N - 1] = e[N - 1] * sol[N - 3] + d[N - 1] * sol[N - 2] + a[N - 1] * sol[N - 1];
    inrhs[N - 2] = e[N - 2] * sol[N - 4] + d[N - 2] * sol[N - 3] + a[N - 2] * sol[N - 2] + b[N - 2] * sol[N - 1];
    for (int i = 2; i < N - 2; i++) {
        inrhs[i] = e[i] * sol[i - 2] + d[i] * sol[i - 1] + a[i] * sol[i] + b[i] * sol[i + 1] + c[i] * sol[i + 2];
    };

    /*
    * Print generated solution
    */
    for (int k = 0; k < N; k++) {
        std::cout << sol[k] << " ";
    };
    std::cout << std::endl;

    pentadiag_top(e, d, a, b, c, inrhs, u);

    /*
    * Print solved solution
    */
    for (int k = 0; k < P_SIZE; k++) {
        std::cout << u[k] << " ";
        if (u[k] - sol[k] > 0.1) {
            fail = true;
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
}
