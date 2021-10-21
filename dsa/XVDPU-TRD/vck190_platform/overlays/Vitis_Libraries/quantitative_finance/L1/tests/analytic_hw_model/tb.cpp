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

#include <ap_int.h>
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <iostream>

#define LEN (12)
#define NUM_TEST_CASES (5)

typedef double DT;
template <typename DT>
struct TestCaseTyp {
    DT a;
    DT sigma;
    DT t;
    DT T;
    DT P;
};

DT dut(DT a, DT sigma, DT maturity[LEN], DT interestRates[LEN], DT t, DT T);

int main() {
    // Zero bond yield curve data
    DT maturity[LEN] = {0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 10.0, 20.0, 30.0};

    DT interestRates[LEN] = {0.0020, 0.0050, 0.0070, 0.0110, 0.0150, 0.0180,
                             0.0200, 0.0220, 0.0250, 0.0288, 0.0310, 0.0340};

    struct TestCaseTyp<DT> tc[NUM_TEST_CASES] = {
        {0.10, 0.01, 0, 1, 0.9895549285}, {0.09, 0.02, 1, 3, 0.9515974490}, {0.08, 0.03, 2, 5, 0.9098468303},
            {0.07, 0.04, 5, 10, 0.8071884801}, {
            0.06, 0.05, 10, 20, 0.4753889516
        }
    };

    // tolerance
    DT err = 1e-8;
    int failCnt = 0;

    std::cout << std::setprecision(10) << std::endl;

    // test cases
    for (int i = 0; i < NUM_TEST_CASES; i++) {
        DT P = dut(tc[i].a, tc[i].sigma, maturity, interestRates, tc[i].t, tc[i].T);

        if (std::abs(P - tc[i].P) > err) {
            std::cout << "Failure at t:" << tc[i].t << "-->" << tc[i].T << " calculated P:" << P
                      << " expected P:" << tc[i].P << std::endl;
            failCnt++;
        }

        std::cout << P << std::endl;
    }

    if (failCnt != 0) {
        return 1;
    } else {
        return 0;
    }
}
