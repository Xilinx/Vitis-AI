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
#include "poisson_distribution.hpp"
#define DT double

DT dut(unsigned k, DT x);

int main() {
    int nerr = 0;
    DT err = 1e-10;
    // pmf
    DT testArray[2][3] = {{1, 5, 0.03368973499542734}, {30, 5, 2.365743446112826e-14}};
    for (int i = 0; i < 2; i++) {
        DT golden_pmf = testArray[i][2];
        DT a = testArray[i][0];
        DT x = testArray[i][1];
        DT pmf = dut((unsigned)a, x);
        if (std::abs(pmf - golden_pmf) / golden_pmf > err) {
            std::cout << "a=" << a << ",x=" << x << ",pmf=" << pmf << ",err=" << std::abs(pmf - golden_pmf) / golden_pmf
                      << std::endl;
            nerr++;
        }
    }
    return nerr;
}
