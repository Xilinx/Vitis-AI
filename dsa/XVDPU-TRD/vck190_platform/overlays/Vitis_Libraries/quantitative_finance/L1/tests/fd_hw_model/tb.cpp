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

#define LEN 16
typedef double DT;

void dut(DT t, DT T, DT flatRate, DT spread, DT a, DT sigma, DT* discountBond, DT* shortRate);

int main() {
    DT discountBondGolden = 1.0;
    DT shortRateGolden = 0.049482681780887;
    DT err = 1e-8;

    DT t = 4.5024657534246577;
    DT T = 4.5024657534246577;
    DT flatRate = 0.04875825;
    DT a = 0.050055733653096922;
    DT sigma = 0.0094424342056787739;
    DT golden = 1.0160774569439277;
    double spread = 0.0;
    DT discountBond, shortRate;
    dut(t, T, flatRate, spread, a, sigma, &discountBond, &shortRate);

    std::cout << "discountBond=" << std::setprecision(16) << discountBond
              << ",diff=" << discountBond - discountBondGolden << std::endl;
    std::cout << "shortRate=" << std::setprecision(16) << shortRate << ",diff=" << shortRate - shortRateGolden
              << std::endl;
    if ((std::abs(discountBond - discountBondGolden) < err) && (std::abs(shortRate - shortRateGolden) < err))
        return 0;
    else
        return 1;
}
