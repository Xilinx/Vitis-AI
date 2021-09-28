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

void dut(DT t, DT T, DT x[2], DT flatRate, DT a, DT sigma, DT b, DT eta, DT rho, DT* discountBond, DT* shortRate);

int main() {
    DT discountBondGolden = 1.0;
    DT shortRateGolden = -0.110675579848115;
    DT err = 1e-8;

    DT t = 4.5024657534246577;
    DT T = 4.5024657534246577;
    DT flatRate = 0.04875825;
    DT a = 0.050055733653096922;
    DT sigma = 0.0094424342056787739;
    DT b = 0.050052910248222851;
    DT eta = 0.0094424313463861171;
    DT rho = -0.76300324120391616;

    DT x[2] = {-0.079888357349334832, -0.079888850463537983};
    DT discountBond, shortRate;
    dut(t, T, x, flatRate, a, sigma, b, eta, rho, &discountBond, &shortRate);

    std::cout << "discountBond=" << discountBond << ",diff=" << discountBond - discountBondGolden << std::endl;
    std::cout << "shortRate=" << shortRate << ",diff=" << shortRate - shortRateGolden << std::endl;
    if ((std::abs(discountBond - discountBondGolden) < err) && (std::abs(shortRate - shortRateGolden) < err))
        return 0;
    else
        return 1;
}
