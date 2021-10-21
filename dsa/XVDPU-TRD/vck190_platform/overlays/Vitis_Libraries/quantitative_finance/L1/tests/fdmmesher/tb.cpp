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

#define GridMax 5
typedef double DT;

void dut(DT maturity, DT invEps, unsigned int grid, DT a, DT sigma, DT locations[GridMax]);

int main() {
    int nerror = 0;
    DT err = 1e-8;
    DT locationsGolden[GridMax] = {-0.07988835734933485, -0.01263399786712028, 0, 0.01263399786712028,
                                   0.07988835734935389};

    DT maturity = 5.0027397260273974;
    DT invEps = 1e-5;
    unsigned int grid = 5;
    DT a = 0.050055733653096922;
    DT sigma = 0.0094424342056787739;

    DT locations[GridMax];
    dut(maturity, invEps, grid, a, sigma, locations);

    for (int i = 0; i < GridMax; i++) {
        std::cout << "locations[" << i << "=]" << std::setprecision(16) << locations[i]
                  << ",diff=" << locations[i] - locationsGolden[i] << std::endl;
        if (locations[i] - locationsGolden[i] >= err) nerror++;
    }

    if (nerror)
        return nerror;
    else
        return 0;
}
