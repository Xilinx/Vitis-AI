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

typedef double DT;

void dut(DT t, DT dt, DT x, DT dw, DT a, DT sigma, DT* expectation, DT* variance, DT* stdDeviation, DT* evolve);

int main() {
    DT expectationGolden = -0.077929764705508;
    DT varianceGolden = 4.31338392533706e-05;
    DT stdDeviationGolden = 0.0065676357430486;
    DT evolveGolden = -0.015251431186560;
    DT err = 1e-8;

    DT t = 4.4986301369863018;
    DT dt = 0.4958904109589044;
    DT dw = -2.3222102721977445;
    DT a = 0.050055733653096922;
    DT sigma = 0.0094424342056787739;

    DT x = -0.079888357349334832;
    DT expectation, variance, stdDeviation, evolve;
    dut(t, dt, x, dw, a, sigma, &expectation, &variance, &stdDeviation, &evolve);

    std::cout << "expectation=" << std::setprecision(16) << expectation << ",diff=" << expectation - expectationGolden
              << std::endl;
    std::cout << "variance=" << std::setprecision(16) << variance << ",diff=" << variance - varianceGolden << std::endl;
    std::cout << "stdDeviation=" << std::setprecision(16) << stdDeviation
              << ",diff=" << stdDeviation - stdDeviationGolden << std::endl;
    std::cout << "evolve=" << std::setprecision(16) << evolve << ",diff=" << evolve - evolveGolden << std::endl;
    if ((std::abs(expectation - expectationGolden) < err) && (std::abs(variance - varianceGolden) < err) &&
        (std::abs(stdDeviation - stdDeviationGolden) < err) && (std::abs(evolve - evolveGolden)) < err)
        return 0;
    else
        return 1;
}
