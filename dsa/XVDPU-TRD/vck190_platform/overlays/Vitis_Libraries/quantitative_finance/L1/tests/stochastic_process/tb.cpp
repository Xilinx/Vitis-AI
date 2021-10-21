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

void dut(DT t, DT dt, DT x, DT a, DT sigma, DT theta, DT k, DT* expectation, DT* variance);

int main() {
    DT expectationGolden = -0.10645875033864;
    DT varianceGolden = 0.0005896109425059;
    DT err = 1e-8;

    DT t = 4.4986301369863018;
    DT dt = 0.4958904109589044;
    DT a = 0.043389447297063261;
    DT sigma = 0.068963597413997324;
    DT theta = 0.10177935032500834;
    DT k = 0.10220445840674211;

    DT x = -0.079888357349334832;
    DT expectation, variance;
    dut(t, dt, x, a, sigma, theta, k, &expectation, &variance);

    std::cout << "expectation=" << std::setprecision(16) << expectation << ",diff=" << expectation - expectationGolden
              << std::endl;
    std::cout << "variance=" << std::setprecision(16) << variance << ",diff=" << variance - varianceGolden << std::endl;
    if ((std::abs(expectation - expectationGolden) < err) && (std::abs(variance - varianceGolden) < err))
        return 0;
    else
        return 1;
}
