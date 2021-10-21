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

DT dut(DT x, DT y);

int main() {
    int nerr = 0;
    DT err = 1e-12;
    DT golden_icdf = 7;
    DT a = 4;
    DT x = 0.9;
    DT icdf = dut(a, x);
    if (std::abs(icdf - golden_icdf) > err) {
        std::cout << "a=" << a << ",x=" << x << ",icdf=" << icdf << ",err=" << std::abs(icdf - golden_icdf)
                  << std::endl;
        nerr++;
    }
    return nerr;
}
