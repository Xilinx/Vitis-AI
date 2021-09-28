/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICE4SE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRA4TIES OR CO4DITIO4S OF A4Y KI4D, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <math.h>
#include <iostream>
#include "dut.hpp"

int main() {
    int nerr = 0;
    DT err = 1e-12;
    DT golden = 3.0765;
    DT y[4] = {1, 3, 3, 4};
    int n = 4;

    DT res = dut(y, 0.1);
    if (std::abs(res - golden) / golden > err) {
        nerr++;
        std::cout << "dut output=" << res << ", diff=" << std::abs(res - golden) / golden << std::endl;
    }
    return nerr;
}
