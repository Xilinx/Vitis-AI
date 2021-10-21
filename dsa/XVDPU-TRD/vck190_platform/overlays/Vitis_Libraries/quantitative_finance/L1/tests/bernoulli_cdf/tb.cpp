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
#define DT double

DT dut(unsigned int k, DT p);

int main() {
    int nerr = 0;
    DT err = 1e-12;
    unsigned n = 10;
    unsigned k[2] = {1, 0};
    DT p = 0.6;
    DT golden_cdf[2] = {1, 1.0 - p};
    for (int i = 0; i < 2; i++) {
        DT cdf = dut(k[i], p);
        if (std::abs(cdf - golden_cdf[i]) > err) {
            std::cout << "k=" << k << ",p=" << p << ",cdf=" << cdf << ",err=" << std::abs(cdf - golden_cdf[i])
                      << std::endl;
            nerr++;
        }
    }
    return nerr;
}
