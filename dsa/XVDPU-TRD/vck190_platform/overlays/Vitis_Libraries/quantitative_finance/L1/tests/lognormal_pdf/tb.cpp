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

DT dut(DT average, DT sigma, DT x);

int main() {
    int nerr = 0;
    DT err = 1e-12;
    DT golden_pdf[2][4] = {{0.0, 1.0, 1.0, 0.2419707245191434}, {1.5, 2.6, 4.2, 0.08948796234693353}};
    for (int i = 0; i < 2; i++) {
        DT average = golden_pdf[i][0];
        DT sigma = golden_pdf[i][1];
        DT x = std::exp(golden_pdf[i][2]);
        DT pdf = dut(average, sigma, x) * x;
        if (std::abs(pdf - golden_pdf[i][3]) > err) {
            std::cout << "average=" << golden_pdf[i][0] << ",sigma=" << golden_pdf[i][1] << ",x=" << golden_pdf[i][2]
                      << ",pdf=" << pdf << ",err=" << std::abs(pdf - golden_pdf[i][3]) << std::endl;
            nerr++;
        }
    }
    return nerr;
}
