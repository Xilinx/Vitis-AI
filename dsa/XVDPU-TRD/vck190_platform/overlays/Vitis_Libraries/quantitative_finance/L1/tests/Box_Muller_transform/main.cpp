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
#include <random>

void bm_cpu(double u1, double u2, double& z1, double& z2) {
    double r = std::sqrt(-2 * std::log(u1));
    double theta = 6.28318530718 * u2;

    z1 = std::cos(theta) * r;
    z2 = std::sin(theta) * r;
}

void dut(double u1, double u2, double& z1, double& z2);

int main() {
    double len = 20;

    bool err = false;

    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            double u1, u2, z1, z2, g1, g2;
            u1 = 1.0 / len / 2 + i * 1.0 / len;
            u2 = 1.0 / len / 2 + j * 1.0 / len;
            bm_cpu(u1, u2, g1, g2);
            dut(u1, u2, z1, z2);
            if (abs(g1 - z1) > 0.000001 || abs(g2 - z2) > 0.000001) {
                err = true;
                std::cout << "golden: " << g1 << " " << g2 << std::endl;
                std::cout << "result: " << z1 << " " << z2 << std::endl;
            }
        }
    }

    if (err) {
        std::cout << "function error!\n";
        return -1;
    } else {
        std::cout << "function correct!\n";
        return 0;
    }
}
