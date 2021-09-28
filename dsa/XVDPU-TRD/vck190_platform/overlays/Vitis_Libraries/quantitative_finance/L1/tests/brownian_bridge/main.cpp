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
#include <cmath>
#include <iostream>
#include <random>

#include "brownianbridge.hpp"
void brownian_bridge_top(int stpes, double a[100], double b[100]);

int main() {
    int steps = 100;
    double input[steps];
    double output[100];
    std::vector<double> inputQL;
    inputQL.resize(steps);
    std::vector<double> resultQL;
    resultQL.resize(steps);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < steps; ++i) {
        // double rd = distribution(generator);
        input[i] = 1;
        inputQL[i] = 1;
    }

    brownian_bridge_top(steps, input, output);

    // Get QuantLib function output
    QuantLib::BrownianBridge ql_bridge(steps);
    ql_bridge.transform(inputQL, resultQL);

    // Compare results
    bool hasDiff = false;
    for (int i = 0; i < steps; ++i) {
        if (std::fabs(output[i] - resultQL[i]) > 0.0001) {
            std::cout << "Error " << output[i] << " " << resultQL[i] << std::endl;
            hasDiff = true;
        }
    }

    if (!hasDiff) {
        std::cout << "The result is correct" << std::endl;
        return 0;
    }
    std::cout << "The result is wrong" << std::endl;
    return -1;
}
