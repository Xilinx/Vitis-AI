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

void bs_cpu(double riskFreeRate,
            double dividendYield,
            double timeLength,
            double volatility,
            double underlying,
            unsigned int timeSteps,
            unsigned int paths,
            double* rand_in,
            double* out) {
    double dt = timeLength / timeSteps;
    for (int i = 0; i < timeSteps; ++i) {
        for (int j = 0; j < paths; ++j) {
            double preS;
            if (i == 0)
                preS = underlying;
            else
                preS = out[(i - 1) * paths + j];
            out[i * paths + j] = preS * std::exp((riskFreeRate - dividendYield - 0.5 * volatility * volatility) * dt +
                                                 volatility * rand_in[i * paths + j] * std::sqrt(dt));
        }
    }
}

void dut(double riskFreeRate,
         double dividendYield,
         double timeLength,
         double volatility,
         double underlying,
         unsigned int timeSteps,
         unsigned int paths,
         double rand_in[10240],
         double out[10240]);

int main() {
    unsigned int timeSteps = 10;
    unsigned int paths = 1024;
    double riskFreeRate = 0.06;
    double dividendYield = 0.0;
    double timeLength = 1;
    double volatility = 0.20;
    double underlying = 36;

    double* rand_in = new double[10 * 1024];
    double* out = new double[10 * 1024];
    double* ref = new double[10 * 1024];

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < timeSteps; ++i) {
        for (int j = 0; j < paths; ++j) {
            double d = distribution(generator);
            rand_in[i * paths + j] = d;
        }
    }
    dut(riskFreeRate, dividendYield, timeLength, volatility, underlying, timeSteps, paths, rand_in, out);
    bs_cpu(riskFreeRate, dividendYield, timeLength, volatility, underlying, timeSteps, paths, rand_in, ref);
    bool err = false;
    for (int i = 0; i < timeSteps; ++i) {
        for (int j = 0; j < paths; ++j) {
            double diff = std::abs(out[i * paths + j] - ref[i * paths + j]);
            if (diff > 0.0001) {
                err = true;
            }
        }
    }
    delete[] rand_in;
    delete[] out;
    delete[] ref;
    if (err) {
        std::cout << "function error!\n";
        return -1;
    } else {
        std::cout << "function correct!\n";
        return 0;
    }
}
