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
#include <iostream>

#define TEST_DT double
#define TEST_MAXVARS 15
#define TEST_MAXOBS 80
#define N_FACTORS 3

extern "C" void dut(unsigned rows,
                    unsigned cols,
                    TEST_DT input[TEST_MAXVARS][TEST_MAXOBS],
                    TEST_DT outputLoadings[TEST_MAXVARS][N_FACTORS]);

int main() {
    TEST_DT inputData[TEST_MAXVARS][TEST_MAXOBS] = {
#include "inputData.golden"
    };
    TEST_DT expectedLoadings[TEST_MAXVARS][N_FACTORS] = {
#include "expectedLoadings.golden"
    };

    TEST_DT outputLoadings[TEST_MAXVARS][N_FACTORS];

    dut(TEST_MAXVARS, TEST_MAXOBS, inputData, outputLoadings);

    bool err = false;
    const double epsilon = 1e-4;
    for (unsigned i = 0; i < TEST_MAXVARS; i++) {
        for (unsigned j = 0; j < N_FACTORS; j++) {
            double diff = outputLoadings[i][j] - expectedLoadings[i][j];
            if ((diff * diff) > epsilon) {
                std::cout << "Error computing PCA @" << i << ", " << j << ": Expected " << expectedLoadings[i][j]
                          << " got loadings " << outputLoadings[i][j] << std::endl;
                err = true;
            }
        }
    }

    if (err) {
        std::cout << "Function error!" << std::endl;
        return 1;
    } else {
        std::cout << "Function correct!" << std::endl;
        return 0;
    }
}
