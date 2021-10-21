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

#define TEST_DT double
// 4th degree polynomial
#define N 5
#define MAX_WIDTH 40

void dut(unsigned n, TEST_DT input[MAX_WIDTH], TEST_DT output[N]);

void eval(TEST_DT coeffs[N], TEST_DT* input, unsigned s) {
    for (unsigned i = 0; i < s; i++) {
        TEST_DT sum = 0.0;
        for (unsigned j = 0; j < N; j++) {
            sum += coeffs[j] * pow((TEST_DT)i, (N - 1) - j);
        }
        input[i] = sum;
    }
}

int main() {
    const unsigned int size = 35;

    TEST_DT inputData[MAX_WIDTH];
    TEST_DT expectedFit[N] = {3.0, 1.2, -0.4, 0.05, -0.63};
    TEST_DT gotFit[N];
    // expected == polyfit(eval(expected))
    eval(expectedFit, inputData, size);

    dut(size, inputData, gotFit);

    bool err = false;
    const double epsilon = 1e-6;
    for (unsigned j = 0; j < N; j++) {
        double diff = gotFit[j] - expectedFit[j];
        if ((diff * diff) > epsilon) {
            std::cout << "Error computing polyfit @" << j << ": Expected " << expectedFit[j] << " got loadings "
                      << gotFit[j] << std::endl;
            err = true;
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
