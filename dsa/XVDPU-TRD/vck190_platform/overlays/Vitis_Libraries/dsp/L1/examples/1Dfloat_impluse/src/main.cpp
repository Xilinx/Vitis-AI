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
//================================== End Lic =================================================
#include "top_module.hpp"
#include <iostream>

int main(int argc, char** argv) {
    T_in inData[SSR][FFT_LEN / SSR];
    T_out outData[SSR][FFT_LEN / SSR];
    for (int r = 0; r < SSR; ++r) {
        for (int t = 0; t < FFT_LEN / SSR; ++t) {
            if (r == 0 && t == 0)
                inData[r][t] = 1;
            else
                inData[r][t] = 0;
        }
    }
    for (int t = 0; t < 4; ++t) {
        // Added Dummy loop iterations
        // to make II measurable in cosim
        fft_top(inData, outData);
    }
    int errs = 0;
    for (int r = 0; r < SSR; ++r) {
        for (int t = 0; t < FFT_LEN / SSR; ++t) {
            if (outData[r][t].real() != 1 || outData[r][t].imag() != 0) errs++;
        }
    }
    std::cout << "===============================================================" << std::endl;
    std::cout << "--Input Impulse:" << std::endl;
    for (int r = 0; r < SSR; ++r) {
        for (int t = 0; t < FFT_LEN / SSR; ++t) {
            std::cout << inData[r][t] << std::endl;
        }
    }
    std::cout << "===============================================================" << std::endl;

    std::cout << "===============================================================" << std::endl;
    std::cout << "--Output Step fuction:" << std::endl;
    for (int r = 0; r < SSR; ++r) {
        for (int t = 0; t < FFT_LEN / SSR; ++t) {
            std::cout << outData[r][t] << std::endl;
        }
    }
    std::cout << "===============================================================" << std::endl;

    return errs;
}
