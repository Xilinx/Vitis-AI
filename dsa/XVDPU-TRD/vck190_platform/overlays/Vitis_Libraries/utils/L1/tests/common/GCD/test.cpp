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
#include "xf_utils_hw/common.hpp"

int dut() {
    static const int a = 99;
    static const int b = 77;
    int c = xf::common::utils_hw::GCD<a, b>::value;
    return c;
}

int main(int argc, const char* argv[]) {
    int c = dut();
    if (c != 11) {
        std::cout << "FAIL" << std::endl;
        return 1;
    }
    std::cout << "PASS" << std::endl;
    return 0;
}
