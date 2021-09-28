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

/**
 * @file dut.cpp
 *
 * @brief This file contains top funtion of the test case.
 */

#include "polyfit.hpp"

#define TEST_DT double
// 4th degree polynomial
#define N 5
#define MAX_WIDTH 40

void dut(unsigned n, TEST_DT input[MAX_WIDTH], TEST_DT output[N]) {
    xf::fintech::polyfit<TEST_DT, N, MAX_WIDTH>(input, n, output);
}
