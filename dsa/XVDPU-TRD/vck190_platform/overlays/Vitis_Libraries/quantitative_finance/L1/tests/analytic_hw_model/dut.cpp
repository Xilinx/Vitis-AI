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
 * @brief This file contains top function of test case.
 */

#include <ap_fixed.h>
#include <ap_int.h>
#include "hls_stream.h"
#include "hw_model.hpp"
#include <iostream>

#define LEN (12)
typedef double DT;
typedef xf::fintech::HWModelAnalytic<DT, LEN> Model;

DT dut(DT a, DT sigma, DT maturity[LEN], DT interestRates[LEN], DT t, DT T) {
    Model model;
    model.initialization(a, sigma, maturity, interestRates);

    // get the short rate at time t to pass into bond price
    DT rate = model.shortRate(t);
    DT P = model.discountBond(t, T, rate);
    return P;
}
