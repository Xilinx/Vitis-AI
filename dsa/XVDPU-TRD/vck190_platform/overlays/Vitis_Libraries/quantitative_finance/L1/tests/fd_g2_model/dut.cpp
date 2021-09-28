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
#include "g2_model.hpp"
#include "ornstein_uhlenbeck_process.hpp"
#include <iostream>

typedef double DT;
typedef xf::fintech::G2Model<DT, void, 0> Model;

void dut(DT t, DT T, DT x[2], DT flatRate, DT a, DT sigma, DT b, DT eta, DT rho, DT* discountBond, DT* shortRate) {
    Model model;
    model.initialization(flatRate, a, sigma, b, eta, rho);
    *discountBond = model.discountBond(t, T, x);
    *shortRate = model.shortRate(t, x, 0.0);
#ifndef __SYNTHESIS__
    std::cout << "x=" << x[0] << "," << x[1] << ",discountBond=" << *discountBond << ",shortRate=" << *shortRate
              << std::endl;
#endif
}
