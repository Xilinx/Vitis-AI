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
#include "ornstein_uhlenbeck_process.hpp"
#include "fdmmesher.hpp"
#include <iostream>

#define GridMax 5

typedef double DT;
typedef xf::fintech::OrnsteinUhlenbeckProcess<DT> Process;
typedef xf::fintech::Fdm1dMesher<DT, GridMax> Mesher;

void dut(DT maturity, DT invEps, unsigned int grid, DT a, DT sigma, DT locations[GridMax]) {
    Process process;
    process.init(a, sigma, 0.0, 0.0);
    Mesher mesher;
    mesher.init(process, maturity, invEps, grid, locations);
}
