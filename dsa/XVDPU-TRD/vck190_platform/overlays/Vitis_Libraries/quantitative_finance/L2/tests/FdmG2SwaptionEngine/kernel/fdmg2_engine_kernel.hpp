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
 * @file fdmg2_engine_kernel.hpp
 *
 * @brief This file contains top function of test case.
 */

#include <ap_int.h>
#include <iomanip>
#include <iostream>

#include "xf_fintech/fd_g2_swaption_engine.hpp"
#include "xf_fintech/fdmmesher.hpp"
#include "xf_fintech/g2_model.hpp"
#include "hls_stream.h"
#include "xf_fintech/types.hpp"

#define SIZE 5
#define XMAX 51
#define YMAX 51
#define STEPS 10
#define EXSize 5

extern "C" void FDMG2_k0(double a,
                         double sigma,
                         double b,
                         double eta,
                         double rho,
                         unsigned int steps,
                         unsigned int xGrid,
                         unsigned int yGrid,
                         double invEps,
                         double theta,
                         double mu,
                         double fixed,
                         double rate,
                         double nominal,
                         double stoppingTimes[EXSize + 1],
                         double fixedAccrualTime[EXSize + 1],
                         double floatingAccrualPeriod[EXSize + 1],
                         double iborTime[EXSize + 1],
                         double iborPeriod[EXSize + 1],
                         double NPV[1]);
