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
 * @file FDMG2_k0.cpp
 *
 * @brief This file contains top function of test case.
 */

#include "fdmg2_engine_kernel.hpp"

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
                         double fixedRate,
                         double rate,
                         double nominal,
                         double stoppingTimes[EXSize + 1],
                         double fixedAccrualTime[EXSize + 1],
                         double floatingAccrualPeriod[EXSize + 1],
                         double iborTime[EXSize + 1],
                         double iborPeriod[EXSize + 1],
                         double NPV[1]) {
#pragma HLS INTERFACE m_axi port = NPV bundle = gmem0 offset = slave
#pragma HLS INTERFACE m_axi port = stoppingTimes bundle = gmem1 offset = slave
#pragma HLS INTERFACE m_axi port = fixedAccrualTime bundle = gmem2 offset = slave
#pragma HLS INTERFACE m_axi port = floatingAccrualPeriod bundle = gmem3 offset = slave
#pragma HLS INTERFACE m_axi port = iborTime bundle = gmem4 offset = slave
#pragma HLS INTERFACE m_axi port = iborPeriod bundle = gmem5 offset = slave

#pragma HLS INTERFACE s_axilite port = a bundle = control
#pragma HLS INTERFACE s_axilite port = sigma bundle = control
#pragma HLS INTERFACE s_axilite port = b bundle = control
#pragma HLS INTERFACE s_axilite port = eta bundle = control
#pragma HLS INTERFACE s_axilite port = rho bundle = control
#pragma HLS INTERFACE s_axilite port = steps bundle = control
#pragma HLS INTERFACE s_axilite port = xGrid bundle = control
#pragma HLS INTERFACE s_axilite port = yGrid bundle = control
#pragma HLS INTERFACE s_axilite port = invEps bundle = control
#pragma HLS INTERFACE s_axilite port = theta bundle = control
#pragma HLS INTERFACE s_axilite port = mu bundle = control
#pragma HLS INTERFACE s_axilite port = fixedRate bundle = control
#pragma HLS INTERFACE s_axilite port = rate bundle = control
#pragma HLS INTERFACE s_axilite port = nominal bundle = control
#pragma HLS INTERFACE s_axilite port = stoppingTimes bundle = control
#pragma HLS INTERFACE s_axilite port = fixedAccrualTime bundle = control
#pragma HLS INTERFACE s_axilite port = floatingAccrualPeriod bundle = control
#pragma HLS INTERFACE s_axilite port = iborTime bundle = control
#pragma HLS INTERFACE s_axilite port = iborPeriod bundle = control
#pragma HLS INTERFACE s_axilite port = NPV bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::fintech::FdG2SwaptionEngine<double, EXSize, XMAX, YMAX> swapEngine;
    if (xGrid % 2 == 0) xGrid += 1;
    if (yGrid % 2 == 0) yGrid += 1;

    swapEngine.init(a, sigma, b, eta, rho, STEPS, xGrid, yGrid, invEps, theta, mu, fixedRate, rate, nominal,
                    stoppingTimes, fixedAccrualTime, floatingAccrualPeriod, iborTime, iborPeriod);

    swapEngine.calculate();

    NPV[0] = swapEngine.getNPV();
}
