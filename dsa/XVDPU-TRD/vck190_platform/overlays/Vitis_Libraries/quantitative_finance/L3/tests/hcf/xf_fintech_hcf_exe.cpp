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
/*--
 * ---------------------------------------------------------------------------------------------------------------------*/
/*-- DISCLAIMER AND CRITICAL APPLICATIONS */
/*--
 * ---------------------------------------------------------------------------------------------------------------------*/
/*-- */
/*-- (c) Copyright 2019 Xilinx, Inc. All rights reserved. */
/*-- */
/*-- This file contains confidential and proprietary information of Xilinx, Inc.
 * and is protected under U.S. and          */
/*-- international copyright and other intellectual property laws. */
/*-- */
/*-- DISCLAIMER */
/*-- This disclaimer is not a license and does not grant any rights to the
 * materials distributed herewith. Except as      */
/*-- otherwise provided in a valid license issued to you by Xilinx, and to the
 * maximum extent permitted by applicable     */
/*-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS,
 * AND XILINX HEREBY DISCLAIMS ALL WARRANTIES  */
/*-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING BUT NOT LIMITED
 * TO WARRANTIES OF MERCHANTABILITY, NON-     */
/*-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall
 * not be liable (whether in contract or tort,*/
/*-- including negligence, or under any other theory of liability) for any loss
 * or damage of any kind or nature           */
/*-- related to, arising under or in connection with these materials, including
 * for any direct, or any indirect,          */
/*-- special, incidental, or consequential loss or damage (including loss of
 * data, profits, goodwill, or any type of      */
/*-- loss or damage suffered as a retVal of any action brought by a third party)
 * even if such damage or loss was          */
/*-- reasonably foreseeable or Xilinx had been advised of the possibility of the
 * same.                                    */
/*-- */
/*-- CRITICAL APPLICATIONS */
/*-- Xilinx products are not designed or intended to be fail-safe, or for use in
 * any application requiring fail-safe      */
/*-- performance, such as life-support or safety devices or systems, Class III
 * medical devices, nuclear facilities,       */
/*-- applications related to the deployment of airbags, or any other
 * applications that could lead to death, personal      */
/*-- injury, or severe property or environmental damage (individually and
 * collectively, "Critical                         */
/*-- Applications"). Customer assumes the sole risk and liability of any use of
 * Xilinx products in Critical               */
/*-- Applications, subject only to applicable laws and regulations governing
 * limitations on product liability.            */
/*-- */
/*-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE
 * AT ALL TIMES.                             */
/*--
 * ---------------------------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>

#include <chrono>
#include <vector>

#include "xf_fintech_api.hpp"

using namespace xf::fintech;

int main(int argc, char** argv) {
    // binomial tree fintech model...
    std::string path = std::string(argv[1]);
    hcf hcf(path);

    std::string device = TOSTRING(DEVICE_PART);
    if (argc == 3) {
        device = std::string(argv[2]);
    }

    int retval = XLNX_OK;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    std::vector<Device*> deviceList;
    Device* pChosenDevice;

    // device passed in command line args
    deviceList = DeviceManager::getDeviceList(device);

    if (deviceList.size() == 0) {
        printf("No matching devices found\n");
        exit(0);
    }

    printf("Found %zu matching devices\n", deviceList.size());

    // we'll just pick the first device in the...
    pChosenDevice = deviceList[0];

    if (retval == XLNX_OK) {
        // turn off trace output...turn it on here if you want extra debug output...
        Trace::setEnabled(true);
    }

    printf("\n\n\n");
    printf("[XF_FINTECH] HCF trying to claim device...\n");

    float K = 100.0;
    float rho = -0.9;
    float T = 1.0;
    float r = 0.05;
    float vvol = 0.3;
    float vbar = 0.04;
    float kappa = 1.5;

    if (retval == XLNX_OK) {
        printf("\n");
        printf("\n");
        printf("[XF_FINTECH] ==========\n");
        printf("[XF_FINTECH] Parameters\n");
        printf("[XF_FINTECH] ==========\n");
        printf("[XF_FINTECH] Strike price                       = %f\n", K);
        printf("[XF_FINTECH] Rho (Weiner process correlation)   = %f\n", rho);
        printf("[XF_FINTECH] Time to maturity                   = %f\n", T);
        printf("[XF_FINTECH] Risk free interest rate            = %f\n", r);
        printf("[XF_FINTECH] Rate of reversion (kappa)          = %f\n", kappa);
        printf("[XF_FINTECH] volatility of volatility (sigma)   = %f\n", vvol);
        printf("[XF_FINTECH] Long term average variance (theta) = %f\n", vbar);
        printf("\n");
    }

    start = std::chrono::high_resolution_clock::now();

    retval = hcf.claimDevice(pChosenDevice);

    end = std::chrono::high_resolution_clock::now();

    if (retval == XLNX_OK) {
        printf("[XF_FINTECH] Device setup time = %lld microseconds\n",
               (long long int)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    } else {
        printf("[XF_FINTECH] Failed to claim device - error = %d\n", retval);
    }

    static const int numberOptions = 16;
    int ret = 0; // assume pass
    if (retval == XLNX_OK) {
        printf("[XF_FINTECH] Multiple Options European Call [%d]\n", numberOptions);

        hcf::hcf_input_data inputData[numberOptions];
        float outputData[numberOptions];
        start = std::chrono::high_resolution_clock::now();

        float s0 = 80.0;
        float v0 = 0.1;

        // populate some data
        for (int i = 0; i < numberOptions; i++) {
            inputData[i].s0 = s0 + (3 * i);
            inputData[i].v0 = v0;
            inputData[i].K = K;
            inputData[i].rho = rho;
            inputData[i].T = T;
            inputData[i].r = r;
            inputData[i].kappa = kappa;
            inputData[i].vvol = vvol;
            inputData[i].vbar = vbar;
        }

        retval = hcf.run(inputData, outputData, numberOptions);

        end = std::chrono::high_resolution_clock::now();

        if (retval == XLNX_OK) {
            for (int i = 0; i < numberOptions; i++) {
                printf("[XF_FINTECH] [%02u] OptionPrice = %f\n", i, outputData[i]);
            }
            long long int executionTime =
                (long long int)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            printf(
                "[XF_FINTECH] ExecutionTime = %lld microseconds (average %lld "
                "microseconds)\n",
                executionTime, executionTime / numberOptions);
        }
        // quick fix to get pass/fail criteria
        if (std::abs(outputData[numberOptions - 1] - 33.002968) > 0.001) {
            ret = 1;
        }
    }

    printf("[XF_FINTECH] HCF releasing device...\n");
    retval = hcf.releaseDevice();

    if (!ret) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
    return ret;
}
