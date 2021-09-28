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
/*-- This file contains confidential and proprietary information of Xilinx, Inc. and is protected under U.S. and */
/*-- international copyright and other intellectual property laws. */
/*-- */
/*-- DISCLAIMER */
/*-- This disclaimer is not a license and does not grant any rights to the materials distributed herewith. Except as */
/*-- otherwise provided in a valid license issued to you by Xilinx, and to the maximum extent permitted by applicable */
/*-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
 */
/*-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON- */
/*-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether in contract or
 * tort,*/
/*-- including negligence, or under any other theory of liability) for any loss or damage of any kind or nature */
/*-- related to, arising under or in connection with these materials, including for any direct, or any indirect, */
/*-- special, incidental, or consequential loss or damage (including loss of data, profits, goodwill, or any type of */
/*-- loss or damage suffered as a retVal of any action brought by a third party) even if such damage or loss was */
/*-- reasonably foreseeable or Xilinx had been advised of the possibility of the same. */
/*-- */
/*-- CRITICAL APPLICATIONS */
/*-- Xilinx products are not designed or intended to be fail-safe, or for use in any application requiring fail-safe */
/*-- performance, such as life-support or safety devices or systems, Class III medical devices, nuclear facilities, */
/*-- applications related to the deployment of airbags, or any other applications that could lead to death, personal */
/*-- injury, or severe property or environmental damage (individually and collectively, "Critical */
/*-- Applications"). Customer assumes the sole risk and liability of any use of Xilinx products in Critical */
/*-- Applications, subject only to applicable laws and regulations governing limitations on product liability. */
/*-- */
/*-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES. */
/*--
 * ---------------------------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>

#include <chrono>
#include <vector>
#include <boost/algorithm/string.hpp>

#include "xf_fintech_api.hpp"
#include "models/xf_fintech_fdbslv.hpp"
using namespace xf::fintech;

struct testcaseParams {
    unsigned int solverN;
    unsigned int solverM;
    double solverTheta;
    double modelS;
    double modelK;
    double boundaryLower;
    double boundaryUpper;
};

int ReadVector(const std::string filename, std::vector<float>& A, const unsigned int size) {
    std::ifstream file(filename);

    std::vector<std::string> v;
    std::string line;

    if (file.good()) {
        std::cout << "Opened " << filename << " OK" << std::endl;

        unsigned int i = 0;
        while (file.good()) {
            getline(file, line);
            if (line.length()) {
                A[i] = std::stof(line);
                i++;
            }
            if (i > size) {
                std::cout << "Warning! File has more than expected " << size << " lines..." << std::endl;
                break;
            }
        }
    } else {
        std::cout << "Couldn't open " << filename << std::endl;
        return 1;
    }
    return 0;
}

/// @brief Read testcase parameters
int ReadTestcaseParameters(std::string filename, testcaseParams& params) {
    std::ifstream file(filename);

    std::vector<std::string> v;
    std::string line;

    if (file.good()) {
        std::cout << "Opened " << filename << " OK" << std::endl;

        getline(file, line);
        boost::split(v, line, [](char c) { return c == ','; });
        params.solverN = std::stoi(v[0]);
        params.solverM = std::stoi(v[1]);
        params.solverTheta = std::stod(v[2]);
        params.modelS = std::stod(v[3]);
        params.modelK = std::stod(v[4]);
        params.boundaryLower = std::stod(v[5]);
        params.boundaryUpper = std::stod(v[6]);
    } else {
        std::cout << "Couldn't open " << filename << std::endl;
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    // xclbin file
    std::string path = std::string(argv[1]);

    // test data
    std::string test_dir = std::string(argv[2]);

    // Get the testcase parameters
    testcaseParams params;
    std::cout << "Loading testcase..." << std::endl;
    if (ReadTestcaseParameters(test_dir + "/parameters.csv", params)) return EXIT_FAILURE;

    // Extract N, M to aid readability
    const unsigned int N = params.solverN;
    const unsigned int M = params.solverM;

    // device
    std::string device = TOSTRING(DEVICE_PART);
    if (argc == 4) {
        device = std::string(argv[3]);
    }

    // Create the FD solver object
    fdbslv fdbslv(N, M, path);

    int retval = XLNX_OK;

    std::vector<Device*> deviceList;
    Device* pChosenDevice;

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

    // clain the device
    printf("\n\n\n");
    printf("[XF_FINTECH] trying to claim device...\n");
    retval = fdbslv.claimDevice(pChosenDevice);
    if (retval != XLNX_OK) {
        printf("[XF_FINTECH] Failed to claim device - error = %d\n", retval);
    }

    // read input data
    std::vector<float> xGrid(N);
    std::vector<float> tGrid(M);
    std::vector<float> sigma(N * M);
    std::vector<float> rate(M);
    std::vector<float> initialCondition(N);
    std::vector<float> solution(N);
    std::vector<float> reference(N);

    if (ReadVector(test_dir + "/xGrid.csv", xGrid, N)) return 1;
    if (ReadVector(test_dir + "/tGrid.csv", tGrid, M)) return 1;
    if (ReadVector(test_dir + "/sigma.csv", sigma, N * M)) return 1;
    if (ReadVector(test_dir + "/rate.csv", rate, M)) return 1;
    if (ReadVector(test_dir + "/initialCondition.csv", initialCondition, N)) return 1;
    if (ReadVector(test_dir + "/reference.csv", reference, N)) return 1;

    // run the kernel
    retval = fdbslv.run(xGrid, tGrid, sigma, rate, initialCondition, params.solverTheta, params.boundaryLower,
                        params.boundaryUpper, solution);

    // check the results
    float max_diff = 0;
    int index = 0;
    std::cout << "actual (expected)" << std::endl;
    for (unsigned int i = 0; i < N; i++) {
        std::cout << solution[i] << "  (" << reference[i] << ")" << std::endl;
        if (std::abs(solution[i] - reference[i]) > max_diff) {
            max_diff = std::abs(solution[i] - reference[i]);
            index = i;
        }
    }
    std::cout << "Max difference = " << max_diff << " at index " << index << std::endl;

    // release the device
    retval = fdbslv.releaseDevice();

    int ret = 0; // assume pass
    if (max_diff > 0.003) {
        std::cout << "FAIL" << std::endl;
        ret = 1;
    } else {
        std::cout << "PASS" << std::endl;
    }

    return ret;
}
