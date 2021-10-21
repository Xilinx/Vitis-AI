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

#include <stdio.h>
#include <string.h>

#include <chrono>
#include <vector>

#include "xf_fintech_api.hpp"

using namespace xf::fintech;

static float tolerance = 0.001;

int main(int argc, char** argv) {
    std::string path = std::string(argv[1]);
    // population mcmc fintech model
    PopMCMC popmcmc(path);

    std::string device = TOSTRING(DEVICE_PART);
    if (argc == 3) {
        device = std::string(argv[2]);
    }

    int retval = XLNX_OK;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    std::vector<Device*> deviceList;
    Device* pChosenDevice;

    deviceList = DeviceManager::getDeviceList(device);

    if (deviceList.size() == 0) {
        printf("No matching devices found\n");
        exit(0);
    }

    printf("Found %zu matching devices\n", deviceList.size());

    // we'll just pick the first device in the
    pChosenDevice = deviceList[0];

    if (retval == XLNX_OK) {
        // turn off trace output...turn it on here if you want extra debug output
        Trace::setEnabled(true);
    }

    printf("\n\n\n");
    printf("[XF_FINTECH] PopMCMC trying to claim device...\n");

    start = std::chrono::high_resolution_clock::now();

    retval = popmcmc.claimDevice(pChosenDevice);

    end = std::chrono::high_resolution_clock::now();

    if (retval == XLNX_OK) {
        printf("[XF_FINTECH] Device setup time = %lld microseconds\n",
               (long long int)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    } else {
        printf("[XF_FINTECH] Failed to claim device - error = %d\n", retval);
    }

    static const int maxSamples = 5000;
    double sigma = 0.4;
    double outputData[maxSamples];
    long long int lastRuntime = 0;
    FILE* fp;

    std::string mode_emu = "hw_emu";
    if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
        mode_emu = std::getenv("XCL_EMULATION_MODE");
    }
    std::cout << "Running in " << mode_emu << " mode" << std::endl;

    int numSamples = 5000;
    int numBurnInSamples = 500;
    double checkValue = 0.847094;
    if (mode_emu == "hw_emu") {
        numSamples = 2;
        numBurnInSamples = 0;
        checkValue = 0.661728;
    }

    retval = popmcmc.run(numSamples, numBurnInSamples, sigma, outputData);
    lastRuntime = popmcmc.getLastRunTime();

    fp = fopen("pop_mcmc_output.csv", "wb");
    if (retval == XLNX_OK) {
        for (int i = 0; i < (numSamples - numBurnInSamples); i++) {
            printf("%f\n", outputData[i]);

            // write to file
            fprintf(fp, "%lf\n", outputData[i]);
            if (i == ((numSamples - numBurnInSamples) - 1)) {
                fprintf(fp, "%lf", outputData[i]);
            }
        }
        printf("[XF_FINTECH] ExecutionTime = %lld microseconds\n", lastRuntime);
    }

    fclose(fp);

    printf("[XF_FINTECH] PopMCMC releasing device...\n");
    retval = popmcmc.releaseDevice();

    // quick fix to get pass/fail criteria
    int ret = 0; // assume pass
    if (std::abs(outputData[(numSamples - numBurnInSamples) - 1] - checkValue) > tolerance) {
        printf("Value = %f\n", outputData[(numSamples - numBurnInSamples) - 1]);
        printf("FAIL\n");
        ret = 1;
    } else {
        printf("PASS\n");
    }

    return ret;
}
