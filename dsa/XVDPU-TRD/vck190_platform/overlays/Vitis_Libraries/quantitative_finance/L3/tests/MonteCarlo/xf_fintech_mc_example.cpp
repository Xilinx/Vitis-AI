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

#include "xf_fintech_mc_example.hpp"

#include "xf_fintech_api.hpp"

using namespace xf::fintech;

double varianceMultiplier;

int main(int argc, char** argv) {
    int retval = 0; // assume pass

    std::string path = std::string(argv[1]);
    MCEuropean mcEuropean(path);

    std::string device = TOSTRING(DEVICE_PART);
    if (argc == 3) {
        device = std::string(argv[2]);
    }

    std::vector<Device*> deviceList;
    Device* pChosenDevice;

    // Get a list of U250s available on the system (just because our current
    // bitstreams are built for U250s)
    deviceList = DeviceManager::getDeviceList(device);

    if (deviceList.size() == 0) {
        printf("[XLNX] No matching devices found\n");
        exit(0);
    }

    printf("[XLNX] Found %zu matching devices\n", deviceList.size());

    // we'll just pick the first device in the...
    pChosenDevice = deviceList[0];

    if (!retval) {
        // turn off trace output...turn it on here if you want extra debug output...
        Trace::setEnabled(false);
    }

    if (!retval) {
        retval = MCDemoRunEuropeanSingle(pChosenDevice, &mcEuropean);
    }

    if (!retval) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
    return retval;
}
