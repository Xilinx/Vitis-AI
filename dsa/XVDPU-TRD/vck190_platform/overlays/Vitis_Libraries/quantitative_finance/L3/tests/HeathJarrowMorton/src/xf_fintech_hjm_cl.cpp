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

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <exception>
#include <getopt.h>

#include "xf_fintech_api.hpp"

#define DEF_PATHS 100
#define DEF_YEARS 10.0

#define TEST_UN (4)
#define TEST_N (3)
#define TEST_DT double

using namespace std;
using namespace xf::fintech;

static float tolerance = 0.02;

template <typename T>
using al_vec = std::vector<T, aligned_allocator<T> >;

static al_vec<unsigned> getFpgaSeeds() {
    al_vec<unsigned> seeds(TEST_UN * TEST_N);
    for (unsigned i = 0; i < TEST_UN * TEST_N; i++) {
        seeds[i] = 42 + i; // Custom seeds generation goes here
    }
    return seeds;
}

static size_t readCsv(const std::string& name, al_vec<TEST_DT>& data) {
    std::ifstream in;
    in.open(name.c_str());
    if (!in.is_open()) {
        std::cerr << "Failed to open file " << name << std::endl;
        exit(-1);
    }

    std::string line;
    size_t rows = 0;
    while (std::getline(in, line)) {
        rows++;
        std::istringstream iss(line);
        std::string word;
        while (std::getline(iss, word, ',')) {
            data.push_back(static_cast<TEST_DT>(atof(word.c_str())));
        }
    }
    in.close();
    return rows;
}

static void printHelp(const char* pgrmName) {
    cout << "Usage: " << string(pgrmName) << " [-h] -d data_file -p no_paths -s sim_years -m zcb_maturity" << endl;
    cout << "Runs example of ZeroCouponBond pricing with the Heath-Jarrow-Morton framework using L3 libraries" << endl;
    cout << "\t-d --data_in     File pointing to historical csv data"
         << "\t-p --num_paths   Number of MonteCarlo paths to generate (Defaults to " << DEF_PATHS << ")"
         << "\t-s --sim_years   Number of years to simulate in the future per HJM path (Defaults to " << DEF_YEARS
         << "\t-m --zcb_mat     Maturity, in years, of ZCB to be priced (Defaults to " << DEF_YEARS << endl;
}

int main(int argc, char* argv[]) {
    const struct option ops[] = {{"help", no_argument, 0, 'h'},
                                 {"data_in", required_argument, 0, 'd'},
                                 {"num_paths", required_argument, 0, 'p'},
                                 {"sim_years", required_argument, 0, 's'},
                                 {"zcb_mat", required_argument, 0, 'm'},
                                 {"xclbin", required_argument, 0, 'x'},
                                 {"device", required_argument, 0, 'f'},
                                 {0, 0, 0, 0}};

    char arg = 0;
    std::string dataLoc;
    float simYears = DEF_YEARS;
    float zcbMaturity = DEF_YEARS;
    unsigned noPaths = DEF_PATHS;
    TEST_DT resZcb;
    std::string path;
    std::string device = TOSTRING(DEVICE_PART);

    while ((arg = getopt_long(argc, argv, "hd:p:s:m:x:f:", ops, NULL)) != -1) {
        switch (arg) {
            case 'd':
                dataLoc = string(optarg);
                break;
            case 'p':
                noPaths = atoi(optarg);
                break;
            case 's':
                simYears = atof(optarg);
                break;
            case 'm':
                zcbMaturity = atof(optarg);
                break;
            case 'x':
                path = std::string(optarg);
                break;
            case 'f':
                device = std::string(optarg);
                break;
            case 'h':
                printHelp(argv[0]);
                return 0;
            case '?':
            default:
                cerr << "Error parsing arguments..." << endl;
                printHelp(argv[0]);
                return -1;
        }
    }

    cout << "===================================\n"
         << "    Heath-Jarrow-Morton Example    \n"
         << "===================================\n";
    cout << "\tdata_in = " << dataLoc << endl
         << "\tnum_paths = " << noPaths << endl
         << "\tsim_years = " << simYears << endl
         << "\tzcb_mat   = " << zcbMaturity << endl;

    int ret = 0; // assume pass
    try {
        al_vec<TEST_DT> dataIn;
        al_vec<unsigned> seeds = getFpgaSeeds();
        unsigned curves;
        curves = readCsv(dataLoc, dataIn);
        const unsigned tenors = dataIn.size() / curves;

        cout << "\ttenors    = " << tenors << endl << "\tno_curves = " << curves << endl;
        cout << "\tseeds     = [";
        for (unsigned i = 0; i < TEST_N * TEST_UN; i++) {
            cout << seeds[i];
            if (i != (TEST_N * TEST_UN - 1)) {
                cout << ", ";
            } else {
                cout << "]" << endl;
            }
        }

        int retval = XLNX_OK;
        std::vector<Device*> deviceList;
        Device* pChosenDevice;

        HJM hjm(path);

        deviceList = DeviceManager::getDeviceList(device);

        if (deviceList.size() == 0) {
            cerr << "No matching devices found" << endl;
            return 0;
        }

        // Pick the first device on the list
        pChosenDevice = deviceList[0];

        retval = hjm.claimDevice(pChosenDevice);

        if (retval == XLNX_OK) {
            // Run HJM framework
            retval = hjm.run(dataIn.data(), tenors, curves, noPaths, simYears, zcbMaturity, seeds.data(), &resZcb);
        }

        if (retval == XLNX_OK) {
            cout << "Duration: " << hjm.getLastRunTime() << " us" << endl;
            cout << "ZCB Price: " << resZcb << endl;
            // quick fix to get pass fail criteria
            if (abs(resZcb - 0.63515) > tolerance) {
                cout << "FAIL" << endl;
                ret = 1;
            }
        } else {
            cerr << "ERROR: Failed to run HJM!" << endl;
        }

        if (retval == XLNX_OK) {
            retval = hjm.releaseDevice();
        }
    } catch (const exception& ex) {
        cerr << ex.what() << endl;
    } catch (const string& ex) {
        cerr << ex << endl;
    } catch (...) {
        cerr << "Unhandled exception caught!" << endl;
    }

    if (!ret) {
        cout << "PASS" << endl;
    }

    return ret;
}
