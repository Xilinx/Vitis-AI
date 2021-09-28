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
#include <getopt.h>

#include "xf_fintech_api.hpp"

#define TEST_UN (4)
#define TEST_MAX_TENORS (10)
#define TEST_DT float

/*
 * Pricer defaults
 */
#define DEF_PATHS (1000)
#define DEF_BETA (0.2f)
#define DEF_NOTIONAL (1e7)
#define DEF_CAPRATE (0.011f)
#define DEF_RFX (0.0015f)
#define DEF_RFY (0.0015f)
#define DEF_RFALPHA (0.0005f)
#define DEF_RCSPREAD (0.005f)
#define DEF_RCKAPPA0 (0.001f)

using namespace std;
using namespace xf::fintech;

template <typename T>
using al_vec = std::vector<T, aligned_allocator<T> >;

static al_vec<unsigned> getFpgaSeeds() {
    al_vec<unsigned> seeds(TEST_UN);
    for (unsigned i = 0; i < TEST_UN; i++) {
        seeds[i] = 42 + i; // Custom seeds generation goes here
    }
    return seeds;
}

static al_vec<TEST_DT> readVectors(const string& fileName) {
    ifstream in;
    in.open(fileName.c_str());
    if (!in.is_open()) {
        cerr << "Failed to open file " << fileName << endl;
        exit(-1);
    }

    string line, word;
    al_vec<TEST_DT> data;
    getline(in, line);
    istringstream iss(line);
    while (getline(iss, word, ',')) {
        data.push_back(static_cast<TEST_DT>(atof(word.c_str())));
    }
    in.close();
    return data;
}

static void printHelp(const char* pgrmName) {
    cout << "Usage: " << string(pgrmName);
    cout << " [-h] -l rates_in -v caplet_volas [-b beta] [-p paths] [-n notional] [-c caprate] [-x rfX] [-y rfY] "
            "[-a rfAlpha] [-s spread] [-k kappa0]"
         << endl;
    cout << "Runs examples of LIBOR Market Model Cap, Ratchet Floater and Ratchet Cap"
            " option pricings using L3 libraries"
         << endl;
    cout << "\t-h --help            Shows this help message." << endl;
    cout << "\t-l --libor_rates     csv file with current LIBOR rates vector. The number of elements must be <= "
         << TEST_MAX_TENORS << endl;
    cout << "\t-v --volatilities    csv file with implied caplet volatilities. Must have one fewer element than the "
            "rates file."
         << endl;
    cout << "\t-b --rho_beta        Beta argument for parametric correlation generator. Must be between 0.0 and 1.0. "
            "Defaults to "
         << DEF_BETA << endl;
    cout << "\t-p --paths           Number of MonteCarlo paths to generate. Defaults to " << DEF_PATHS << endl;
    cout << "\t-n --notional        Notional parameter for all the pricers. Defaults to " << DEF_NOTIONAL << endl;
    cout << "\t-c --caprate         Caprate parameter (K) for Cap pricing. Defaults to " << DEF_CAPRATE << endl;
    cout << "\t-x --rf_x            X parameter for Ratchet Floater pricing. Defaults to " << DEF_RFX << endl;
    cout << "\t-y --rf_y            Y parameter for Ratchet Floater pricing. Defaults to " << DEF_RFY << endl;
    cout << "\t-a --rf_alpha        Alpha parameter for Ratchet Floater pricing. Defaults to " << DEF_RFALPHA << endl;
    cout << "\t-s --rc_spread       Spread parameter for Ratchet Cap pricing. Defaults to " << DEF_RCSPREAD << endl;
    cout << "\t-k --rc_kappa0       Kappa0 parameter for Ratchet Cap pricing. Defaults to " << DEF_RCKAPPA0 << endl;
}

int main(int argc, char* argv[]) {
    const struct option ops[] = {{"help", no_argument, 0, 'h'},
                                 {"libor_rates", required_argument, 0, 'l'},
                                 {"volatilities", required_argument, 0, 'v'},
                                 {"rho_beta", required_argument, 0, 'b'},
                                 {"paths", required_argument, 0, 'p'},
                                 {"notional", required_argument, 0, 'n'},
                                 {"caprate", required_argument, 0, 'c'},
                                 {"rf_x", required_argument, 0, 'x'},
                                 {"rf_y", required_argument, 0, 'y'},
                                 {"rf_alpha", required_argument, 0, 'a'},
                                 {"rc_spread", required_argument, 0, 's'},
                                 {"rc_kappa0", required_argument, 0, 'k'},
                                 {"xclbin", required_argument, 0, 'f'},
                                 {"device", required_argument, 0, 'd'},
                                 {0, 0, 0, 0}};

    char arg = 0;
    int retval = XLNX_OK;
    std::string liborFile, volaFile, path;
    TEST_DT rhoBeta = DEF_BETA;
    unsigned paths = DEF_PATHS;
    TEST_DT notional = DEF_NOTIONAL;
    TEST_DT caprate = DEF_CAPRATE;
    TEST_DT rfX = DEF_RFX;
    TEST_DT rfY = DEF_RFY;
    TEST_DT rfAlpha = DEF_RFALPHA;
    TEST_DT rcSpread = DEF_RCSPREAD;
    TEST_DT rcKappa0 = DEF_RCKAPPA0;

    std::string device = TOSTRING(DEVICE_PART);
    while ((arg = getopt_long(argc, argv, "hl:v:b:p:n:c:x:y:a:s:k:f:d:", ops, NULL)) != -1) {
        switch (arg) {
            case 'h':
                printHelp(argv[0]);
                return 1;
            case 'l':
                liborFile = string(optarg);
                break;
            case 'v':
                volaFile = string(optarg);
                break;
            case 'b':
                rhoBeta = atof(optarg);
                break;
            case 'p':
                paths = atoi(optarg);
                break;
            case 'n':
                notional = atof(optarg);
                break;
            case 'c':
                caprate = atof(optarg);
                break;
            case 'x':
                rfX = atof(optarg);
                break;
            case 'y':
                rfY = atof(optarg);
                break;
            case 'a':
                rfAlpha = atof(optarg);
                break;
            case 's':
                rcSpread = atof(optarg);
                break;
            case 'k':
                rcKappa0 = atof(optarg);
                break;
            case 'f':
                path = string(optarg);
                break;
            case 'd':
                device = string(optarg);
                break;
            case '?':
            default:
                cerr << "Error parsing arguments..." << endl;
                printHelp(argv[0]);
                return -1;
        }
    }

    if (liborFile.empty()) {
        cerr << "Missing mandatory argument 'libor_rates'" << endl;
        return -1;
    }
    if (volaFile.empty()) {
        cerr << "Missing mandatory argument 'volatilities'" << endl;
    }

    cout << "========================================\n"
         << "    LIBOR Market Model (BGM) Example    \n"
         << "========================================\n";
    cout << "\tlibor_rates  = " << liborFile << endl;
    cout << "\tvolatilities = " << volaFile << endl;
    cout << "\trho_beta     = " << rhoBeta << endl;
    cout << "\tpaths        = " << paths << endl;
    cout << "\tnotional     = " << notional << endl;
    cout << "\tcaprate      = " << caprate << endl;
    cout << "\trf_x         = " << rfX << endl;
    cout << "\trf_y         = " << rfY << endl;
    cout << "\trf_alpha     = " << rfAlpha << endl;
    cout << "\trc_spread    = " << rcSpread << endl;
    cout << "\trc_kappa0    = " << rcKappa0 << endl;
    cout << "========================================" << endl;

    int ret = 1; // assume fail
    try {
        al_vec<TEST_DT> presentRates = readVectors(liborFile);
        if (presentRates.size() <= 2) {
            cerr << "Invalid number of tenors provided. Must be larger than 2" << endl;
            return -1;
        }
        if (presentRates.size() > TEST_MAX_TENORS) {
            cerr << "Invalid number of tenors provided. Must be <= than maximum synthetisable tenors ("
                 << TEST_MAX_TENORS << ")" << endl;
        }
        al_vec<TEST_DT> capletVolas = readVectors(volaFile);
        if (capletVolas.size() != presentRates.size() - 1) {
            cerr << "Invalid number of caplet volatilities. Must be equal to the number of LIBOR rates - 1" << endl;
            return -1;
        }
        al_vec<unsigned> seeds = getFpgaSeeds();
        const unsigned noTenors = presentRates.size();

        vector<Device*> deviceList;
        Device* pChosenDevice;
        LMM lmm(path);

        deviceList = DeviceManager::getDeviceList(device);
        if (deviceList.size() == 0) {
            cerr << "No matching devices found" << endl;
            return -1;
        }

        // Pick the first device on the list
        pChosenDevice = deviceList[0];
        float outPrice;

        if ((retval = lmm.claimDeviceCap(pChosenDevice)) == XLNX_OK) {
            cout << "Running Cap pricing..." << endl;
            // Run Cap pricing
            retval = lmm.runCap(noTenors, paths, presentRates.data(), rhoBeta, capletVolas.data(), notional, caprate,
                                seeds.data(), &outPrice);
        }
        if (retval == XLNX_OK) {
            cout << "Cap price = " << outPrice << endl;
            cout << "Duration = " << lmm.getLastRunTime() << " us" << endl;
            if (std::abs(outPrice - 156903) <= 1500) {
                ret = 0;
            }
        } else {
            cerr << "ERROR: Failed to run LMM!" << endl;
        }
        if (retval == XLNX_OK) {
            retval = lmm.releaseDevice();
        }
    } catch (const exception& ex) {
        cerr << ex.what() << endl;
    } catch (const string& ex) {
        cerr << ex << endl;
    } catch (...) {
        cerr << "Unhandled exception caught!" << endl;
    }

    if (ret != 0) {
        std::cout << "Fail with " << ret << " errors." << std::endl;
    } else {
        std::cout << "Pass validation." << std::endl;
    }

    return ret;
}
