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

#ifndef BENCH_HELPER_HPP
#define BENCH_HELPER_HPP

#include <string>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <sstream>
#include <assert.h>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimePointType;

using namespace std;

void showTimeData(string p_Task, TimePointType& t1, TimePointType& t2, double* p_TimeMsOut = 0) {
    t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> l_durationSec = t2 - t1;
    double l_timeMs = l_durationSec.count() * 1e3;
    if (p_TimeMsOut) {
        *p_TimeMsOut = l_timeMs;
    }
    cout << p_Task << "  " << fixed << setprecision(6) << l_timeMs << " msec\n";
}

float getBoardFreqMHz(string xclbin) {
    string l_freqCmd = "xclbinutil --info --input " + xclbin;
    float l_freq = -1;
    char l_lineBuf[256];
    shared_ptr<FILE> l_pipe(popen(l_freqCmd.c_str(), "r"), pclose);
    // if (!l_pipe) throw std::runtime_error("ERROR: popen(" + l_freqCmd + ") failed");
    if (!l_pipe) cout << ("ERROR: popen(" + l_freqCmd + ") failed");
    bool l_nextLine_isFreq = false;
    while (l_pipe && fgets(l_lineBuf, 256, l_pipe.get())) {
        std::string l_line(l_lineBuf);
        // std::cout << "DEBUG: read line " << l_line << std::endl;
        if (l_nextLine_isFreq) {
            std::string l_prefix, l_val, l_mhz;
            std::stringstream l_ss(l_line);
            l_ss >> l_prefix >> l_val >> l_mhz;
            l_freq = std::stof(l_val);
            assert(l_mhz == "MHz");
            break;
        } else if (l_line.find("Type:      DATA") != std::string::npos) {
            l_nextLine_isFreq = true;
        }
    }
    if (l_freq == -1) {
        // if xbutil does not work, user could put the vitis achieved kernel frequcy here
        l_freq = 250;
        std::cout << "INFO: Failed to get board frequency by xclbinutil. This is normal for cpu and hw emulation, "
                     "using 250 MHz for reporting.\n";
    }
    return (l_freq);
}

bool readConfigDict(string p_configFile, unordered_map<string, string>* p_configDict) {
    unordered_map<string, string> l_configDict;
    ifstream l_configInfo(p_configFile);
    bool l_good = l_configInfo.good();
    if (!l_good) {
        return false;
    }
    if (l_configInfo.is_open()) {
        string line;
        string key;
        string value;
        string equalSign = "=";
        while (getline(l_configInfo, line)) {
            int index = line.find(equalSign);
            if (index == 0) continue;
            key = line.substr(0, index);
            value = line.substr(index + 1);
            l_configDict[key] = value;
        }
    }

    l_configInfo.close();

    *p_configDict = l_configDict;
    return true;
}

#endif
