/*
 * (c) Copyright 2019 Xilinx, Inc. All rights reserved.
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
 *
 */

// fake dut in case run in HLS
int dut(int i) {
    return i;
}

#ifndef __SYNTHESIS__

#include <iostream>
#include <string>
#include <vector>

#include "xf_utils_sw/arg_parser.hpp"

int main(int argc, const char* argv[]) {
    for (int i = 0; i < argc; ++i) {
        std::cout << "DEBUG: argv[" << i << "]='" << argv[i] << "'\n";
    }
    xf::common::utils_sw::ArgParser parser(argc, argv);

    parser.addFlag("", "--demo", "Run in demo mode, other options will be ignored");
    parser.addFlag("-c", "--check", "Check result");
    parser.addFlag("-v", "--verbose", "Be verbose");
    parser.addOption("-x", "--xclbin", "Path to xclbin", "", true);
    parser.addOption("-i", "--index", "Device index", "0");
    parser.addOption("-q", "--quick", "Bool flag for quick method or not", "", true);

    if (parser.getAs<bool>("help")) {
        parser.showUsage();
        return 0;
    }

    if (parser.getAs<bool>("demo")) {
        std::cout << "Running demo... done!" << std::endl;
        return 0;
    }

    bool v = parser.getAs<bool>("verbose");
    std::string xclbin_path = parser.getAs<std::string>("xclbin");
    int idx = parser.getAs<int>("index");
    bool q = parser.getAs<bool>("q");

    if (v) {
        std::cout << "INFO: verbose mode on" << std::endl;
    }

    std::cout << "INFO: xclbin: " << xclbin_path << std::endl;
    std::cout << "INFO: index: " << idx << std::endl;
    std::cout << "INFO: quick: " << std::boolalpha << q << std::endl;

    // checks
    int nerror = 0;
    if (xclbin_path != "vadd.xclbin") ++nerror;
    if (q != false) ++nerror;
    if (v != true) ++nerror;
    if (idx != 2) ++nerror;

    dut(0);

    if (!nerror)
        std::cout << "PASS!" << std::endl;
    else
        std::cout << "FAIL with " << nerror << " errors!" << std::endl;
    return nerror;
}

#endif
