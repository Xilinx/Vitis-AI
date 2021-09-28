/*
 * Copyright 2021 Xilinx, Inc.
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

//#include "xcl2.hpp"
#include "utils.hpp"
#include "dup_match.hpp"

int main(int argc, const char* argv[]) {
    std::cout << "\n---------------------Duplicate Record Matching Flow-----------------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // cmd parser
    ArgParser parser(argc, argv);

    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return -1;
    }
    if (!exist_file(xclbin_path)) {
        std::cout << "ERROR: xclbin file is not exist\n";
        return -1;
    }

    std::string in_file;
    if (!parser.getCmdOption("-in", in_file)) {
        std::cout << "ERROR: input file path is not set!\n";
        return -1;
    }
    if (!exist_file(in_file)) {
        std::cout << "ERROR: input file is not exist\n";
        return -1;
    }

    std::string golden_file;
    bool en_check = true;
    if (!parser.getCmdOption("-golden", golden_file)) {
        en_check = false;
    } else if (!exist_file(golden_file)) {
        std::cout << "ERROR: golden file is not exist\n";
        return -1;
    }

    int nerr = 0;
    struct timeval tk1, tk2;
    gettimeofday(&tk1, 0);

    const std::vector<std::string> field_name{"Site name", "Address", "Zip", "Phone"};
    DupMatch dup_match = DupMatch(in_file, golden_file, field_name, xclbin_path);
    std::vector<std::pair<uint32_t, double> > cluster_membership;
    dup_match.run(cluster_membership);
    gettimeofday(&tk2, 0);
    std::cout << "Execution time " << tvdiff(&tk1, &tk2) / 1000.0 << "s" << std::endl;

    if (en_check) {
        std::ifstream f(golden_file, std::ios::in);
        std::string line_str;
        uint32_t ii = 0;
        while (getline(f, line_str)) {
            if (cluster_membership[ii++].first != std::stoi(line_str)) nerr++;
        }
        f.close();
    }
    // for (int i = 0; i < cluster_membership.size(); i++) {
    //    std::cout << cluster_membership[i].first << " " << std::setprecision(10) << cluster_membership[i].second
    //              << std::endl;
    //}

    nerr ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
         : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return nerr;
}
