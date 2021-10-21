/**
* Copyright (C) 2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <unistd.h>
#include <sys/ioctl.h>
#include "xf_data_analytics/text/aml_checker.hpp"
#include "arg_parser.hpp"

using namespace xf::data_analytics::text;

const std::string KO[2] = {"OK", "KO"};
const std::string DES[2] = {"", "Description"};
const std::string SWC[2] = {"", "SwiftCode"};
const std::string ENT[2] = {"", "Entity"};
const std::string SND[2] = {"", "Sender"};
std::string print_result(SwiftMT103CheckResult& r) {
    std::string res = "";
    res += std::to_string(r.id);
    res += ",";
    res += KO[r.isMatch];
    res += ",";
    res += DES[r.matchField[0]];
    if (r.matchField[0]) res += ":";
    res += SWC[r.matchField[1]];
    if (r.matchField[0] || r.matchField[1]) res += ":";
    res += ENT[r.matchField[4] || r.matchField[5]];
    if (r.matchField[0] || r.matchField[1] || r.matchField[4] || r.matchField[5]) res += ":";
    res += SND[r.matchField[2] || r.matchField[3]];
    res += ",";
    res += std::to_string(r.timeTaken);

    return res;
}

int main(int argc, const char* argv[]) {
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // cmd arg parser
    xf::common::utils_sw::ArgParser parser(argc, argv);

    parser.addOption("-i", "--in-dir", "Folder of watch list CSV files", "");
    parser.addOption("-d", "--device-id", "Set Device id by user, if not set, choose the first available one", "-1");
    parser.addOption("-m", "--mode", "Work mode, 0 for FPGA only, 1 for CPU only, 2 for both and comparing results",
                     "0");
    parser.addOption("", "--xclbin", "xclbin path", "");
    parser.addFlag("", "--accept-EULA", "Skip printing license");
    parser.addFlag("", "--demo", "Demo");

    if (parser.getAs<bool>("help")) {
        parser.showUsage();
        return 0;
    }

    bool demo = parser.getAs<bool>("demo");
    bool skip = parser.getAs<bool>("accept-EULA");
    std::string in_dir = parser.getAs<std::string>("in-dir");
    std::string xclbin_path = parser.getAs<std::string>("xclbin");
    int work_mode = parser.getAs<int>("mode");
    int device_id_by_user = parser.getAs<int>("device-id");
    // when no user setting
    int device_id = 0;
    bool user_setting = false;
    if (device_id_by_user != -1) {
        user_setting = true;
        device_id = device_id_by_user;
    }

    if (demo) {
        in_dir = "/home/nimbix/demo_data/";
    } else {
        if (in_dir == "") {
            std::cout << "ERROR: input watch list csv file path is not set!\n";
            return -1;
        }
    }

    if (work_mode == 0)
        std::cout << "Select FPGA-only work mode\n";
    else if (work_mode == 1)
        std::cout << "Select CPU-only work mode\n";
    else if (work_mode == 2)
        std::cout << "Select both FPGA and CPU checker\n";
    else {
        std::cout << "ERROR: work mode out of range [0,2]" << std::endl;
        return -1;
    }

    // std::string out_result = "/home/nimbix/results";
    // Path for full dataset
    std::cerr << "----------------------------------------------------------------\n"
                 " NOTICE: The people.csv included in the repo has been tailored.\n"
                 " To download full deny-list for hardware test, please refer to\n"
                 " the README file in data folder.\n"
                 "----------------------------------------------------------------"
              << std::endl;

    if (!skip) {
        //// print license
        std::ifstream file("/opt/xilinx/apps/vt_data_analytis/aml/docs/license.txt");
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        std::string str;
        std::string file_contents;
        int row_counter = 0;
        while (std::getline(file, str)) {
            file_contents += str;
            file_contents.push_back('\n');
            row_counter++;
            if (row_counter == w.ws_row - 3) {
                std::cout << file_contents;
                row_counter = 0;
                file_contents = "";
                // std::cin.ignore();
                printf("\n[Press Enter Key to Continue]\n");
                std::cin.get();
            }
        }
        if (row_counter != w.ws_row - 1) std::cout << file_contents;
        ////end of printing license

        std::string acknow = "";
        while (acknow != "yes" && acknow != "no") {
            std::cout << "Please input yes/no to acknowledge the agreement. yes/no: ";
            std::cin >> acknow;
        }
        if (acknow == "no") {
            exit(1);
        }
        if (acknow == "yes") {
            setenv("XILINX_LICENCE", "pass", 1);
        }
    }

    // Add Watch List CSV Files
    std::ifstream f;
    const std::string stopKeywordFile = in_dir + "/" + "stopkeywords.csv";
    f.open(stopKeywordFile);
    if (f.good()) {
        f.close();
        f.clear();
    } else {
        std::cout << "Error: " << stopKeywordFile << " cannot be found, please check and re-run.\n\n";
        exit(1);
    }

    const std::string peopleFile = in_dir + "/" + "people.csv";
    f.open(peopleFile);
    if (f.good()) {
        f.close();
        f.clear();
    } else {
        std::cout << "Error: File " << peopleFile << " cannot be found, please check and re-run.\n\n";
        exit(1);
    }

    const std::string entityFile = in_dir + "/" + "entities.csv";
    f.open(stopKeywordFile);
    if (f.good()) {
        f.close();
        f.clear();
    } else {
        std::cout << "Error: File " << entityFile << " cannot be found, please check and re-run.\n\n";
        exit(1);
    }

    const std::string BICRefFile = in_dir + "/" + "BIC_ref_data.csv";
    f.open(BICRefFile);
    if (f.good()) {
        f.close();
        f.clear();
    } else {
        std::cout << "Error: File " << BICRefFile << " cannot be found, please check and re-run.\n\n";
        exit(1);
    }

    // Read some transactions
    const int trans_num = 100;
    std::string test_input = in_dir + "/" + "txdata.csv";
    f.open(test_input);
    if (f.good()) {
        f.close();
        f.clear();
    } else {
        std::cout << "Error: <Input transaction> File " << test_input
                  << " cannot be found, please check and re-run.\n\n";
        exit(1);
    }

    std::vector<std::vector<std::string> > list_trans(7);
    load_csv(trans_num, -1U, test_input, 10, list_trans[0]); // TransactionDescription
    load_csv(trans_num, -1U, test_input, 11, list_trans[1]); // SwiftCode1
    load_csv(trans_num, -1U, test_input, 12, list_trans[2]); // Bank1
    load_csv(trans_num, -1U, test_input, 13, list_trans[3]); // SwiftCode2
    load_csv(trans_num, -1U, test_input, 14, list_trans[4]); // Bank2
    load_csv(trans_num, -1U, test_input, 15, list_trans[5]); // NombrePersona1
    load_csv(trans_num, -1U, test_input, 18, list_trans[6]); // NombrePersona2

    std::vector<SwiftMT103> test_transaction(trans_num);
    for (int i = 0; i < trans_num; i++) {
        test_transaction[i].id = i;
        test_transaction[i].transactionDescription = list_trans[0][i];
        test_transaction[i].swiftCode1 = list_trans[1][i];
        test_transaction[i].bank1 = list_trans[2][i];
        test_transaction[i].swiftCode2 = list_trans[3][i];
        test_transaction[i].bank2 = list_trans[4][i];
        test_transaction[i].nombrePersona1 = list_trans[5][i];
        test_transaction[i].nombrePersona2 = list_trans[6][i];
    }

    std::vector<SwiftMT103CheckResult> result_set(trans_num);

    // Begin to analyze if on mode 0 or 2
    if (work_mode == 0 || work_mode == 2) {
        SwiftMT103Checker checker;
        checker.initialize(xclbin_path, stopKeywordFile, peopleFile, entityFile, BICRefFile, device_id,
                           user_setting); // card 0

        float min = std::numeric_limits<float>::max(), max = 0.0, sum = 0.0;
        for (int i = 0; i < trans_num; i++) {
            auto ts = std::chrono::high_resolution_clock::now();
            result_set[i] = checker.check(test_transaction[i]);
            auto te = std::chrono::high_resolution_clock::now();
            float timeTaken = std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count() / 1000.0f;
            result_set[i].timeTaken = timeTaken;

            if (min > timeTaken) min = timeTaken;
            if (max < timeTaken) max = timeTaken;
            sum += timeTaken;
        }

        // print the result
        std::cout << "\nTransaction Id, OK/KO, Field of match, Time taken(:ms)" << std::endl;
        for (int i = 0; i < trans_num; i++) {
            std::string s = print_result(result_set[i]);
            std::cout << s << std::endl;
        }

        std::cout << "\nFor FPGA, ";
        std::cout << trans_num << " transactions were processed.\n";

        std::cout << "Min(ms)\t\tMax(ms)\t\tAvg(ms)\n";
        std::cout << "----------------------------------------" << std::endl;
        std::cout << min << "\t\t" << max << "\t\t" << sum / trans_num << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    // check the result
    internal::SwiftMT103CPUChecker cpu_checker;
    int nerror = 0;
    if (work_mode == 1 || work_mode == 2) {
        if (work_mode == 2) std::cout << "\nStart to check...\n";
        cpu_checker.initialize(stopKeywordFile, peopleFile, entityFile, BICRefFile);

        for (int i = 0; i < trans_num; i++) {
            SwiftMT103CheckResult t = cpu_checker.check(test_transaction[i]);
            if (work_mode == 2 && t != result_set[i]) {
                std::cout << "Trans-" << i << std::endl;
                for (int j = 0; j < 6; j++) std::cout << t.matchField[j] << " ";
                std::cout << std::endl;
                for (int j = 0; j < 6; j++) std::cout << result_set[i].matchField[j] << " ";
                std::cout << std::endl;
                nerror++;
            }
        }
    }
    nerror ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
           : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);

    return nerror;
}
