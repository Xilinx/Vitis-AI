// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// Example server implementation to use for unit testing and benchmarking
// purposes

#include <signal.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <CL/cl_ext_xilinx.h>
#include <CL/cl.h>

//#include <gflags/gflags.h>
#include "db_intpair_sort_1g.hpp"
#include "xf_utils_sw/arg_parser.hpp"
#include "x_utils.hpp"
inline int tvdiff(const timeval& tv0, const timeval& tv1) {
    return (tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec);
}

inline int tvdiff(const timeval& tv0, const timeval& tv1, const char* info) {
    int exec_us = tvdiff(tv0, tv1);
    printf("%s: %d.%03d msec\n", info, (exec_us / 1000), (exec_us % 1000));
    return exec_us;
}
bool check_sorted(uint64_t* out, int len, int order = 1) {
    int prev = out[0];
    for (int i = 0; i < len; i++) {
        ap_uint<64> data_t = out[i];
        int key = data_t.range(31, 0);
        int data = data_t.range(63, 32);
        if (order == 1) {
            if (key < prev || key != (data - 1)) return false;
        } else {
            if (key > prev || key != (data - 1)) return false;
        }
        prev = key;
    }
    // std::cout << "check kv success!" << std::endl;
    return true;
}

xf::database::intpair_sort::sortAPI sortapi;

int single_mode_fun(std::string input, std::string ouput, int FLAGS_order) {
    using namespace xf::database::intpair_sort;
    std::cout << "Single mode: input data loading...!" << std::endl;
    uint64_t* user_in;
    int size = x_utils::load_dat<uint64_t>(user_in, input);
    uint64_t* user_out = x_utils::aligned_alloc<uint64_t>(size);
    if (size == 0) {
        std::cout << "Error, data size = 0" << std::endl;
        return 1;
    }
    std::cout << "Single mode: input data load done!" << std::endl;
    timeval t_1, t_2;
    gettimeofday(&t_1, 0);

    std::future<ErrCode> err = sortapi.sort(user_in, user_out, size, FLAGS_order);
    err.wait();
    gettimeofday(&t_2, 0);
    std::cout << "Single mode: data sort done!" << std::endl;
    tvdiff(t_1, t_2, "End-to-End time");

    x_utils::gen_dat<uint64_t>(user_out, ouput, size);
    std::cout << "Single mode: output data save done!" << std::endl;
    std::cout << "Single mode: start validating result..." << std::endl;
    bool chk = check_sorted(user_out, size, FLAGS_order);
    if (!chk)
        std::cout << "Single mode, Status: Error" << std::endl;
    else
        std::cout << "Single mode, Status: Pass" << std::endl;

    return 0;
}
int batch_mode_fun(std::string input_list, std::string output_list, int FLAGS_order) {
    using namespace xf::database::intpair_sort;
    std::cout << "Batch mode: input data loading...!" << std::endl;
    std::vector<uint64_t*> user_in;
    std::vector<uint64_t*> user_out;
    std::vector<int32_t> sizes;
    std::vector<std::string> output_file_list;
    std::ifstream file(input_list);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            uint64_t* user_in_tmp;
            int size = x_utils::load_dat<uint64_t>(user_in_tmp, line);
            uint64_t* user_out_tmp = x_utils::aligned_alloc<uint64_t>(size);
            user_in.push_back(user_in_tmp);
            user_out.push_back(user_out_tmp);
            sizes.push_back(size);
            std::cout << "Batch mode: " << line << " load done!" << std::endl;
        }
        file.close();
    }
    std::ifstream file_o(output_list);
    if (file_o.is_open()) {
        std::string line;
        while (std::getline(file_o, line)) {
            output_file_list.push_back(line);
        }
        file_o.close();
    }
    int batch_num = user_in.size();
    std::cout << "Batch mode: input data load done, total " << batch_num << " files." << std::endl;

    timeval t_1, t_2;
    gettimeofday(&t_1, 0);
    std::vector<std::future<ErrCode> > futures;
    for (int i = 0; i < batch_num; i++) {
        futures.push_back(sortapi.sort(user_in[i], user_out[i], sizes[i], FLAGS_order));
    }
    for (int i = 0; i < batch_num; i++) {
        futures[i].wait();
        futures[i].get();
    }
    gettimeofday(&t_2, 0);
    std::cout << "Batch mode: data sort done!" << std::endl;
    tvdiff(t_1, t_2, "End-to-End time");

    for (int i = 0; i < batch_num; i++) {
        x_utils::gen_dat<uint64_t>(user_out[i], output_file_list[i], sizes[i]);
        std::cout << "Batch mode: " << output_file_list[i] << " saved, start validate..." << std::endl;
        bool chk = check_sorted(user_out[i], sizes[i], FLAGS_order);
        if (!chk)
            std::cout << "Batch mode, " << output_file_list[i] << " Status: Error." << std::endl;
        else
            std::cout << "Batch mode, " << output_file_list[i] << " Status: Pass." << std::endl;
    }
    std::cout << "Batch mode: output files saved, total: " << batch_num << " files." << std::endl;
    return 0;
}

int main(int argc, const char* argv[]) {
    xf::common::utils_sw::ArgParser parser(argc, argv);
    parser.addOption("-d", "--device-id", "Set Device id by user, if not set, choose the first available one", "-1");
    parser.addOption("-i", "--in", "Single run, input dat", "");
    parser.addOption("-o", "--out", "Single run, output dat", "");
    parser.addOption("-I", "--files-in", "Batch run, input list txt", "");
    parser.addOption("-O", "--files-out", "Batch run, output list txt", "");
    parser.addFlag("", "--accept-EULA", "Skip printing license");
    parser.addFlag("", "--demo", "Demo");
    parser.addOption("-a", "--asc", "Ascending order", "1");
    parser.addOption("", "--xclbin", "XCLBIN path", "");
    if (parser.getAs<bool>("help")) {
        parser.showUsage();
        return 0;
    }
    bool FLAGS_y = parser.getAs<bool>("accept-EULA");
    bool FLAGS_demo = parser.getAs<bool>("demo");
    int FLAGS_order = parser.getAs<int>("asc");
    std::string FLAGS_in = parser.getAs<std::string>("in");
    std::string FLAGS_out = parser.getAs<std::string>("out");
    std::string FLAGS_files_in = parser.getAs<std::string>("files-in");
    std::string FLAGS_files_out = parser.getAs<std::string>("files-out");
    std::string FLAGS_xclbin_path = parser.getAs<std::string>("xclbin");
    int device_id_by_user = parser.getAs<int>("device-id");
    int device_id = 0;
    bool user_setting = false;
    if (device_id_by_user >= 0) {
        user_setting = true;
        device_id = device_id_by_user;
    }

    if (FLAGS_demo) {
        sortapi.init(FLAGS_xclbin_path, device_id, user_setting);
        std::cout << "Platform init done!" << std::endl;

        std::cout << std::endl;
        std::cout << "Demo1: Single Mode Args: --in /home/nimbix/demo_data/input_1M_0.dat --out "
                     "/home/nimbix/demo_data/single_mode_out.dat"
                  << std::endl;
        int err =
            single_mode_fun("/home/nimbix/demo_data/input_1M_0.dat", "/home/nimbix/demo_data/1M_out0.dat", FLAGS_order);
        if (err) return err;

        std::cout << std::endl;
        std::cout << "Demo2: Batch Mode Args: --files-in /home/nimbix/demo_data/input.txt --files-out "
                     "/home/nimbix/demo_data/output.txt"
                  << std::endl;
        err = batch_mode_fun("/home/nimbix/demo_data/input.txt", "/home/nimbix/demo_data/output.txt", FLAGS_order);
        return err;
    }
    bool single_mode = (FLAGS_in != "" && FLAGS_out != "");
    bool batch_mode = (FLAGS_files_in != "" && FLAGS_files_out != "");
    if (single_mode && batch_mode) {
        std::cout << "please select one mode!" << std::endl;
        return 1;
    }

    if ((!single_mode) && (!batch_mode)) {
        std::cout << "wrong mode!" << std::endl;
        parser.showUsage();
        return 1;
    }

    if (!FLAGS_y) {
        //// print license
        std::ifstream file("/opt/xilinx/apps/vt_database/sort/docs/license.txt");
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

    if (single_mode) {
        sortapi.init(FLAGS_xclbin_path, device_id, user_setting);
        std::cout << "Platform init done!" << std::endl;
        int err = single_mode_fun(FLAGS_in, FLAGS_out, FLAGS_order);
        return err;
    } else if (batch_mode) {
        sortapi.init(FLAGS_xclbin_path, device_id, user_setting);
        std::cout << "Platform init done!" << std::endl;
        int err = batch_mode_fun(FLAGS_files_in, FLAGS_files_out, FLAGS_order);
        return err;
    }

    return 0;
}
