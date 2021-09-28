/*
 * (c) Copyright 2019-2021 Xilinx, Inc. All rights reserved.
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
#include "zlib.hpp"
#include <fstream>
#include <vector>
#include "cmdlineparser.h"
#include <sys/wait.h>
#include <unistd.h>

// Default values
constexpr auto M_PROC = 1;
constexpr auto NUM_ITER = 10;

using namespace xf::compression;

// Bandwidth measurement API
void xil_compress_bandwidth(
    const std::string& single_bin, uint8_t* in, uint8_t* out, uint32_t input_size, uint8_t device_id, uint8_t max_cr) {
    uint32_t enbytes = 0;
    uint32_t num_iter = NUM_ITER;
    xfZlib xlz(single_bin, max_cr, COMP_ONLY, device_id);
    ERROR_STATUS(xlz.error_code());
    std::chrono::duration<double, std::nano> compress_API_time_ns_1(0);
    std::chrono::duration<double, std::milli> compress_API_time_ms_1(0);
    auto compress_API_start = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < num_iter; i++) enbytes = xlz.compress_buffer(in, out, input_size);

    auto compress_API_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::nano>(compress_API_end - compress_API_start);
    auto duration_ms = std::chrono::duration<double, std::milli>(compress_API_end - compress_API_start);

    compress_API_time_ns_1 = duration / num_iter;
    compress_API_time_ms_1 = duration_ms / num_iter;

    float throughput_in_mbps_1 = (float)input_size * 1000 / compress_API_time_ns_1.count();
    std::cout << "Input Size: " << input_size / 1024 << "KB ";
    std::cout << "Compressed Size: " << enbytes / 1024 << "KB ";
    std::cout << "PID: " << getpid();
    std::cout << " PPID: " << getppid();
    std::cout << " API: " << std::fixed << std::setprecision(2) << throughput_in_mbps_1 << "MB/s";
    std::cout << " Time: " << std::fixed << std::setprecision(2) << compress_API_time_ms_1.count() << std::endl;
}

// Bandwidth measurement API
void xil_decompress_bandwidth(const std::string& single_bin,
                              uint8_t* in,
                              uint8_t* out,
                              uint32_t input_size,
                              uint32_t cu,
                              uint8_t device_id,
                              uint8_t max_cr) {
    uint32_t debytes = 0;
    uint32_t num_iter = NUM_ITER;
    xfZlib xlz(single_bin, max_cr, DECOMP_ONLY, device_id, 0, FULL);
    ERROR_STATUS(xlz.error_code());
    std::chrono::duration<double, std::nano> decompress_API_time_ns_1(0);
    std::chrono::duration<double, std::milli> decompress_API_time_ms_1(0);
    auto decompress_API_start = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < num_iter; i++) debytes = xlz.decompress(in, out, input_size, cu);

    auto decompress_API_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::nano>(decompress_API_end - decompress_API_start);
    auto duration_ms = std::chrono::duration<double, std::milli>(decompress_API_end - decompress_API_start);

    decompress_API_time_ns_1 = duration / num_iter;
    decompress_API_time_ms_1 = duration_ms / num_iter;
    float throughput_in_mbps_1 = (float)debytes * 1000 / decompress_API_time_ns_1.count();
    std::cout << "Input Size: " << input_size / 1024 << "KB ";
    std::cout << "Compressed Size: " << debytes / 1024 << "KB ";
    std::cout << "CU: " << cu;
    std::cout << " PID: " << getpid();
    std::cout << " PPID: " << getppid();
    std::cout << " API: " << std::fixed << std::setprecision(2) << throughput_in_mbps_1 << "MB/s";
    std::cout << " Time: " << std::fixed << std::setprecision(2) << decompress_API_time_ms_1.count() << std::endl;
}

int main(int argc, char* argv[]) {
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--compress", "-c", "Compress", "");
    parser.addSwitch("--decompress", "-d", "DeCompress", "");
    parser.addSwitch("--single_xclbin", "-sx", "Single XCLBIN", "single");
    parser.addSwitch("--max_cr", "-mcr", "Maximum CR", "20");
    parser.addSwitch("--multi_process", "-p", "Multiple Process", "1");
    parser.addSwitch("--id", "-id", "Device ID", "0");

    parser.parse(argc, argv);

    std::string compress_mod = parser.value("compress");
    std::string decompress_mod = parser.value("decompress");
    std::string single_bin = parser.value("single_xclbin");
    std::string mcr = parser.value("max_cr");
    std::string mproc = parser.value("multi_process");
    std::string device_ids = parser.value("id");

    uint8_t device_id = 0;
    if (!(device_ids.empty())) device_id = atoi(device_ids.c_str());

    uint8_t max_cr_val = MAX_CR;
    if (!(mcr.empty())) {
        max_cr_val = atoi(mcr.c_str());
    }

    uint8_t multi_proc = M_PROC;
    if (!(mproc.empty())) multi_proc = atoi(mproc.c_str());

    std::string mprocess_ent = "XCL_MULTIPROCESS_MODE=1";
    if (putenv((char*)mprocess_ent.c_str()) != 0) {
        std::cerr << "putenv failed" << std::endl;
    } else {
        std::cout << "Environmnet Variable: XCL_MULTIPROCESS_MODE: " << getenv("XCL_MULTIPROCESS_MODE") << std::endl;
    }

    if (!compress_mod.empty()) {
        if (multi_proc > 2) {
            multi_proc = 2;
            std::cout << "More than two processes may crash, resetting to 2" << std::endl;
        }
        // "-c" - Compress Mode
        std::string inFile_name = compress_mod;
        std::ifstream inFile(inFile_name.c_str(), std::ifstream::binary);
        uint64_t input_size = get_file_size(inFile);

        std::vector<uint8_t, zlib_aligned_allocator<uint8_t> > in[multi_proc];
        std::vector<uint8_t, zlib_aligned_allocator<uint8_t> > out[multi_proc];

        std::cout << "\n";
        // Allocate buffers for input and output
        for (int i = 0; i < multi_proc; i++) {
            in[i].resize(input_size);
            out[i].resize(input_size * 2);
        }

        // Read input file
        inFile.read((char*)in[0].data(), input_size);
        inFile.close();

        // Copy input data into multiple buffers
        for (int i = 1; i < multi_proc; i++) {
            std::memcpy(in[i].data(), in[0].data(), input_size);
        }

        std::cout << "\n";

        std::cout << "No of Process " << (int)multi_proc << std::endl;
        for (int i = 0; i < multi_proc; i++) {
            if (fork() == 0) {
                xil_compress_bandwidth(single_bin, in[i].data(), out[i].data(), input_size, device_id, max_cr_val);
                exit(0);
            }
        }

        for (int i = 0; i < multi_proc; i++) wait(NULL);

    } else if (!decompress_mod.empty()) {
        // "-d" - DeCompress Mode
        std::string inFile_name = decompress_mod;
        std::ifstream inFile(inFile_name.c_str(), std::ifstream::binary);
        uint64_t input_size = get_file_size(inFile);

        std::vector<uint8_t, zlib_aligned_allocator<uint8_t> > in[multi_proc];
        std::vector<uint8_t, zlib_aligned_allocator<uint8_t> > out[multi_proc];

        std::cout << "\n";
        // Allocate buffers for input and output
        for (int i = 0; i < multi_proc; i++) {
            in[i].resize(input_size);
            out[i].resize(input_size * max_cr_val);
        }

        // Read input file
        inFile.read((char*)in[0].data(), input_size);
        inFile.close();

        // Copy input data into multiple buffers
        for (int i = 1; i < multi_proc; i++) {
            std::memcpy(in[i].data(), in[0].data(), input_size);
        }

        std::cout << "No of Process " << (int)multi_proc << std::endl;
        for (int i = 0; i < multi_proc; i++) {
            if (fork() == 0) {
                xil_decompress_bandwidth(single_bin, in[i].data(), out[i].data(), input_size, i, device_id, max_cr_val);
                exit(0);
            }
        }

        for (int i = 0; i < multi_proc; i++) wait(NULL);
    }
}
