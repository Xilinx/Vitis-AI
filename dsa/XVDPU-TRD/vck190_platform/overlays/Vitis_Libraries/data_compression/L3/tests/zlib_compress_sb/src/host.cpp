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
#include "zlib.hpp"
#include <fstream>
#include <vector>
#include "cmdlineparser.h"

using namespace xf::compression;

void xil_compress_list(std::string& file_list,
                       std::string& ext,
                       int cu,
                       std::string& compress_bin,
                       uint8_t max_cr,
                       uint8_t device_id = 0) {
    // Create xfZlib object
    xfZlib xlz(compress_bin, true, max_cr, BOTH, device_id, 0, FULL, XILINX_ZLIB);
    ERROR_STATUS(xlz.error_code());

    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "                     Xilinx Zlib Compress                          " << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;

    std::cout << "\n";
    std::cout << "E2E(MBps)\tCR\t\tFile Size(MB)\t\tFile Name" << std::endl;
    std::cout << "\n";

    std::ifstream infilelist(file_list.c_str());
    std::string line;

    // Compress list of files
    // This loop does LZ4 compression on list
    // of files.
    while (std::getline(infilelist, line)) {
        std::ifstream inFile(line.c_str(), std::ifstream::binary);
        if (!inFile) {
            std::cout << "Unable to open file";
            exit(1);
        }

        uint64_t input_size = get_file_size(inFile);
        inFile.close();

        std::string compress_in = line;
        std::string compress_out = line;
        compress_out = compress_out + ext;

        // Call Zlib compression
        uint64_t enbytes = xlz.compress_file(compress_in, compress_out, input_size);

        std::cout << "\t\t" << (double)input_size / enbytes << "\t\t" << std::fixed << std::setprecision(3)
                  << (double)input_size / 1000000 << "\t\t\t" << compress_in << std::endl;
    }
}

void xil_batch_verify(std::string& file_list, int cu, std::string& compress_bin, uint8_t device_id, uint8_t max_cr) {
    std::string ext;

    // Xilinx ZLIB Compression
    ext = ".xe2xd.zlib";

    xil_compress_list(file_list, ext, cu, compress_bin, max_cr, device_id);
}

void xil_compress_top(std::string& compress_mod, std::string& compress_bin, uint8_t device_id, uint8_t max_cr) {
    // Xilinx ZLIB object
    xfZlib xlz(compress_bin, true, max_cr, COMP_ONLY, device_id, 0, FULL, XILINX_ZLIB);
    ERROR_STATUS(xlz.error_code());

    std::cout << std::fixed << std::setprecision(2) << "E2E(MBps)\t\t:";

    std::ifstream inFile(compress_mod.c_str(), std::ifstream::binary);
    if (!inFile) {
        std::cout << "Unable to open file";
        exit(1);
    }
    uint64_t input_size = get_file_size(inFile);

    std::string lz_compress_in = compress_mod;
    std::string lz_compress_out = compress_mod;
    lz_compress_out = lz_compress_out + ".zlib";

    // Call ZLIB compression
    uint64_t enbytes = xlz.compress_file(lz_compress_in, lz_compress_out, input_size);

    std::cout.precision(3);
    std::cout << std::fixed << std::setprecision(2) << std::endl
              << "ZLIB_CR\t\t\t:" << (double)input_size / enbytes << std::endl
              << std::fixed << std::setprecision(3) << "File Size(MB)\t\t:" << (double)input_size / 1000000 << std::endl
              << "File Name\t\t:" << lz_compress_in << std::endl;
    std::cout << "\n";
    std::cout << "Output Location: " << lz_compress_out.c_str() << std::endl;
}

int main(int argc, char* argv[]) {
    int cu_run;
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--compress", "-c", "Compress", "");
    parser.addSwitch("--compress_xclbin", "-cx", "Compress XCLBIN", "compress");

    parser.addSwitch("--file_list", "-l", "List of Input Files", "");
    parser.addSwitch("--cu", "-k", "CU", "0");
    parser.addSwitch("--id", "-id", "Device ID", "0");
    parser.addSwitch("--max_cr", "-mcr", "Maximum CR", "10");
    parser.parse(argc, argv);

    std::string compress_mod = parser.value("compress");
    std::string filelist = parser.value("file_list");
    std::string compress_bin = parser.value("compress_xclbin");
    std::string cu = parser.value("cu");
    std::string device_ids = parser.value("id");
    std::string mcr = parser.value("max_cr");

    uint8_t max_cr_val = 0;
    if (!(mcr.empty())) {
        max_cr_val = atoi(mcr.c_str());
    } else {
        // Default block size
        max_cr_val = MAX_CR;
    }

    uint8_t device_id = 0;

    if (!(device_ids.empty())) device_id = atoi(device_ids.c_str());

    if (cu.empty()) {
        std::cout << "please provide -k option for cu" << std::endl;
        exit(0);
    } else {
        cu_run = atoi(cu.c_str());
    }

    if (!filelist.empty()) {
        // "-l" - List Compress Mode
        xil_batch_verify(filelist, cu_run, compress_bin, device_id, max_cr_val);
    } else if (!compress_mod.empty()) {
        // "-c" - Compress Mode
        xil_compress_top(compress_mod, compress_bin, device_id, max_cr_val);
    }
}
