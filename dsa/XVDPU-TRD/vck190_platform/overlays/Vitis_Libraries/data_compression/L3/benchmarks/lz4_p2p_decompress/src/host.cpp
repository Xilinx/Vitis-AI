/*
 * Copyright 2019-2021 Xilinx, Inc.
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
#include "lz4_p2p.hpp"
#include "lz4_p2p_dec.hpp"
#include <fstream>
#include <iostream>
#include <cassert>
#include "cmdlineparser.h"

// The default value set as non-P2P, so that design can work for all platforms.
// For P2P enabled platform, user need to manually change this macro value to true.
#ifndef ENABLE_P2P
#define ENABLE_P2P false
#endif

int validate(std::string& inFile_name, std::string& outFile_name) {
    std::string command = "cmp " + inFile_name + " " + outFile_name;
    int ret = system(command.c_str());
    return ret;
}

void decompress_multiple_files(const std::vector<std::string>& inFileVec,
                               const std::vector<std::string>& outFileVec,
                               const std::string& decompress_bin,
                               bool enable_p2p,
                               uint8_t maxCR) {
    std::vector<char*> outVec;
    std::vector<uint64_t> orgSizeVec;
    std::vector<uint32_t> inSizeVec;
    std::vector<int> fd_p2p_vec;
    std::vector<cl_event> userEventVec;
    uint64_t total_in_size = 0;

    std::cout << "\n";
    std::cout << "\x1B[31m[Disk Operation]\033[0m Reading Input Files Started ..." << std::endl;
    for (uint32_t fid = 0; fid < inFileVec.size(); fid++) {
        std::string inFile_name = inFileVec[fid];
        std::ifstream inFile(inFile_name.c_str(), std::ifstream::binary);
        uint32_t input_size = xfLz4::get_file_size(inFile);
        inFile.close();

        int fd_p2p_c_in = open(inFile_name.c_str(), O_RDONLY | O_DIRECT);
        if (fd_p2p_c_in <= 0) {
            std::cout << "P2P: Unable to open input file, fd: " << fd_p2p_c_in << std::endl;
            exit(1);
        }
        std::vector<uint8_t, aligned_allocator<uint8_t> > in_4kbytes(4 * KB);
        read(fd_p2p_c_in, (char*)in_4kbytes.data(), 4 * KB);
        lseek(fd_p2p_c_in, 0, SEEK_SET);
        fd_p2p_vec.push_back(fd_p2p_c_in);
        total_in_size += input_size;
        char* out = (char*)aligned_alloc(4096, maxCR * input_size);
        uint64_t orgSize;
        outVec.push_back(out);
        orgSizeVec.push_back(orgSize);
        inSizeVec.push_back(input_size);
    }
    std::cout << "\x1B[31m[Disk Operation]\033[0m Reading Input Files Done ..." << std::endl;
    std::cout << "\n\n";
    std::cout << "\x1B[32m[OpenCL Setup]\033[0m OpenCL/Host/Device Buffer Setup Started ..." << std::endl;
    xfLz4 xlz(decompress_bin);
    std::cout << "\x1B[32m[OpenCL Setup]\033[0m OpenCL/Host/Device Buffer Setup Done ..." << std::endl;
    std::cout << "\n";
    std::cout << "\x1B[36m[FPGA LZ4]\033[0m LZ4 P2P DeCompression Started ..." << std::endl;
    std::cout << "\n";
    xlz.decompress_in_line_multiple_files(inFileVec, fd_p2p_vec, outVec, orgSizeVec, inSizeVec, enable_p2p, maxCR);
    std::cout << "\n";
    std::cout << "\x1B[36m[FPGA LZ4]\033[0m LZ4 P2P DeCompression Done ..." << std::endl;
    std::cout << "\n";
    std::cout << "\x1B[31m[Disk Operation]\033[0m Writing Output Files Started ..." << std::endl;
    for (uint32_t fid = 0; fid < inFileVec.size(); fid++) {
        std::string outFile_name = outFileVec[fid];
        std::ofstream outFile(outFile_name.c_str(), std::ofstream::binary);
        outFile.write((char*)outVec[fid], orgSizeVec[fid]);
        close(fd_p2p_vec[fid]);
        outFile.close();
    }
    std::cout << "\x1B[31m[Disk Operation]\033[0m Writing Output Files Done ..." << std::endl;
}

void xil_decompress_file_list(std::string& file_list, std::string& decompress_bin, bool enable_p2p, uint8_t maxCR) {
    std::ifstream infilelist_dec(file_list.c_str());
    std::string line_dec;
    std::vector<std::string> inFileList;
    std::vector<std::string> outFileList;
    std::vector<std::string> orgFileList;
    while (std::getline(infilelist_dec, line_dec)) {
        std::string in_file = line_dec;
        std::string out_file = line_dec + ".org";
        inFileList.push_back(in_file);
        std::string delimiter = ".lz4";
        std::string token = line_dec.substr(0, line_dec.find(delimiter));
        orgFileList.push_back(token);
        outFileList.push_back(out_file);
    }
    decompress_multiple_files(inFileList, outFileList, decompress_bin, enable_p2p, maxCR);
    std::cout << std::endl;
    for (size_t i = 0; i < inFileList.size(); i++) {
        auto ret = validate(orgFileList[i], outFileList[i]) ? "FAILED: " : "PASSED: ";
        std::cout << ret << inFileList[i] << std::endl;
    }
}

void xil_decompress_file(std::string& file, std::string& decompress_bin, bool enable_p2p, uint8_t maxCR) {
    std::string line_dec = file.c_str();
    std::vector<std::string> inFileList;
    std::vector<std::string> outFileList;
    std::vector<std::string> orgFileList;
    std::string in_file = line_dec;
    std::string out_file = line_dec + ".org";
    inFileList.push_back(in_file);
    std::string delimiter = ".lz4";
    std::string token = line_dec.substr(0, line_dec.find(delimiter));
    orgFileList.push_back(token);
    outFileList.push_back(out_file);
    decompress_multiple_files(inFileList, outFileList, decompress_bin, enable_p2p, maxCR);
    std::cout << std::endl;
    for (size_t i = 0; i < inFileList.size(); i++) {
        auto ret = validate(orgFileList[i], outFileList[i]) ? "FAILED: " : "PASSED: ";
        std::cout << ret << inFileList[i] << std::endl;
    }
}

int main(int argc, char* argv[]) {
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--decompress_xclbin", "-dx", "Decompress XCLBIN", "decompress");
    parser.addSwitch("--decompress_mode", "-d", "Decompress Mode", "");
    parser.addSwitch("--p2p_mod", "-p2p", "P2P Mode", "");
    parser.addSwitch("--max_cr", "-mcr", "Maximum CR", "8");
    parser.addSwitch("--single_xclbin", "-sx", "Single XCLBIN", "p2p_decompress");
    parser.addSwitch("--file_list", "-l", "List of Input Files", "");
    parser.parse(argc, argv);

    std::string decompress_xclbin = parser.value("decompress_xclbin");
    std::string decompress_mod = parser.value("decompress_mode");
    std::string p2pMode = parser.value("p2p_mod");
    std::string maxCR = parser.value("max_cr");
    std::string single_bin = parser.value("single_xclbin");
    std::string filelist = parser.value("file_list");

    uint8_t maximCR = stoi(maxCR);

    bool enable_p2p = ENABLE_P2P;
    if (!p2pMode.empty()) enable_p2p = std::stoi(p2pMode);

    if (!decompress_mod.empty()) xil_decompress_file(decompress_mod, decompress_xclbin, enable_p2p, maximCR);

    // "-l" List of Files
    if (!filelist.empty()) xil_decompress_file_list(filelist, decompress_xclbin, enable_p2p, maximCR);
}
