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
#include "lz4_p2p_comp.hpp"
#include <fstream>
#include <vector>
#include "cmdlineparser.h"

// The default value set as non-P2P, so that design can work for all platforms.
// For P2P enabled platform, user need to manually change this macro value to true.
#ifndef ENABLE_P2P
#define ENABLE_P2P false
#endif

void compress_multiple_files(const std::vector<std::string>& inFileVec,
                             const std::vector<std::string>& outFileVec,
                             uint32_t block_size,
                             const std::string& compress_bin,
                             bool enable_p2p) {
    std::vector<char*> inVec;
    std::vector<int> fd_p2p_vec;
    std::vector<char*> outVec;
    std::vector<uint32_t> inSizeVec;

    std::cout << "\n\nNumFiles:" << inFileVec.size() << std::endl;

    std::cout << "\x1B[31m[Disk Operation]\033[0m Reading Input Files Started ..." << std::endl;
    for (uint32_t fid = 0; fid < inFileVec.size(); fid++) {
        std::string inFile_name = inFileVec[fid];
        std::ifstream inFile(inFile_name.c_str(), std::ifstream::binary);
        if (!inFile) {
            std::cout << "Unable to open file";
            exit(1);
        }
        uint32_t input_size = xflz4::get_file_size(inFile);

        std::string outFile_name = outFileVec[fid];

        char* in = (char*)aligned_alloc(4096, input_size);
        inFile.read(in, input_size);
        inVec.push_back(in);
        inSizeVec.push_back(input_size);
    }
    std::cout << "\x1B[31m[Disk Operation]\033[0m Reading Input Files Done ..." << std::endl;
    std::cout << "\n\n";
    std::cout << "\x1B[32m[OpenCL Setup]\033[0m OpenCL/Host/Device Buffer Setup Started ..." << std::endl;
    xflz4 xlz(compress_bin, 0, block_size);
    std::cout << "\x1B[32m[OpenCL Setup]\033[0m OpenCL/Host/Device Buffer Setup Done ..." << std::endl;
    std::cout << "\n";
    std::cout << "\x1B[36m[FPGA LZ4]\033[0m LZ4 P2P Compression Started ..." << std::endl;
    xlz.compress_in_line_multiple_files(inVec, outFileVec, inSizeVec, enable_p2p);
    std::cout << "\n\n";
    std::cout << "\x1B[36m[FPGA LZ4]\033[0m LZ4 P2P Compression Done ..." << std::endl;
}

int validateFile(std::string& inFile_name, std::string& origFile_name) {
    std::string command = "cmp " + inFile_name + " " + origFile_name;
    int ret = system(command.c_str());
    return ret;
}

void xil_compress_file_list(std::string& file_list, uint32_t block_size, std::string& compress_bin, bool enable_p2p) {
    std::ifstream infilelist_comp(file_list.c_str());
    std::string line_comp;

    std::vector<std::string> inFileList;
    std::vector<std::string> outFileList;
    std::vector<std::string> origFileList;

    while (std::getline(infilelist_comp, line_comp)) {
        std::string orig_file = line_comp;
        std::string out_file = line_comp + ".lz4";
        inFileList.push_back(line_comp);
        origFileList.push_back(orig_file);
        outFileList.push_back(out_file);
    }
    compress_multiple_files(inFileList, outFileList, block_size, compress_bin, enable_p2p);
    std::cout << std::endl;
}

void xil_compress_file(std::string& file, uint32_t block_size, std::string& compress_bin, bool enable_p2p) {
    std::string line_comp = file.c_str();

    std::vector<std::string> inFileList;
    std::vector<std::string> outFileList;
    std::vector<std::string> origFileList;

    std::string orig_file = line_comp;
    std::string out_file = line_comp + ".lz4";
    inFileList.push_back(line_comp);
    origFileList.push_back(orig_file);
    outFileList.push_back(out_file);
    compress_multiple_files(inFileList, outFileList, block_size, compress_bin, enable_p2p);
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--compress_xclbin", "-cx", "Compress XCLBIN", "compress");
    parser.addSwitch("--compress", "-c", "Compress", "");
    parser.addSwitch("--file_list", "-l", "List of Input Files", "");
    parser.addSwitch("--p2p_mod", "-p2p", "P2P Mode", "");
    parser.addSwitch("--block_size", "-B", "Compress Block Size [0-64: 1-256: 2-1024: 3-4096]", "0");
    parser.addSwitch("--id", "-id", "Device ID", "0");
    parser.parse(argc, argv);

    std::string compress_file = parser.value("compress");
    std::string compress_bin = parser.value("compress_xclbin");
    std::string p2pMode = parser.value("p2p_mod");
    std::string filelist = parser.value("file_list");
    std::string block_size = parser.value("block_size");
    std::string device_ids = parser.value("id");

    uint32_t bSize = 0;

    bool enable_p2p = ENABLE_P2P;
    if (!p2pMode.empty()) enable_p2p = std::stoi(p2pMode);

    // Block Size
    if (!(block_size.empty())) {
        bSize = stoi(block_size);

        switch (bSize) {
            case 0:
                bSize = 64;
                break;
            case 1:
                bSize = 256;
                break;
            case 2:
                bSize = 1024;
                break;
            case 3:
                bSize = 4096;
                break;
            default:
                std::cout << "Invalid Block Size provided" << std::endl;
                parser.printHelp();
                exit(1);
        }
    } else {
        // Default Block Size - 64KB
        bSize = BLOCK_SIZE_IN_KB;
    }

    // "-c" - Compress Mode
    if (!compress_file.empty()) xil_compress_file(compress_file, bSize, compress_bin, enable_p2p);

    // "-l" List of Files
    if (!filelist.empty()) {
        std::cout << "\n" << std::endl;
        xil_compress_file_list(filelist, bSize, compress_bin, enable_p2p);
    }
}
