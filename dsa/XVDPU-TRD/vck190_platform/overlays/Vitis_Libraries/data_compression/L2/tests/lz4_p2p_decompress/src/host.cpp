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
#include "cmdlineparser.h"
#include "lz4_stream.hpp"
#include <fstream>
#include <vector>

static uint32_t getFileSize(std::ifstream& file) {
    file.seekg(0, file.end);
    uint32_t file_size = file.tellg();
    file.seekg(0, file.beg);
    return file_size;
}

void xilValidate(std::string& file_list) {
    std::cout << "\n";
    std::cout << "Status\t\tFile Name" << std::endl;
    std::cout << "\n";

    std::ifstream infilelist_val(file_list.c_str());
    std::string line_val;

    while (std::getline(infilelist_val, line_val)) {
        std::string line_in = line_val;
        std::string line_orig = line_in + ".orig";
        std::string delimiter = ".lz4";
        std::string token = line_in.substr(0, line_in.find(delimiter));
        token = token.substr(0, token.find(".xe2xd"));
        std::string line_raw = token;

        int ret = 0;
        // Validate input and output files
        ret = validate(line_raw, line_orig);
        if (ret == 0) {
            std::cout << (ret ? "FAILED\t" : "PASSED\t") << "\t" << line_in << std::endl;
        } else {
            std::cout << "Validation Failed" << line_orig.c_str() << std::endl;
        }
    }
}

void xilCompressDecompressList(
    std::string& file_list, bool d_flow, uint32_t block_size, std::string& decompress_bin, bool enable_p2p) {
    // De-Compression
    xfLz4Streaming* xlz = new xfLz4Streaming(decompress_bin, 0, block_size);

    std::ifstream infilelist_dec(file_list.c_str());
    std::string line_dec;

    if (!(infilelist_dec.good())) {
        std::cout << "Unable to open the list of files" << std::endl;
        exit(1);
    }

    std::getline(infilelist_dec, line_dec);
    if (!line_dec.length()) {
        std::cout << "Input list of file is empty" << std::endl;
        exit(1);
    }

    std::cout << "\n";
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "                     Xilinx De-Compress                       " << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    if (d_flow == 0) {
        std::cout << "\n";
        std::cout << "KT(MBps)\tFile Size(MB)\t\tFile Name" << std::endl;
        std::cout << "\n";
    }

    // Decompress list of files
    // This loop does LZ4 decompress on list
    // of files.
    do {
        std::string file_line = line_dec;
        std::ifstream inFile_dec(file_line.c_str(), std::ifstream::binary);
        if (!inFile_dec) {
            std::cout << "Unable to open file";
            exit(1);
        }

        uint32_t input_size = getFileSize(inFile_dec);
        inFile_dec.close();

        std::string lz_decompress_in = file_line;
        std::string lz_decompress_out = file_line;
        lz_decompress_out = lz_decompress_out + ".orig";

        // Call LZ4 decompression
        xlz->decompressFileFull(lz_decompress_in, lz_decompress_out, input_size, d_flow, enable_p2p);

        if (d_flow == 0) {
            std::cout << "\t\t" << (double)input_size / 1000000 << "\t\t\t" << lz_decompress_in << std::endl;
        }
    } while (std::getline(infilelist_dec, line_dec)); // While loop ends

    delete (xlz);
}
void xilBatchVerify(std::string& file_list, uint32_t block_size, std::string& decompress_bin, bool enable_p2p) {
    { // Start of Flow : Standard LZ4 Compress vs Xilinx LZ4 Decompress

        xilCompressDecompressList(file_list, 0, block_size, decompress_bin, enable_p2p);

        // Validate
        std::cout << "\n";
        std::cout << "-------------------------------------------------------------"
                     "---------------------------"
                  << std::endl;
        std::cout << "                       Validate: Xilinx LZ4 Decompress             " << std::endl;
        std::cout << "-------------------------------------------------------------"
                     "---------------------------"
                  << std::endl;
        xilValidate(file_list);

    } // End of Flow : Standard LZ4 Compress vs Xilinx LZ4 Decompress
}

void xilDecompressTop(std::string& decompress_mod, uint32_t block_size, std::string& decompress_bin, bool enable_p2p) {
    // Create xilLz4Streaming object
    xfLz4Streaming* xlz = new xfLz4Streaming(decompress_bin, 0, block_size);

    std::string file = decompress_mod;
    std::ifstream inFile_dec(file.c_str(), std::ifstream::binary);
    if (!inFile_dec) {
        std::cout << "Unable to open file";
        exit(1);
    }

    uint32_t input_size = getFileSize(inFile_dec);
    inFile_dec.close();

    const char* sizes[] = {"B", "kB", "MB", "GB", "TB"};
    double len = input_size;
    int order = 0;
    while (len >= 1000) {
        order++;
        len = len / 1000;
    }

    string lz_decompress_in = decompress_mod;
    string lz_decompress_out = lz_decompress_in;
    lz_decompress_out = lz_decompress_out + ".orig";

    std::cout << "KT(MB/s)\t\t:";
    // Call LZ4 decompression
    xlz->decompressFileFull(lz_decompress_in, lz_decompress_out, input_size, 0, enable_p2p);
#ifdef VERBOSE
    std::cout << std::fixed << std::setprecision(2) << "\nFile Size(" << sizes[order] << ")\t\t:" << len << std::endl
              << "File Name\t\t:" << lz_decompress_in << std::endl;
    std::cout << "\n";
    std::cout << "Output Location: " << lz_decompress_out.c_str() << std::endl;
#endif
    delete (xlz);
}

int main(int argc, char* argv[]) {
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--decompress_xclbin", "-dx", "DeCompress XCLBIN", "decompress");
    parser.addSwitch("--decompress", "-d", "Decompress", "");
    parser.addSwitch("--p2p_mode", "-p2p", "P2P Mode", "");
    parser.addSwitch("--file_list", "-l", "List of Input Files", "");
    parser.addSwitch("--block_size", "-B", "Compress Block Size [0-64: 1-256: 2-1024: 3-4096]", "0");
    parser.parse(argc, argv);

    std::string decompress_bin = parser.value("decompress_xclbin");
    std::string decompress_mod = parser.value("decompress");
    std::string p2pMode = parser.value("p2p_mode");
    std::string filelist = parser.value("file_list");
    std::string block_size = parser.value("block_size");

    uint32_t bSize = 0;
    // Block Size
    if (!(block_size.empty())) {
        bSize = atoi(block_size.c_str());

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

    // p2p flow enable/disable
    bool enable_p2p = ENABLE_P2P;
    if (!p2pMode.empty()) enable_p2p = std::stoi(p2pMode);

    // "-d" Decompress Mode
    if (!decompress_mod.empty()) xilDecompressTop(decompress_mod, bSize, decompress_bin, enable_p2p);

    // "-l" List of Files
    if (!filelist.empty()) {
        xilBatchVerify(filelist, bSize, decompress_bin, enable_p2p);
    }
}
