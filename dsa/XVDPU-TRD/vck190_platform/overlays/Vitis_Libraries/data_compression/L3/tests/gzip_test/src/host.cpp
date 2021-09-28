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

using namespace xf::compression;

void xil_validate(std::string& file_list, std::string& ext);

void xil_compress_decompress_list(std::string& file_list,
                                  std::string& ext1,
                                  std::string& ext2,
                                  int ccu,
                                  int dcu,
                                  std::string& single_bin,
                                  uint8_t max_cr,
                                  enum list_mode mode = ONLY_DECOMPRESS,
                                  uint8_t device_id = 0,
                                  design_flow dflow = XILINX_GZIP) {
    // Create xfZlib object
    xfZlib xlz(single_bin, false, max_cr, BOTH, device_id, 0, FULL, dflow, ccu, dcu);
    ERROR_STATUS(xlz.error_code());

    if (mode != ONLY_DECOMPRESS) {
        std::cout << "--------------------------------------------------------------" << std::endl;
        if (dflow)
            std::cout << "                     Xilinx ZLIB Compress" << std::endl;
        else
            std::cout << "                     Xilinx GZIP Compress " << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;

        std::cout << "\n";
        std::cout << "E2E(MBps)\tCR\t\tFile Size(MB)\t\tFile Name" << std::endl;
        std::cout << "\n";

        std::ifstream infilelist(file_list);
        std::string line;

        // Compress list of files
        // This loop does LZ4 compression on list
        // of files.
        while (std::getline(infilelist, line)) {
            std::ifstream inFile(line, std::ifstream::binary);
            if (!inFile) {
                std::cerr << "Unable to open file";
                exit(1);
            }

            uint64_t input_size = get_file_size(inFile);
            inFile.close();

            std::string compress_in = line;
            std::string compress_out = line;
            compress_out = compress_out + ext1;

            // Call Zlib compression
            uint64_t enbytes = xlz.compress_file(compress_in, compress_out, input_size, ccu);

            std::cout << "\t\t" << (double)input_size / enbytes << "\t\t" << std::fixed << std::setprecision(3)
                      << (double)input_size / 1000000 << "\t\t\t" << compress_in << std::endl;
        }
    }

    // Decompress
    if (mode != ONLY_COMPRESS) {
        std::ifstream infilelist_dec(file_list);
        std::string line_dec;

        std::cout << "\n";
        std::cout << "--------------------------------------------------------------" << std::endl;
        if (dflow)
            std::cout << "                     Xilinx ZLIB Decompress" << std::endl;
        else
            std::cout << "                     Xilinx GZIP Decompress" << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;
        std::cout << "\n";
        std::cout << "E2E(MBps)\tFile Size(MB)\t\tFile Name" << std::endl;
        std::cout << "\n";
        // Decompress list of files
        while (std::getline(infilelist_dec, line_dec)) {
            std::string file_line = line_dec;
            file_line = file_line + ext2;

            std::ifstream inFile_dec(file_line, std::ifstream::binary);
            if (!inFile_dec) {
                std::cerr << "Unable to open file";
                exit(1);
            }

            uint64_t input_size = get_file_size(inFile_dec);
            inFile_dec.close();

            std::string decompress_in = file_line;
            std::string decompress_out = file_line;
            decompress_out = decompress_out + ".orig";
            // Call Zlib decompression

            auto t1 = std::chrono::high_resolution_clock::now();
            xlz.decompress_file(decompress_in, decompress_out, input_size, dcu);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(t2 - t1).count();

            std::cout << std::fixed << std::setprecision(3) << "\t\t" << (double)input_size / 1000000 << "\t\t"
                      << decompress_in << "\t\t" << duration << std::endl;
        } // While loop ends
    }
}

void xil_batch_verify(std::string& file_list,
                      int ccu,
                      int dcu,
                      list_mode mode,
                      std::string& single_bin,
                      uint8_t device_id,
                      uint8_t max_cr,
                      design_flow dflow) {
    std::string ext1, ext2, ext3;

    if (dflow) {
        ext1 = ".xe2xd.xz";
        ext2 = ".xe2xd.xz";
        ext3 = ".xe2xd.xz.orig";
    } else {
        ext1 = ".xe2xd.gz";
        ext2 = ".xe2xd.gz";
        ext3 = ".xe2xd.gz.orig";
    }

    xil_compress_decompress_list(file_list, ext1, ext2, ccu, dcu, single_bin, max_cr, mode, device_id, dflow);

    // Validate
    std::cout << "\n";
    std::cout << "----------------------------------------------------------------------------------------"
              << std::endl;
    if (dflow)
        std::cout << "                       Validate: Xilinx ZLIB Compress vs Xilinx ZLIB Decompress " << std::endl;
    else
        std::cout << "                       Validate: Xilinx GZIP Compress vs Xilinx GZIP Decompress           "
                  << std::endl;
    std::cout << "----------------------------------------------------------------------------------------"
              << std::endl;
    xil_validate(file_list, ext3);
}

void xil_decompress_top(std::string& decompress_mod,
                        int cu,
                        std::string& single_bin,
                        uint8_t device_id,
                        uint8_t max_cr,
                        bool list_flow = false,
                        design_flow dflow = XILINX_GZIP) {
    // Xilinx ZLIB object
    xfZlib xlz(single_bin, false, max_cr, DECOMP_ONLY, device_id, 0, FULL, dflow, 0, cu);
    ERROR_STATUS(xlz.error_code());
    if (!list_flow) std::cout << std::fixed << std::setprecision(2) << "E2E(MBps)\t\t:";

    std::ifstream inFile(decompress_mod, std::ifstream::binary);
    if (!inFile) {
        std::cerr << "Unable to open file";
        exit(1);
    }
    uint64_t input_size = get_file_size(inFile);

    std::string lz_decompress_in = decompress_mod;
    std::string lz_decompress_out = decompress_mod;
    lz_decompress_out = lz_decompress_out + ".raw";
    // Call ZLIB compression
    xlz.decompress_file(lz_decompress_in, lz_decompress_out, input_size, cu);

    if (list_flow) {
        std::cout << std::fixed << std::setprecision(3) << "\t\t" << (double)input_size / 1000000 << "\t\t" << cu
                  << "\t" << lz_decompress_in << std::endl;

    } else {
        std::cout << std::fixed << std::setprecision(3) << std::endl
                  << "File Size(MB)\t\t:" << (double)input_size / 1000000 << std::endl
                  << "File Name\t\t:" << lz_decompress_in << std::endl;
    }
}

void xil_compress_top(std::string& compress_mod,
                      std::string& single_bin,
                      uint8_t device_id,
                      uint8_t max_cr,
                      int cu_id,
                      design_flow dflow = XILINX_GZIP,
                      bool list_flow = false) {
    // Xilinx ZLIB object
    xfZlib xlz(single_bin, false, max_cr, COMP_ONLY, device_id, 0, FULL, dflow, cu_id);
    ERROR_STATUS(xlz.error_code());
    if (!list_flow) std::cout << std::fixed << std::setprecision(2) << "E2E(MBps)\t\t:";

    std::ifstream inFile(compress_mod, std::ifstream::binary);
    if (!inFile) {
        std::cerr << "Unable to open file";
        exit(1);
    }
    uint64_t input_size = get_file_size(inFile);

    std::string lz_compress_in = compress_mod;
    std::string lz_compress_out = compress_mod;
    if (dflow)
        lz_compress_out = lz_compress_out + ".xz";
    else
        lz_compress_out = lz_compress_out + ".gz";
    // Call ZLIB compression
    uint64_t enbytes = xlz.compress_file(lz_compress_in, lz_compress_out, input_size, cu_id);
    if (list_flow) {
        std::cout << "\t\t" << (double)input_size / enbytes << "\t\t" << std::fixed << std::setprecision(3)
                  << (double)input_size / 1000000 << "\t\t\t" << lz_compress_in << "\t\t" << cu_id << std::endl;
    } else {
        std::cout.precision(3);
        std::cout << std::fixed << std::setprecision(2) << std::endl
                  << "ZLIB_CR\t\t\t:" << (double)input_size / enbytes << std::endl
                  << std::fixed << std::setprecision(3) << "File Size(MB)\t\t:" << (double)input_size / 1000000
                  << std::endl
                  << "File Name\t\t:" << lz_compress_in << std::endl;
        std::cout << "\n";
        std::cout << "Output Location: " << lz_compress_out.c_str() << std::endl;
    }
}

void xil_validate(std::string& file_list, std::string& ext) {
    std::cout << "\n";
    std::cout << "Status\t\tFile Name" << std::endl;
    std::cout << "\n";

    std::ifstream infilelist_val(file_list);
    std::string line_val;

    while (std::getline(infilelist_val, line_val)) {
        std::string line_in = line_val;
        std::string line_out = line_in + ext;

        int ret = 0;
        // Validate input and output files
        ret = validate(line_in, line_out);
        if (ret == 0) {
            std::cout << (ret ? "FAILED\t" : "PASSED\t") << "\t" << line_in << std::endl;
        } else {
            std::cerr << "Validation Failed" << line_out.c_str() << std::endl;
            exit(1);
        }
    }
}

void xilCompressDecompressTop(std::string& compress_decompress_mod,
                              std::string& single_bin,
                              uint8_t device_id,
                              uint8_t max_cr_val,
                              design_flow dflow) {
    // Create xfZlib object
    xfZlib xlz(single_bin, false, max_cr_val, BOTH, device_id, 0, FULL, dflow);
    ERROR_STATUS(xlz.error_code());

    std::cout << "--------------------------------------------------------------" << std::endl;
    if (dflow)
        std::cout << "                     Xilinx ZLIB Compress" << std::endl;
    else
        std::cout << "                     Xilinx GZIP Compress" << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;

    std::cout << "\n";

    std::cout << std::fixed << std::setprecision(2) << "E2E(MBps)\t\t:";

    std::ifstream inFile(compress_decompress_mod, std::ifstream::binary);
    if (!inFile) {
        std::cerr << "Unable to open file";
        exit(1);
    }

    uint64_t input_size = get_file_size(inFile);
    inFile.close();

    std::string compress_in = compress_decompress_mod;
    std::string compress_out = compress_decompress_mod;
    if (dflow)
        compress_out = compress_out + ".xz";
    else
        compress_out = compress_out + ".gz";

    // Call Zlib compression
    uint64_t enbytes = xlz.compress_file(compress_in, compress_out, input_size, 0);
    std::cout << std::fixed << std::setprecision(2) << std::endl
              << "CR\t\t\t:" << (double)input_size / enbytes << std::endl
              << std::fixed << std::setprecision(3) << "File Size(MB)\t\t:" << (double)input_size / 1000000 << std::endl
              << "File Name\t\t:" << compress_in << std::endl;

    // Decompress

    std::cout << "\n";
    std::cout << "--------------------------------------------------------------" << std::endl;
    if (dflow)
        std::cout << "                     Xilinx ZLIB Decompress" << std::endl;
    else
        std::cout << "                     Xilinx GZIP Decompress" << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "\n";

    std::cout << std::fixed << std::setprecision(2) << "E2E(MBps)\t\t:";

    // Decompress list of files

    std::string lz_decompress_in, lz_decompress_out;
    if (dflow)
        lz_decompress_in = compress_decompress_mod + ".xz";
    else
        lz_decompress_in = compress_decompress_mod + ".gz";
    lz_decompress_out = lz_decompress_in + ".orig";

    std::ifstream inFile_dec(lz_decompress_in, std::ifstream::binary);
    if (!inFile_dec) {
        std::cerr << "Unable to open file";
        exit(1);
    }

    input_size = get_file_size(inFile_dec);
    inFile_dec.close();

    // Call Zlib decompression
    xlz.decompress_file(lz_decompress_in, lz_decompress_out, input_size, 0);

    std::cout << std::fixed << std::setprecision(2) << std::endl
              << std::fixed << std::setprecision(3) << "File Size(MB)\t\t:" << (double)input_size / 1000000 << std::endl
              << "File Name\t\t:" << lz_decompress_in << std::endl;

    // Validate
    std::cout << "\n";
    std::string outputFile = compress_decompress_mod;
    std::string inputFile = compress_decompress_mod;
    if (dflow)
        outputFile = outputFile + ".xz" + ".orig";
    else
        outputFile = outputFile + ".gz" + ".orig";

    int ret = validate(inputFile, outputFile);
    if (ret == 0) {
        std::cout << (ret ? "FAILED\t" : "PASSED\t") << "\t" << inputFile << std::endl;
    } else {
        std::cout << "Validation Failed" << outputFile.c_str() << std::endl;
    }
}

void get_list_filenames(std::string& filelist, std::vector<std::string>& fname_vec) {
    std::ifstream infilelist(filelist);
    if (!infilelist) {
        std::cerr << "Unable to Open File List: " + filelist;
        exit(1);
    }

    std::string line;

    // Pick File names
    while (std::getline(infilelist, line)) fname_vec.push_back(line);

    infilelist.close();

    if (fname_vec.empty()) {
        std::cerr << "Failed to find files under " + filelist;
        exit(1);
    }
}
int main(int argc, char* argv[]) {
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--compress", "-c", "Compress", "");
    parser.addSwitch("--decompress", "-d", "DeCompress", "");
    parser.addSwitch("--single_xclbin", "-sx", "Single XCLBIN [Optional]", "");
    parser.addSwitch("--test", "-t", "Compress Decompress", "");

    parser.addSwitch("--c_file_list", "-cfl", "Compress list files", "");
    parser.addSwitch("--d_file_list", "-dfl", "Decompress list files", "");
    parser.addSwitch("--file_list", "-l", "List of Input Files", "");
    parser.addSwitch("--zlib", "-zlib", "[0:GZIP, 1:ZLIB]", "0");
    parser.addSwitch("--ck", "-ck", "Compress CU", "0");
    parser.addSwitch("--dk", "-dk", "Decompress CU", "0");
    parser.addSwitch("--id", "-id", "Device ID", "0");
    parser.addSwitch("--max_cr", "-mcr", "Maximum CR", "10");
    parser.parse(argc, argv);

    std::string compress_mod = parser.value("compress");
    std::string filelist = parser.value("file_list");
    std::string cfilelist = parser.value("c_file_list");
    std::string dfilelist = parser.value("d_file_list");
    std::string decompress_mod = parser.value("decompress");
    std::string single_bin = parser.value("single_xclbin");
    std::string test_mod = parser.value("test");
    std::string is_Zlib = parser.value("zlib");
    std::string ccu = parser.value("ck");
    std::string dcu = parser.value("dk");
    std::string device_ids = parser.value("id");
    std::string mcr = parser.value("max_cr");

    bool list_flow = false;
    uint8_t device_id = 0;
    int ccu_id, dcu_id;
    uint8_t max_cr_val = 0;

    if (single_bin.empty()) single_bin = getenv("XILINX_LIBZ_XCLBIN");
    if (!mcr.empty()) {
        max_cr_val = std::stoi(mcr);
    } else {
        // Default block size
        max_cr_val = MAX_CR;
    }

    design_flow dflow = std::stoi(is_Zlib) ? XILINX_ZLIB : XILINX_GZIP;

    if (!device_ids.empty()) device_id = std::stoi(device_ids);

    if (ccu.empty()) {
        std::cout << "please provide -ck option for cu" << std::endl;
        exit(0);
    } else {
        ccu_id = std::stoi(ccu);
        if (ccu_id > C_COMPUTE_UNIT) {
            std::cerr << "Compute unit value not valid, resetting it to 0 " << std::endl;
            ccu_id = 0;
        }
    }
    if (dcu.empty()) {
        std::cout << "please provide -dk option for cu" << std::endl;
        exit(0);
    } else {
        dcu_id = std::stoi(dcu);
        if (dcu_id > D_COMPUTE_UNIT) {
            std::cerr << "Compute unit value not valid, resetting it to 0 " << std::endl;
            dcu_id = 0;
        }
    }

    if (!test_mod.empty()) xilCompressDecompressTop(test_mod, single_bin, device_id, max_cr_val, dflow);
    if (!cfilelist.empty() || !dfilelist.empty() || !filelist.empty()) {
        list_flow = true;
        std::vector<std::string> fname_vec;
        if (!cfilelist.empty()) {
            get_list_filenames(cfilelist, fname_vec);
            int batch_proc = (fname_vec.size() < C_COMPUTE_UNIT) ? fname_vec.size() : C_COMPUTE_UNIT;
            std::cout << "\n";
            std::cout << "E2E(MBps)\tCR\t\tFile Size(MB)\t\tFile Name\tCU" << std::endl;
            std::cout << "\n";
            for (uint8_t i = 0; i < fname_vec.size(); i += C_COMPUTE_UNIT) {
                if (fname_vec.size() - i < C_COMPUTE_UNIT) batch_proc = fname_vec.size() - i;
                for (uint8_t j = 0; j < batch_proc; j++) {
                    if (fork() == 0) {
                        xil_compress_top(fname_vec[i + j], single_bin, device_id, max_cr_val, j, dflow, list_flow);
                        exit(0);
                    }
                }

                while (wait(NULL) > 0)
                    ;
            }
        } else if (!dfilelist.empty()) {
            get_list_filenames(dfilelist, fname_vec);
            int batch_proc = (fname_vec.size() < D_COMPUTE_UNIT) ? fname_vec.size() : D_COMPUTE_UNIT;
            std::cout << "\n";
            std::cout << "E2E(MBps)\tFile Size(MB)\tCU\tFile Name" << std::endl;
            std::cout << "\n";
            for (uint8_t i = 0; i < fname_vec.size(); i += D_COMPUTE_UNIT) {
                if (fname_vec.size() - i < D_COMPUTE_UNIT) batch_proc = fname_vec.size() - i;
                for (uint8_t j = 0; j < batch_proc; j++) {
                    if (fork() == 0) {
                        xil_decompress_top(fname_vec[j + i], j, single_bin, device_id, max_cr_val, list_flow);
                        exit(0);
                    }
                }
                while (wait(NULL) > 0)
                    ;
            }
        } else if (!filelist.empty()) {
        }
    }

    if (!filelist.empty()) {
        list_mode lMode;
        // "-l" - List of files
        if (!compress_mod.empty()) {
            lMode = ONLY_COMPRESS;
        } else if (!decompress_mod.empty()) {
            lMode = ONLY_DECOMPRESS;
        } else {
            lMode = COMP_DECOMP;
        }
        xil_batch_verify(filelist, ccu_id, dcu_id, lMode, single_bin, device_id, max_cr_val, dflow);
    } else if (!compress_mod.empty()) {
        // "-c" - Compress Mode
        xil_compress_top(compress_mod, single_bin, device_id, max_cr_val, ccu_id, dflow);
    } else if (!decompress_mod.empty())
        // "-d" - DeCompress Mode
        xil_decompress_top(decompress_mod, dcu_id, single_bin, device_id, max_cr_val, dflow);
}
