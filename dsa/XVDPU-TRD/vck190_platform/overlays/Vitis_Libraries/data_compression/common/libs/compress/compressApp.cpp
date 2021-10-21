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
#include "compressApp.hpp"
#include "compressBase.hpp"
#include <unistd.h>
#include <string>
#include <sys/wait.h>
#include <typeinfo>
#include <iomanip>
#include <iostream>
#include <fstream>

// Input file preliminary checks
void compressApp::inputFilePreCheck(std::ifstream& file) {
    if (!file) {
        std::cerr << "Unable to open input file" << std::endl;
        exit(1);
    }

    file.seekg(0, file.end);
    size_t file_size = file.tellg();
    if (file_size == 0) {
        cerr << "File is empty!" << std::endl;
        exit(1);
    }
    file.seekg(0, file.beg);
    m_inputsize = file_size;
}

// API to fetch the file size unit
static std::string getFileSizeUnit(double& input_size) {
    const char* sizes[] = {"B", "kB", "MB", "GB", "TB"};
    int order = 0;
    while (input_size >= 1000) {
        order++;
        input_size = input_size / 1000;
    }
    std::string res = sizes[order];
    return res;
}

std::string compressApp::getXclbin(void) const {
    return m_xclbin;
}
uint8_t compressApp::getDeviceId(void) const {
    return std::stoi(m_device_id);
}

std::string& compressApp::getInFileName(void) {
    return m_uncompressed_file;
}

uint8_t compressApp::getMCR(void) const {
    return std::stoi(m_mcr);
}

void compressApp::getListFilenames(std::string& filelist, std::vector<std::string>& fname_vec) {
    std::ifstream infilelist(filelist.c_str());
    inputFilePreCheck(infilelist);
    std::string line;
    // Pick File names
    while (std::getline(infilelist, line)) {
        fname_vec.push_back(line);
    }
    infilelist.close();
}

int compressApp::validate(std::string& inFile, std::string& outFile) {
    std::string command = "cmp " + inFile + " " + outFile;
    int ret = system(command.c_str());
    return ret;
}

void compressApp::parser(int argc, char** argv) {
    m_parser.parse(argc, argv);
    m_uncompressed_file = m_parser.value("compress");
    m_compressed_file = m_parser.value("decompress");
    if (m_parser.value("test") != "") {
        m_compress_decompress = true;
        m_uncompressed_file = m_parser.value("test");
    }
    m_uncompressed_filelist = m_parser.value("compress_list");
    m_compressed_filelist = m_parser.value("decompress_list");
    m_filelist = m_parser.value("test_list");
    m_xclbin = m_parser.value("xclbin");
    m_device_id = m_parser.value("device_id");
    m_mcr = m_parser.value("max_cr");
}

// compressApp Constructor: parse CLI opions and set the driver class memebr variables
compressApp::compressApp(int argc, char** argv, bool is_seq, bool enable_profile) {
    m_enableProfile = enable_profile;
    m_isSeq = is_seq;
    m_parser.addSwitch("--compress", "-c", "Compress", "");
    m_parser.addSwitch("--decompress", "-d", "Decompress", "");
    m_parser.addSwitch("--test", "-t", "Xilinx compress & Decompress", "");
    m_parser.addSwitch("--compress_list", "-cfl", "Compress List of Input Files", "");
    m_parser.addSwitch("--decompress_list", "-dfl", "Decompress List of compressed Input Files", "");
    m_parser.addSwitch("--test_list", "-l", "Xilinx Compress & Decompress on Input Files", "");
    m_parser.addSwitch("--max_cr", "-mcr", "Maximum CR", "20");
    m_parser.addSwitch("--xclbin", "-xbin", "XCLBIN", "");
    m_parser.addSwitch("--device_id", "-id", "Device ID", "0");
}

void compressApp::printTestSummaryHeader() {
    if (m_compress_flow) {
        std::cout << "--------------------------------------------------------------" << std::endl;
        std::cout << "                     Xilinx Compress" << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;
    } else {
        std::cout << "--------------------------------------------------------------" << std::endl;
        std::cout << "                     Xilinx Decompress" << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;
    }
    if (m_list_flow) {
        (m_isSeq) ? std::cout << "KT(MBps)\t" : std::cout << "E2E(MBps)\t";
        (m_compress_flow) ? std::cout << "CR\t" : std::cout << "";
        std::cout << "File Size(MB)\tFile Name" << std::endl;
    } else {
        (m_isSeq) ? std::cout << std::fixed << std::setprecision(2) << "KT(MBps)\t\t:"
                  : std::cout << std::fixed << std::setprecision(2) << "E2E(MBps)\t\t:";
    }
}

void compressApp::printTestSummaryFooter(const std::string& testFile) {
    double input_size = m_inputsize;
    std::string fileUnit = getFileSizeUnit(input_size);
    if (m_list_flow) {
        (m_compress_flow) ? std::cout << "\t\t" << (double)m_inputsize / m_enbytes : std::cout << "";
        std::cout << "\t" << std::fixed << std::setprecision(3);
        (m_compress_flow) ? std::cout << (double)m_inputsize / 1000000 << "\t\t" << testFile
                          : std::cout << "\t" << (double)m_inputsize / 1000000 << "\t\t" << testFile;
        std::cout << std::endl;
    } else {
        (m_compress_flow)
            ? std::cout << std::endl
                        << "CR\t\t\t:" << (double)m_inputsize / m_enbytes << std::fixed << std::setprecision(3)
            : std::cout << "\t";
        std::cout << std::endl;
        std::cout << "File Size(" << fileUnit << ")\t\t:" << input_size << std::endl;
        std::cout << "File Name\t\t:";
        std::cout << testFile;
        std::cout << "\n";
        std::cout << "Output Location: " << m_outFile_name.c_str() << std::endl;
        double enbytes = m_enbytes;
        fileUnit = getFileSizeUnit(enbytes);
        (m_compress_flow) ? std::cout << "Compressed file size(" << fileUnit << ")\t\t:" << enbytes : std::cout << "\t";
        std::cout << std::endl;
    }
}

// compressApp run API: entry point of the program
void compressApp::run(compressBase* b, uint16_t maxCR) {
    b->m_maxCR = maxCR;
    if (!m_uncompressed_file.empty()) {
        runCompress(b, m_uncompressed_file);
    }

    if (!m_compressed_file.empty()) {
        runDecompress(b, m_compressed_file);
    }

    if (!m_uncompressed_filelist.empty() || !m_filelist.empty()) {
        if (!m_filelist.empty()) {
            m_uncompressed_filelist = m_filelist;
        }
        m_list_flow = true;
        m_compress_flow = true;
        printTestSummaryHeader();
        std::vector<std::string> filename_vec;
        getListFilenames(m_uncompressed_filelist, filename_vec);
        for (auto file_itr : filename_vec) {
            m_uncompressed_file = file_itr;
            runCompress(b, file_itr);
        }
    }
    if (!m_compressed_filelist.empty() || !m_filelist.empty()) {
        if (!m_filelist.empty()) {
            m_compressed_filelist = m_filelist;
        }
        m_list_flow = true;
        m_compress_flow = false;
        printTestSummaryHeader();
        std::vector<std::string> filename_vec;
        getListFilenames(m_compressed_filelist, filename_vec);
        for (auto file_itr : filename_vec) {
            if (!m_filelist.empty()) {
                file_itr = file_itr + m_extn;
            }
            runDecompress(b, file_itr);
        }

        if (!m_filelist.empty()) {
            std::string decompressed_file;

            std::cout << "--------------------------------------------------------------" << std::endl;
            std::cout << "                     Validation" << std::endl;
            std::cout << "--------------------------------------------------------------" << std::endl;

            for (auto file_name : filename_vec) {
                decompressed_file = file_name + m_extn + ".orig";
                int ret = validate(file_name, decompressed_file);
                if (ret == 0) {
                    std::cout << (ret ? "FAILED\t" : "PASSED\t") << "\t" << file_name << std::endl;
                } else {
                    std::cout << "Validation Failed: " << file_name << std::endl;
                }
            }
        }
    }
}

// File Compression API
void compressApp::runCompress(compressBase* b, const std::string& m_uncompressed_file) {
    std::ifstream inFile(m_uncompressed_file, std::ifstream::binary);
    inputFilePreCheck(inFile);
    double input_size = m_inputsize;
    double output_size = m_inputsize + ((m_inputsize - 1) / m_stdBSize + 1) * 100;

    std::vector<uint8_t> in(input_size);
    std::vector<uint8_t> out(output_size);

    inFile.read(reinterpret_cast<char*>(in.data()), m_inputsize);
    inFile.close();
    m_compress_flow = true;
    if (!m_list_flow) {
        printTestSummaryHeader();
    }
    std::string fileUnit = getFileSizeUnit(input_size);
    std::string uncompressed_file = m_uncompressed_file;
    m_outFile_name = m_uncompressed_file;
    m_outFile_name += m_extn;

    // Invoking design class virtaul decompress api
    m_enbytes = b->xilCompress(in.data(), out.data(), m_inputsize);

    // Writing compressed data
    std::ofstream outFile(m_outFile_name, std::ofstream::binary);
    if (!outFile) {
        std::cerr << "\nOutfile file directory incorrect!" << std::endl;
        exit(1);
    }
    outFile.write((char*)out.data(), m_enbytes);

    printTestSummaryFooter(uncompressed_file);
    // Close file
    outFile.close();
    if (m_compress_decompress) {
        m_golden_file = m_uncompressed_file;
        runDecompress(b, m_outFile_name);
    }
}

// File Decompress API
void compressApp::runDecompress(compressBase* b, const std::string& m_compressed_file) {
    std::ifstream inFile(m_compressed_file, std::ifstream::binary);
    inputFilePreCheck(inFile);
    double input_size = m_inputsize;
    std::vector<uint8_t> in(m_inputsize);
    std::vector<uint8_t> out;
    uint8_t mcr = std::stoi(m_mcr);
    out.reserve(mcr * m_inputsize);

    inFile.read((char*)in.data(), m_inputsize);
    inFile.close();
    std::string fileUnit = getFileSizeUnit(input_size);
    std::string compressed_file = m_compressed_file;
    m_outFile_name = m_compressed_file;
    m_outFile_name += ".orig";
    m_compress_flow = false;
    if (!m_list_flow) {
        printTestSummaryHeader();
    }
    // Invoking design class virtaul decompress api
    m_debytes = b->xilDecompress(in.data(), out.data(), m_inputsize);

    std::ofstream outFile(m_outFile_name, std::ofstream::binary);

    outFile.write((char*)out.data(), m_debytes);
    // Close file
    outFile.close();
    printTestSummaryFooter(compressed_file);
    if (m_compress_decompress) {
        int ret = validate(m_golden_file, m_outFile_name);
        if (ret == 0) {
            std::cout << (ret ? "FAILED\t" : "PASSED\t") << "\t" << m_golden_file << std::endl;
        } else {
            std::cout << "Validation Failed: " << m_golden_file << std::endl;
        }
    }
}
