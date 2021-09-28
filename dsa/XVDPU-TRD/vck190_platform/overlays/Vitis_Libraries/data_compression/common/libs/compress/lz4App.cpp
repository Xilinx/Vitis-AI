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
#include "lz4App.hpp"
#include "compressBase.hpp"
#include <unistd.h>
#include <string>
#include <sys/wait.h>
#include <typeinfo>
#include <iomanip>
#include <iostream>

// compressApp Constructor: parse CLI opions and set the driver class memebr variables
lz4App::lz4App(int argc, char** argv, bool is_seq, bool enable_profile)
    : compressApp(argc, argv, is_seq, enable_profile) {
    m_extn = ".lz4";
    m_parser.addSwitch("--block_size", "-B", "Compress Block Size [0-64: 1-256: 2-1024: 3-4096]", "0");
    m_parser.parse(argc, argv);
    m_blocksize = m_parser.value("block_size");
    compressApp::parser(argc, argv);
}

uint32_t lz4App::getBlockSize(void) {
    uint32_t bSize;
    std::map<std::string, uint32_t> blockSet;
    blockSet["0"] = 64;
    blockSet["1"] = 256;
    blockSet["2"] = 1024;
    blockSet["3"] = 4096;

    if (blockSet.find(m_blocksize) == blockSet.end()) {
        std::cout << "Invalid Block Size provided" << std::endl;
        exit(1);
    } else {
        bSize = blockSet[m_blocksize];
        return bSize;
    }
}
