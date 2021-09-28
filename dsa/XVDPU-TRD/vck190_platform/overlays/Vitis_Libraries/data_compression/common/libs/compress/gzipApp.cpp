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
#include "gzipApp.hpp"
#include "compressBase.hpp"
#include <unistd.h>
#include <string>
#include <sys/wait.h>
#include <typeinfo>
#include <iomanip>

// compressApp Constructor: parse CLI opions and set the driver class memebr variables
gzipApp::gzipApp(int argc, char** argv, bool is_seq, bool enable_profile)
    : compressApp(argc, argv, is_seq, enable_profile) {
    m_extn = ".gz";
    m_parser.addSwitch("--zlib", "-zlib", "[0:GZip, 1:Zlib]", "0");
    m_parser.parse(argc, argv);
    m_zlibFlow = (uint8_t)stoi(m_parser.value("zlib"));
    if (m_zlibFlow) m_extn = ".xz";
    compressApp::parser(argc, argv);
}

uint8_t gzipApp::getDesignFlow(void) {
    return m_zlibFlow;
}
