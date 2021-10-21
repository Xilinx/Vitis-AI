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
#include "zstdApp.hpp"
#include "compressBase.hpp"
#include <unistd.h>
#include <string>
#include <sys/wait.h>
#include <typeinfo>
#include <iomanip>

// compressApp Constructor: parse CLI opions and set the driver class memebr variables
zstdApp::zstdApp(int argc, char** argv, bool is_seq, bool enable_profile)
    : compressApp(argc, argv, is_seq, enable_profile) {
    m_extn = ".zst";
    m_parser.parse(argc, argv);
    compressApp::parser(argc, argv);
}
