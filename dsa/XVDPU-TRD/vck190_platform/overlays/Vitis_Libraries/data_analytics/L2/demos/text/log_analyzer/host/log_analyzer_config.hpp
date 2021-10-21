
/*
 * Copyright 2020 Xilinx, Inc.
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
#pragma once
#ifndef _LOG_ANALYZER_CONFIG_HPP_
#define _LOG_ANALYZER_CONFIG_HPP_

#include <iostream>
#include <thread>
#include <atomic>
#include <algorithm>
#include <queue>
#include <cstring>
#include "x_utils.hpp"

extern "C" {
#include "xf_data_analytics/text/xf_re_compile.h"
}

namespace xf {
namespace search {

enum ErrCode {
    SUCCESS = 0,
    PATTERN_ERR = 1,
    MEM_ERR = 2,
    DEV_ERR = 3,
    OPCODE_ERR = 4,
    INSTR_NM_ERR = 5,
    CCLASS_NM_ERR = 6,
    CPGP_NM_ERR = 7
};

enum { INSTR_DEPTH = 4096, CCLASS_NM = 128, CPGP_NM = 512, REPEAT_CNT = 65536, STACK_DEPTH = 8192, MSG_SZ = 4096 };

enum {
    SLICE_MSG_SZ = 5242880,
    // SLICE_MSG_SZ = 8192,
    MAX_SLC_NM = 256,
    GEO_DB_LNM = 5000000
};

class logAnalyzerConfig {
   private:
    x_utils::MM mm;
    // buffer for configuration
    uint64_t* re_cfg;
    // bitset buffer
    uint32_t* bitset;

    // number of instruction
    uint32_t instr_num = 0;
    // number of character class
    uint32_t cclass_num = 0;
    // number of capture group
    uint32_t cpgp_num = 0;

   public:
    // field name buferr
    uint8_t* fldName;
    // offset buffer of field name
    uint32_t* fldOfft;
    // pattern
    std::string pattern;
    // constructor
    logAnalyzerConfig();
    // compile and check the pattern
    ErrCode compile(std::string pattern);
    // generate config bits
    uint64_t* getConfigBits();
    //
    uint32_t getCpgpNm() const;
};
} // search namespace
} // xf namespace
#endif
