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
#ifndef XF_TEXT_RE_ENGNINE_CONFIG_H
#define XF_TEXT_RE_ENGNINE_CONFIG_H

#include <iostream>
#include <thread>
#include <atomic>
#include <algorithm>
#include <queue>
#include <cstring>

#include "xf_data_analytics/text/helper.hpp"

extern "C" {
#include "xf_data_analytics/text/xf_re_compile.h"
}

namespace xf {
namespace data_analytics {
namespace text {
namespace re {

/// Error code enumerations
enum ErrCode {
    SUCCESS = 0,       /*!< RE pre-compilation success */
    PATTERN_ERR = 1,   /*!< Invalid RE pattern */
    MEM_ERR = 2,       /*!< CL buffer allocation failed */
    DEV_ERR = 3,       /*!< Kernel creation failed */
    OPCODE_ERR = 4,    /*!< Wrong OP code */
    INSTR_NM_ERR = 5,  /*!< Wrong number of instructions */
    CCLASS_NM_ERR = 6, /*!< Wrong number of character classes */
    CPGP_NM_ERR = 7    /*!< Wrong number of capturing groups */
};

/**
 * @brief Pre-compiling the regular expression and generating configurations under the requirement of ``reEngineKernel``
 * if the pattern is valid
 */
class reConfig {
   private:
    /// aligned buffer allocator
    details::MM mm;
    /// buffer for configuration
    uint64_t* re_cfg;
    /// bit-set map buffer
    uint32_t* bitset;

    /// number of instructions
    uint32_t instr_num = 0;
    /// number of character classes
    uint32_t cclass_num = 0;
    /// number of capture groups
    uint32_t cpgp_num = 0;

    /// Max number of instructions
    const uint32_t kInstrDepth;
    /// Max number of character classes
    const uint32_t kCharClassNum;
    /// Max number of capturing groups
    const uint32_t kCaptureGrpNum;

   public:
    /// Regular expression pattern
    std::string pattern;

    /**
     * @brief Allocates buffers for configurations and bit-set map, and initializes number of
     * instructions/character-classes/capturing-groups.
     *
     * @param instr_depth max number of instructions.
     * @param char_class_num max number of character classes.
     * @param capture_grp_num max number of capturing groups.
     */
    reConfig(const uint32_t instr_depth, const uint32_t char_class_num, const uint32_t capture_grp_num);

    /**
     * @brief Pre-compiles pattern and gives error code correspondingly
     *
     * @param pattern Input regular expression
     */
    ErrCode compile(std::string pattern);
    /**
     * @brief Prepares configurations under the requirement of L2 ``reEngineKernel``
     */
    const uint64_t* getConfigBits();
    /**
     * @brief Gets the number of capturing groups for current pattern
     */
    uint32_t getCpgpNm() const;
};

} // re namespace
} // text namespace
} // data_analytics namespace
} // xf namespace
#endif
