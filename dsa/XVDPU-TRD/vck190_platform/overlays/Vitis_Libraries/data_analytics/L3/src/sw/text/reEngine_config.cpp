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
#include "xf_data_analytics/text/reEngine_config.hpp"

namespace xf {
namespace data_analytics {
namespace text {
namespace re {
// detailed implemetion
reConfig::reConfig(const uint32_t instr_depth, const uint32_t char_class_num, const uint32_t capture_grp_num)
    : kInstrDepth(instr_depth), kCharClassNum(char_class_num), kCaptureGrpNum(capture_grp_num) {
    // allocate buffer
    re_cfg = mm.aligned_alloc<uint64_t>(kInstrDepth + kCharClassNum * 4 + 2);
    bitset = mm.aligned_alloc<uint32_t>(kCharClassNum * 8);
    // initialization
    this->instr_num = 0;
    this->cclass_num = 0;
    this->cpgp_num = 0;
}
ErrCode reConfig::compile(const std::string pattern) {
    this->pattern = pattern;
    // call compiler to compile the pattern
    int r = xf_re_compile(pattern.c_str(), bitset, &re_cfg[2], &instr_num, &cclass_num, &cpgp_num, NULL, NULL);
    // leave one more space for match result
    cpgp_num++;
    ErrCode err;
    // check result
    if (r == 0) {
        if (instr_num > kInstrDepth)
            err = INSTR_NM_ERR;
        else if (cclass_num > kCharClassNum)
            err = CCLASS_NM_ERR;
        else if (cpgp_num > kCaptureGrpNum)
            err = CPGP_NM_ERR;
        else
            err = SUCCESS;
    } else {
        err = PATTERN_ERR;
    }

    return err;
}
uint32_t reConfig::getCpgpNm() const {
    return this->cpgp_num;
}
const uint64_t* reConfig::getConfigBits() {
    uint32_t cfg_nm = 2 + instr_num;
    for (unsigned int i = 0; i < cclass_num * 4; ++i) {
        uint64_t tmp = bitset[i * 2 + 1];
        tmp = tmp << 32;
        tmp += bitset[i * 2];
        re_cfg[cfg_nm++] = tmp;
    }
    typedef union {
        struct {
            uint32_t instr_nm;
            uint16_t cc_nm;
            uint16_t gp_nm;
        } head_st;
        uint64_t d;
    } cfg_info;

    cfg_info cfg_h;
    cfg_h.head_st.instr_nm = instr_num;
    cfg_h.head_st.cc_nm = cclass_num;
    cfg_h.head_st.gp_nm = cpgp_num;

    re_cfg[0] = cfg_nm;
    re_cfg[1] = cfg_h.d;
    return re_cfg;
}
} // namespace re
} // namespace text
} // namespace data_analytics
} // namespace xf
