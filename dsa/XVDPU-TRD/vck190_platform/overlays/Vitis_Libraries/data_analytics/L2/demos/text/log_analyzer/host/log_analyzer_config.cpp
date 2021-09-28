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
#include "log_analyzer_config.hpp"
namespace xf {
namespace search {
// detailed implementation
logAnalyzerConfig::logAnalyzerConfig() {
    // allocate buffer
    re_cfg = mm.aligned_alloc<uint64_t>(INSTR_DEPTH + CCLASS_NM * 4 + 2 + CPGP_NM + CPGP_NM * 4);
    bitset = mm.aligned_alloc<uint32_t>(CCLASS_NM * 4);
    fldName = mm.aligned_alloc<uint8_t>(CPGP_NM * 256);
    fldOfft = mm.aligned_alloc<uint32_t>(CPGP_NM + 1);
    // initialization
    this->instr_num = 0;
    this->cclass_num = 0;
    this->cpgp_num = 0;
}
ErrCode logAnalyzerConfig::compile(const std::string pattern) {
    this->pattern = pattern;
    // call compiler to compile the pattern
    int r = xf_re_compile(pattern.c_str(), bitset, &re_cfg[2], &instr_num, &cclass_num, &cpgp_num, fldName, fldOfft);
    // leave one more space for match result
    cpgp_num++;
    ErrCode err;
    // check result
    if (r == 0) {
        if (instr_num > INSTR_DEPTH)
            err = INSTR_NM_ERR;
        else if (cclass_num > CCLASS_NM)
            err = CCLASS_NM_ERR;
        else if (cpgp_num > CPGP_NM)
            err = CPGP_NM_ERR;
        else
            err = SUCCESS;
    } else {
        err = PATTERN_ERR;
    }

    return err;
}
uint32_t logAnalyzerConfig::getCpgpNm() const {
    return this->cpgp_num;
}
uint64_t* logAnalyzerConfig::getConfigBits() {
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

    // store field name and field length
    for (unsigned int i = 0; i < cpgp_num - 1; ++i) {
        uint32_t len = fldOfft[i + 1] - fldOfft[i];
        if (i == 0)
            re_cfg[cfg_nm++] = len + 5;
        else
            re_cfg[cfg_nm++] = len + 6;
    }
    // ToDO: need to fix if the length of capture name is longer than 256 bits.
    for (unsigned int i = 0; i < cpgp_num - 1; ++i) {
        std::string fld;
        size_t fld_sz = fldOfft[i + 1] - fldOfft[i];
        fld.assign((char*)(fldName + fldOfft[i]), fld_sz);
        if (i == 0) {
            fld = "{\"" + fld + "\":\"";
        } else {
            fld = "\",\"" + fld + "\":\"";
        }
        memcpy(re_cfg + cfg_nm, fld.c_str(), fld.size());
        cfg_nm += 256 / 64;
    }
    return re_cfg;
}

} // search
} // xf
