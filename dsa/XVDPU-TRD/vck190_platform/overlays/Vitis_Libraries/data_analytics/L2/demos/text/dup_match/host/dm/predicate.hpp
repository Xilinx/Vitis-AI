/*
 * Copyright 2021 Xilinx, Inc.
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
#ifndef _DUP_MATCH_PREDICATE_HPP_
#define _DUP_MATCH_PREDICATE_HPP_

#include <algorithm>
#include "dm/common.hpp"
#include "utils.hpp"
#include "xcl2.hpp"
#include "predicate_kernel.hpp"
#include "xf_utils_sw/logger.hpp"

namespace dup_match {
namespace internal {

const int CU = 2;

class Predicate {
   public:
   private:
};

class TwoGramPredicate : public Predicate {
   public:
    const int BS = 1024 * 1024 * 256;       // Buffer Size
    static const int RN = 1024 * 1024 * 64; // Record Number
    const int TFLEN = 1024 * 1024 * 32;
    TwoGramPredicate(std::string& xclbinPath, std::vector<std::string>& column, uint32_t* indexId[2]) {
        index(column);
        search(xclbinPath, column, indexId);
    }
    void finish() { queue_->finish(); }

   private:
    cl::CommandQueue* queue_;
    double* idf_value_ = aligned_alloc<double>(4096);
    uint64_t* tf_addr_ = aligned_alloc<uint64_t>(4096);
    uint64_t* tf_value_ = aligned_alloc<uint64_t>(TFLEN);
    char charFilter(char in);
    char charEncode(char in);
    std::string preTwoGram(std::string& inStr);
    void twoGram(std::string& inStr, std::vector<uint16_t>& terms);
    void index(const std::vector<std::string>& column);
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;
    std::vector<cl::Event> events_write;
    std::vector<cl::Event> events_kernel;
    std::vector<cl::Event> events_read;

    void search(std::string& xclbinPath, std::vector<std::string>& column, uint32_t* indexId[2]);
};

class WordPredicate : public Predicate {
   public:
    WordPredicate(std::vector<std::string>& column, std::vector<uint32_t>& indexId) {
        index(column);
        search(column, indexId);
    }

   private:
    std::map<std::string, uint32_t> doc_to_id_;
    std::map<std::string, uint32_t> term_to_id_;
    std::vector<std::vector<udPT> > tf_value_;
    std::vector<double> idf_value_;
    void index(const std::vector<std::string>& column);
    void search(std::vector<std::string>& column, std::vector<uint32_t>& indexId);
};

#include "oniguruma.h"
class AlphaNumPredicate : public Predicate {
   public:
    AlphaNumPredicate(const std::vector<std::string>& column, std::vector<std::vector<std::string> >& blockKey) {
        const std::string pattern = "(?=[a-zA-Z0-9_]*[0-9])[a-zA-Z0-9]+";
        regex_t* reg;
        OnigRegion* region;
        region = onig_region_new();
        OnigErrorInfo einfo;
        OnigEncoding use_encs[1];
        use_encs[0] = ONIG_ENCODING_ASCII;
        onig_initialize(use_encs, sizeof(use_encs) / sizeof(use_encs[0]));
        UChar* pattern_c = (UChar*)pattern.c_str();
        int r = onig_new(&reg, pattern_c, pattern_c + strlen((char*)pattern_c), ONIG_OPTION_DEFAULT,
                         ONIG_ENCODING_ASCII, ONIG_SYNTAX_DEFAULT, &einfo);

        if (r != ONIG_NORMAL) {
            char s[ONIG_MAX_ERROR_MESSAGE_LEN];
            onig_error_code_to_str((UChar*)s, r, &einfo);
            fprintf(stderr, "ERROR: %s\n", s);
            return;
        }
        for (int i = 0; i < column.size(); i++) {
            std::string line_str;
            for (int j = 0; j < column[i].size(); j++) {
                if (checkAlphaNum(column[i][j])) line_str += column[i][j];
            }
            int offset = 0;
            UChar* str = (UChar*)line_str.data();
            while (1) {
                r = onig_search(reg, str, str + line_str.size(), str + offset, str + line_str.size(), region,
                                ONIG_OPTION_NONE);
                if (r >= 0) {
                    blockKey[i].push_back(line_str.substr(region->beg[0], region->end[0] - region->beg[0]));
                    offset = region->end[0];
                } else {
                    break;
                }
            }
        }
        onig_region_free(region, 1 /* 1:free self, 0:free contents only */);
        onig_free(reg);
        onig_end();
    }
};

class StringPredicate : public Predicate {
   public:
    StringPredicate(const std::vector<std::string>& column, std::vector<std::string>& blockKey) {
        blockKey.resize(column.size());
        for (int i = 0; i < column.size(); i++) {
            std::string line_str;
            for (int j = 0; j < column[i].size(); j++) {
                if (checkAlphaNum(column[i][j])) line_str += column[i][j];
            }
            blockKey[i] = line_str;
        }
    }
};

class SimplePredicate : public Predicate {
   public:
    SimplePredicate(const std::vector<std::string>& column, std::vector<std::string>& blockKey) {
        blockKey.resize(column.size());
        for (int i = 0; i < column.size(); i++) {
            blockKey[i] = column[i];
        }
    }
};

} // internal
} // dup_match

#endif
