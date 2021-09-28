/**
* Copyright (C) 2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#ifndef _XILINX_AML_CHECKER_HEADER_
#define _XILINX_AML_CHECKER_HEADER_

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <vector>
#include "logger.hpp"

namespace xf {
namespace data_analytics {
namespace text {

const int max_validated_people = 10000000;
const int max_validated_entity = 10000;
const int max_validated_BIC = 100;
const int max_validated_stopword = 200;

struct SwiftMT103 {
    int id;
    std::string company;
    std::string channel;
    std::string operationType;
    std::string contract;
    std::string product;
    std::string productSubtype;
    std::string operationTypeForAML;
    std::string currency;
    std::string amount;
    std::string transactionDescription;
    std::string swiftCode1;
    std::string bank1;
    std::string swiftCode2;
    std::string bank2;
    std::string nombrePersona1;
    std::string tipoPersona1;
    std::string codigoPersona1;
    std::string nombrePersona2;
    std::string tipoPersona2;
    std::string codigoPersona2;
    std::string fechaDeCorte;
    std::string fechaDeValor;
};

struct SwiftMT103CheckResult {
    int id;
    int isMatch;                 // 0-unmatched, 1-matched
    std::vector<int> matchField; // description - swiftCode - bank - nombrePersona
    float timeTaken;

    bool operator==(const SwiftMT103CheckResult& r) {
        return this->id == r.id && this->isMatch == r.isMatch && this->matchField[0] == r.matchField[0] &&
               this->matchField[1] == r.matchField[1] && this->matchField[2] == r.matchField[2] &&
               this->matchField[3] == r.matchField[3] && this->matchField[4] == r.matchField[4] &&
               this->matchField[5] == r.matchField[5];
    }

    bool operator!=(const SwiftMT103CheckResult& r) {
        return this->id != r.id || this->isMatch != r.isMatch || this->matchField[0] != r.matchField[0] ||
               this->matchField[1] != r.matchField[1] || this->matchField[2] != r.matchField[2] ||
               this->matchField[3] != r.matchField[3] || this->matchField[4] != r.matchField[4] ||
               this->matchField[5] != r.matchField[5];
    }
};

// extract select column from CSV file.
// column id starts from 0.
// pass -1 to max_entry_num to get all lines.
int load_csv(const size_t max_entry_num,
             const size_t max_field_len,
             const std::string& file_path,
             const unsigned col,
             std::vector<std::string>& vec_str);

namespace internal {

class SwiftMT103CPUChecker {
   public:
    SwiftMT103CPUChecker() {
        max_fuzzy_len = -1;
        max_contain_len = -1;
        max_equan_len = 12;
    }

    //  initialize the FACTIVA tables and do pre-sort
    int initialize(const std::string& stopKeywordsFileName,
                   const std::string& peopleFileName,
                   const std::string& entitiesFileName,
                   const std::string& BICRefDataFileName);

    // The check method returns whether the transaction is okay, and triggering condition if any.
    SwiftMT103CheckResult check(const SwiftMT103& t);

   protected:
    size_t max_fuzzy_len;
    size_t max_contain_len;
    size_t max_equan_len;

    std::vector<std::vector<std::string> > vec_grp_people =
        std::vector<std::vector<std::string> >(max_people_len_in_char);
    std::vector<std::vector<std::string> > vec_grp_entity =
        std::vector<std::vector<std::string> >(max_entity_len_in_char);
    std::vector<std::string> vec_stopword;
    std::vector<std::string> vec_bic;

    // do one fuzzy process
    bool doFuzzyTask(int id,
                     const size_t upper_limit,
                     const std::string& ptn_str,
                     const std::vector<std::vector<std::string> >& vec_grp);
    // do fuzzy match against given list, return true if matched.
    bool strFuzzy(const size_t upper_limit,
                  const std::string& ptn_str,
                  std::vector<std::vector<std::string> >& vec_grp_str);
    // do equal match against given list, return true if matched.
    bool strEqual(const std::string& code1, const std::string& code2);
    // do string contain against given list, return true if matched.
    bool strContain(const std::string& description);

   private:
    static const size_t max_people_len_in_char = 1024 * 1024; // in char
    static const size_t max_entity_len_in_char = 1024 * 1024; // in char

    std::future<bool> worker[100];
    unsigned int totalThreadNum = std::thread::hardware_concurrency();

}; // end class SwiftMT103CPUChecker
} // namespace internal

class SwiftMT103Checker : public internal::SwiftMT103CPUChecker {
   public:
    SwiftMT103Checker() { this->max_fuzzy_len = 35; }

    // The intialize process will download FPGA binary to FPGA card, and initialize the HBM/DDR FACTIVA tables.
    int initialize(const std::string& xclbinPath,
                   const std::string& stopKeywordsFileName,
                   const std::string& peopleFileName,
                   const std::string& entitiesFileName,
                   const std::string& BICRefDataFileName,
                   int cardID,
                   bool user_setting = false);

    // The check method returns whether the transaction is okay, and triggering condition if any.
    SwiftMT103CheckResult check(const SwiftMT103& t);

   private:
    static const int PU_NUM = 8;
    int boost;
    cl::Context ctx;
    cl::Program prg;
    cl::CommandQueue queue;
    cl::Kernel fuzzy[4];

    int sum_line[2];
    std::vector<std::vector<int> > vec_base = std::vector<std::vector<int> >(2);
    std::vector<std::vector<int> > vec_offset = std::vector<std::vector<int> >(2);

    cl::Buffer buf_field_i1[2];
    cl::Buffer buf_field_i2[2];
    cl::Buffer buf_field_i3[2];
    cl::Buffer buf_field_i4[2];
    cl::Buffer buf_csv[4 * PU_NUM];
    cl::Buffer buf_field_o1[2];
    cl::Buffer buf_field_o2[2];
    cl::Buffer buf_field_o3[2];
    cl::Buffer buf_field_o4[2];

    std::vector<std::vector<std::vector<cl::Event> > > events_write =
        std::vector<std::vector<std::vector<cl::Event> > >(
            2, std::vector<std::vector<cl::Event> >(4, std::vector<cl::Event>(1)));
    std::vector<std::vector<std::vector<cl::Event> > > events_kernel =
        std::vector<std::vector<std::vector<cl::Event> > >(
            2, std::vector<std::vector<cl::Event> >(4, std::vector<cl::Event>(1)));
    std::vector<std::vector<std::vector<cl::Event> > > events_read = std::vector<std::vector<std::vector<cl::Event> > >(
        2, std::vector<std::vector<cl::Event> >(4, std::vector<cl::Event>(1)));

    void preSort(std::vector<std::vector<std::string> >& vec_grp_str1,
                 std::vector<std::vector<std::string> >& vec_grp_str2,
                 std::vector<std::vector<int> >& vec_base,
                 std::vector<std::vector<int> >& vec_offset);

}; // end class SwiftMT103Checker

} // namespace text
} // namespace analytics
} // namespace xf

#endif
