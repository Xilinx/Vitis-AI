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

#ifndef _GQE_CONFIG_BASE_L3_
#define _GQE_CONFIG_BASE_L3_
// commmon
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <iomanip>
// HLS
#include <ap_int.h>
// L2
#include "xf_database/gqe_utils.hpp"
// L3
#include "xf_database/gqe_table.hpp"

namespace xf {
namespace database {
namespace gqe {

enum { INNER_JOIN = 0, SEMI_JOIN = 1, ANTI_JOIN = 2 };

class BaseConfig {
   protected:
    // memory allocate
    gqe::utils::MM mm;

    void CHECK_0(std::vector<std::string> str, size_t len, std::string sinfo);

    // get join_keys
    std::vector<std::vector<std::string> > extractKeys(std::string join_str);

    // get filter_keys
    std::vector<std::string> extractKey(std::string input_str);

    // get join wr_cols
    std::vector<std::string> extractWcols(std::vector<std::vector<std::string> > join_keys, std::string outputs);

    // get filter wr_cols
    std::vector<std::string> extractWcol(std::string outputs);

    // align the key name from two tables to the same, e.g. o_orderkey, l_orderkey align to o_orderkey
    void AlignColName(std::string& str, const std::string& from, const std::string& to);

    // Calculate the ss position in join_str
    int findStrInd(std::vector<std::string> join_str, std::string ss);

    // sw_shuffle for scan
    void ShuffleScan(std::string& filter_str_,
                     std::vector<std::string> join_keys_,
                     std::vector<std::string> write_out_cols_,
                     std::vector<std::string>& col_names_,
                     std::vector<int8_t>& sw_shuffle_scan_);

    // sw_shuffle for wr
    std::vector<int8_t> ShuffleWrite(std::vector<std::string> col_in_names, std::vector<std::string> col_wr_names);

    void UpdateRowIDCol(std::string& filter_str,
                        std::vector<std::string>& col_names_,
                        std::string rowID_str_,
                        std::vector<std::string> strs_ = {"a", "b", "c", "d"});
};
}
}
}
#endif
