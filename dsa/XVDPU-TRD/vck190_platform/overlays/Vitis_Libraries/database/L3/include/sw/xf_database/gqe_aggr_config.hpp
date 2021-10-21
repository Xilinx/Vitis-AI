/*
 * Copyright 2019 Xilinx, Inc.
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

#ifndef _GQE_AGGR_CONFIG_L3_
#define _GQE_AGGR_CONFIG_L3_
// commmon
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <map>
// HLS
#include <ap_int.h>
// L1
#include "xf_database/enums.hpp"
// L2
#include "xf_database/gqe_utils.hpp"
// L3
#include "xf_database/gqe_table.hpp"

namespace xf {
namespace database {
namespace gqe {

struct EvaluationInfo {
    std::string eval_str;
    std::vector<int> eval_const;
    // int div_scale;
    // EvaluationInfo() : eval_str(""), eval_const({0}), div_scale(0){};
};

class AggrConfig {
   private:
    gqe::utils::MM mm; // memory manager for following pointers.
    ap_uint<32>* table_cfg;
    ap_uint<32>* table_out_cfg;
    ap_uint<512>* table_cfg_part;
    std::vector<int8_t> scan_list;
    std::vector<int8_t> part_list;
    std::vector<bool> write_flag;
    int output_col_num;
    // for convert avg into sum/count
    std::string count_key; // when "" no AOP_COUNT OP
    bool avg_to_sum;

    struct hashGrpInfo {
        std::string real_output_str;
        int real_ouput_ind; // the real sorted ouput
        int merge_ind;      // when kernel done, the index

        int key_info;         // if -1, then aggr_info, else, real_in_ind
        ap_uint<4> aggr_info; // sum:0 avg:1 count:2 min:3 max:4

        // for avg_to_sum
        // route flag if use self value
        // when route -1, values above is valid
        // else use the current location wont set aggregation and will
        // use route
        // when avg_to_sum, it will route the sum location
        int route;
    };
    // init
    std::vector<std::vector<std::string> > ref_col_evals;
    std::vector<std::string> ref_col_filter;
    std::vector<std::string> ref_col_writeout;

    std::map<std::string, std::string> avg_to_sum_map;
    std::map<std::string, struct hashGrpInfo> hashGrpInfoMap;
    std::vector<std::string> read_sort_keys;
    std::vector<std::string> pld_keys;
    std::vector<std::string> key_keys;
    std::vector<std::string> group_keys;
    void CHECK_0(std::vector<std::string> str, size_t len, std::string sinfo);
    std::string getAggrInfo(int key_info);

    // check output non-aggr column if group keys
    void CHECK_1(std::string key);
    // check input contain all group keys
    void CHECK_2(std::vector<std::string> col_names);
    void compressCol(std::vector<int8_t>& shuffle,
                     std::vector<std::string>& col_names,
                     std::vector<std::string> ref_cols);
    void ReplaceAll(std::string& str, const std::string& from, const std::string& to);
    void extractGroupKeys(std::string group_keys_str);
    void extractCompressRefKeys(std::string input_str,
                                std::vector<std::string> col_names,
                                std::vector<std::string>& ref_col);
    ap_uint<4> checkIsAggr(std::string& str);
    void extractWcols(std::string outputs, std::vector<std::string>& col_names, bool avg_to_sum);
    std::vector<int8_t> shuffle(size_t shuf_id,
                                std::string& eval_str, // or filter_str
                                std::vector<std::string>& col_names,
                                std::vector<std::string> strs = {"strm1", "strm2", "strm3", "strm4"},
                                bool replace = true);
    void setPartList(std::vector<std::string> init_col_names);

   public:
    /**
     * @brief construct of AggrConfig.
     *
     * The class generate aggregation configure bits by column names,
     *
     * Input filter_str like "19940101<=o_orderdate && o_orderdate<19950101",
     * o_orderdate and o_orderdate must be exsisted colunm names in input table
     * when no filter conditions, input ""
     *
     * Input evaluation information as a struct EvaluationInfo, creata a valid Evaluation
     * struct using initializer list, e.g.
     * {"l_extendedprice * (-l_discount+c2) / 100", {0, 100}}
     * EvaluationInfo has two members: evaluation string and evaluation constants.
     * In the evaluation string, you can input a final division calculation. Divisor only supports:
     * 10,100,1000,10000
     * In the evaluation constants, input a constant for each column, if no constant, like
     * "l_extendedprice" above, input zero.
     *
     * Input Group keys in a string, like "group_key0, group_key1", use comma as seperator
     *
     * Output strings are like "c0=tab_in_col1, c1=tab_in_col2",
     * when contains several columns, use comma as seperator
     * Usage:
     *
     * \rst
     * ::
     *
     *  AggrConfig aggr_config(tab_in,
     *  		       {{"l_extendedprice * (-l_discount+c2) / 100", {0, 100}}},
     *  		       "l_shipdate<=19980902",
     *  		       "l_returnflag,l_linestatus",
     *  		       "c0=l_returnflag, c1=l_linestatus,c2=sum(eval0),c3=sum(eval1)");
     *
     * \endrst
     *
     * @param tab_in input table
     * @param evals_info Evalutaion information
     * @param filter_str filter condition
     * @param group_keys_str group keys
     * @param output_str output list, output1 = tab_a_col1
     * @param avg_to_sum_ if auto fix the avg config
     *
     */
    AggrConfig(Table tab_in,
               std::vector<EvaluationInfo> evals_info,
               std::string filter_str,
               std::string group_keys_str,
               std::string output_str,
               bool avg_to_sum_ = false);

    /**
     * @brief software shuffle list.
     *
     * @return software shuffle array to adjust the kernel input
     */
    std::vector<int8_t> getScanList() const;
    /**
     * @brief software shuffle list.
     *
     * @return software shuffle array to adjust the kernel input when using partition kernel
     */
    std::vector<int8_t> getPartList() const;
    /**
     * @brief Write out flags.
     *
     * @return the write flags for each column, 0-invalid 1-valid
     */
    std::vector<bool> getWriteFlag() const;
    /**
     * @brief return partition config bits.
     *
     * @return partition config bits
     */
    ap_uint<512>* getPartConfigBits() const;
    /**
     * @brief return aggregation config bits.
     *
     * @return aggregation config bits
     */
    ap_uint<32>* getAggrConfigBits() const;
    ap_uint<32>* getAggrConfigOutBits() const;
    /**
     * @brief return merge info for each output (32bits impl).
     *
     * @return return merge info for column i
     */
    std::vector<int> getResults(int i);
    /**
     * @brief return output column number.
     *
     * @return return output column number
     */
    int getOutputColNum() const;
    /**
     * @brief return group key number.
     *
     * @return return group key number
     */
    int getGrpKeyNum() const;
};
}
}
}

#endif
