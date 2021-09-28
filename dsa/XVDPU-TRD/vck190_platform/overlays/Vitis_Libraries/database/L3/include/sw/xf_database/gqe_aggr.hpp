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

#ifndef _GQE_AGGR_L3_
#define _GQE_AGGR_L3_

#include <iostream>
#include <thread>
#include <atomic>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <queue>
// L3
#include "xf_database/gqe_ocl.hpp"
#include "xf_database/gqe_aggr_config.hpp"
#include "xf_database/gqe_aggr_strategy.hpp"

namespace xf {
namespace database {
namespace gqe {
enum DIV_SCALE_1 { NODIV = 0, DIV_10 = 4, DIV_100 = 5, DIV_1K = 6, DIV_10K = 7 };

struct Key {
    int keys[8];
    int key_num = 8;
    bool operator==(const Key& other) const {
        bool re = true;
        for (int i = 0; i < key_num; i++) {
            re &= (keys[i] == other.keys[i]);
        }
        return re;
    }
};

struct Payloads {
    int64_t values[10];
};

struct KeyHasher {
    std::size_t operator()(const xf::database::gqe::Key& k) const {
        using std::size_t;
        using std::hash;
        size_t hash_value = 0;
        for (int i = 0; i < k.key_num; i++) {
            hash_value += hash<int>()(k.keys[i]);
        }
        return hash_value;
    }
};

/**
 * @class Aggregator gqe_aggr.hpp "xf_database/gqe_aggr.hpp"
 */
class Aggregator {
   private:
    // for platform init
    cl_int err;
    cl_context ctx;
    cl_device_id dev_id;
    cl_command_queue cq;
    cl_program prg;
    std::string xclbin_path;
    enum { PU_NM = 8, VEC_SCAN = 8, S_BUFF_DEPTH = (1 << 25) };
    // solutions:
    ErrCode aggr_sol0(Table& tab_in, Table& tab_out, AggrConfig& aggr_cfg, std::vector<size_t> params);
    ErrCode aggr_sol1(Table& tab_in, Table& tab_out, AggrConfig& aggr_cfg, std::vector<size_t> params);
    ErrCode aggr_sol2(Table& tab_in, Table& tab_out, AggrConfig& aggr_cfg, std::vector<size_t> params);
    ErrCode aggr_all(Table& tab_in, Table& tab_out, AggrConfig& aggr_cfg, std::vector<size_t> params);

   public:
    // constructer
    /**
     * @brief construct of Aggregator.
     *
     * @param xclbin xclbin path
     *
     */
    Aggregator(std::string xclbin);

    ~Aggregator();

    /**
     * @brief aggregate function.
     *
     * Usage:
     *
     * \rst
     * ::
     *
     *     err_code = bigaggr.aggregate(tab_l, //input table
     *                                  {{"l_extendedprice * (-l_discount+c2) / 100", {0, 100}},
     *                                   {"l_extendedprice * (-l_discount+c2) * (l_tax+c3) / 10000", {0, 100, 100}}
     *                                  }, // evaluation
     *                                  "l_shipdate<=19980902", //filter
     *                                  "l_returnflag,l_linestatus", // group keys
     *                                  "c0=l_returnflag, c1=l_linestatus,c2=sum(eval0),c3=sum(eval1)", // mapping
     *                                  tab_c, //output table
     *                                  sptr); //strategy
     * \endrst
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
     *
     * StrategyImp class pointer of derived class of AggrStrategyBase.
     *
     *
     * @param tab_in input table
     * @param evals_info Evalutaion information
     * @param filter_str filter condition
     * @param group_keys_str group keys
     * @param out_ptr output list, output1 = tab_a_col1
     * @param tab_out result table
     * @param strategyImp pointer to an object of AggrStrategyBase or its derived type.
     *
     *
     */
    ErrCode aggregate(Table& tab_in,
                      std::vector<EvaluationInfo> evals_info,
                      std::string filter_str,
                      std::string group_keys_str,
                      std::string output_str,
                      Table& tab_out,
                      AggrStrategyBase* strategyImp = nullptr);
};
//-----------------------------------------------------------------------------------------------

} // gqe
} // database
} // xf
#endif
