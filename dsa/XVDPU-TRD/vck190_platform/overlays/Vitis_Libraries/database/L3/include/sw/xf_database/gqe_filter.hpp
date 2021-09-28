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

#ifndef _GQE_FILTER_L3_
#define _GQE_FILTER_L3_

#include <mutex>
#include <unistd.h>
#include <condition_variable>

#include <iostream>
#include <thread>
#include <atomic>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <queue>

// commmon
// for opencl
// helpers for OpenCL C API
#ifndef HLS_TEST
#include "xf_database/gqe_ocl.hpp"
#endif

#include "xf_database/gqe_init.hpp"
#include "xf_database/gqe_base.hpp"
#include "xf_database/gqe_table.hpp"

// gqeFilter config
#include "xf_database/gqe_filter_config.hpp"
// StrategySet struct
#include "xf_database/gqe_join_strategy.hpp"
// Bloom-filter create/build/merge
#include "xf_database/gqe_bloomfilter.hpp"

// meta
#include "xf_database/meta_table.hpp"

namespace xf {
namespace database {
namespace gqe {

class Filter : public Base {
   private:
    // pipelined N x probes
    ErrCode filter_sol(Table& tab_in,
                       Table& tab_out,
                       BloomFilterConfig& fcfg,
                       uint64_t bf_size_in_bits,
                       ap_uint<256>** hash_table,
                       StrategySet params);

   public:
    /**
     * @brief constructor of Filter.
     *
     * Initializes hardware as well as loads binary to FPGA by class Base & FpgaInit
     *
     * @param obj FpgaInit class object
     *
     */
    Filter(FpgaInit& obj) : Base(obj){};

    /**
     * @brief deconstructor of Filter.
     *
     * clProgram, commandQueue, and Context will be released by class Base
     *
     */
    ~Filter(){};

    /**
     * @brief gqeFilter run function.
     *
     * Usage:
     *
     * \rst
     * ::
     *
     *   err_code = Filter.run(
     *       tab_in,
     *       "l_orderkey",
     *       bf_in,
     *       "19940101<=l_orderdate && l_orderdate<19950101",
     *       tab_c1,
     *       "c1=l_extendedprice, c2=l_discount, c3=o_orderdate, c4=l_orderkey",
     *       params);
     *
     * \endrst
     *
     * Input filter_condition like "19940101<=l_orderdate && l_orderdate<19950101",
     * l_orderdate must be exsisted in colunm names of the input table,
     * when no filter conditions, input ""
     *
     * Input key name(s) string like "l_orderkey_0",
     * when enable dual key join, use comma as seperator,
     * "l_orderkey_0, l_orderkey_1"
     *
     * Output mapping is like "output_c0 = tab_in_col",
     * when contains several columns, use comma as seperator
     *
     * @param tab_in input table
     * @param input_str key column names(s) of the input table to be bloom-filtered
     * @param bf_in input bloom-filter from which the hash-table used
     * @param filter_condition filter condition used in dynamic filter
     * @param tab_out result table
     * @param output_str output column mapping
     * @param params StrategySet struct contatins number of sections of the input table. params.sec_l = 0: uses section
     * info from input table; params.sec_l >= 1: separates input table into params.sec_l sections evenly
     * @return error code
     *
     */
    ErrCode run(Table& tab_in,
                std::string input_str, // comma separated
                BloomFilter& bf_in,
                std::string filter_condition,
                Table& tab_out,
                std::string output_str, // comma separated
                StrategySet params);

}; // end class Filter

} // namespace gqe
} // namespace database
} // namespace xf
#endif
