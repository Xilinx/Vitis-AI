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

#ifndef _GQE_JOIN_L3_
#define _GQE_JOIN_L3_

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
// gqe_table
#include "xf_database/gqe_join_strategy.hpp"

// gqe_join_config
#include "xf_database/gqe_join_config.hpp"
#include "xf_database/gqe_partjoin_config.hpp"

// meta
#include "xf_database/meta_table.hpp"

namespace xf {
namespace database {
namespace gqe {

class Joiner : public Base {
   private:
    // solutions:
    // direct join: 1 x build + 1 x probe
    ErrCode join_sol0(Table& tab_a, Table& tab_b, Table& tab_c, JoinConfig& jcmd, StrategySet params);

    // pipelined join: 1 x build + pipelined N x probes
    ErrCode join_sol1(Table& tab_a, Table& tab_b, Table& tab_c, JoinConfig& jcmd, StrategySet params);

    // pipelined partition + pipelined join
    ErrCode join_sol2(Table& tab_a, Table& tab_b, Table& tab_c, PartJoinConfig& jcmd, StrategySet params);

   public:
    /**
    * @brief constructor of Joiner.
    *
    * @param obj the FpgaInit instance.
    *
    * Passing FpgaInit obj to Joiner class. Splitting FpgaInit(OpenCL context, program, commandqueue, host/device
    * buffers creation/allocation etc.) and Joiner Init, guaranteens OpenCL stuff are not released after each join call.
    * So the joiner may launch multi-times.
    *
    **/
    Joiner(FpgaInit& obj) : Base(obj){};

    ~Joiner(){};

    /**
    * @brief Run join with the input arguments defined strategy, which includes
    * - solution: the join solution (direct-join or partation-join)
    * - sec_o: left table sec number
    * - sec_l: right table sec number
    * - slice_num: the slice number that used in probe
    * - log_part, the partition number of left/right table
    * - coef_exp_partO: the expansion coefficient of table O result buffer size / input buffer size, this param affects
    * the output buffer size, but not the perf
    * - coef_exp_partL: the expansion coefficient of table L result buffer size / input buffer size, this param affects
    * the output buffer size, but not the perf
    * - coef_exp_join: the expansion coefficient of result buffer size / input buffer size, this param affects the
    * output buffer size, but not the perf
    *
    * Usage:
    *
    * \rst
    * ::
    *
    *   auto smanual = new gqe::JoinStrategyManualSet(solution, sec_o, sec_l, slice_num, log_part, coef_exp_partO,
    * coef_exp_partL, coef_exp_join);
    *
    *   ErrCode err = bigjoin.run(
    *       tab_o, "o_rowid > 0",
    *       tab_l, "",
    *       "o_orderkey = l_orderkey",
    *       tab_c, "c1=l_orderkey, c2=o_rowid, c3=l_rowid",
    *       gqe::INNER_JOIN,
    *       smanual);
    *   delete smanual;
    *
    * \endrst
    *
    * Table tab_o filter condition like "o_rowid > 0", o_rowid is the col name of tab_o
    * when no filter conditions, given empty fitler condition ""
    *
    * The join condition like "left_join_key_0=right_join_key_0"
    * when dual key join is enabled, using comma as the seperator in join condition, e.g.
    * "left_join_key_0=right_join_key_0,left_join_key_1=right_join_key_1"
    *
    * Output strings are like "output_c0 = tab_a_col/tab_b_col",
    * when several columns are output, using comma as the seperator
    *
    *
    * @param tab_a left table
    * @param filter_a filter condition of left table
    * @param tab_b right table
    * @param filter_b filter condition of right table
    * @param join_str join condition(s)
    * @param tab_c result table
    * @param output_str output columns
    * @param join_type INNER_JOIN(default) | SEMI_JOIN | ANTI_JOIN.
    * @param strategyimp pointer to an object of JoinStrategyBase or its derived type.
    *
    *
    */

    ErrCode run(Table& tab_a,
                std::string filter_a,
                Table& tab_b,
                std::string filter_b,
                std::string join_str, // comma seperated
                Table& tab_c,
                std::string output_str, // comma seperated
                int join_type = INNER_JOIN,
                JoinStrategyBase* strategyimp = nullptr);
};
//-----------------------------------------------------------------------------------------------
// put the implementations here for solving compiling deduction

} // gqe
} // database
} // xf
#endif
