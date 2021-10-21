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

#ifndef _GQE_AGGR_STRATEGY_L3_
#define _GQE_AGGR_STRATEGY_L3_

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cstdio>

// L3
#include "xf_database/gqe_table.hpp"

namespace xf {
namespace database {
namespace gqe {

class AggrStrategyBase {
   public:
    /**
     * @brief construct of AggrStrategyBase.
     *
     */
    AggrStrategyBase(){};
    virtual ~AggrStrategyBase(){};

    /**
     * @brief get solution id and parameters.
     *
     * @param tab_a input table
     *
     */
    virtual std::vector<size_t> getSolutionParams(Table tab_a) { return {0, 0, 0, 0}; };
    //
};
//--------------------------------------------------------------------------------//
// Derived Classes
//--------------------------------------------------------------------------------//

class AggrStrategyManualSet : public AggrStrategyBase {
   private:
    size_t sol;
    size_t sec_l;
    size_t slice_num;
    size_t log_part;

   public:
    AggrStrategyManualSet(){};
    ~AggrStrategyManualSet(){};

    /**
     * @brief construct of AggrStrategyManualSet.
     *
     * derived class of AggrStrategyBase, for set solution and parameters manually
     *
     * @param sol solution id SOL0 | SOL1 | SOL2.
     * @param sec_l section number of input table
     * @param slice_num slice number of probe kernel.
     * @param log_part log number of hash partition.
     *
     */
    AggrStrategyManualSet(size_t _sol, size_t _sec_l, size_t _slice_num, size_t _log_part) {
        sol = _sol;
        sec_l = _sec_l;
        slice_num = _slice_num;
        log_part = _log_part;
    };

    /**
     * @brief get solution id and parameters.
     *
     * @param tab_a table to do aggregation
     *
     */
    std::vector<size_t> getSolutionParams(Table tab_a) {
        std::cout << "create gqe::SOL" << sol << std::endl;
        std::cout << "sec_l:" << sec_l << ", log_part:" << log_part << ", slice_num:" << slice_num << std::endl;
        return {sol, sec_l, slice_num, log_part};
    };
};
//-----------------------------------strategy--------------------------------------------//

} // gqe
} // database
} // xf
#endif
