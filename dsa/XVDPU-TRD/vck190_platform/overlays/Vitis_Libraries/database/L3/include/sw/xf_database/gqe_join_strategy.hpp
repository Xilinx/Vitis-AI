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
#ifndef _GQE_JOIN_STRATEGY_L3_
#define _GQE_JOIN_STRATEGY_L3_

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>
// L3
#include "xf_database/gqe_table.hpp"

namespace xf {
namespace database {
namespace gqe {

enum DATASIZE : int64_t {
    sz_1g = 1024 * 1024 * 1024,
    sz_2g = 2 * sz_1g,
    sz_4g = 4 * sz_1g,
    sz_10g = 10 * sz_1g,
    sz_20g = 20 * sz_1g,
    sz_100g = 100 * sz_1g,
    sz_200g = 200 * sz_1g,
    sz_1000g = 1000 * sz_1g,
    sf1_o_n = 1500000,
    sf1_l_n = 6001215,
    sf8_o_n = 8 * 1500000,
    sf8_l_n = 8 * 6001215,
    sf20_o_n = 20 * 1500000,
    sf20_l_n = 20 * 6001215
};

enum SOLUTION { SOL0 = 0, SOL1 = 1, SOL2 = 2 };

struct StrategySet {
    size_t sol;
    size_t sec_o;
    size_t sec_l;
    size_t slice_num;
    size_t log_part;
    float coef_expansion_partO;
    float coef_expansion_partL;
    float coef_expansion_join;
};

class JoinStrategyBase {
   public:
    /**
     * @brief construct of JoinStrategyBase.
     *
     */
    JoinStrategyBase(){};
    virtual ~JoinStrategyBase(){};

    /**
     * @brief get solution id and parameters.
     *
     * @param tab_a left table
     * @param tab_b right table
     *
     */
    virtual StrategySet getSolutionParams(Table tab_a, Table tab_b) { return StrategySet(); };
    //
};

//--------------------------------------------------------------------------------//
// Derived Classes
//--------------------------------------------------------------------------------//

class JoinStrategyV1 : public JoinStrategyBase {
   public:
    /**
     * @brief construct of JoinStrategyV1.
     *
     */
    JoinStrategyV1(){};
    ~JoinStrategyV1(){};

    /**
     * @brief get solution id and parameters.
     *
     * @param tab_a left table
     * @param tab_b right table
     *
     */
    StrategySet getSolutionParams(Table tab_a, Table tab_b) {
        // int64_t tb0_one_col_sz = tb0_n * type_size_tb0;
        // int64_t tb1_one_col_sz = tb1_n * type_size_tb1;
        size_t tb0_n = tab_a.getRowNum();
        size_t tb1_n = tab_b.getRowNum();
        size_t type_size_tb0 = sizeof(int64_t);
        size_t type_size_tb1 = sizeof(int64_t);
        size_t valid_col_num_tb0 = tab_a.getColNum();
        size_t valid_col_num_tb1 = tab_b.getColNum();

        int64_t tb0_sz = valid_col_num_tb0 * tb0_n * type_size_tb0;
        int64_t tb1_sz = valid_col_num_tb1 * tb1_n * type_size_tb1;
        std::cout << "tab 0 data size: " << (double)tb0_sz / 1024 / 1024 << " MB" << std::endl;
        std::cout << "tab 1 data size: " << (double)tb1_sz / 1024 / 1024 << " MB" << std::endl;

        size_t _solution;
        size_t sec_o;
        size_t sec_l;
        size_t slice_num;
        size_t log_part;
        // when sf1
        if (tb0_n <= sf1_o_n && tb1_n <= sf1_l_n) {
            _solution = 0;
            // not wrok
            slice_num = 1;
            sec_o = 1;
            sec_l = 1;
            log_part = 0;
        } else if (tb0_n <= sf20_o_n) { // when left table  < sf20, we will compare the perf between sol1 and sol2
            if (valid_col_num_tb0 <= 3 &&
                valid_col_num_tb1 <= 3) { // partition maybe gets better perf because partition kernel has its best perf
                if (tb0_n <= sf8_o_n && tb1_n <= sf8_l_n) { // solution 1 is better
                    _solution = 1;
                    slice_num = tb1_n / sf1_l_n;
                    // not wrok
                    sec_o = 1;
                    sec_l = 1;
                    log_part = 0;
                } else { // solution 2 is better
                    _solution = 2;
                    slice_num = 1;
                    sec_o = tb0_n / sf1_o_n;
                    log_part = std::log(sec_o) / std::log(2);
                }
            } else {
                _solution = 1;
                slice_num = tb1_n / sf1_l_n;
                // not wrok
                sec_o = 1;
                sec_l = 1;
                log_part = 0;
            }
        } else {
            _solution = 2;
            slice_num = 1;
            int32_t sec_o_1 = tb0_sz / sz_1g;
            int32_t sec_o_2 = tb0_n / sf1_o_n;
            sec_o = (sec_o_1 > sec_o_2) ? sec_o_1 : sec_o_2;
            log_part = std::log(sec_o) / std::log(2);
        }

#ifdef USER_DEBUG
        std::cout << "create gqe::SOL" << _solution << std::endl;
        std::cout << "sec_o:" << sec_o << ", sec_l:" << sec_l << ", log_part:" << log_part
                  << ", slice_num:" << slice_num << std::endl;
#endif
        StrategySet params;
        params.sol = _solution;
        params.sec_o = sec_o;
        params.sec_l = sec_l;
        params.slice_num = slice_num;
        params.log_part = log_part;

        return params;
    }
};
class JoinStrategyManualSet : public JoinStrategyBase {
   private:
    size_t sol;
    size_t sec_o;
    size_t sec_l;
    size_t slice_num;
    size_t log_part;
    float coef_expansion_partO;
    float coef_expansion_partL;
    float coef_expansion_join;

   public:
    JoinStrategyManualSet(){};
    ~JoinStrategyManualSet(){};

    /**
     * @brief construct of JoinStrategyManualSet.
     *
     * derived class of JoinStrategyBase, for set solution and parameters manually
     *
     * @param sol solution id SOL0 | SOL1 | SOL2.
     * @param sec_o section number of left table
     * @param sec_l section number of right table
     * @param slice_num slice number of probe kernel.
     * @param log_part log number of hash partition.
     * @param _expansion_partO partition O output_buffer_size = _expansion_partO * input_buffer_size
     * @param _expansion_partL partition L output_buffer_size = _expansion_partL * input_buffer_size
     * @param _expansion_join join output_buffer_size = _expansion_join * input_buffer_size
     *
     */
    JoinStrategyManualSet(size_t _sol,
                          size_t _sec_o,
                          size_t _sec_l,
                          size_t _slice_num,
                          size_t _log_part,
                          float _expansion_partO = 2,
                          float _expansion_partL = 2,
                          float _expansion_join = 1) {
        sol = _sol;
        sec_o = _sec_o;
        sec_l = _sec_l;
        slice_num = _slice_num;
        log_part = _log_part;
        coef_expansion_partO = _expansion_partO;
        coef_expansion_partL = _expansion_partL;
        coef_expansion_join = _expansion_join;
    };

    /**
     * @brief get solution id and parameters.
     *
     * @param tab_a left table
     * @param tab_b right table
     *
     */
    StrategySet getSolutionParams(Table tab_a, Table tab_b) {
#ifdef USER_DEBUG
        std::cout << "create gqe::SOL" << sol << std::endl;
        std::cout << "sec_o:" << sec_o << ", sec_l:" << sec_l << ", log_part:" << log_part
                  << ", slice_num:" << slice_num << ", expansion_partO:" << _expansion_partO
                  << " , expansion_partL: " << _expansion_partL << " , expansion_join: " << _expansion_join
                  << std::endl;
#endif

        StrategySet params;

        params.sol = sol;
        params.sec_o = sec_o;
        params.sec_l = sec_l;
        params.slice_num = slice_num;
        params.log_part = log_part;
        params.coef_expansion_partO = coef_expansion_partO;
        params.coef_expansion_partL = coef_expansion_partL;
        params.coef_expansion_join = coef_expansion_join;

        return params;
    };
};

} // gqe
} // database
} // xf
#endif
