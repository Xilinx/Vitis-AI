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

#ifndef BT_TESTCASES_H
#define BT_TESTCASES_H

static const std::string TestCasesFileName = "bt_testcases.csv";
static const std::string TestCasesFileEmulationName = "bt_testcases_emulation.csv";
static const std::string SVGridFileName = "bt_testcases_sv_grid.csv";
static const std::string BinomialTreeEuropeanPutName = "european_put";
static const std::string BinomialTreeEuropeanCallName = "european_call";
static const std::string BinomialTreeAmericanPutName = "american_put";
static const std::string BinomialTreeAmericanCallName = "american_call";

#define BINOMIAL_TESTCASE_NUM_S_GRID_VALUES (7)
#define BINOMIAL_TESTCASE_NUM_V_GRID_VALUES (7)

template <typename DT>
struct BinomialTestCase {
    std::string name;
    DT K;
    DT rf;
    DT T;
    DT N;
};

template <typename DT>
struct BinomialTestSVGrid {
    DT s[BINOMIAL_TESTCASE_NUM_S_GRID_VALUES];
    DT v[BINOMIAL_TESTCASE_NUM_V_GRID_VALUES];
};

#endif // BT_TESTCASES_H